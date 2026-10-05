"""Compare trial-balancing strategies for lick/no-lick Facemap decoding.

This is a deliberately small-session experiment.  It loads each session once,
interpolates the Facemap features once, and then applies several trial-selection
strategies to the same feature matrix.  Feature values are never baseline
subtracted or otherwise changed.

The default sessions were chosen from the existing leave-one-block-out results:
one has a large pre-stimulus offset and the other has a large class imbalance.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
from warnings import catch_warnings, filterwarnings

import dr_datacube
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.svm import LinearSVC

from dynamic_routing_facemap.decode_facemap import (
    DEFAULT_FACEMAP_TABLE,
    _interpolate_trials,
)


DEFAULT_SESSIONS = (
    "644866_2023-02-10",
    "741137_2024-10-08",
)
DEFAULT_WINDOW = (-0.25, 0.5)
DEFAULT_SEEDS = tuple(range(5))
N_POSITION_BINS = 5

# Every strategy is an undersampling scheme.  An empty tuple means that only
# response class is matched.  The grouping fields are in addition to block for
# every method whose name begins with ``block_``.
STRATEGIES: dict[str, tuple[str, ...] | None] = {
    "all_trials": None,
    "global_response": (),
    "block_response": ("block_index",),
    "block_position": ("block_index", "position_bin"),
    "block_previous_response": ("block_index", "previous_response"),
    "block_position_previous_response": (
        "block_index",
        "position_bin",
        "previous_response",
    ),
    # This uses special handling in ``run_session``: it has the same per-block
    # response counts as block_stimulus, but the trials are sampled without
    # regard to stimulus identity.  It separates nuisance matching from the
    # large loss of sample size caused by exact stimulus matching.
    "block_response_stimulus_size": ("block_index",),
    "block_stimulus": ("block_index", "stim_name"),
}

STRATEGY_LABELS = {
    "all_trials": "all trials",
    "global_response": "global response",
    "block_response": "block × response",
    "block_position": "block × position",
    "block_previous_response": "block × previous response",
    "block_position_previous_response": "block × position × previous response",
    "block_response_stimulus_size": "block × response (stimulus-size control)",
    "block_stimulus": "block × stimulus",
}


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _prepare_trials(session_id: str) -> pl.DataFrame:
    """Load trials and add nuisance variables known before the current response."""

    trials = (
        dr_datacube.get_lf("trials", session_id=session_id)
        .sort("trial_index")
        .with_columns(
            # Shift before dropping instruction trials so this is genuinely the
            # immediately preceding trial, not merely the preceding retained row.
            previous_response=pl.col("is_response").shift(1).over("block_index"),
        )
        .filter(
            pl.col("is_instruction").eq(False),
            pl.col("is_response").is_not_null(),
            pl.col("stim_start_time").is_not_null(),
            pl.col("stim_start_time").is_finite(),
            pl.col("block_index").is_not_null(),
        )
        .collect()
        .sort("trial_index")
    )

    # Equal-width bins in rank, separately in each block, control slow within-
    # block drift without using Facemap values or the future response label.
    trials = trials.with_columns(
        position_fraction=(
            pl.col("trial_index_in_block").rank("ordinal").over("block_index") - 1
        )
        / pl.len().over("block_index")
    ).with_columns(
        position_bin=(pl.col("position_fraction") * N_POSITION_BINS)
        .floor()
        .clip(0, N_POSITION_BINS - 1)
        .cast(pl.Int8)
    )
    return trials


def _balanced_indices(
    trials: pl.DataFrame,
    strata: Sequence[str],
    *,
    seed: int,
) -> np.ndarray:
    """Return equal lick/no-lick samples within every requested stratum."""

    rng = np.random.default_rng(seed)
    responses = trials["is_response"].to_numpy()
    if strata:
        keys: Iterable[object] = zip(
            *(trials[column].to_list() for column in strata),
            strict=True,
        )
    else:
        keys = [()] * trials.height

    grouped: dict[object, dict[bool, list[int]]] = {}
    for index, (key, response) in enumerate(zip(keys, responses, strict=True)):
        grouped.setdefault(key, {False: [], True: []})[bool(response)].append(index)

    retained: list[int] = []
    for response_indices in grouped.values():
        n = min(len(response_indices[False]), len(response_indices[True]))
        if n == 0:
            continue
        for response in (False, True):
            retained.extend(
                rng.choice(response_indices[response], size=n, replace=False).tolist()
            )
    return np.asarray(sorted(retained), dtype=int)


def _validate_selection(trials: pl.DataFrame, indices: np.ndarray) -> None:
    selected = trials[indices]
    if selected.is_empty():
        raise ValueError("balancing retained no trials")

    counts = (
        selected.group_by("block_index")
        .agg(pl.col("is_response").n_unique().alias("n_classes"))
        .sort("block_index")
    )
    invalid = counts.filter(pl.col("n_classes") != 2)
    if not invalid.is_empty():
        raise ValueError(
            "each held-out block must contain both response classes; invalid blocks: "
            f"{invalid['block_index'].to_list()}"
        )


def _match_block_response_counts(
    trials: pl.DataFrame,
    target_indices: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Sample without stimulus matching but copy target block/class counts."""

    target = trials[target_indices]
    target_counts = {
        (row["block_index"], bool(row["is_response"])): row["len"]
        for row in target.group_by("block_index", "is_response")
        .len()
        .iter_rows(named=True)
    }
    blocks = trials["block_index"].to_list()
    responses = trials["is_response"].to_list()
    available: dict[tuple[object, bool], list[int]] = {}
    for index, (block, response) in enumerate(zip(blocks, responses, strict=True)):
        available.setdefault((block, bool(response)), []).append(index)

    rng = np.random.default_rng(seed)
    retained: list[int] = []
    for key, n in target_counts.items():
        retained.extend(rng.choice(available[key], size=n, replace=False).tolist())
    return np.asarray(sorted(retained), dtype=int)


def _decode_selection(
    features: np.ndarray,
    trials: pl.DataFrame,
    indices: np.ndarray,
    *,
    n_jobs: int,
) -> tuple[list[float], list[float]]:
    """Decode one fixed trial selection and return both score definitions."""

    _validate_selection(trials, indices)
    selected_features = features[indices]
    y = trials["is_response"].cast(pl.Int8).to_numpy()[indices]
    groups = trials["block_index"].to_numpy()[indices]
    splitter = LeaveOneGroupOut()

    accuracy: list[float] = []
    balanced_accuracy: list[float] = []
    for sample_index in range(selected_features.shape[2]):
        X = selected_features[:, :, sample_index]
        classifier = LinearSVC(
            class_weight="balanced",
            max_iter=10_000,
            dual=True,
            random_state=0,
        )
        with catch_warnings():
            filterwarnings("ignore", category=UserWarning)
            fold_scores = cross_validate(
                classifier,
                X,
                y,
                groups=groups,
                cv=splitter,
                scoring={
                    "accuracy": "accuracy",
                    "balanced_accuracy": "balanced_accuracy",
                },
                n_jobs=n_jobs,
            )
        accuracy.append(float(np.mean(fold_scores["test_accuracy"])))
        balanced_accuracy.append(
            float(np.mean(fold_scores["test_balanced_accuracy"]))
        )
    return accuracy, balanced_accuracy


def _selection_summary(trials: pl.DataFrame, indices: np.ndarray) -> dict[str, object]:
    selected = trials[indices]
    fold_counts = (
        selected.group_by("block_index")
        .agg(
            n_trials=pl.len(),
            n_lick=pl.col("is_response").sum(),
        )
        .with_columns(lick_fraction=pl.col("n_lick") / pl.col("n_trials"))
    )
    return {
        "n_trials_retained": selected.height,
        "n_lick_retained": int(selected["is_response"].sum()),
        "n_no_lick_retained": int((~selected["is_response"]).sum()),
        "min_fold_lick_fraction": float(fold_counts["lick_fraction"].min()),
        "max_fold_lick_fraction": float(fold_counts["lick_fraction"].max()),
    }


def run_session(
    session_id: str,
    *,
    seeds: Sequence[int],
    window: tuple[float, float],
    features_to_use: int,
    sampling_rate: float,
    n_jobs: int,
) -> list[dict[str, object]]:
    """Run all balancing strategies for one session."""

    print(f"Loading {session_id}", flush=True)
    trials = _prepare_trials(session_id)
    facemap = dr_datacube.get_lf(DEFAULT_FACEMAP_TABLE, session_id=session_id)
    features = _interpolate_trials(
        trials,
        facemap,
        window=window,
        features_to_use=features_to_use,
        sampling_rate=sampling_rate,
    )
    window_centers = (
        window[0] + (np.arange(features.shape[2]) + 0.5) / sampling_rate
    )

    rows: list[dict[str, object]] = []
    for strategy, strata in STRATEGIES.items():
        strategy_seeds: Sequence[int | None] = (None,) if strata is None else seeds
        for seed in strategy_seeds:
            print(f"  {strategy}, seed={seed}", flush=True)
            if strata is None:
                indices = np.arange(trials.height)
            elif strategy == "block_response_stimulus_size":
                stimulus_matched = _balanced_indices(
                    trials,
                    STRATEGIES["block_stimulus"] or (),
                    seed=int(seed),
                )
                indices = _match_block_response_counts(
                    trials,
                    stimulus_matched,
                    seed=int(seed),
                )
            else:
                indices = _balanced_indices(trials, strata, seed=int(seed))
            accuracy, balanced_accuracy = _decode_selection(
                features,
                trials,
                indices,
                n_jobs=n_jobs,
            )
            scores = np.asarray(accuracy)
            balanced_scores = np.asarray(balanced_accuracy)
            prestim = (window_centers >= window[0]) & (window_centers < 0)
            early = (window_centers >= 0) & (window_centers < 0.2)
            row: dict[str, object] = {
                "session_id": session_id,
                "strategy": strategy,
                "seed": seed,
                "n_trials_available": trials.height,
                **_selection_summary(trials, indices),
                "window_center_s": window_centers.tolist(),
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "prestim_accuracy": float(scores[prestim].mean()),
                "prestim_balanced_accuracy": float(
                    balanced_scores[prestim].mean()
                ),
                "early_accuracy": float(scores[early].mean()),
                "early_balanced_accuracy": float(balanced_scores[early].mean()),
            }
            rows.append(row)
    return rows


def _make_summary(results: pl.DataFrame) -> pl.DataFrame:
    return (
        results.group_by("session_id", "strategy")
        .agg(
            n_resamples=pl.len(),
            n_trials_retained_mean=pl.col("n_trials_retained").mean(),
            n_trials_retained_min=pl.col("n_trials_retained").min(),
            n_trials_retained_max=pl.col("n_trials_retained").max(),
            prestim_accuracy_mean=pl.col("prestim_accuracy").mean(),
            prestim_accuracy_sd=pl.col("prestim_accuracy").std().fill_null(0),
            prestim_balanced_accuracy_mean=pl.col(
                "prestim_balanced_accuracy"
            ).mean(),
            prestim_balanced_accuracy_sd=pl.col(
                "prestim_balanced_accuracy"
            )
            .std()
            .fill_null(0),
            early_accuracy_mean=pl.col("early_accuracy").mean(),
            early_accuracy_sd=pl.col("early_accuracy").std().fill_null(0),
            early_balanced_accuracy_mean=pl.col(
                "early_balanced_accuracy"
            ).mean(),
            early_balanced_accuracy_sd=pl.col("early_balanced_accuracy")
            .std()
            .fill_null(0),
        )
        .sort("session_id", "strategy")
    )


def _plot_results(results: pl.DataFrame, output_path: Path) -> None:
    sessions = results["session_id"].unique(maintain_order=True).to_list()
    colors = dict(
        zip(STRATEGIES, plt.cm.tab10.colors[: len(STRATEGIES)], strict=True)
    )
    figure, axes = plt.subplots(
        len(sessions),
        2,
        figsize=(13, 4.2 * len(sessions)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, session_id in enumerate(sessions):
        session_results = results.filter(pl.col("session_id") == session_id)
        curve_axis, summary_axis = axes[row_index]
        for strategy in STRATEGIES:
            strategy_results = session_results.filter(
                pl.col("strategy") == strategy
            )
            times = np.asarray(strategy_results["window_center_s"][0])
            curves = np.stack(strategy_results["accuracy"].to_list())
            mean = curves.mean(axis=0)
            curve_axis.plot(
                times,
                mean,
                color=colors[strategy],
                label=STRATEGY_LABELS[strategy],
            )
            if curves.shape[0] > 1:
                curve_axis.fill_between(
                    times,
                    mean - curves.std(axis=0),
                    mean + curves.std(axis=0),
                    color=colors[strategy],
                    alpha=0.12,
                    linewidth=0,
                )

        all_trial_results = session_results.filter(
            pl.col("strategy") == "all_trials"
        )
        curve_axis.plot(
            np.asarray(all_trial_results["window_center_s"][0]),
            np.asarray(all_trial_results["balanced_accuracy"][0]),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="all trials, balanced accuracy",
        )

        curve_axis.axhline(0.5, color="0.3", linestyle="--", linewidth=1)
        curve_axis.axvline(0, color="0.3", linestyle=":", linewidth=1)
        curve_axis.set(
            title=f"{session_id}: leave-one-block-out accuracy",
            xlabel="Time from stimulus (s)",
            ylabel="Accuracy",
            xlim=DEFAULT_WINDOW,
        )
        curve_axis.legend(fontsize=8, ncol=2)

        strategy_names = list(STRATEGIES)
        means = []
        errors = []
        retained = []
        for strategy in strategy_names:
            values = session_results.filter(pl.col("strategy") == strategy)
            means.append(values["prestim_accuracy"].mean())
            errors.append(values["prestim_accuracy"].std() or 0)
            retained.append(values["n_trials_retained"].mean())
        x = np.arange(len(strategy_names))
        summary_axis.errorbar(
            x,
            means,
            yerr=errors,
            marker="o",
            linestyle="none",
            capsize=3,
            color="black",
        )
        summary_axis.axhline(0.5, color="0.3", linestyle="--", linewidth=1)
        summary_axis.set_xticks(
            x,
            [
                f"{STRATEGY_LABELS[name]}\n(n={retained[i]:.0f})"
                for i, name in enumerate(strategy_names)
            ],
            rotation=35,
            ha="right",
        )
        summary_axis.set(
            title="Mean pre-stimulus accuracy (error: resample SD)",
            ylabel="Accuracy",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions",
        type=_parse_csv,
        default=DEFAULT_SESSIONS,
        help="comma-separated session IDs",
    )
    parser.add_argument(
        "--seeds",
        type=lambda value: tuple(int(seed) for seed in _parse_csv(value)),
        default=DEFAULT_SEEDS,
        help="comma-separated random seeds for undersampling (default: 0,1,2,3,4)",
    )
    parser.add_argument("--features", type=int, default=200)
    parser.add_argument("--sampling-rate", type=float, default=60.0)
    parser.add_argument(
        "--jobs",
        type=int,
        default=6,
        help="parallel held-out-block fits per time point (default: 6)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/trial_balancing"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows: list[dict[str, object]] = []
    for session_id in args.sessions:
        rows.extend(
            run_session(
                session_id,
                seeds=args.seeds,
                window=DEFAULT_WINDOW,
                features_to_use=args.features,
                sampling_rate=args.sampling_rate,
                n_jobs=args.jobs,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pl.DataFrame(rows)
    summary = _make_summary(results)
    results.write_parquet(args.output_dir / "trial_balancing_curves.parquet")
    summary.write_csv(args.output_dir / "trial_balancing_summary.csv")
    _plot_results(results, args.output_dir / "trial_balancing_comparison.png")
    print("\nSummary")
    print(summary)


if __name__ == "__main__":
    main()
