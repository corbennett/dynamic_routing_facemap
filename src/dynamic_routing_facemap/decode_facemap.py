# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dr-datacube @ git+https://github.com/AllenNeuralDynamics/dr-datacube.git@366809a4b2b3f39a7f36e054a4650dd79faadbe2",
#     "numpy>=2.4.6",
#     "polars>=1.44.2",
#     "scikit-learn>=1.9.1",
#     "scipy>=1.17.1",
#     "tqdm>=4.67.1",
# ]
# ///

"""Decode lick versus no-lick trials from side-camera Facemap data.

The main entry point, :func:`decode_all_sessions`, mirrors the workflow in
``scratch.ipynb`` but applies it to every session in a trials LazyFrame.  The
returned table has one row per session.  The CV accuracy for each sliding
window is stored in the ``cv_scores`` list column alongside the corresponding
window times and session-level behavior metadata.

The module can also be run as a command-line program::

    python -m dynamic_routing_facemap.decode_facemap \
        --output results/facemap_lick_decoding.parquet

The datacube configuration is controlled by ``dr_datacube`` in the same way as
in the notebook.  In particular, the default query is the default
behavior-passing brainwide session set.  Use ``--all-sessions`` to query all
session types without the behavior filter.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing as mp
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from warnings import catch_warnings, filterwarnings

import dr_datacube
import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.svm import LinearSVC
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)

DEFAULT_FACEMAP_TABLE = "processing/behavior/facemap_side_camera"
DEFAULT_WINDOW = (-0.25, 0.5)
DEFAULT_FEATURES = 200
DEFAULT_SAMPLING_RATE = 60.0
DEFAULT_WINDOW_LENGTH = 1

STIMULUS_MODALITY_BOOLEAN_COLUMNS = {
    "visual": "is_vis_stim",
    "auditory": "is_aud_stim",
}

# These are the behavior fields currently exposed by the Dynamic Routing
# performance table.  The code checks the schema before selecting them so it
# remains usable if a datacube version omits one of the fields.
BEHAVIOR_METRIC_COLUMNS = (
    "hit_rate",
    "false_alarm_rate",
    "cross_modality_dprime",
    "aud_dprime",
    "vis_dprime",
    "n_contingent_rewards",
)


def _collect(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Collect a LazyFrame while also accepting an already-collected frame."""

    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def _require_columns(frame: pl.DataFrame, required: Sequence[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalize_stimulus_modality(stimulus_modality: str | None) -> str | None:
    """Normalize the public visual/auditory modality selector."""

    if stimulus_modality is None:
        return None

    aliases = {
        "vis": "visual",
        "visual": "visual",
        "aud": "auditory",
        "audio": "auditory",
        "auditory": "auditory",
    }
    normalized = aliases.get(stimulus_modality.strip().lower())
    if normalized is None:
        raise ValueError(
            "stimulus_modality must be one of visual or auditory, "
            f"got {stimulus_modality!r}"
        )
    return normalized


def _filter_stimulus_modality(
    trials: pl.DataFrame,
    stimulus_modality: str | None,
) -> pl.DataFrame:
    """Keep only trials from one stimulus modality when requested.

    Raise instead of silently returning all trials if the expected modality
    field cannot be identified.
    """

    modality = _normalize_stimulus_modality(stimulus_modality)
    if modality is None:
        return trials

    column = STIMULUS_MODALITY_BOOLEAN_COLUMNS[modality]
    if column in trials.columns:
        return trials.filter(pl.col(column).cast(pl.Boolean, strict=False).eq(True))

    raise ValueError(
        "stimulus_modality was requested, but no recognizable stimulus "
        f"modality column {column!r} was found; available "
        f"trial columns are {trials.columns}"
    )


def _trial_metadata(trials: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Return trial counts and lick fraction keyed by session ID."""

    non_instruction = trials.filter(
        pl.col("is_instruction").eq(False),
        pl.col("is_response").is_not_null(),
    )
    counts = (
        non_instruction.with_columns(
            is_lick=pl.col("is_response").cast(pl.Int64, strict=False),
        )
        .group_by("session_id")
        .agg(
            n_behavior_trials=pl.len(),
            n_lick_trials=pl.col("is_lick").sum(),
            n_no_lick_trials=(pl.lit(1) - pl.col("is_lick")).sum(),
        )
        .with_columns(
            lick_fraction=pl.when(pl.col("n_behavior_trials") > 0)
            .then(pl.col("n_lick_trials") / pl.col("n_behavior_trials"))
            .otherwise(None),
        )
    )

    return {row["session_id"]: row for row in counts.to_dicts()}


def _behavior_metadata(
    performance: pl.DataFrame,
    session_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Aggregate block-level behavior metrics to one row per session."""

    if performance.is_empty():
        return {}

    _require_columns(performance, ["session_id"], "performance")
    performance = performance.filter(pl.col("session_id").is_in(session_ids))
    if performance.is_empty():
        return {}

    available_metrics = [
        column for column in BEHAVIOR_METRIC_COLUMNS if column in performance.columns
    ]
    aggregations: list[pl.Expr] = [pl.len().alias("behavior_n_blocks")]

    for column in available_metrics:
        numeric = pl.col(column).cast(pl.Float64, strict=False)
        aggregations.append(numeric.mean().alias(f"behavior_{column}_mean"))

        if column == "n_contingent_rewards":
            aggregations.append(numeric.sum().alias("behavior_n_contingent_rewards_sum"))

    if "cross_modality_dprime" in performance.columns:
        aggregations.append(
            pl.col("cross_modality_dprime")
            .cast(pl.Float64, strict=False)
            .ge(1.0)
            .sum()
            .alias("behavior_n_good_blocks")
        )
    if "n_contingent_rewards" in performance.columns:
        aggregations.append(
            pl.col("n_contingent_rewards")
            .cast(pl.Float64, strict=False)
            .ge(10.0)
            .sum()
            .alias("behavior_n_engaged_blocks")
        )

    summary = performance.group_by("session_id").agg(aggregations)
    return {row["session_id"]: row for row in summary.to_dicts()}


def _facemap_array(
    facemap: pl.DataFrame,
    *,
    features_to_use: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the timestamp and list-of-feature columns to sorted arrays."""

    _require_columns(facemap, ["timestamps", "data"], "facemap")
    if facemap.is_empty():
        raise ValueError("the Facemap table has no samples in the requested time range")

    timestamps = np.asarray(facemap["timestamps"].to_numpy(), dtype=float)
    data = np.asarray(facemap["data"].to_list(), dtype=float)
    if data.ndim != 2:
        raise ValueError(f"expected Facemap data to be 2-D, got shape {data.shape}")
    if data.shape[1] < features_to_use:
        raise ValueError(
            f"requested {features_to_use} Facemap features, but only "
            f"{data.shape[1]} are available"
        )

    finite = np.isfinite(timestamps)
    timestamps = timestamps[finite]
    data = data[finite]
    if timestamps.size < 2:
        raise ValueError("at least two finite Facemap timestamps are required")

    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    data = data[order, :features_to_use]

    # interp1d requires strictly increasing x values.  Keeping the first
    # sample at duplicate timestamps is deterministic and matches the usual
    # interpretation of a sampled camera stream.
    timestamps, unique_indices = np.unique(timestamps, return_index=True)
    data = data[unique_indices]
    if timestamps.size < 2:
        raise ValueError("at least two unique Facemap timestamps are required")
    return timestamps, data


def _interpolate_trials(
    condition_trials: pl.DataFrame,
    facemap_lf: pl.LazyFrame | pl.DataFrame,
    *,
    window: tuple[float, float],
    features_to_use: int,
    sampling_rate: float,
) -> np.ndarray:
    """Interpolate Facemap features into trial-aligned arrays."""

    if condition_trials.is_empty():
        raise ValueError("there are no trials for this decoding condition")

    stim_start = np.asarray(condition_trials["stim_start_time"].to_numpy(), dtype=float)
    stim_start = stim_start[np.isfinite(stim_start)]
    if stim_start.size == 0:
        raise ValueError("the decoding trials have no finite stim_start_time values")

    n_samples = int(round((window[1] - window[0]) * sampling_rate))
    if n_samples < 2:
        raise ValueError("the capture window must contain at least two samples")

    trial_starts = stim_start + window[0]
    target_times = trial_starts[:, None] + np.arange(n_samples)[None, :] / sampling_rate
    signal_start = float(trial_starts.min())
    signal_end = float(stim_start.max() + window[1])

    facemap = _collect(
        facemap_lf.filter(
            pl.col("timestamps").is_between(signal_start, signal_end, closed="both")
        ).select("timestamps", "data")
        if isinstance(facemap_lf, pl.LazyFrame)
        else facemap_lf.filter(
            pl.col("timestamps").is_between(signal_start, signal_end, closed="both")
        ).select("timestamps", "data")
    )
    timestamps, data = _facemap_array(facemap, features_to_use=features_to_use)

    interpolator = interp1d(
        timestamps,
        data,
        axis=0,
        bounds_error=False,
        # Preserve the notebook behavior for trials near the edge of the
        # available camera recording: clamp to the nearest recorded sample.
        fill_value=(data[0], data[-1]),
    )
    # Shape: trials x features x samples.
    return interpolator(target_times).transpose(0, 2, 1)


def decode_session(
    trials: pl.DataFrame,
    facemap_lf: pl.LazyFrame | pl.DataFrame,
    *,
    stimulus_modality: str | None = None,
    window: tuple[float, float] = DEFAULT_WINDOW,
    features_to_use: int = DEFAULT_FEATURES,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    window_length: int = DEFAULT_WINDOW_LENGTH,
) -> dict[str, Any]:
    """Decode one session and return its windowed CV scores.

    The classifier and scoring behavior intentionally match the notebook:
    unscaled Facemap features, a class-balanced ``LinearSVC``, and accuracy
    scoring. Cross-validation leaves out one task block at a time, using the
    six ``block_index`` values as groups.
    """

    _require_columns(
        trials,
        [
            "session_id",
            "is_instruction",
            "is_response",
            "stim_start_time",
            "block_index",
        ],
        "trials",
    )
    if window[1] <= window[0]:
        raise ValueError(f"window end must be greater than start, got {window}")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if window_length < 1:
        raise ValueError("window_length must be at least one sample")

    decoding_trials = trials.filter(
        pl.col("is_instruction").eq(False),
        pl.col("is_response").is_not_null(),
        pl.col("stim_start_time").is_not_null(),
        pl.col("stim_start_time").is_finite(),
        pl.col("block_index").is_not_null(),
    )
    decoding_trials = _filter_stimulus_modality(
        decoding_trials,
        stimulus_modality,
    )
    lick_trials = decoding_trials.filter(pl.col("is_response").eq(True))
    no_lick_trials = decoding_trials.filter(pl.col("is_response").eq(False))

    unique_blocks = np.unique(decoding_trials["block_index"].to_numpy())
    if unique_blocks.size < 2:
        raise ValueError(
            "leave-one-block-out cross-validation requires at least two blocks "
            f"with decodable trials (found {unique_blocks.size})"
        )

    # Every training split must retain both response classes. Check this
    # before fitting so incomplete block structures produce a clear
    # session-level error instead of an invalid LinearSVC score.
    block_class_counts = (
        decoding_trials
        .with_columns(is_lick=pl.col("is_response").cast(pl.Int8))
        .group_by("block_index")
        .agg(
            n_lick=pl.col("is_lick").sum(),
            n_no_lick=(pl.lit(1) - pl.col("is_lick")).sum(),
        )
    )
    total_lick = int(lick_trials.height)
    total_no_lick = int(no_lick_trials.height)
    invalid_training_blocks = block_class_counts.filter(
        (pl.lit(total_lick) - pl.col("n_lick") <= 0)
        | (pl.lit(total_no_lick) - pl.col("n_no_lick") <= 0)
    )
    if not invalid_training_blocks.is_empty():
        raise ValueError(
            "each leave-one-block-out training split needs both lick and "
            "no-lick trials; the following held-out blocks leave one class "
            f"absent: {invalid_training_blocks['block_index'].to_list()}"
        )

    # Read the camera samples once per session.  Each condition is then
    # filtered from this in-memory slice, avoiding two remote reads per
    # session.
    stim_start = np.asarray(decoding_trials["stim_start_time"].to_numpy(), dtype=float)
    finite_stim_start = stim_start[np.isfinite(stim_start)]
    if finite_stim_start.size == 0:
        raise ValueError("the session has no finite stim_start_time values")
    signal_start = float(finite_stim_start.min() + window[0])
    signal_end = float(finite_stim_start.max() + window[1])
    facemap = _collect(
        facemap_lf.filter(
            pl.col("timestamps").is_between(signal_start, signal_end, closed="both")
        ).select("timestamps", "data")
        if isinstance(facemap_lf, pl.LazyFrame)
        else facemap_lf.filter(
            pl.col("timestamps").is_between(signal_start, signal_end, closed="both")
        ).select("timestamps", "data")
    )

    condition_arrays = {
        "lick": _interpolate_trials(
            lick_trials,
            facemap,
            window=window,
            features_to_use=features_to_use,
            sampling_rate=sampling_rate,
        ),
        "no_lick": _interpolate_trials(
            no_lick_trials,
            facemap,
            window=window,
            features_to_use=features_to_use,
            sampling_rate=sampling_rate,
        ),
    }

    n_samples = condition_arrays["lick"].shape[2]
    if window_length > n_samples:
        raise ValueError(
            f"window_length ({window_length}) exceeds the {n_samples} samples "
            "in the capture window"
        )

    y = np.concatenate(
        [
            np.ones(condition_arrays["lick"].shape[0], dtype=int),
            np.zeros(condition_arrays["no_lick"].shape[0], dtype=int),
        ]
    )
    # X and y are concatenated in lick/no-lick order above, so groups must
    # use that same order rather than the original trial-table order.
    block_indices = np.concatenate(
        [
            lick_trials["block_index"].to_numpy(),
            no_lick_trials["block_index"].to_numpy(),
        ]
    )

    scores: list[float] = []
    for window_start in range(n_samples - window_length + 1):
        X = np.concatenate(
            [
                condition_arrays["lick"][:, :, window_start : window_start + window_length],
                condition_arrays["no_lick"][:, :, window_start : window_start + window_length],
            ],
            axis=0,
        )
        X_window = X.reshape(X.shape[0], -1)

        # The notebook suppresses LinearSVC convergence warnings.  Increase
        # max_iter enough for batch processing. ``balanced`` gives each class
        # inverse-frequency weight within each training fold, so the more
        # numerous class does not dominate the SVM objective.
        classifier = LinearSVC(
            class_weight="balanced",
            max_iter=10_000,
            # The high-dimensional windowed features are substantially faster
            # with the dual formulation, including when window_length=1.
            dual=True,
        )
        with catch_warnings():
            filterwarnings("ignore", category=UserWarning)
            fold_scores = cross_val_score(
                classifier,
                X_window,
                y,
                groups=block_indices,
                cv=LeaveOneGroupOut(),
                scoring="accuracy",
            )
        scores.append(float(fold_scores.mean()))

    window_starts = window[0] + np.arange(len(scores)) / sampling_rate
    window_ends = window_starts + window_length / sampling_rate
    window_centers = window[0] + (
        np.arange(len(scores)) + window_length / 2
    ) / sampling_rate

    return {
        "cv_scores": scores,
        "window_start_s": window_starts.tolist(),
        "window_end_s": window_ends.tolist(),
        "window_center_s": window_centers.tolist(),
        "cv_score_mean": float(np.mean(scores)),
        "cv_score_max": float(np.max(scores)),
        "best_window_center_s": float(window_centers[int(np.argmax(scores))]),
        "cv_folds": int(unique_blocks.size),
    }


def _base_result_row(
    session_id: str,
    trials: pl.DataFrame,
    trial_metadata: Mapping[str, Any],
    behavior_metadata: Mapping[str, Any],
    *,
    stimulus_modality: str | None,
    window: tuple[float, float],
    features_to_use: int,
    sampling_rate: float,
    window_length: int,
) -> dict[str, Any]:
    """Build the session metadata shared by successful and failed sessions."""

    row: dict[str, Any] = {
        "session_id": session_id,
        "n_trials": trials.height,
        "n_behavior_trials": trial_metadata.get("n_behavior_trials", 0),
        "n_lick_trials": trial_metadata.get("n_lick_trials", 0),
        "n_no_lick_trials": trial_metadata.get("n_no_lick_trials", 0),
        "lick_fraction": trial_metadata.get("lick_fraction"),
        "stimulus_modality_filter": stimulus_modality,
        "window_capture_start_s": window[0],
        "window_capture_end_s": window[1],
        "sampling_rate_hz": sampling_rate,
        "features_used": features_to_use,
        "window_length_samples": window_length,
        # Filled by decode_session after the available task blocks are known.
        "cv_folds": None,
        "status": "success",
        "error_type": None,
        "error_message": None,
    }
    row.update(behavior_metadata)
    return row


def _new_checkpoint_dir(output_path: Path) -> Path:
    """Create a unique checkpoint directory for one decoding run."""

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_id}-{uuid4().hex[:8]}"
    checkpoint_dir = output_path.with_name(f"{output_path.stem}.checkpoints") / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    return checkpoint_dir


def _write_parquet_atomically(frame: pl.DataFrame, output_path: Path) -> None:
    """Write a Parquet part so an interruption cannot expose a partial part."""

    temporary_path = output_path.with_suffix(".tmp")
    frame.write_parquet(temporary_path)
    temporary_path.replace(output_path)


def _decode_one_session(
    index: int,
    session_id: str,
    trials: pl.DataFrame,
    trial_metadata: Mapping[str, Mapping[str, Any]],
    behavior_metadata: Mapping[str, Mapping[str, Any]],
    *,
    facemap_table: str,
    datacube_kwargs: Mapping[str, Any],
    stimulus_modality: str | None,
    window: tuple[float, float],
    features_to_use: int,
    sampling_rate: float,
    window_length: int,
    fail_fast: bool,
    get_lf: Callable[..., pl.LazyFrame],
) -> dict[str, Any]:
    """Decode one session in a process-pool worker."""

    LOGGER.info("Decoding session %d: %s", index, session_id)
    session_trials = trials.filter(pl.col("session_id").eq(session_id))
    row = _base_result_row(
        session_id,
        session_trials,
        trial_metadata.get(session_id, {}),
        behavior_metadata.get(session_id, {}),
        stimulus_modality=stimulus_modality,
        window=window,
        features_to_use=features_to_use,
        sampling_rate=sampling_rate,
        window_length=window_length,
    )

    try:
        facemap_lf = get_lf(
            facemap_table,
            session_id=session_id,
            **datacube_kwargs,
        )
        row.update(
            decode_session(
                session_trials,
                facemap_lf,
                stimulus_modality=stimulus_modality,
                window=window,
                features_to_use=features_to_use,
                sampling_rate=sampling_rate,
                window_length=window_length,
            )
        )
    except Exception as exc:  # one problematic session should not stop a batch
        if fail_fast:
            raise
        LOGGER.exception("Could not decode session %s", session_id)
        row.update(
            status="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            cv_scores=None,
            window_start_s=None,
            window_end_s=None,
            window_center_s=None,
            cv_score_mean=None,
            cv_score_max=None,
            best_window_center_s=None,
        )
    return row


def decode_all_sessions(
    trials_lf: pl.LazyFrame | pl.DataFrame | None = None,
    *,
    performance_lf: pl.LazyFrame | pl.DataFrame | None = None,
    output_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    facemap_table: str = DEFAULT_FACEMAP_TABLE,
    session_type: str | Sequence[str] | None = "brainwide",
    with_behavior_filter: bool = True,
    only_in_data_asset: bool = True,
    stimulus_modality: str | None = None,
    window: tuple[float, float] = DEFAULT_WINDOW,
    features_to_use: int = DEFAULT_FEATURES,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    fail_fast: bool = False,
    parallelize_sessions: bool = False,
    max_workers: int | None = None,
    get_lf: Callable[..., pl.LazyFrame] = dr_datacube.get_lf,
) -> pl.DataFrame:
    """Decode every unique session in ``trials_lf``.

    Parameters
    ----------
    trials_lf:
        Trials LazyFrame or DataFrame.  If omitted, it is loaded from
        ``dr_datacube.get_lf("trials", ...)``.
    performance_lf:
        Optional performance LazyFrame or DataFrame.  If omitted, the
        datacube performance table is loaded and aggregated by session.
    output_path:
        Optional final Parquet destination.  The output contains one row per
        input session, including failed sessions with ``status="error"``.
        When supplied, a separate checkpoint Parquet part is written after
        every session under ``<output stem>.checkpoints``.
    checkpoint_dir:
        Optional directory for per-session checkpoint Parquet parts.  If not
        supplied, a unique run directory is created beside ``output_path``.
        Checkpoint parts are retained after the final consolidated output is
        written so an interrupted run can be recovered.
    fail_fast:
        Raise on the first session-level error instead of recording it and
        continuing with the other sessions.
    stimulus_modality:
        Optional stimulus filter.  Set to ``"visual"`` or ``"auditory"`` to
        decode only trials from that stimulus modality.  The default includes
        both modalities.
    parallelize_sessions:
        Decode sessions concurrently. Defaults to ``False`` so the default
        behavior remains sequential and uses one session at a time.
    max_workers:
        Maximum number of concurrent session decodes when
        ``parallelize_sessions`` is true. If omitted, the executor default is
        used. This option has no effect for sequential decoding.
    get_lf:
        Dependency-injection hook useful for tests or a local data loader.  A
        custom loader must be picklable when ``parallelize_sessions`` is true.
    """

    datacube_kwargs = {
        "session_type": session_type,
        "with_behavior_filter": with_behavior_filter,
        "only_in_data_asset": only_in_data_asset,
    }
    if trials_lf is None:
        trials_lf = get_lf("trials", **datacube_kwargs)
    trials = _collect(trials_lf)
    _require_columns(
        trials,
        [
            "session_id",
            "is_instruction",
            "is_response",
            "stim_start_time",
            "block_index",
        ],
        "trials",
    )
    stimulus_modality = _normalize_stimulus_modality(stimulus_modality)
    trials = _filter_stimulus_modality(trials, stimulus_modality)

    session_ids = (
        trials.select("session_id")
        .drop_nulls()
        .unique()
        .sort("session_id")["session_id"]
        .to_list()
    )
    if not session_ids:
        raise ValueError("trials contains no non-null session_id values")

    if performance_lf is None:
        performance_lf = get_lf("performance", **datacube_kwargs)
    performance = _collect(performance_lf)

    trial_metadata = _trial_metadata(trials)
    behavior_metadata = _behavior_metadata(performance, session_ids)
    output = Path(output_path) if output_path is not None else None
    if checkpoint_dir is not None:
        checkpoint = Path(checkpoint_dir)
        checkpoint.mkdir(parents=True, exist_ok=True)
    elif output is not None:
        checkpoint = _new_checkpoint_dir(output)
    else:
        checkpoint = None

    if checkpoint is not None:
        LOGGER.info("Writing per-session checkpoints to %s", checkpoint)

    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be at least one")

    rows: list[dict[str, Any] | None] = [None] * len(session_ids)

    def save_result(index: int, row: dict[str, Any]) -> None:
        rows[index - 1] = row
        if checkpoint is not None:
            part_path = checkpoint / f"part-{index:06d}.parquet"
            _write_parquet_atomically(pl.DataFrame([row]), part_path)
            LOGGER.info("Wrote checkpoint for session %s to %s", row["session_id"], part_path)

    # Futures may finish out of order, but the consolidated result and stable
    # checkpoint names retain the sorted session order.
    with tqdm(
        total=len(session_ids),
        desc="Decoding sessions",
        unit="session",
        file=sys.stdout,
    ) as progress:
        if parallelize_sessions:
            # Polars uses native threads and must not be forked on Unix.  The
            # explicit spawn context is required on Linux, whose default is
            # fork.  The worker is module-level so it is picklable by spawn.
            spawn_context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=spawn_context,
            ) as executor:
                futures = {
                    executor.submit(
                        _decode_one_session,
                        index,
                        session_id,
                        trials,
                        trial_metadata,
                        behavior_metadata,
                        facemap_table=facemap_table,
                        datacube_kwargs=datacube_kwargs,
                        stimulus_modality=stimulus_modality,
                        window=window,
                        features_to_use=features_to_use,
                        sampling_rate=sampling_rate,
                        window_length=window_length,
                        fail_fast=fail_fast,
                        get_lf=get_lf,
                    ): (index, session_id)
                    for index, session_id in enumerate(session_ids, start=1)
                }
                try:
                    for future in as_completed(futures):
                        index, _session_id = futures[future]
                        save_result(index, future.result())
                        progress.update(1)
                except Exception:
                    for future in futures:
                        future.cancel()
                    raise
        else:
            for index, session_id in enumerate(session_ids, start=1):
                save_result(
                    index,
                    _decode_one_session(
                        index,
                        session_id,
                        trials,
                        trial_metadata,
                        behavior_metadata,
                        facemap_table=facemap_table,
                        datacube_kwargs=datacube_kwargs,
                        stimulus_modality=stimulus_modality,
                        window=window,
                        features_to_use=features_to_use,
                        sampling_rate=sampling_rate,
                        window_length=window_length,
                        fail_fast=fail_fast,
                        get_lf=get_lf,
                    ),
                )
                progress.update(1)

    result = pl.DataFrame([row for row in rows if row is not None])
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet_atomically(result, output)
        LOGGER.info("Wrote %d session results to %s", result.height, output)
    return result


def _parse_window(value: str) -> tuple[float, float]:
    try:
        start, end = (float(part.strip()) for part in value.split(",", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("window must be two comma-separated numbers") from exc
    if end <= start:
        raise argparse.ArgumentTypeError("window end must be greater than window start")
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("facemap_lick_decoding.parquet"),
        help="output Parquet file (default: %(default)s)",
    )
    parser.add_argument(
        "--window",
        type=_parse_window,
        default=DEFAULT_WINDOW,
        metavar="START,END",
        help="stimulus-aligned capture window in seconds (default: -1,1)",
    )
    parser.add_argument("--features", type=int, default=DEFAULT_FEATURES)
    parser.add_argument("--sampling-rate", type=float, default=DEFAULT_SAMPLING_RATE)
    parser.add_argument("--window-length", type=int, default=DEFAULT_WINDOW_LENGTH)
    parser.add_argument(
        "--stimulus-modality",
        choices=("visual", "auditory"),
        default=None,
        help="restrict decoding to visual or auditory stimulus trials",
    )
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="include all datacube session types and disable the behavior filter",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop instead of recording an error when one session fails",
    )
    parser.add_argument(
        "--parallelize-sessions",
        action="store_true",
        help="decode sessions concurrently (default: sequential)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="maximum concurrent session decodes when parallelized",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    decode_all_sessions(
        output_path=args.output,
        session_type=None if args.all_sessions else "brainwide",
        with_behavior_filter=not args.all_sessions,
        stimulus_modality=args.stimulus_modality,
        window=args.window,
        features_to_use=args.features,
        sampling_rate=args.sampling_rate,
        window_length=args.window_length,
        fail_fast=args.fail_fast,
        parallelize_sessions=args.parallelize_sessions,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
