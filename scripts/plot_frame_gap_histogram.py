"""Plot the distribution of >25 ms Facemap frame gaps across sessions."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import Sequence

import dr_datacube
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


DEFAULT_DECODING_RESULTS = Path(
    "results/decoding_results_leave_block_out.parquet"
)
DEFAULT_OUTPUT_DIR = Path("results/frame_gap_qc")
DEFAULT_GAP_THRESHOLD_MS = 25.0

HIGH_BALANCED_BASELINE_SESSIONS = {
    "644866_2023-02-10",
    "666986_2023-08-15",
    "666986_2023-08-17",
}


def collect_frame_gap_summary(
    session_ids: Sequence[str],
    *,
    threshold_ms: float,
    max_workers: int,
) -> pl.DataFrame:
    """Aggregate frame-interval statistics without collecting Facemap features."""

    spawn_context = mp.get_context("spawn")
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=spawn_context,
    ) as executor:
        futures = {
            executor.submit(_collect_one_session, session_id, threshold_ms): session_id
            for session_id in session_ids
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index}/{len(session_ids)}] {row['session_id']}: "
                f"{row['n_gaps_gt_25_ms']} gaps",
                flush=True,
            )

    return (
        pl.DataFrame(rows)
        .with_columns(
            n_intervals=pl.col("n_frames") - 1,
            gap_fraction_percent=(
                100 * pl.col("n_gaps_gt_25_ms") / (pl.col("n_frames") - 1)
            ),
            high_balanced_baseline=pl.col("session_id").is_in(
                HIGH_BALANCED_BASELINE_SESSIONS
            ),
        )
        .sort("n_gaps_gt_25_ms", descending=True)
    )


def _collect_one_session(
    session_id: str,
    threshold_ms: float,
) -> dict[str, object]:
    """Collect timestamp-gap statistics for one session in a spawned worker."""

    threshold_s = threshold_ms / 1_000
    summary = (
        dr_datacube.get_lf(
            "processing/behavior/facemap_side_camera",
            session_id=session_id,
        )
        .select("timestamps")
        .sort("timestamps")
        .with_columns(
            frame_interval_s=pl.col("timestamps").diff()
        )
        .select(
            n_frames=pl.len(),
            median_frame_interval_ms=(
                pl.col("frame_interval_s").median() * 1_000
            ),
            p99_frame_interval_ms=(
                pl.col("frame_interval_s").quantile(0.99) * 1_000
            ),
            max_frame_interval_ms=(pl.col("frame_interval_s").max() * 1_000),
            n_gaps_gt_25_ms=(
                pl.col("frame_interval_s").gt(threshold_s).sum()
            ),
        )
        .collect()
    )
    if summary.height != 1 or summary["n_frames"][0] < 2:
        raise ValueError(f"Insufficient Facemap timestamps for {session_id}")
    return {"session_id": session_id, **summary.row(0, named=True)}


def plot_frame_gap_summary(summary: pl.DataFrame, output_path: Path) -> None:
    """Plot a log-scaled histogram and the most extreme sessions."""

    counts = summary["n_gaps_gt_25_ms"].to_numpy()
    log_counts = np.log10(counts + 1)
    max_log_count = max(1.0, float(log_counts.max()))
    bins = np.linspace(0, max_log_count + 0.05, 26)

    figure, (histogram_axis, rank_axis) = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.25, 1)},
    )

    histogram_axis.hist(
        log_counts,
        bins=bins,
        color="0.35",
        edgecolor="white",
        linewidth=0.8,
    )
    tick_counts = np.asarray([0, 1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000])
    tick_counts = tick_counts[np.log10(tick_counts + 1) <= max_log_count + 0.05]
    histogram_axis.set_xticks(
        np.log10(tick_counts + 1),
        [f"{count:,}" for count in tick_counts],
        rotation=35,
        ha="right",
    )
    histogram_axis.set(
        title="Distribution across sessions",
        xlabel="Number of frame intervals >25 ms (log-scaled)",
        ylabel="Number of sessions",
    )

    for session_id in HIGH_BALANCED_BASELINE_SESSIONS:
        row = summary.filter(pl.col("session_id") == session_id)
        if row.height:
            histogram_axis.axvline(
                np.log10(row["n_gaps_gt_25_ms"][0] + 1),
                color="#d62728",
                alpha=0.75,
                linewidth=1.5,
            )

    top = summary.head(12).sort("n_gaps_gt_25_ms")
    colors = [
        "#d62728" if is_high else "0.55"
        for is_high in top["high_balanced_baseline"].to_list()
    ]
    rank_axis.barh(
        top["session_id"].to_list(),
        top["n_gaps_gt_25_ms"].to_numpy(),
        color=colors,
    )
    rank_axis.set_xscale("log")
    rank_axis.set(
        title="Sessions with the most gaps",
        xlabel="Number of frame intervals >25 ms (log scale)",
    )
    rank_axis.grid(axis="x", which="both", alpha=0.2)

    figure.suptitle(
        f"Facemap frame-gap QC ({summary.height} decoded sessions)",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.005,
        "Red marks sessions with confirmed high pre-stimulus balanced accuracy.",
        ha="center",
        color="#a61b1b",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decoding-results",
        type=Path,
        default=DEFAULT_DECODING_RESULTS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--gap-threshold-ms",
        type=float,
        default=DEFAULT_GAP_THRESHOLD_MS,
    )
    parser.add_argument("--max-workers", type=int, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    decoded = pl.read_parquet(args.decoding_results).filter(
        pl.col("status") == "success"
    )
    session_ids = decoded["session_id"].unique().sort().to_list()
    summary = collect_frame_gap_summary(
        session_ids,
        threshold_ms=args.gap_threshold_ms,
        max_workers=args.max_workers,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "frame_gap_summary.csv"
    figure_path = args.output_dir / "frame_gap_histogram.png"
    summary.write_csv(csv_path)
    plot_frame_gap_summary(summary, figure_path)
    print(summary)
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
