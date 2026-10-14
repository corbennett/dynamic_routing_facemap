# Dynamic Routing Facemap decoding

The decoding workflow from `scratch.ipynb` is available as a reusable module.
It finds every unique session in the trials table, decodes lick versus no-lick
trials from `processing/behavior/facemap_side_camera`, and writes one Parquet
row per session.

Run it from the project environment with:

```bash
uv run dynamic-routing-facemap \
  --output results/facemap_lick_decoding.parquet \
  --verbose
```

To analyze one stimulus modality at a time, add `--stimulus-modality visual`
or `--stimulus-modality auditory`. With no flag, both modalities are included.
Add `--target-trials-only` to restrict decoding to target trials. When combined
with a modality flag, the corresponding `is_vis_target` or `is_aud_target`
column is used; without one, either target flag qualifies.
Add `--balance-trials-across-blocks` to downsample the larger response class
within each block so that block has equal lick and no-lick trial counts.
Before classifier fitting, the batch decoder also counts intervals longer than
25 ms in each session's full Facemap timestamp stream. Sessions with more than
1,000 such intervals are excluded by default. Use
`--frame-gap-threshold-ms` and `--max-frame-gaps` to change those cutoffs.

The output includes:

- `cv_scores`: mean leave-one-block-out balanced accuracy for each sliding window;
- `cv_folds`: number of nonempty task blocks used as CV folds (normally six);
- `window_start_s`, `window_end_s`, and `window_center_s`: matching time lists;
- `n_lick_trials`, `n_no_lick_trials`, and `lick_fraction`;
- `stimulus_modality_filter`, recording any visual/auditory filter used;
- `target_trials_only`, recording whether target-only filtering was enabled;
- `balance_trials_across_blocks`, recording whether per-block class balancing
  was enabled;
- `facemap_n_frames`, `facemap_n_frame_gaps`, and
  `facemap_frame_gap_fraction`, recording the frame-timing QC result;
- `frame_gap_threshold_ms`, `max_frame_gaps`, and `qc_exclusion_reason`,
  recording the QC rule and why a session was excluded;
- aggregated performance-table fields such as `behavior_hit_rate_mean`,
  `behavior_false_alarm_rate_mean`, `behavior_cross_modality_dprime_mean`,
  `behavior_aud_dprime_mean`, and `behavior_vis_dprime_mean`;
- `status`, `error_type`, and `error_message`, so sessions with too few trials,
  missing Facemap data, or failed frame-timing QC remain visible in the output.
  QC-excluded sessions have `status="excluded_qc"` and null decoding scores.

Each completed session is also written immediately to a unique checkpoint
directory beside the output, for example
`results/facemap_lick_decoding.checkpoints/<run-id>/part-000001.parquet`.
These parts are retained if the run is interrupted and can be recovered with:

```python
import polars as pl

partial_results = pl.read_parquet(
    "results/facemap_lick_decoding.checkpoints/<run-id>/part-*.parquet"
)
```

The defaults are a `[-0.25, 0.5)` second capture window, 200
Facemap features, 60 Hz sampling, and one-sample sliding windows. Cross-
validation holds out each of the six task blocks in turn, identified by the
trial `block_index` (including both auditory- and visually-rewarded blocks).
Use
`--all-sessions` to remove the datacube's default brainwide behavior filter.
Sessions are decoded sequentially by default. Pass `--parallelize-sessions` to
decode them concurrently; optionally use `--max-workers N` to limit the number
of concurrent sessions. A progress bar is printed to standard output as each
session completes. On Linux, parallel sessions use Python's `spawn` process
context because Polars must not be used from forked child processes.

For use from Python:

```python
from dynamic_routing_facemap.decode_facemap import decode_all_sessions

results = decode_all_sessions(
    output_path="results/facemap_lick_decoding.parquet",
    stimulus_modality="visual",  # or "auditory"; omit to include both
    max_frame_gaps=1_000,         # None disables frame-gap QC
    parallelize_sessions=False,  # set True to decode sessions concurrently
    # max_workers=4,             # optional limit when parallelized
)
```

To turn the list-valued scores into a long table for plotting:

```python
long_results = results.explode(
    ["window_start_s", "window_end_s", "window_center_s", "cv_scores"]
)
```

## Train at a fixed time and save continuous decoder scores

Use `--train-time` to train one window centered at a specified time relative
to stimulus onset. For example, train a six-sample window centered at 0.5 s:

```bash
uv run dynamic-routing-facemap \
  --train-time 0.5 \
  --window-length 6 \
  --projection-dir results/lick_axis_frames \
  --output results/lick_axis.parquet
```

Or from Python:

```python
results = decode_all_sessions(
    train_time_s=0.5,
    window_length=6,  # use 1 for a single instantaneous sample
    projection_dir="results/lick_axis_frames",
    output_path="results/lick_axis.parquet",
)
```

Samples are linearly interpolated at offsets
`(arange(window_length) - (window_length - 1) / 2) / sampling_rate` from the
requested center. Six samples at 60 Hz span 0.45833–0.54167 s when centered at
0.5 s. Features at successive sample offsets are concatenated, as in the
sliding-window decoder, rather than averaged. Fixed-time training is independent
of `--window`; it can train at 0.5 s even though the default capture interval
ends there. All requested training samples must lie within the camera recording.

The decoder is fit only at `train_time_s`, then each leave-one-block-out model
is applied to its held-out trials across the full `--window`. The resulting
`cv_scores` list is therefore a temporal-generalization curve: every entry is
the mean held-out balanced accuracy of the same fixed decoder axes, not a model
refit at that time. `window_start_s`, `window_end_s`, and `window_center_s`
describe the evaluation windows and coincide for a one-sample decoder.
The following additional fields are saved:

- `train_time_s`: the requested stimulus-relative window center.
- `train_cv_score`: mean held-out balanced accuracy at the exact training time.
- `decoder_axes`: one entry for `all_trials` and one per CV fold (`holdout_0`,
  `holdout_1`, etc.). Each entry contains `model_id`, `held_out_block`,
  `feature_indices`, `sample_offsets_s`, `coefficients`, `intercept`, training
  and test trial counts, and the fold's `balanced_accuracy` (null for the full fit).
  The three coefficient arrays have matching lengths: each coefficient belongs
  to one feature at one time offset. Feature indices are zero-based indices into
  the original Facemap `data` vector. Weights use unscaled features and point
  toward the lick-response class. They are decoder coefficients, not correlations
  between individual features and licking.
- `frame_projections_path`: path to a per-session Parquet file, if
  `projection_dir` was supplied. Files are named `<session_id>.parquet`, for
  example `644866_2023-02-10.parquet` (special characters are URL-escaped).
  Files live in a unique run subdirectory so
  subsequent runs do not overwrite previous projections. Without this option,
  axes and CV scores are still saved but full-session projections are not computed.

The summary table maps `session_id` to `frame_projections_path`, and each
projection file also stores `session_id` on every row. Earlier runs with
numbered filenames (`session-000001.parquet`, etc.) use the same mapping.

Each projection file contains `session_id`, `frame_index`, the original
`timestamps`, and one score column per `model_id`. Every original camera row is
preserved, including intertrial frames and duplicate timestamps. At each
timestamp the same centered window is sampled, and its score is
`X_window @ coefficients + intercept`. For one-sample projections the original
feature vector is used directly. Multi-sample projections use linear interpolation
onto the offsets above (the first row at a duplicate timestamp is used for
interpolation). Scores are null when the full window extends outside the
recording or its inputs are non-finite. These centered windows include samples
after their timestamp and are intended for offline analysis.

These are **decoder scores, not calibrated lick probabilities**. The `all_trials`
column uses a single model fitted to all eligible training trials. Each `holdout_*`
column also covers the entire recording, but is a held-out prediction **only for
trials in the block identified by that axis's `held_out_block`**. Choose that
column when analyzing that block; the fold number need not equal the block index.
The score columns are kept separate to preserve which axis generated each value.
Filtering and balancing affect model training, not which camera frames are saved.

For example, with the default `[-0.25, 0.5)` capture window, 60 Hz sampling,
and a one-sample decoder, the output curve has 45 scores spanning -0.25 s
through 0.4833 s. A multi-sample decoder has fewer evaluation windows because
all of its samples must fit inside the capture window. `cv_score_mean`,
`cv_score_max`, and `best_window_center_s` summarize this full curve; use
`train_cv_score` for performance at the fixed training time. The training time
does not need to fall inside the evaluation window.

Read the frames and reconstruct a coefficient matrix with:

```python
import numpy as np

row = results.filter(pl.col("status") == "success").to_dicts()[0]
frames = pl.read_parquet(row["frame_projections_path"])
axis = row["decoder_axes"][0]  # all_trials; other entries identify held-out blocks
weights = np.asarray(axis["coefficients"]).reshape(
    row["features_used"], row["window_length_samples"]
)
# weights[feature_index, sample_index], ordered from earliest to latest offset
```

For direct use, `decode_session(..., train_time_s=0.5,
project_full_session=True)` returns the axes and an in-memory
`frame_projections` DataFrame without writing files. Full-session arrays are
projected in chunks to limit memory use for multi-sample windows.

Synthetic checks can be run without downloading data:

```bash
uv run python -m unittest discover -s tests -v
```
