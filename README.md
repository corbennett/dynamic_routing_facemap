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

The defaults match the notebook: a `[-1, 1]` second capture window, 200
Facemap features, 60 Hz sampling, and six-sample sliding windows. Cross-
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
