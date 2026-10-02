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

The output includes:

- `cv_scores`: mean five-fold accuracy for each sliding window;
- `window_start_s`, `window_end_s`, and `window_center_s`: matching time lists;
- `n_lick_trials`, `n_no_lick_trials`, and `lick_fraction`;
- aggregated performance-table fields such as `behavior_hit_rate_mean`,
  `behavior_false_alarm_rate_mean`, `behavior_cross_modality_dprime_mean`,
  `behavior_aud_dprime_mean`, and `behavior_vis_dprime_mean`;
- `status`, `error_type`, and `error_message`, so sessions with too few trials
  or missing Facemap data remain visible in the output.

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
Facemap features, 60 Hz sampling, and six-sample sliding windows. Use
`--all-sessions` to remove the datacube's default brainwide behavior filter.
Sessions are decoded sequentially by default. Pass `--parallelize-sessions` to
decode them concurrently; optionally use `--max-workers N` to limit the number
of concurrent sessions. A progress bar is printed to standard output as each
session completes.

For use from Python:

```python
from dynamic_routing_facemap.decode_facemap import decode_all_sessions

results = decode_all_sessions(
    output_path="results/facemap_lick_decoding.parquet",
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
