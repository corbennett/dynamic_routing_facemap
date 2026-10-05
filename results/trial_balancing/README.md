# Trial-balancing comparison

This experiment compares lick/no-lick decoder offsets without subtracting or
otherwise changing the Facemap features. It uses 200 features sampled at 60 Hz,
`LinearSVC(class_weight="balanced")`, and leave-one-block-out cross-validation.
The pre-stimulus summary is the mean score from -0.25 to 0 seconds; the early
summary is the mean from 0 to 0.2 seconds. Undersampling results are the mean of
five fixed resamples (seeds 0–4).

## Main result

There are at least two different causes of the apparent offset, and trial
balancing only fixes one of them.

| Session | Strategy | Trials | Pre-stim accuracy | Pre-stim balanced accuracy | Early accuracy |
|---|---:|---:|---:|---:|---:|
| 644866_2023-02-10 | all trials | 498 | 0.675 | 0.690 | 0.644 |
| 644866_2023-02-10 | block x response | 394 | 0.681 | 0.681 | 0.662 |
| 644866_2023-02-10 | block x stimulus | 96 | 0.560 | 0.560 | 0.571 |
| 644866_2023-02-10 | same-size block x response control | 96 | 0.696 | 0.696 | 0.677 |
| 741137_2024-10-08 | all trials | 514 | 0.609 | 0.509 | 0.653 |
| 741137_2024-10-08 | block x response | 214 | 0.501 | 0.501 | 0.559 |
| 741137_2024-10-08 | block x stimulus | 108 | 0.497 | 0.497 | 0.534 |
| 741137_2024-10-08 | same-size block x response control | 108 | 0.507 | 0.507 | 0.548 |

For session 741137, raw accuracy is offset because held-out folds contain many
more no-lick than lick trials. Merely scoring the unchanged predictions with
balanced accuracy reduces the pre-stimulus score from 0.609 to 0.509. Equal
lick/no-lick undersampling inside every block has the same effect (0.501), but
throws away 300 trials.

For session 644866, balanced accuracy remains high (0.690), and matching
response count within block, trial-position bin, previous response, or both
does not reduce it (pre-stimulus accuracy 0.680–0.687). This is therefore not a
class-prevalence artifact. Exact response matching within each block and
stimulus reduces it to 0.560. A random block/response sample with the identical
96-trial block/class counts scores 0.696, showing that the reduction is not
explained by sample size alone. However, exact stimulus matching drops cells
with deterministic responses and changes the question to response prediction
conditional on stimulus identity; it also substantially weakens the
post-stimulus signal.

## Recommendation

1. Use mean fold-wise balanced accuracy as the default score. It corrects
   response-prevalence offsets without discarding trials.
2. If ordinary accuracy is required, balance lick/no-lick trials independently
   inside every held-out block and average several fixed resamples. This is the
   closest sampling analogue of balanced accuracy.
3. Diagnose any remaining pre-stimulus decoding using block x stimulus matching.
   Treat this as a different, conditional decoder rather than a generic
   correction. Report retained counts because it can remove most trials.
4. Do not add trial-position or previous-response matching by default: neither
   helped beyond block-wise response balancing in these examples.

The full resample-level curves are in `trial_balancing_curves.parquet`, the
aggregated values are in `trial_balancing_summary.csv`, and the comparison plot
is `trial_balancing_comparison.png`. Re-run the analysis from the repository
root with:

```bash
uv run python scripts/compare_trial_balancing.py
```
