"""Synthetic checks for fixed-time decoding; no datacube access required."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from sklearn.svm import LinearSVC

from dynamic_routing_facemap.decode_facemap import (
    build_parser,
    decode_all_sessions,
    decode_session,
)


def fixture():
    starts = 2.0 + 2 * np.arange(12)
    trials = pl.DataFrame({
        "session_id": ["good"] * 12,
        "stim_start_time": starts,
        "is_response": [True, False] * 6,
        "is_instruction": [False] * 12,
        "block_index": np.repeat([0, 2, 5], 4),
    })
    times = np.arange(0, 28.1, 0.1)
    data = np.column_stack((np.cos(np.pi * (times - 2.5) / 2),
                            0.2 * np.sin(times)))
    return trials, pl.DataFrame({"timestamps": times, "data": data.tolist()})


def synthetic_loader(table, **kwargs):
    """Module-level loader also exercises spawned session workers."""
    return fixture()[1].lazy()


class SingleTimeTests(unittest.TestCase):
    def test_coefficients_reconstruct_windows_and_full_session(self):
        trials, facemap = fixture()
        times = facemap["timestamps"].to_numpy()
        data = np.asarray(facemap["data"].to_list())
        interpolate = interp1d(times, data, axis=0)
        for length in (1, 3, 4):
            with self.subTest(window_length=length):
                result = decode_session(
                    trials, facemap.lazy(), features_to_use=2,
                    train_time_s=0.5, window_length=length, sampling_rate=10,
                    project_full_session=True,
                )
                self.assertEqual(len(result["cv_scores"]), 1)
                self.assertEqual(result["cv_folds"], 3)
                self.assertGreater(result["cv_scores"][0], 0.9)
                self.assertEqual(len(result["decoder_axes"]), 4)
                offsets = (np.arange(length) - (length - 1) / 2) / 10
                target = trials["stim_start_time"].to_numpy()[:, None] + 0.5 + offsets
                X = interpolate(target).transpose(0, 2, 1).reshape(12, -1)
                y = trials["is_response"].cast(pl.Int8).to_numpy()
                frames = result["frame_projections"]
                np.testing.assert_array_equal(frames["timestamps"], times)
                self.assertEqual(frames.height, facemap.height)
                for axis in result["decoder_axes"]:
                    excluded = axis["held_out_block"]
                    train = np.ones(12, dtype=bool) if excluded is None else (
                        trials["block_index"].to_numpy() != excluded
                    )
                    reference = LinearSVC(class_weight="balanced", max_iter=10_000,
                                          dual=True, random_state=0).fit(X[train], y[train])
                    np.testing.assert_allclose(axis["coefficients"], reference.coef_[0])
                    np.testing.assert_allclose(axis["intercept"], reference.intercept_[0])
                    np.testing.assert_array_equal(axis["feature_indices"], np.repeat([0, 1], length))
                    np.testing.assert_allclose(axis["sample_offsets_s"], np.tile(offsets, 2))
                    frame = 100
                    frame_X = interpolate(times[frame] + offsets).T.reshape(1, -1)
                    self.assertAlmostEqual(frames[axis["model_id"]][frame],
                                           reference.decision_function(frame_X)[0])
                if length > 1:
                    self.assertIsNone(frames["all_trials"][0])
                    self.assertIsNone(frames["all_trials"][-1])

    def test_held_out_labels_do_not_change_fold_axis(self):
        trials, facemap = fixture()
        changed = trials.with_columns(
            pl.when(pl.col("block_index") == 2).then(~pl.col("is_response"))
            .otherwise(pl.col("is_response")).alias("is_response")
        )
        results = [decode_session(t, facemap, features_to_use=2, train_time_s=0.5)
                   for t in (trials, changed)]
        axes = [next(a for a in r["decoder_axes"] if a["held_out_block"] == 2)
                for r in results]
        np.testing.assert_array_equal(axes[0]["coefficients"], axes[1]["coefficients"])
        self.assertEqual(axes[0]["intercept"], axes[1]["intercept"])
        self.assertAlmostEqual(axes[0]["balanced_accuracy"], 1 - axes[1]["balanced_accuracy"])

    def test_preserves_duplicate_and_invalid_frames(self):
        trials, facemap = fixture()
        extra = pl.DataFrame({"timestamps": [1.0, None, 29.0],
                              "data": [[0.1, 0.2], [0.1, 0.2], [float("nan"), 0.2]]})
        facemap = pl.concat([facemap, extra])
        result = decode_session(trials, facemap, features_to_use=2,
                                train_time_s=0.5, project_full_session=True)
        frames = result["frame_projections"]
        self.assertEqual(frames.height, facemap.height)
        self.assertEqual(frames["timestamps"].to_list(), facemap["timestamps"].to_list())
        self.assertIsNotNone(frames["all_trials"][-3])
        self.assertIsNone(frames["all_trials"][-2])
        self.assertIsNone(frames["all_trials"][-1])

    def test_validation_and_legacy_mode(self):
        trials, facemap = fixture()
        for options in ({"project_full_session": True}, {"train_time_s": np.nan},
                        {"train_time_s": 100.0}, {"train_time_s": 0.5, "window_length": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                decode_session(trials, facemap, features_to_use=2, **options)
        result = decode_session(trials, facemap, features_to_use=2,
                                window=(-0.2, 0.6), sampling_rate=10, window_length=2)
        self.assertEqual(len(result["cv_scores"]), 7)
        self.assertNotIn("decoder_axes", result)
        result = decode_session(trials, facemap, features_to_use=2, train_time_s=0.5)
        self.assertNotIn("frame_projections", result)
        args = build_parser().parse_args(["--train-time", "0.5", "--window-length", "6",
                                          "--projection-dir", "/tmp/projections"])
        self.assertEqual(args.train_time, 0.5)
        self.assertEqual(args.window_length, 6)

    def test_batch_persistence_and_parallel_workers(self):
        trials, _ = fixture()
        # An error before the successful session exercises null nested metadata.
        bad = trials.filter(pl.col("block_index") == 0).with_columns(
            pl.lit("bad").alias("session_id")
        )
        trials = pl.concat([bad, trials])
        for parallel in (False, True):
            with self.subTest(parallel=parallel), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                result = decode_all_sessions(
                    trials, performance_lf=pl.DataFrame(), output_path=root / "results.parquet",
                    projection_dir=root / "frames", train_time_s=0.5, window_length=4,
                    sampling_rate=10, features_to_use=2, max_frame_gaps=None,
                    get_lf=synthetic_loader, parallelize_sessions=parallel, max_workers=2,
                )
                self.assertEqual(result["status"].to_list(), ["error", "success"])
                saved = pl.read_parquet(root / "results.parquet")
                row = saved.filter(pl.col("session_id") == "good").to_dicts()[0]
                self.assertEqual(len(row["decoder_axes"]), 4)
                self.assertEqual(Path(row["frame_projections_path"]).name, "good.parquet")
                frames = pl.read_parquet(row["frame_projections_path"])
                self.assertEqual(frames.height, fixture()[1].height)
                self.assertEqual(frames["session_id"].unique().to_list(), ["good"])
                self.assertIn("holdout_2", frames.columns)
                self.assertEqual(len(list(root.glob("results.checkpoints/*/part-*.parquet"))), 2)


if __name__ == "__main__":
    unittest.main()
