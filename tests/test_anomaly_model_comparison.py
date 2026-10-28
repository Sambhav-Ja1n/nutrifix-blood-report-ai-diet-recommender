import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.nutrifix.anomaly_model_comparison import (
    ID_COLUMN,
    fit_reference_transform,
    fit_score_models,
    make_split,
    robust_mahalanobis_scores,
    run_comparison,
    top_k_selection,
)


def small_frame(n=180):
    rng = np.random.default_rng(717)
    values = rng.normal(size=(n, 4))
    values[:, 1] = values[:, 0] * 0.8 + rng.normal(scale=0.2, size=n)
    values[-6] += 9.0
    values[::11, 0] = np.nan
    values[::17, 2] = np.nan
    values[-5:, :] = np.nan
    return pd.DataFrame({
        ID_COLUMN: np.arange(10000, 10000 + n),
        "a": values[:, 0], "b": values[:, 1],
        "c": values[:, 2], "d": values[:, 3],
    })


SCHEMA = {
    "biomarkers": {
        f"biomarker_{col}": {
            "source_column": col,
            "display_name": col,
            "canonical_unit": "unit",
        }
        for col in ["a", "b", "c", "d"]
    }
}


class ComparisonTests(unittest.TestCase):
    def test_eligibility_and_missingness_are_explicit(self):
        frame = small_frame()
        split = make_split(frame, ["a", "b", "c", "d"], random_seed=9)
        self.assertEqual(split.excluded_all_missing_count, 5)
        self.assertEqual(split.eligible_count, len(frame) - 5)
        self.assertGreater(int(split.missing_train.sum() + split.missing_test.sum()), 0)
        self.assertTrue(set(split.ids_train).isdisjoint(set(split.ids_test)))
        self.assertTrue(np.all(np.diff(split.test_positions) >= 0))

    def test_id_not_in_model_matrices_and_rows_align(self):
        frame = small_frame()
        split = make_split(frame, ["a", "b", "c", "d"], random_seed=19)
        train_x, test_x, _ = fit_reference_transform(
            split.raw_train, split.raw_test, split.missing_train,
            split.missing_test, include_missingness=True,
        )
        scores, _ = fit_score_models(train_x, test_x)
        self.assertEqual(train_x.shape[1], 8)
        self.assertEqual(len(split.ids_test), test_x.shape[0])
        self.assertTrue(all(len(score) == len(split.ids_test) for score in scores.values()))
        self.assertEqual(set(scores), {
            "isolation_forest", "local_outlier_factor", "one_class_svm",
            "regularized_autoencoder", "robust_mahalanobis",
        })

    def test_anomaly_score_direction_for_all_models(self):
        rng = np.random.default_rng(91)
        train = rng.normal(size=(140, 3))
        test = np.vstack([np.zeros(3), np.array([10.0, -9.0, 11.0])])
        scores, _ = fit_score_models(train, test, random_seed=17)
        for model_name, values in scores.items():
            self.assertGreater(
                values[1], values[0],
                f"{model_name} should assign a higher anomaly score to the extreme profile",
            )

    def test_insufficient_reference_sample_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "LOF requires more than"):
            fit_score_models(np.zeros((20, 3)), np.ones((2, 3)))

    def test_robust_covariance_estimation_failure_is_reported(self):
        rng = np.random.default_rng(92)
        base = rng.normal(size=(120, 2))
        train = np.column_stack([base, base[:, 0] + base[:, 1]])
        test = train[:4]
        with self.assertRaisesRegex(RuntimeError, "covariance estimation failed"):
            robust_mahalanobis_scores(train, test, random_seed=12)

    def test_score_direction_and_top_k_tie_policy(self):
        scores = np.array([0.2, 8.0, 1.0, 8.0, -1.0])
        flags, order = top_k_selection(scores, fractions=(0.4,))
        self.assertEqual(int(flags["top_40pct"].sum()), 2)
        self.assertEqual(order[:2].tolist(), [1, 3])
        self.assertTrue(flags["top_40pct"][1])
        self.assertTrue(flags["top_40pct"][3])

    def test_reproducible_seed_and_output_integrity(self):
        frame = small_frame()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "experiment"
            first = run_comparison(
                frame, SCHEMA, output, random_seed=31, test_size=0.25,
                feature_columns=["a", "b", "c", "d"],
            )
            first_scores = pd.read_csv(output / "held_out_scores.csv")
            self.assertEqual(first["cohort_counts"]["excluded_all_selected_features_missing"], 5)
            self.assertTrue((output / "pairwise_rank_correlations.csv").exists())
            self.assertTrue((output / "pairwise_top_k_overlap.csv").exists())
            self.assertTrue((output / "ranking_stability.csv").exists())
            metadata = json.loads((output / "experiment_metadata.json").read_text())
            self.assertFalse(metadata["participant_id_used_as_model_feature"])
            self.assertEqual(
                set(first_scores["model"]),
                {
                    "isolation_forest", "local_outlier_factor", "one_class_svm",
                    "regularized_autoencoder", "robust_mahalanobis",
                },
            )
            for _, group in first_scores.groupby(["preprocessing", "model"]):
                self.assertEqual(group[ID_COLUMN].nunique(), len(group))
            id_sets = [
                set(group[ID_COLUMN])
                for _, group in first_scores.groupby(["preprocessing", "model"])
            ]
            self.assertTrue(all(ids == id_sets[0] for ids in id_sets))
            with self.assertRaises(FileExistsError):
                run_comparison(
                    frame, SCHEMA, output, random_seed=31,
                    feature_columns=["a", "b", "c", "d"],
                )

            second = Path(temporary) / "experiment_repeat"
            run_comparison(
                frame, SCHEMA, second, random_seed=31, test_size=0.25,
                feature_columns=["a", "b", "c", "d"],
            )
            second_scores = pd.read_csv(second / "held_out_scores.csv")
            pd.testing.assert_frame_equal(first_scores, second_scores)

    def test_non_finite_score_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            top_k_selection(np.array([1.0, np.inf]))


if __name__ == "__main__":
    unittest.main()
