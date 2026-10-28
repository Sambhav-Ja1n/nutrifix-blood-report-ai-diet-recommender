import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.nutrifix.anomaly_model_comparison import FEATURE_COLUMNS, ID_COLUMN
from src.nutrifix.report_scoring import (
    InsufficientReportProfile,
    fit_report_scoring_bundle,
    save_report_scoring_bundle,
    score_profile_with_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas" / "canonical_biomarkers.json").read_text())


def reference_frame(size=400):
    rng = np.random.default_rng(612)
    base = rng.normal(size=(size, len(FEATURE_COLUMNS)))
    for index in range(len(FEATURE_COLUMNS)):
        base[:, index] += 0.15 * base[:, index - 1]
        base[::25, index] = np.nan
    frame = pd.DataFrame(base, columns=FEATURE_COLUMNS)
    frame.insert(0, ID_COLUMN, np.arange(50000, 50000 + size))
    return frame


class ReportScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = fit_report_scoring_bundle(reference_frame(), SCHEMA)

    def test_scoring_requires_all_model_features_and_excludes_ids(self):
        self.assertEqual(len(self.bundle["feature_columns"]), 15)
        self.assertNotIn(ID_COLUMN, self.bundle["feature_columns"])
        profile = {column: 0.2 for column in FEATURE_COLUMNS}
        profile.pop(FEATURE_COLUMNS[-1])
        with self.assertRaises(InsufficientReportProfile) as caught:
            score_profile_with_bundle(profile, self.bundle)
        self.assertEqual(caught.exception.missing_features, [FEATURE_COLUMNS[-1]])

    def test_all_five_scores_are_finite_aligned_and_reproducible(self):
        profile = {column: 0.25 for column in FEATURE_COLUMNS}
        first = score_profile_with_bundle(profile, self.bundle)
        second = score_profile_with_bundle(profile, self.bundle)
        self.assertEqual(first["status"], "scored")
        self.assertEqual(set(first["models"]), {
            "isolation_forest",
            "local_outlier_factor",
            "one_class_svm",
            "regularized_autoencoder",
            "robust_mahalanobis",
        })
        for model_name, result in first["models"].items():
            self.assertTrue(np.isfinite(result["anomaly_score"]))
            self.assertEqual(result["score_direction"], "higher_is_more_anomalous")
            self.assertEqual(set(result["candidate_flags"]), {
                "top_2pct_candidate", "top_5pct_candidate", "top_10pct_candidate"
            })
            self.assertEqual(
                result["anomaly_score"], second["models"][model_name]["anomaly_score"]
            )
        self.assertFalse(self.bundle["metadata"]["participant_ids_stored_in_artifact"])
        self.assertFalse(self.bundle["metadata"]["participant_ids_used_as_model_features"])

    def test_bundle_saves_without_overwriting_and_records_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "reference_bundle"
            artifact, metadata_file = save_report_scoring_bundle(self.bundle, output)
            self.assertTrue(artifact.is_file())
            metadata = json.loads(metadata_file.read_text())
            self.assertEqual(metadata["artifact_format_version"], 1)
            # Every 25th row is entirely missing in the fixture and is excluded
            # before splitting: 384 eligible participants remain.
            self.assertEqual(metadata["eligible_reference_population"], 384)
            self.assertEqual(metadata["reference_training_participants"], 288)
            self.assertEqual(metadata["held_out_calibration_participants"], 96)
            self.assertEqual(metadata["excluded_all_selected_features_missing"], 16)
            with self.assertRaises(FileExistsError):
                save_report_scoring_bundle(self.bundle, output)


if __name__ == "__main__":
    unittest.main()
