"""Reference-only five-model scoring for complete, reviewed CBC profiles.

The scores are exploratory outlier rankings. They are not disease probabilities,
clinical reference intervals, or diagnostic outputs.
"""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

from src.nutrifix.anomaly_model_comparison import (
    AE_ALPHA,
    AE_EARLY_STOPPING,
    AE_MAX_ITER,
    AE_N_ITER_NO_CHANGE,
    AE_TOL,
    FEATURE_COLUMNS,
    ID_COLUMN,
    IF_N_ESTIMATORS,
    LOF_N_NEIGHBORS,
    OCSVM_NU,
    PRIMARY_SEED,
    TEST_SIZE,
    TOP_FRACTIONS,
    fit_reference_models,
    fit_reference_transform,
    make_split,
    score_reference_models,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_FILE = ROOT / "schemas" / "canonical_biomarkers.json"
DEFAULT_MODEL_DIR = ROOT / "outputs" / "models" / "report_scoring_v1"
ARTIFACT_FILE_NAME = "scorer.joblib"
METADATA_FILE_NAME = "scorer_metadata.json"
ARTIFACT_FORMAT_VERSION = 1
MODEL_NAMES = (
    "isolation_forest",
    "local_outlier_factor",
    "one_class_svm",
    "regularized_autoencoder",
    "robust_mahalanobis",
)


class ReportScoringError(RuntimeError):
    """The profile could not be scored safely."""


class InsufficientReportProfile(ReportScoringError):
    """The report does not contain the complete model feature set."""

    def __init__(self, missing_features: list[str]):
        self.missing_features = list(missing_features)
        super().__init__(
            "All model biomarkers are required; missing: " + ", ".join(self.missing_features)
        )


class ScoringArtifactUnavailable(ReportScoringError):
    """A compatible locally generated reference model is unavailable."""


def _load_schema(schema_path: Path = SCHEMA_FILE) -> dict:
    with Path(schema_path).open(encoding="utf-8") as file:
        return json.load(file)


def _feature_definitions(schema: dict) -> list[dict]:
    lookup = {
        item["source_column"]: (name, item)
        for name, item in schema["biomarkers"].items()
    }
    definitions = []
    for column in FEATURE_COLUMNS:
        if column not in lookup:
            raise ValueError(f"Model feature {column} is absent from canonical schema")
        name, item = lookup[column]
        definitions.append({
            "source_column": column,
            "canonical_name": name,
            "display_name": item["display_name"],
            "unit": item["canonical_unit"],
        })
    return definitions


def _calibration_cutoffs(scores: np.ndarray) -> tuple[dict[str, float], dict[str, int]]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be a nonempty finite vector")
    order = np.argsort(-scores, kind="stable")
    cutoffs = {}
    selected_counts = {}
    for fraction in TOP_FRACTIONS:
        count = int(math.ceil(fraction * len(scores)))
        label = f"top_{int(round(fraction * 100))}pct"
        cutoffs[label] = float(scores[order[count - 1]])
        selected_counts[label] = count
    return cutoffs, selected_counts


def fit_report_scoring_bundle(
    frame: pd.DataFrame,
    schema: dict,
    random_seed: int = PRIMARY_SEED,
    test_size: float = TEST_SIZE,
) -> dict:
    """Fit models on the reference partition and calibrate on held-out rows."""
    split = make_split(
        frame,
        feature_columns=list(FEATURE_COLUMNS),
        random_seed=random_seed,
        test_size=test_size,
    )
    if set(split.ids_train) & set(split.ids_test):
        raise RuntimeError("Reference and calibration participant IDs overlap")

    train_x, calibration_x, preprocessing = fit_reference_transform(
        split.raw_train,
        split.raw_test,
        split.missing_train,
        split.missing_test,
        include_missingness=True,
    )
    # Save the precise scaler that corresponds to fit_reference_transform.
    raw_medians = np.nanmedian(split.raw_train, axis=0)
    if not np.isfinite(raw_medians).all():
        raise ValueError("Reference data cannot provide a finite median for every feature")
    imputed_train = np.where(np.isnan(split.raw_train), raw_medians, split.raw_train)
    scaler = StandardScaler().fit(imputed_train)
    if not np.allclose(scaler.mean_, preprocessing["training_scaler_means"]):
        raise RuntimeError("Saved scaler does not match the reference preprocessing")
    if not np.allclose(scaler.scale_, preprocessing["training_scaler_scales"]):
        raise RuntimeError("Saved scaler scales do not match the reference preprocessing")

    models, model_fit_details = fit_reference_models(train_x, random_seed=random_seed)
    calibration_scores = score_reference_models(models, calibration_x)
    if set(calibration_scores) != set(MODEL_NAMES):
        raise RuntimeError("Not all five models returned calibration scores")

    sorted_calibration_scores = {}
    cutoffs = {}
    selected_counts = {}
    for model_name in MODEL_NAMES:
        values = np.asarray(calibration_scores[model_name], dtype=np.float64)
        if values.shape != (len(split.ids_test),) or not np.isfinite(values).all():
            raise RuntimeError(f"{model_name} produced invalid calibration scores")
        sorted_calibration_scores[model_name] = np.sort(values, kind="stable")
        cutoffs[model_name], selected_counts[model_name] = _calibration_cutoffs(values)

    metadata = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "schema_version": schema.get("schema_version"),
        "artifact_purpose": "exploratory individual CBC profile scoring against an NHANES reference cohort",
        "reference_data": "NHANES 2017-2018 merged CBC cohort; survey weights are not used",
        "clinical_validation": False,
        "participant_id_column": ID_COLUMN,
        "participant_ids_stored_in_artifact": False,
        "participant_ids_used_as_model_features": False,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_definitions": _feature_definitions(schema),
        "required_profile_coverage": "all 15 model biomarkers, with exact canonical units, must be user-confirmed",
        "missingness_indicators_in_reference_models": True,
        "input_participants": int(len(frame)),
        "eligible_reference_population": int(split.eligible_count),
        "reference_training_participants": int(len(split.ids_train)),
        "held_out_calibration_participants": int(len(split.ids_test)),
        "excluded_all_selected_features_missing": int(split.excluded_all_missing_count),
        "reference_split": {
            "method": "seeded train_test_split over participants with at least one observed feature",
            "random_seed": random_seed,
            "calibration_fraction": test_size,
            "scaler_and_models_fit_on_reference_partition_only": True,
        },
        "preprocessing": {
            **preprocessing,
            "required_report_profiles_with_missing_features": "not scored",
        },
        "models": {
            "isolation_forest": {
                "n_estimators": IF_N_ESTIMATORS,
                "contamination": "auto; native decision threshold is not used",
                "score": "negative score_samples; higher means more anomalous",
            },
            "local_outlier_factor": {
                "n_neighbors": LOF_N_NEIGHBORS,
                "novelty": True,
                "score": "negative score_samples; higher means more anomalous",
            },
            "one_class_svm": {
                "kernel": "rbf",
                "gamma": "scale",
                "nu": OCSVM_NU,
                "score": "negative score_samples; higher means more anomalous",
            },
            "regularized_autoencoder": {
                **model_fit_details["regularized_autoencoder"],
                "alpha_l2_regularization": AE_ALPHA,
                "max_iter": AE_MAX_ITER,
                "early_stopping": AE_EARLY_STOPPING,
                "tolerance": AE_TOL,
                "n_iter_no_change": AE_N_ITER_NO_CHANGE,
            },
            "robust_mahalanobis": model_fit_details["robust_mahalanobis"],
        },
        "calibration": {
            "method": "empirical score distribution on the held-out calibration partition",
            "participant_count": int(len(split.ids_test)),
            "fractions": list(TOP_FRACTIONS),
            "selected_counts_by_model": selected_counts,
            "cutoffs_by_model": cutoffs,
            "candidate_tie_policy": "Scores equal to the cutoff are included; ties can make a candidate set larger than the nominal fraction.",
            "percentile_interpretation": "empirical reference rank only; not a probability or clinical percentile",
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "Scores describe similarity to an NHANES 2017-2018 CBC reference cohort and are not diagnoses.",
            "All 15 model features are required because broad profile missingness was not validated.",
            "The held-out calibration set is an unsupervised reference distribution, not labeled normal or disease data.",
            "Population, age, sex, assay, specimen, and laboratory differences can change score meaning.",
            "No clinical accuracy, risk probability, or treatment action is estimated.",
        ],
    }
    return {
        "metadata": metadata,
        "feature_columns": list(FEATURE_COLUMNS),
        "schema_version": schema.get("schema_version"),
        "medians": raw_medians,
        "scaler": scaler,
        "models": models,
        "calibration_scores_sorted": sorted_calibration_scores,
        "cutoffs": cutoffs,
    }


def save_report_scoring_bundle(bundle: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write a new ignored model bundle directory without overwriting results."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite report scoring bundle: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    artifact_path = output_dir / ARTIFACT_FILE_NAME
    metadata_path = output_dir / METADATA_FILE_NAME
    joblib.dump(bundle, artifact_path, compress=3)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(bundle["metadata"], file, indent=2, allow_nan=False)
    return artifact_path, metadata_path


@lru_cache(maxsize=1)
def _load_report_scoring_bundle() -> dict:
    artifact_path = DEFAULT_MODEL_DIR / ARTIFACT_FILE_NAME
    metadata_path = DEFAULT_MODEL_DIR / METADATA_FILE_NAME
    if not artifact_path.is_file() or not metadata_path.is_file():
        raise ScoringArtifactUnavailable(
            "No local report scoring model is built; run scripts/build_report_scoring_model.py"
        )
    try:
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
            raise ScoringArtifactUnavailable("Report scoring model format is incompatible")
        current_schema = _load_schema()
        if metadata.get("feature_columns") != list(FEATURE_COLUMNS):
            raise ScoringArtifactUnavailable("Report scoring feature definitions are incompatible")
        if metadata.get("schema_version") != current_schema.get("schema_version"):
            raise ScoringArtifactUnavailable("Canonical schema changed; rebuild the report scoring model")
        bundle = joblib.load(artifact_path)
        if bundle.get("metadata") != metadata:
            raise ScoringArtifactUnavailable("Report scoring artifact metadata does not match")
        return bundle
    except ScoringArtifactUnavailable:
        raise
    except Exception as exc:
        raise ScoringArtifactUnavailable("The local report scoring model could not be loaded") from exc


def score_profile_with_bundle(values_by_column: dict[str, float], bundle: dict) -> dict:
    """Score one complete reviewed CBC profile using already-loaded model state."""
    required = list(FEATURE_COLUMNS)
    missing = [column for column in required if column not in values_by_column]
    if missing:
        raise InsufficientReportProfile(missing)
    if bundle.get("feature_columns") != required:
        raise ScoringArtifactUnavailable("Report scoring feature definitions are incompatible")
    try:
        raw = np.asarray([[values_by_column[column] for column in required]], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReportScoringError("Report values cannot be represented as finite numbers") from exc
    if raw.shape != (1, len(required)) or not np.isfinite(raw).all():
        raise ReportScoringError("Report scoring requires finite values for every model feature")

    missingness = np.isnan(raw)
    if missingness.any():
        missing_columns = [column for column, is_missing in zip(required, missingness[0]) if is_missing]
        raise InsufficientReportProfile(missing_columns)
    imputed = np.where(missingness, bundle["medians"], raw)
    scaled = bundle["scaler"].transform(imputed)
    model_input = np.column_stack([scaled, missingness.astype(np.float64)])
    if not np.isfinite(model_input).all():
        raise ReportScoringError("Report preprocessing produced non-finite model inputs")

    scores_by_model = score_reference_models(bundle["models"], model_input)
    output = {}
    calibration_count = int(bundle["metadata"]["calibration"]["participant_count"])
    for model_name in MODEL_NAMES:
        score = float(scores_by_model[model_name][0])
        if not math.isfinite(score):
            raise ReportScoringError(f"{model_name} returned a non-finite score")
        distribution = bundle["calibration_scores_sorted"][model_name]
        percentile = 100.0 * float(np.searchsorted(distribution, score, side="right")) / len(distribution)
        thresholds = bundle["cutoffs"][model_name]
        flags = {
            "top_2pct_candidate": score >= thresholds["top_2pct"],
            "top_5pct_candidate": score >= thresholds["top_5pct"],
            "top_10pct_candidate": score >= thresholds["top_10pct"],
        }
        output[model_name] = {
            "anomaly_score": score,
            "score_direction": "higher_is_more_anomalous",
            "empirical_reference_percentile": percentile,
            "calibration_reference_participants": calibration_count,
            "candidate_flags": flags,
            "calibration_cutoffs": thresholds,
            "interpretation": (
                "Exploratory statistical outlier score relative to an NHANES reference cohort; "
                "not a probability, diagnosis, or treatment recommendation."
            ),
        }
    return {
        "status": "scored",
        "reference_population": "NHANES 2017-2018",
        "feature_count": len(required),
        "models": output,
        "limitations": list(bundle["metadata"]["limitations"]),
    }


def score_reviewed_profile(values_by_column: dict[str, float]) -> dict:
    """Score one complete reviewed CBC profile against the fixed local reference."""
    missing = [column for column in FEATURE_COLUMNS if column not in values_by_column]
    if missing:
        raise InsufficientReportProfile(missing)
    return score_profile_with_bundle(values_by_column, _load_report_scoring_bundle())
