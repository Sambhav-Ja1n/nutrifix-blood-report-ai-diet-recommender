"""Reproducible held-out comparison of exploratory CBC anomaly methods.

Scores are statistical ranking signals, not clinical diagnoses. All learned
preprocessing and model parameters are fitted on the reference partition only.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "nhanes_2017_2018_merged.parquet"
SCHEMA_FILE = ROOT / "schemas" / "canonical_biomarkers.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "metrics" / "anomaly_model_comparison"
ID_COLUMN = "SEQN"
FEATURE_COLUMNS = [
    "LBXWBCSI",
    "LBXLYPCT",
    "LBXMOPCT",
    "LBXNEPCT",
    "LBXEOPCT",
    "LBXBAPCT",
    "LBXRBCSI",
    "LBXHGB",
    "LBXHCT",
    "LBXMCVSI",
    "LBXMCHSI",
    "LBXMC",
    "LBXRDW",
    "LBXPLTSI",
    "LBXMPSI",
]
TOP_FRACTIONS = (0.02, 0.05, 0.10)
PRIMARY_SEED = 42
TEST_SIZE = 0.25
STABILITY_SEEDS = (43, 44)
IF_N_ESTIMATORS = 300
LOF_N_NEIGHBORS = 35
OCSVM_NU = 0.05
AE_ALPHA = 1e-3
AE_MAX_ITER = 400
AE_EARLY_STOPPING = True
AE_TOL = 1e-3
AE_N_ITER_NO_CHANGE = 10
MCD_EIGENVALUE_FLOOR_RATIO = 1e-6
MAX_REGULARIZED_CONDITION_NUMBER = 1e8


@dataclass
class PreparedSplit:
    ids_train: np.ndarray
    ids_test: np.ndarray
    raw_train: np.ndarray
    raw_test: np.ndarray
    missing_train: np.ndarray
    missing_test: np.ndarray
    source_test_order: np.ndarray
    train_positions: np.ndarray
    test_positions: np.ndarray
    non_numeric_count: int
    eligible_count: int
    excluded_all_missing_count: int


@dataclass
class RobustMahalanobisScorer:
    """Regularized MCD score state reusable for previously unseen profiles."""

    location: np.ndarray
    eigenvectors: np.ndarray
    regularized_eigenvalues: np.ndarray

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.location):
            raise ValueError("Robust Mahalanobis scoring features are misaligned")
        _require_finite("Robust Mahalanobis scoring inputs", values)
        projected = (values - self.location) @ self.eigenvectors
        squared = np.sum(
            (projected * projected) / self.regularized_eigenvalues,
            axis=1,
        )
        scores = np.sqrt(np.maximum(squared, 0.0))
        _require_finite("Robust Mahalanobis anomaly scores", scores)
        return scores


def _validate_raw_frame(frame: pd.DataFrame, feature_columns: list[str]) -> None:
    required = [ID_COLUMN, *feature_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError("Participant IDs must be present and unique")


def make_split(
    frame: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    random_seed: int = PRIMARY_SEED,
    test_size: float = TEST_SIZE,
) -> PreparedSplit:
    """Apply the eligibility rule and create a reproducible ID-aligned split."""
    _validate_raw_frame(frame, feature_columns)
    converted = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    non_numeric_count = int(
        sum((frame[col].notna() & converted[col].isna()).sum() for col in feature_columns)
    )
    raw_all = converted.to_numpy(dtype=np.float64)
    if np.isinf(raw_all).any():
        raise ValueError("Raw selected biomarkers contain infinite values")

    missing_all = np.isnan(raw_all)
    eligible_mask = (~missing_all).any(axis=1)
    eligible_positions = np.flatnonzero(eligible_mask)
    excluded_count = int((~eligible_mask).sum())
    if len(eligible_positions) < 4:
        raise ValueError("At least four eligible participants are required")

    train_positions, test_positions = train_test_split(
        eligible_positions,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )
    train_positions = np.asarray(train_positions, dtype=np.int64)
    # Preserve original merged-file order in the held-out arrays so stable
    # score ties use source row order rather than split-generator order.
    test_positions = np.sort(np.asarray(test_positions, dtype=np.int64), kind="stable")

    return PreparedSplit(
        ids_train=frame.iloc[train_positions][ID_COLUMN].to_numpy(copy=True),
        ids_test=frame.iloc[test_positions][ID_COLUMN].to_numpy(copy=True),
        raw_train=raw_all[train_positions],
        raw_test=raw_all[test_positions],
        missing_train=missing_all[train_positions],
        missing_test=missing_all[test_positions],
        source_test_order=np.arange(len(test_positions), dtype=np.int64),
        train_positions=train_positions,
        test_positions=test_positions,
        non_numeric_count=non_numeric_count,
        eligible_count=len(eligible_positions),
        excluded_all_missing_count=excluded_count,
    )


def fit_reference_transform(
    raw_train: np.ndarray,
    raw_test: np.ndarray,
    missing_train: np.ndarray,
    missing_test: np.ndarray,
    include_missingness: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit median imputation and scaling on training observations only."""
    raw_train = np.asarray(raw_train, dtype=np.float64)
    raw_test = np.asarray(raw_test, dtype=np.float64)
    if raw_train.ndim != 2 or raw_test.ndim != 2:
        raise ValueError("Training and test features must be two-dimensional")
    if raw_train.shape[1] != raw_test.shape[1]:
        raise ValueError("Training and test feature definitions are misaligned")
    if not np.isfinite(raw_train[~np.isnan(raw_train)]).all() or not np.isfinite(
        raw_test[~np.isnan(raw_test)]
    ).all():
        raise ValueError("Observed raw feature values must be finite")

    medians = np.nanmedian(raw_train, axis=0)
    if not np.isfinite(medians).all():
        bad = np.flatnonzero(~np.isfinite(medians)).tolist()
        raise ValueError(
            "Reference partition has features with no observed values; "
            f"cannot fit medians for feature indexes {bad}"
        )

    train_imputed = np.where(np.isnan(raw_train), medians, raw_train)
    test_imputed = np.where(np.isnan(raw_test), medians, raw_test)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)

    if include_missingness:
        train_ready = np.column_stack([train_scaled, missing_train.astype(np.float64)])
        test_ready = np.column_stack([test_scaled, missing_test.astype(np.float64)])
    else:
        train_ready = train_scaled
        test_ready = test_scaled

    if not np.isfinite(train_ready).all() or not np.isfinite(test_ready).all():
        raise ValueError("Preprocessing produced non-finite model inputs")

    details = {
        "imputation": "feature-wise median fitted on reference partition",
        "scaling": "StandardScaler fitted on imputed reference partition; indicators left as 0/1",
        "missingness_indicators_in_model": include_missingness,
        "training_imputed_cells": int(np.isnan(raw_train).sum()),
        "held_out_imputed_cells": int(np.isnan(raw_test).sum()),
        "training_participants_with_partial_missingness": int(
            (np.isnan(raw_train).any(axis=1)).sum()
        ),
        "held_out_participants_with_partial_missingness": int(
            (np.isnan(raw_test).any(axis=1)).sum()
        ),
        "training_medians": medians.tolist(),
        "training_scaler_means": scaler.mean_.tolist(),
        "training_scaler_scales": scaler.scale_.tolist(),
        "model_input_feature_count": int(train_ready.shape[1]),
    }
    return train_ready, test_ready, details


def _require_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


def fit_reference_models(
    train_x: np.ndarray,
    random_seed: int = PRIMARY_SEED,
) -> tuple[dict, dict]:
    """Fit reusable estimator state on reference rows only."""
    train_x = np.asarray(train_x, dtype=np.float64)
    if train_x.ndim != 2:
        raise ValueError("Reference model inputs must be a two-dimensional matrix")
    _require_finite("Reference model inputs", train_x)
    n_train, n_features = train_x.shape
    if n_train < 2:
        raise ValueError("At least two reference participants are required")
    if n_train <= LOF_N_NEIGHBORS:
        raise ValueError(
            f"LOF requires more than {LOF_N_NEIGHBORS} reference participants; got {n_train}"
        )
    if n_train <= max(50, n_features + 1):
        raise ValueError(
            "Robust Mahalanobis requires more than max(50, feature_count + 1) "
            f"reference participants; got {n_train} for {n_features} features"
        )

    models = {
        "isolation_forest": IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            contamination="auto",
            random_state=random_seed,
            n_jobs=1,
        ),
        "local_outlier_factor": LocalOutlierFactor(
            n_neighbors=LOF_N_NEIGHBORS,
            novelty=True,
            contamination="auto",
            n_jobs=1,
        ),
        "one_class_svm": OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=OCSVM_NU,
        ),
    }
    for model in models.values():
        model.fit(train_x)

    # A one-hidden-layer MLP trained to reconstruct its own reference inputs
    # provides a compact regularized autoencoder without a new deep-learning
    # dependency. The held-out rows are only passed to predict().
    latent_width = max(2, min(8, n_features // 2))
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(latent_width,),
        activation="relu",
        solver="adam",
        alpha=AE_ALPHA,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=1e-3,
        max_iter=AE_MAX_ITER,
        early_stopping=AE_EARLY_STOPPING,
        validation_fraction=0.1,
        n_iter_no_change=AE_N_ITER_NO_CHANGE,
        random_state=random_seed,
        shuffle=True,
        tol=AE_TOL,
    )
    autoencoder.fit(train_x, train_x)
    models["regularized_autoencoder"] = autoencoder

    robust_model, covariance_details = fit_robust_mahalanobis(
        train_x, random_seed=random_seed
    )
    models["robust_mahalanobis"] = robust_model
    fit_details = {
        "robust_mahalanobis": covariance_details,
        "regularized_autoencoder": {
            "estimator": "one-hidden-layer MLPRegressor reconstruction autoencoder",
            "hidden_layer_sizes": [latent_width],
            "activation": "relu",
            "solver": "adam",
            "alpha_l2_regularization": AE_ALPHA,
            "max_iter": AE_MAX_ITER,
            "early_stopping": AE_EARLY_STOPPING,
            "validation_fraction_from_reference_only": 0.1,
            "n_iter_no_change": AE_N_ITER_NO_CHANGE,
            "tolerance": AE_TOL,
            "random_seed": random_seed,
            "iterations_run": int(autoencoder.n_iter_),
            "converged_before_iteration_limit": bool(autoencoder.n_iter_ < AE_MAX_ITER),
            "reached_iteration_limit": bool(autoencoder.n_iter_ >= AE_MAX_ITER),
            "best_validation_score": float(autoencoder.best_validation_score_),
            "final_training_loss": float(autoencoder.loss_),
            "score": "mean squared reconstruction error per held-out row; higher is more anomalous",
        },
    }
    return models, fit_details


def score_reference_models(models: dict, values: np.ndarray) -> dict[str, np.ndarray]:
    """Score rows with fitted models; higher scores mean more anomalous."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Scoring inputs must be a two-dimensional matrix")
    _require_finite("Scoring inputs", values)
    expected = {
        "isolation_forest", "local_outlier_factor", "one_class_svm",
        "regularized_autoencoder", "robust_mahalanobis",
    }
    if set(models) != expected:
        raise ValueError("Fitted model collection does not contain the five expected models")
    scores = {}
    for name in ("isolation_forest", "local_outlier_factor", "one_class_svm"):
        model = models[name]
        if values.shape[1] != model.n_features_in_:
            raise ValueError(f"{name} scoring feature count does not match fitted model")
        # These APIs return larger values for more inlier-like rows.
        scores[name] = -np.asarray(model.score_samples(values), dtype=np.float64)
        _require_finite(f"{name} anomaly scores", scores[name])

    autoencoder = models["regularized_autoencoder"]
    if values.shape[1] != autoencoder.n_features_in_:
        raise ValueError("Autoencoder scoring feature count does not match fitted model")
    reconstructed = np.asarray(autoencoder.predict(values), dtype=np.float64)
    if reconstructed.shape != values.shape:
        raise RuntimeError("Autoencoder reconstruction has an unexpected shape")
    scores["regularized_autoencoder"] = np.mean(
        np.square(values - reconstructed), axis=1
    )
    _require_finite("regularized_autoencoder anomaly scores", scores["regularized_autoencoder"])
    scores["robust_mahalanobis"] = models["robust_mahalanobis"].score_samples(values)
    _require_finite("Robust Mahalanobis anomaly scores", scores["robust_mahalanobis"])
    return scores


def fit_score_models(
    train_x: np.ndarray,
    test_x: np.ndarray,
    random_seed: int = PRIMARY_SEED,
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit on reference rows and score held-out rows with shared model code."""
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("Model input matrices have incompatible dimensions")
    _require_finite("Held-out model inputs", test_x)
    models, fit_details = fit_reference_models(train_x, random_seed=random_seed)
    return score_reference_models(models, test_x), fit_details


def fit_robust_mahalanobis(
    train_x: np.ndarray,
    random_seed: int = PRIMARY_SEED,
) -> tuple[RobustMahalanobisScorer, dict]:
    """Fit reusable MCD covariance state with an explicit eigenvalue floor."""
    train_x = np.asarray(train_x, dtype=np.float64)
    if train_x.ndim != 2:
        raise ValueError("Robust covariance reference inputs must be two-dimensional")
    _require_finite("Robust covariance reference inputs", train_x)
    if train_x.shape[0] <= train_x.shape[1] + 1:
        raise ValueError(
            "Robust covariance estimation requires more training rows than features + 1"
        )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            estimator = MinCovDet(random_state=random_seed, assume_centered=False)
            estimator.fit(train_x)
    except Exception as exc:
        raise RuntimeError(
            "Robust Mahalanobis covariance estimation failed on the reference "
            f"partition ({type(exc).__name__}: {exc})"
        ) from exc

    location = np.asarray(estimator.location_, dtype=np.float64)
    covariance = np.asarray(estimator.covariance_, dtype=np.float64)
    if covariance.shape != (train_x.shape[1], train_x.shape[1]):
        raise RuntimeError("Robust covariance estimator returned an unexpected matrix shape")
    _require_finite("Robust covariance location", location)
    _require_finite("Robust covariance matrix", covariance)
    covariance = (covariance + covariance.T) / 2.0

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"Robust covariance eigendecomposition failed: {exc}") from exc
    _require_finite("Robust covariance eigenvalues", eigenvalues)
    max_eigenvalue = float(np.max(eigenvalues))
    if max_eigenvalue <= 0:
        raise RuntimeError("Robust covariance is not positive in any direction")
    tolerance = np.finfo(np.float64).eps * max(1, covariance.shape[0]) * max_eigenvalue
    if float(np.min(eigenvalues)) < -100.0 * tolerance:
        raise RuntimeError(
            "Robust covariance has materially negative eigenvalues; refusing to score"
        )

    singular = bool(np.any(eigenvalues <= tolerance))
    raw_condition = (
        float(max_eigenvalue / np.min(eigenvalues))
        if np.min(eigenvalues) > 0
        else None
    )
    eigenvalue_floor = max_eigenvalue * MCD_EIGENVALUE_FLOOR_RATIO
    regularized_eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
    regularized_condition = float(
        np.max(regularized_eigenvalues) / np.min(regularized_eigenvalues)
    )
    if not math.isfinite(regularized_condition) or (
        regularized_condition > MAX_REGULARIZED_CONDITION_NUMBER
    ):
        raise RuntimeError(
            "Regularized robust covariance exceeds the maximum condition number: "
            f"{regularized_condition}"
        )

    model = RobustMahalanobisScorer(
        location=location,
        eigenvectors=eigenvectors,
        regularized_eigenvalues=regularized_eigenvalues,
    )
    details = {
        "estimator": "Minimum Covariance Determinant (MinCovDet)",
        "random_seed": random_seed,
        "covariance_singular_before_regularization": singular,
        "covariance_condition_number_before_regularization": raw_condition,
        "eigenvalue_floor_ratio": MCD_EIGENVALUE_FLOOR_RATIO,
        "eigenvalue_floor": eigenvalue_floor,
        "covariance_condition_number_after_regularization": regularized_condition,
        "covariance_regularized": singular or bool(np.min(eigenvalues) < eigenvalue_floor),
        "score_direction": "larger robust Mahalanobis distance is more anomalous",
    }
    return model, details


def robust_mahalanobis_scores(
    train_x: np.ndarray,
    test_x: np.ndarray,
    random_seed: int = PRIMARY_SEED,
) -> tuple[np.ndarray, dict]:
    """Fit and score MCD covariance, preserving the comparison API."""
    model, details = fit_robust_mahalanobis(train_x, random_seed=random_seed)
    return model.score_samples(test_x), details


def top_k_selection(scores: np.ndarray, fractions=TOP_FRACTIONS) -> tuple[dict, np.ndarray]:
    """Select exactly ceil(fraction * denominator) rows; ties use source row order."""
    scores = np.asarray(scores, dtype=np.float64)
    _require_finite("Scores used for top-k selection", scores)
    denominator = len(scores)
    if denominator == 0:
        raise ValueError("Cannot select top-k candidates from an empty held-out set")
    order = np.argsort(-scores, kind="stable")
    flags = {}
    for fraction in fractions:
        k = int(math.ceil(fraction * denominator))
        selected = np.zeros(denominator, dtype=bool)
        selected[order[:k]] = True
        label = f"top_{int(round(fraction * 100))}pct"
        flags[label] = selected
    return flags, order


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    a = pd.Series(first).rank(method="average", ascending=True)
    b = pd.Series(second).rank(method="average", ascending=True)
    value = a.corr(b, method="pearson")
    return None if pd.isna(value) else float(value)


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _summarize_comparisons(scores_by_variant: dict, ids_test: np.ndarray):
    score_rows = []
    rank_rows = []
    overlap_rows = []
    denominators = {"held_out_eligible_participants": int(len(ids_test))}
    for variant, scores in scores_by_variant.items():
        model_flags = {}
        for model_name, score_values in scores.items():
            score_values = np.asarray(score_values, dtype=np.float64)
            if len(score_values) != len(ids_test):
                raise ValueError(f"{variant}/{model_name} score rows do not align with held-out IDs")
            flags, order = top_k_selection(score_values)
            denominator = len(score_values)
            ranks = pd.Series(score_values).rank(method="average", ascending=False).to_numpy()
            for row_index, participant_id in enumerate(ids_test):
                row = {
                    ID_COLUMN: participant_id,
                    "preprocessing": variant,
                    "model": model_name,
                    "anomaly_score": float(score_values[row_index]),
                    "anomaly_score_direction": "higher_is_more_anomalous",
                    "anomaly_rank": float(ranks[row_index]),
                    "held_out_denominator": denominator,
                }
                for label, flag_values in flags.items():
                    row[f"{label}_candidate"] = bool(flag_values[row_index])
                score_rows.append(row)
            model_flags[model_name] = flags

        for first, second in combinations(scores, 2):
            corr = _rank_correlation(scores[first], scores[second])
            rank_rows.append({
                "preprocessing": variant,
                "model_a": first,
                "model_b": second,
                "spearman_rank_correlation": corr,
                "held_out_denominator": len(ids_test),
                "interpretation": "ranking agreement only; not clinical validity",
            })
            for label, first_flag_values in model_flags[first].items():
                second_flags = model_flags[second][label]
                a = set(ids_test[first_flag_values])
                b = set(ids_test[second_flags])
                overlap_rows.append({
                    "preprocessing": variant,
                    "model_a": first,
                    "model_b": second,
                    "top_k": label.removeprefix("top_").removesuffix("pct"),
                    "held_out_denominator": len(ids_test),
                    "selected_count_a": len(a),
                    "selected_count_b": len(b),
                    "intersection_count": len(a & b),
                    "jaccard_overlap": _jaccard(a, b),
                    "tie_policy": "stable source-row order; exactly ceil(fraction * denominator) selected",
                })
    return pd.DataFrame(score_rows), pd.DataFrame(rank_rows), pd.DataFrame(overlap_rows)


def _stability_rows(split: PreparedSplit, include_missingness: bool) -> list[dict]:
    variant = "with_missingness_indicators" if include_missingness else "without_missingness_indicators"
    train_x, test_x, _ = fit_reference_transform(
        split.raw_train, split.raw_test, split.missing_train, split.missing_test,
        include_missingness,
    )
    primary_scores, _ = fit_score_models(train_x, test_x, PRIMARY_SEED)
    stability = []
    for seed in STABILITY_SEEDS:
        repeat_scores, _ = fit_score_models(train_x, test_x, seed)
        for model_name in primary_scores:
            if model_name in {"local_outlier_factor", "one_class_svm"}:
                continue  # Deterministic for a fixed split and parameters.
            for label, flags in top_k_selection(primary_scores[model_name])[0].items():
                repeated_flags = top_k_selection(repeat_scores[model_name])[0][label]
                a = set(split.ids_test[flags])
                b = set(split.ids_test[repeated_flags])
                stability.append({
                    "preprocessing": variant,
                    "model": model_name,
                    "primary_seed": PRIMARY_SEED,
                    "repeat_seed": seed,
                    "top_k": label.removeprefix("top_").removesuffix("pct"),
                    "held_out_denominator": len(split.ids_test),
                    "spearman_rank_correlation": _rank_correlation(
                        primary_scores[model_name], repeat_scores[model_name]
                    ),
                    "top_k_jaccard_overlap": _jaccard(a, b),
                    "interpretation": "algorithm-seed ranking stability on one fixed split; not clinical validity",
                })
    return stability


def _feature_definitions(schema: dict, feature_columns: list[str]) -> list[dict]:
    by_column = {
        entry["source_column"]: (name, entry)
        for name, entry in schema["biomarkers"].items()
    }
    result = []
    for column in feature_columns:
        if column not in by_column:
            raise ValueError(f"Feature {column} is absent from canonical biomarker schema")
        canonical_name, entry = by_column[column]
        result.append({
            "source_column": column,
            "canonical_name": canonical_name,
            "display_name": entry["display_name"],
            "unit": entry["canonical_unit"],
            "included": True,
            "selection_note": "Existing CBC anomaly feature set; absolute differential counts omitted as derived from total WBC and differential percentages.",
        })
    return result


def run_comparison(
    frame: pd.DataFrame,
    schema: dict,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    random_seed: int = PRIMARY_SEED,
    test_size: float = TEST_SIZE,
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment directory: {output_dir}. "
            "Choose a new directory with --output-dir."
        )

    split = make_split(
        frame, feature_columns=feature_columns,
        random_seed=random_seed, test_size=test_size,
    )
    if set(split.ids_train) & set(split.ids_test):
        raise RuntimeError("Reference and held-out participant IDs overlap")
    if len(split.ids_test) != int(math.ceil(split.eligible_count * test_size)):
        # sklearn rounds according to ceil for float test_size; keep explicit.
        raise RuntimeError("Unexpected held-out participant count")

    scores_by_variant = {}
    preprocessing_details = {}
    fit_details_by_variant = {}
    variant_models = {}
    for include in (False, True):
        variant = "with_missingness_indicators" if include else "without_missingness_indicators"
        train_x, test_x, prep = fit_reference_transform(
            split.raw_train, split.raw_test, split.missing_train, split.missing_test,
            include,
        )
        scores, fit_details = fit_score_models(train_x, test_x, random_seed)
        if any(len(values) != len(split.ids_test) for values in scores.values()):
            raise RuntimeError(f"Models did not score identical held-out rows for {variant}")
        scores_by_variant[variant] = scores
        preprocessing_details[variant] = prep
        fit_details_by_variant[variant] = fit_details
        variant_models[variant] = {
            "reference_model_input_shape": list(train_x.shape),
            "held_out_model_input_shape": list(test_x.shape),
            "all_scores_finite": all(np.isfinite(value).all() for value in scores.values()),
            "held_out_ids_identical_across_models": True,
        }

    score_frame, rank_frame, overlap_frame = _summarize_comparisons(
        scores_by_variant, split.ids_test
    )
    stability_frame = pd.DataFrame(
        _stability_rows(split, include_missingness=False)
        + _stability_rows(split, include_missingness=True)
    )

    # Save in original source-row order within each model/condition for easy joins.
    source_order_map = {
        participant_id: int(position)
        for position, participant_id in enumerate(frame[ID_COLUMN].to_numpy())
    }
    score_frame["_source_order"] = score_frame[ID_COLUMN].map(source_order_map)
    score_frame = score_frame.sort_values(
        ["preprocessing", "model", "_source_order"], kind="stable"
    ).drop(columns="_source_order").reset_index(drop=True)

    metadata = {
        "experiment": "NutriFix held-out CBC anomaly model comparison",
        "purpose": "Exploratory statistical ranking comparison; not a clinically validated diagnostic model",
        "input_file": str(INPUT_FILE),
        "participant_id_column": ID_COLUMN,
        "participant_id_used_as_model_feature": False,
        "feature_definitions": _feature_definitions(schema, feature_columns),
        "excluded_features": {
            "absolute_differential_counts": "Omitted as derived from total WBC and differential percentages",
            "ferritin": "Not part of the existing CBC anomaly feature set",
        },
        "eligibility_rule": "Include participant if at least one of the 15 selected CBC features is observed; exclude participants with all 15 missing.",
        "cohort_counts": {
            "input_participants": int(len(frame)),
            "eligible_participants": int(split.eligible_count),
            "excluded_all_selected_features_missing": int(split.excluded_all_missing_count),
            "reference_participants": int(len(split.ids_train)),
            "held_out_participants": int(len(split.ids_test)),
            "eligible_participants_with_partial_missingness": int(
                (np.isnan(np.vstack([split.raw_train, split.raw_test])).any(axis=1)).sum()
            ),
            "unexpected_non_numeric_cells_coerced_to_missing": split.non_numeric_count,
        },
        "split": {
            "method": "train_test_split over eligible participant row positions",
            "random_seed": random_seed,
            "test_fraction": test_size,
            "shuffle": True,
            "stratified": False,
            "single_split_limitation": "One random split can be sensitive to cohort composition. The repeat-fit stability check changes estimator seeds on this same split and does not measure split-to-split or prospective generalization.",
        },
        "preprocessing_by_condition": preprocessing_details,
        "models": {
            "isolation_forest": {
                "n_estimators": IF_N_ESTIMATORS,
                "contamination": "auto; native threshold is not used for model comparison",
                "score": "negative score_samples; higher means more anomalous",
                "random_seed": random_seed,
            },
            "local_outlier_factor": {
                "n_neighbors": LOF_N_NEIGHBORS,
                "novelty": True,
                "contamination": "auto; native threshold is not used for model comparison",
                "score": "negative score_samples; higher means more anomalous",
                "random_seed": None,
            },
            "one_class_svm": {
                "kernel": "rbf",
                "gamma": "scale",
                "nu": OCSVM_NU,
                "native_threshold": "not used for cross-model comparison",
                "score": "negative score_samples; higher means more anomalous",
                "random_seed": None,
            },
            "robust_mahalanobis": {
                "estimator": "Minimum Covariance Determinant (MinCovDet)",
                "covariance_diagnostics_by_condition": {
                    variant: details["robust_mahalanobis"]
                    for variant, details in fit_details_by_variant.items()
                },
                "score": "regularized robust Mahalanobis distance; higher means more anomalous",
            },
            "regularized_autoencoder": {
                "estimator": "one-hidden-layer sklearn MLPRegressor trained for input reconstruction",
                "activation": "relu",
                "solver": "adam",
                "latent_width_rule": "max(2, min(8, input_feature_count // 2))",
                "alpha_l2_regularization": AE_ALPHA,
                "max_iter": AE_MAX_ITER,
                "early_stopping": AE_EARLY_STOPPING,
                "validation_fraction_from_reference_only": 0.1,
                "tolerance": AE_TOL,
                "n_iter_no_change": AE_N_ITER_NO_CHANGE,
                "random_seed": random_seed,
                "fit_diagnostics_by_condition": {
                    variant: details["regularized_autoencoder"]
                    for variant, details in fit_details_by_variant.items()
                },
                "score": "mean squared reconstruction error per held-out row; higher means more anomalous",
            },
        },
        "held_out_scoring": variant_models,
        "top_k_policy": {
            "denominator": "number of eligible held-out participants, identical for all models within a condition",
            "fractions": list(TOP_FRACTIONS),
            "count_rule": "ceil(fraction * held-out denominator), so each selection has exactly that many participants",
            "tie_rule": "Stable sort by descending score; exact ties retain original merged-dataset row order. SEQN is not used for ranking or tie-breaking.",
        },
        "stability": {
            "repeat_seeds": list(STABILITY_SEEDS),
            "measures": "Spearman score-rank correlation and top-k Jaccard overlap against primary seed",
            "limitation": "Algorithm-seed ranking stability on one fixed split only; not clinical validity.",
        },
        "clinical_reference_ranges": {
            "implemented": False,
            "reason": "The canonical schema supplies identities and units but no authoritative applicable intervals. No clinical thresholds were invented.",
            "required_metadata": [
                "authoritative source, citation, version, and effective date for each interval",
                "lower and upper bounds with unit and inclusive/exclusive semantics",
                "population applicability, including age and sex where relevant",
                "pregnancy status and other inclusion/exclusion conditions where relevant",
                "specimen, assay, and laboratory-method context needed to apply the interval",
                "policy for unavailable or mismatched demographic/context data",
            ],
        },
        "software_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "limitations": [
            "No validated clinical labels are used; no diagnostic accuracy is estimated.",
            "Agreement, rank correlation, and stability describe statistical behavior only.",
            "A single reference/test split does not establish prospective generalization.",
            "Median imputation may alter multivariate structure; missingness-indicator sensitivity is reported separately.",
            "Robust covariance is regularized when singular or ill-conditioned; the regularization diagnostics are reported.",
            "NHANES survey design and weights are not used in this exploratory model comparison.",
        ],
    }

    # Recheck just before creating the output directory; mkdir(exist_ok=False)
    # is the final guard against overwriting an earlier run.
    output_dir.mkdir(parents=True, exist_ok=False)
    score_frame.to_csv(output_dir / "held_out_scores.csv", index=False)
    rank_frame.to_csv(output_dir / "pairwise_rank_correlations.csv", index=False)
    overlap_frame.to_csv(output_dir / "pairwise_top_k_overlap.csv", index=False)
    stability_frame.to_csv(output_dir / "ranking_stability.csv", index=False)
    with (output_dir / "experiment_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
    return metadata


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--schema", type=Path, default=SCHEMA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(
            f"output directory exists; refusing to overwrite: {args.output_dir}. "
            "Choose a new path with --output-dir."
        )
    if not args.input.exists():
        parser.error(f"input dataset does not exist: {args.input}")
    if not args.schema.exists():
        parser.error(f"canonical schema does not exist: {args.schema}")

    frame = pd.read_parquet(args.input)
    with args.schema.open(encoding="utf-8") as file:
        schema = json.load(file)
    metadata = run_comparison(
        frame, schema, output_dir=args.output_dir,
        random_seed=args.seed, test_size=args.test_size,
    )
    print(json.dumps(metadata["cohort_counts"], indent=2))
    print(f"Comparison outputs saved to: {args.output_dir}")
    print("Scores are exploratory statistical signals, not medical diagnoses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
