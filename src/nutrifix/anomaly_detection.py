
from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import IsolationForest


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "nhanes_anomaly_features.parquet"
METRICS_DIR = ROOT / "outputs" / "metrics"

ID_COLUMN = "SEQN"

CONTAMINATION = 0.05
RANDOM_STATE = 42
N_ESTIMATORS = 300


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Feature matrix not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    if ID_COLUMN not in df.columns:
        raise ValueError(f"Missing identifier column: {ID_COLUMN}")

    if df[ID_COLUMN].isna().any() or df[ID_COLUMN].duplicated().any():
        raise ValueError("Participant IDs must be present and unique")

    # Only numeric features are passed to the model; IDs are never features.
    feature_columns = [col for col in df.columns if col != ID_COLUMN]
    X = df[feature_columns]

    if X.empty:
        raise ValueError("No model input features found")

    if X.isna().any().any():
        raise ValueError("Feature matrix contains missing values")

    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        raise ValueError("All model input features must be numeric")

    # Missingness indicators identify participants with no observed
    # selected biomarkers. Such participants should not be scored.
    biomarker_columns = [
        col for col in feature_columns if not col.endswith("_was_missing")
    ]

    if not biomarker_columns:
        raise ValueError("No biomarker feature columns found")

    observed_any = (
        df[[f"{col}_was_missing" for col in biomarker_columns]]
        .eq(0)
        .any(axis=1)
    )

    eligible = observed_any
    X_eligible = X.loc[eligible]

    if len(X_eligible) < 2:
        raise ValueError("Not enough participants with observed biomarkers")

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_eligible)

    predictions = model.predict(X_eligible)
    decision_scores = model.decision_function(X_eligible)
    anomaly_scores = -model.score_samples(X_eligible)

    # Preserve all participants. Those with no observed selected biomarkers
    # remain in the report but are not assigned an anomaly prediction.
    results = pd.DataFrame({
        ID_COLUMN: df[ID_COLUMN],
        "isolation_forest_prediction": pd.Series(
            pd.NA, index=df.index, dtype="Int64"
        ),
        "is_anomaly_candidate": pd.Series(
            pd.NA, index=df.index, dtype="boolean"
        ),
        "decision_score": float("nan"),
        "anomaly_score": float("nan"),
        "scoring_status": "insufficient_biomarker_data",
    })

    results.loc[eligible, "isolation_forest_prediction"] = predictions
    results.loc[eligible, "is_anomaly_candidate"] = predictions == -1
    results.loc[eligible, "decision_score"] = decision_scores
    results.loc[eligible, "anomaly_score"] = anomaly_scores
    results.loc[eligible, "scoring_status"] = "scored"

    results = results.sort_values(
        "anomaly_score", ascending=False, na_position="last"
    ).reset_index(drop=True)

    output_file = METRICS_DIR / "isolation_forest_baseline_predictions.csv"
    results.to_csv(output_file, index=False)

    scored = results["scoring_status"].eq("scored")
    candidate_count = int(
        results.loc[scored, "is_anomaly_candidate"].sum()
    )

    scored_results = results.loc[scored]

    summary = {
        "model": "IsolationForest",
        "n_estimators": N_ESTIMATORS,
        "contamination": CONTAMINATION,
        "random_state": RANDOM_STATE,
        "participants_in_cohort": int(len(df)),
        "participants_scored": int(scored.sum()),
        "participants_insufficient_biomarker_data": int((~scored).sum()),
        "input_feature_count": int(len(feature_columns)),
        "anomaly_candidates": candidate_count,
        "anomaly_candidate_percentage": round(
            candidate_count / int(scored.sum()) * 100, 2
        ),
        "minimum_anomaly_score": float(scored_results["anomaly_score"].min()),
        "maximum_anomaly_score": float(scored_results["anomaly_score"].max()),
        "interpretation": (
            "Exploratory statistical anomaly candidates only. "
            "Not medical diagnoses or validated abnormal-result classifications."
        ),
        "preprocessing_note": (
            "The current feature matrix was imputed and scaled using the "
            "full cohort. This is a baseline experiment, not a held-out "
            "prospective evaluation."
        ),
    }

    summary_file = METRICS_DIR / "isolation_forest_baseline_summary.json"
    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("NutriFix Isolation Forest baseline completed.")
    print(json.dumps(summary, indent=2))
    print(f"\nPredictions saved: {output_file}")
    print(f"Summary saved: {summary_file}")
    print("\nTop 10 scored anomaly candidates:")
    print(scored_results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
