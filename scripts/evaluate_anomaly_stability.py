
from pathlib import Path
from itertools import combinations
import json

import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "nhanes_anomaly_features.parquet"
OUTPUT_DIR = ROOT / "outputs" / "metrics"

ID_COLUMN = "SEQN"
CONTAMINATIONS = [0.02, 0.05, 0.10]
N_ESTIMATORS = 300
RANDOM_STATE = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_FILE)

    if ID_COLUMN not in df.columns:
        raise ValueError(f"Missing ID column: {ID_COLUMN}")

    if df[ID_COLUMN].isna().any() or df[ID_COLUMN].duplicated().any():
        raise ValueError("Participant IDs must be unique and non-missing")

    feature_columns = [c for c in df.columns if c != ID_COLUMN]
    X = df[feature_columns]

    if X.isna().any().any():
        raise ValueError("Feature matrix contains missing values")

    flagged_ids = {}
    summaries = []

    for contamination in CONTAMINATIONS:
        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X)

        predictions = model.predict(X)
        scores = -model.score_samples(X)
        flagged = predictions == -1

        flagged_ids[contamination] = set(df.loc[flagged, ID_COLUMN])

        summaries.append({
            "contamination_setting": contamination,
            "participants": len(df),
            "flagged_count": int(flagged.sum()),
            "flagged_percentage": round(float(flagged.mean() * 100), 2),
            "mean_anomaly_score_flagged": float(scores[flagged].mean()),
            "median_anomaly_score_flagged": float(
                pd.Series(scores[flagged]).median()
            ),
        })

    pairwise = []
    for first, second in combinations(CONTAMINATIONS, 2):
        a = flagged_ids[first]
        b = flagged_ids[second]
        union = a | b
        intersection = a & b

        pairwise.append({
            "setting_a": first,
            "setting_b": second,
            "overlap_count": len(intersection),
            "jaccard_similarity": round(
                len(intersection) / len(union), 4
            ) if union else 1.0,
        })

    summary_path = OUTPUT_DIR / "anomaly_stability_summary.csv"
    overlap_path = OUTPUT_DIR / "anomaly_stability_overlap.csv"

    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.DataFrame(pairwise).to_csv(overlap_path, index=False)

    print("Isolation Forest threshold stability evaluation completed.")
    print("\nCandidate counts:")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\nPairwise overlap:")
    print(pd.DataFrame(pairwise).to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {overlap_path}")

    report = {
        "purpose": "Exploratory threshold sensitivity, not clinical validation",
        "contamination_settings": CONTAMINATIONS,
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "feature_count": len(feature_columns),
        "note": (
            "The same full-cohort prepared matrix is used for each setting. "
            "This measures threshold sensitivity, not generalization to new data."
        ),
    }
    with (OUTPUT_DIR / "anomaly_stability_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()

