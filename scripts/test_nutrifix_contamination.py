
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "nhanes_anomaly_features.parquet"

ID_COLUMN = "SEQN"
SETTINGS = [0.02, 0.05, 0.10]


def main():
    df = pd.read_parquet(INPUT_FILE)
    X = df.drop(columns=[ID_COLUMN])

    indicator_columns = [
        c for c in X.columns if c.endswith("_was_missing")
    ]

    # Exclude participants whose every selected CBC value was missing.
    has_observed_biomarker = X[indicator_columns].sum(axis=1) < len(
        indicator_columns
    )
    X_observed = X.loc[has_observed_biomarker].copy()

    print("All participants:", len(X))
    print("Participants with at least one observed CBC biomarker:", len(X_observed))
    print("Excluded from this diagnostic only:", len(X) - len(X_observed))
    print()

    for contamination in SETTINGS:
        model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_observed)

        predictions = model.predict(X_observed)
        scores = model.score_samples(X_observed)

        print(f"Contamination: {contamination}")
        print("  offset:", model.offset_)
        print("  flagged:", int((predictions == -1).sum()))
        print("  flagged percentage:", round((predictions == -1).mean() * 100, 2))
        print("  score min/max:", float(scores.min()), float(scores.max()))
        print()


if __name__ == "__main__":
    main()
