
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "nhanes_anomaly_features.parquet"

ID_COLUMN = "SEQN"
CONTAMINATIONS = [0.02, 0.05, 0.10]


def main():
    df = pd.read_parquet(INPUT_FILE)
    X = df.drop(columns=[ID_COLUMN])

    print("scikit-learn version:", sklearn.__version__)
    print("Dataset shape:", X.shape)
    print("Missing values:", int(X.isna().sum().sum()))
    print("Unique rows:", int(X.drop_duplicates().shape[0]))
    print()

    for contamination in CONTAMINATIONS:
        model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)

        scores = model.score_samples(X)
        decision = model.decision_function(X)
        predictions = model.predict(X)

        print(f"Contamination: {contamination}")
        print("  offset_:", model.offset_)
        print("  prediction counts:", pd.Series(predictions).value_counts().to_dict())
        print("  score min/max:", float(scores.min()), float(scores.max()))
        print("  decision min/max:", float(decision.min()), float(decision.max()))
        print("  anomalies from predict:", int((predictions == -1).sum()))
        print("  anomalies from decision < 0:", int((decision < 0).sum()))
        print()


if __name__ == "__main__":
    main()
