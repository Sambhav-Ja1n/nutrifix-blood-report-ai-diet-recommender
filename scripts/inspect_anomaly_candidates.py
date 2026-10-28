
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "nhanes_2017_2018_merged.parquet"
PREDICTIONS_FILE = (
    ROOT / "outputs" / "metrics" / "isolation_forest_baseline_predictions.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "metrics"

ID_COLUMN = "SEQN"

BIOMARKERS = [
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(DATA_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    required = [ID_COLUMN, *BIOMARKERS]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Missing biomarker columns: {missing}")

    top = predictions.sort_values(
        "anomaly_score", ascending=False
    ).head(10)

    # Keep the original measurements, including missing values.
    candidate_data = top[[ID_COLUMN, "anomaly_score"]].merge(
        data[required],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    # Descriptive cohort statistics: no values are modified.
    reference = data[BIOMARKERS].agg(
        ["count", "median"]
    ).T

    reference["q1"] = data[BIOMARKERS].quantile(0.25)
    reference["q3"] = data[BIOMARKERS].quantile(0.75)
    reference["iqr"] = reference["q3"] - reference["q1"]

    reference_file = OUTPUT_DIR / "cohort_biomarker_reference.csv"
    reference.to_csv(reference_file, index_label="biomarker")

    candidate_file = OUTPUT_DIR / "top10_anomaly_biomarkers.csv"
    candidate_data.to_csv(candidate_file, index=False)

    print("Top 10 anomaly candidates with original biomarker values:")
    print(candidate_data.to_string(index=False))

    print("\nCohort biomarker reference statistics:")
    print(reference.to_string())

    print(f"\nCandidate report: {candidate_file}")
    print(f"Reference report: {reference_file}")


if __name__ == "__main__":
    main()
