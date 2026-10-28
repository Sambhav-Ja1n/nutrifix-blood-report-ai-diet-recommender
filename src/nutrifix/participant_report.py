
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    ROOT / "outputs" / "metrics"
    / "isolation_forest_baseline_predictions.csv"
)
DATA_FILE = (
    ROOT / "data" / "processed"
    / "nhanes_2017_2018_merged.parquet"
)
EXPLANATIONS_FILE = (
    ROOT / "outputs" / "metrics"
    / "anomaly_explanations.csv"
)
OUTPUT_FILE = (
    ROOT / "outputs" / "metrics"
    / "participant_anomaly_report.csv"
)

ID_COLUMN = "SEQN"


def main():
    predictions = pd.read_csv(PREDICTIONS_FILE)
    data = pd.read_parquet(DATA_FILE)
    explanations = pd.read_csv(EXPLANATIONS_FILE)

    for frame in (predictions, data, explanations):
        if ID_COLUMN in frame.columns:
            frame[ID_COLUMN] = pd.to_numeric(
                frame[ID_COLUMN], errors="raise"
            ).astype("int64")

    if data[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate participant IDs in source dataset.")

    report = predictions.merge(
        data,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    # Summarize the strongest cohort-relative deviations per participant.
    if not explanations.empty:
        summary = (
            explanations.sort_values(
                [ID_COLUMN, "absolute_z_score"],
                ascending=[True, False],
            )
            .groupby(ID_COLUMN)
            .agg(
                explained_biomarker_count=("biomarker_feature", "count"),
                top_biomarker=("biomarker_feature", "first"),
                top_observed_value=("observed_value", "first"),
                top_cohort_z_score=("cohort_relative_z_score", "first"),
            )
            .reset_index()
        )

        report = report.merge(
            summary,
            on=ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
    else:
        report["explained_biomarker_count"] = 0
        report["top_biomarker"] = pd.NA
        report["top_observed_value"] = pd.NA
        report["top_cohort_z_score"] = pd.NA

    report["explained_biomarker_count"] = (
        report["explained_biomarker_count"].fillna(0).astype(int)
    )

    report["report_note"] = (
        "Exploratory statistical screening only; not a medical diagnosis."
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_FILE, index=False)

    print("Participant report generated.")
    print("Total participants:", len(report))
    print(
        "Anomaly candidates:",
        int(report["is_anomaly_candidate"].eq(True).sum()),
    )
    print(
        "Candidates with at least one observed explanation:",
        int(report["top_biomarker"].notna().sum()),
    )
    print("Output:", OUTPUT_FILE)

    print("\nSample anomaly reports:")
    sample = report[report["is_anomaly_candidate"].eq(True)].head(5)
    columns = [
        ID_COLUMN,
        "anomaly_score",
        "top_biomarker",
        "top_observed_value",
        "top_cohort_z_score",
        "report_note",
    ]
    print(sample[columns].to_string(index=False))


if __name__ == "__main__":
    main()
