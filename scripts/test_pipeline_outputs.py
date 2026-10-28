
from pathlib import Path

import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    ROOT / "outputs" / "metrics"
    / "isolation_forest_baseline_predictions.csv"
)
EXPLANATIONS_FILE = (
    ROOT / "outputs" / "metrics"
    / "anomaly_explanations.csv"
)
REPORT_FILE = (
    ROOT / "outputs" / "metrics"
    / "participant_anomaly_report.csv"
)
SUMMARY_FILE = (
    ROOT / "outputs" / "metrics"
    / "isolation_forest_baseline_summary.json"
)


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main():
    for path in [
        PREDICTIONS_FILE,
        EXPLANATIONS_FILE,
        REPORT_FILE,
        SUMMARY_FILE,
    ]:
        check(path.exists(), f"Output exists: {path.name}")

    predictions = pd.read_csv(PREDICTIONS_FILE)
    explanations = pd.read_csv(EXPLANATIONS_FILE)
    report = pd.read_csv(REPORT_FILE, low_memory=False)
    summary = json.loads(SUMMARY_FILE.read_text())

    check(
        predictions["SEQN"].is_unique,
        "Prediction participant IDs are unique",
    )
    check(
        len(predictions) == 8366,
        "Prediction file covers all 8,366 participants",
    )
    check(
        int(predictions["is_anomaly_candidate"].eq(True).sum()) == 377,
        "Expected 377 anomaly candidates at current model settings",
    )
    check(
        len(report) == len(predictions),
        "Participant report preserves all participants",
    )
    check(
        report["SEQN"].is_unique,
        "Participant report IDs are unique",
    )
    check(
        set(explanations["SEQN"]).issubset(set(predictions["SEQN"])),
        "Every explanation belongs to a known participant",
    )
    check(
        explanations["observed_value"].notna().all(),
        "Explanations contain observed, nonmissing measurements",
    )
    check(
        report["anomaly_score"].notna().equals(
            predictions["anomaly_score"].notna()
        ),
        "Report retains prediction scores",
    )
    check(
        summary.get("participants_in_cohort") == len(predictions),
        "Model summary participant count matches predictions",
    )

    print("\nAll pipeline output checks passed.")


if __name__ == "__main__":
    main()
