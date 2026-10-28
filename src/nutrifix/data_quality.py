
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "nhanes_2017_2018_merged.parquet"
SCHEMA_FILE = ROOT / "schemas" / "canonical_biomarkers.json"
AUDIT_DIR = ROOT / "outputs" / "dataset_audits"

ID_COLUMN = "SEQN"

# Broad technical screening bounds, not clinical diagnostic cutoffs.
CHECKS = {
    "LBXWBCSI": (0, 500, "WBC; investigate unusually high values"),
    "LBXRBCSI": (0, 20, "RBC; investigate unusually high values"),
    "LBXHGB": (0, 30, "Hemoglobin; investigate unusually high values"),
    "LBXHCT": (0, 100, "Hematocrit percentage"),
    "LBXMCVSI": (0, 200, "MCV; investigate unusual values"),
    "LBXMCHSI": (0, 100, "MCH; investigate unusual values"),
    "LBXMC": (0, 100, "MCHC; investigate unusual values"),
    "LBXRDW": (0, 100, "RDW percentage"),
    "LBXPLTSI": (0, 5000, "Platelets; investigate unusual values"),
    "LBXMPSI": (0, 100, "MPV; investigate unusual values"),
    "LBXFER": (0, 100000, "Ferritin; investigate unusual values"),
}


def load_dataset():
    """Load the merged NHANES dataset and canonical biomarker schema."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_FILE}")

    df = pd.read_parquet(DATA_FILE)

    with SCHEMA_FILE.open(encoding="utf-8") as file:
        schema = json.load(file)

    expected_columns = {
        definition["source_column"]
        for definition in schema["biomarkers"].values()
    }

    missing_columns = sorted(expected_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Schema columns missing from dataset: {missing_columns}"
        )

    if ID_COLUMN not in df.columns:
        raise ValueError(f"Identifier column {ID_COLUMN} is missing")

    return df, expected_columns


def analyze_missingness(df, expected_columns):
    """Create the existing biomarker missingness report."""
    columns = sorted(expected_columns)

    summary = pd.DataFrame({
        "biomarker_column": columns,
        "missing_count": [int(df[col].isna().sum()) for col in columns],
    })

    summary["missing_percentage"] = (
        summary["missing_count"] / len(df) * 100
    ).round(2)

    summary["non_missing_count"] = len(df) - summary["missing_count"]

    return summary.sort_values(
        "missing_percentage", ascending=False
    ).reset_index(drop=True)


def validate_dataset(df, expected_columns):
    """Generate numeric quality, range flag, and dataset summary reports."""
    numeric_rows = []

    for column in sorted(expected_columns):
        original = df[column]
        converted = pd.to_numeric(original, errors="coerce")

        numeric_rows.append({
            "column": column,
            "missing_count": int(original.isna().sum()),
            "missing_percentage": round(original.isna().mean() * 100, 2),
            "non_numeric_count": int(
                (original.notna() & converted.isna()).sum()
            ),
            "unique_non_missing_values": int(
                original.nunique(dropna=True)
            ),
        })

    numeric_report = pd.DataFrame(numeric_rows)

    range_rows = []
    for column, (lower, upper, reason) in CHECKS.items():
        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce")
        flagged = values.notna() & (
            (values < lower) | (values > upper)
        )

        range_rows.append({
            "column": column,
            "lower_screening_bound": lower,
            "upper_screening_bound": upper,
            "flagged_count": int(flagged.sum()),
            "flagged_percentage": round(flagged.mean() * 100, 2),
            "purpose": reason,
        })

    range_report = pd.DataFrame(range_rows)

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "unique_participant_ids": int(df[ID_COLUMN].nunique()),
        "null_participant_ids": int(df[ID_COLUMN].isna().sum()),
        "duplicate_participant_ids": int(df[ID_COLUMN].duplicated().sum()),
        "schema_biomarkers": int(len(expected_columns)),
        "missing_schema_columns": [],
        "numeric_columns_checked": int(len(expected_columns)),
        "range_checks_run": int(len(range_rows)),
        "note": (
            "Range flags are broad screening checks, not clinical diagnoses. "
            "No records or values were modified."
        ),
    }

    return numeric_report, range_report, summary


def main():
    """Run all data-quality checks and save their reports."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    df, expected_columns = load_dataset()

    missingness = analyze_missingness(df, expected_columns)
    numeric_report, range_report, summary = validate_dataset(
        df, expected_columns
    )

    missingness.to_csv(
        AUDIT_DIR / "biomarker_missingness.csv", index=False
    )
    numeric_report.to_csv(
        AUDIT_DIR / "dataset_numeric_quality.csv", index=False
    )
    range_report.to_csv(
        AUDIT_DIR / "dataset_range_flags.csv", index=False
    )

    with (AUDIT_DIR / "dataset_quality_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("NutriFix dataset quality checks completed.")
    print(f"Rows: {len(df):,}")
    print(f"Schema biomarkers checked: {len(expected_columns)}")
    print("\nMissingness report:")
    print(missingness.to_string(index=False))
    print("\nRange screening results:")
    print(range_report.to_string(index=False))
    print("\nDataset summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nReports saved in: {AUDIT_DIR}")


if __name__ == "__main__":
    main()
