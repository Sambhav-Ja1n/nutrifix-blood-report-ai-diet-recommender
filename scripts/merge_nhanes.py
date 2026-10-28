
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nhanes" / "2017_2018"
OUTPUT = ROOT / "data" / "processed"
AUDIT_OUTPUT = ROOT / "outputs" / "dataset_audits"

CBC_FILE = RAW / "CBC_J.xpt"
DEMO_FILE = RAW / "DEMO_J.xpt"
FERRITIN_FILE = RAW / "FERTIN_J.xpt"

ID_COLUMN = "SEQN"


def load_xpt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_sas(path, format="xport")

    if ID_COLUMN not in df.columns:
        raise ValueError(f"{ID_COLUMN} missing from {path.name}")

    if df[ID_COLUMN].isna().any():
        raise ValueError(f"Null participant IDs found in {path.name}")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError(f"Duplicate participant IDs found in {path.name}")

    return df


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    AUDIT_OUTPUT.mkdir(parents=True, exist_ok=True)

    cbc = load_xpt(CBC_FILE)
    demo = load_xpt(DEMO_FILE)
    ferritin = load_xpt(FERRITIN_FILE)

    # Keep every CBC participant and attach matching demographic/ferritin data.
    merged = cbc.merge(
        demo,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
        suffixes=("", "_demo"),
        indicator="_demo_merge",
    )

    merged = merged.merge(
        ferritin,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
        suffixes=("", "_ferritin"),
        indicator="_ferritin_merge",
    )

    audit = {
        "merge_type": "left joins, with CBC as the base cohort",
        "input_rows": {
            "cbc": len(cbc),
            "demographics": len(demo),
            "ferritin": len(ferritin),
        },
        "output_rows": len(merged),
        "unique_participant_ids": int(merged[ID_COLUMN].nunique()),
        "duplicate_participant_ids": int(merged[ID_COLUMN].duplicated().sum()),
        "demographics_matched": int(
            (merged["_demo_merge"] == "both").sum()
        ),
        "demographics_unmatched": int(
            (merged["_demo_merge"] == "left_only").sum()
        ),
        "ferritin_matched": int(
            (merged["_ferritin_merge"] == "both").sum()
        ),
        "ferritin_unmatched": int(
            (merged["_ferritin_merge"] == "left_only").sum()
        ),
        "all_three_matched": int(
            (
                (merged["_demo_merge"] == "both")
                & (merged["_ferritin_merge"] == "both")
            ).sum()
        ),
    }

    # Merge indicators are useful for auditing but are not model features.
    merged = merged.drop(
        columns=["_demo_merge", "_ferritin_merge"]
    )

    output_csv = OUTPUT / "nhanes_2017_2018_merged.csv"
    output_parquet = OUTPUT / "nhanes_2017_2018_merged.parquet"
    audit_json = AUDIT_OUTPUT / "merge_audit.json"

    merged.to_csv(output_csv, index=False)
    merged.to_parquet(output_parquet, index=False)

    with audit_json.open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)

    print("NHANES merge completed successfully.")
    print(f"Output rows: {len(merged):,}")
    print(f"Output columns: {len(merged.columns):,}")
    print(f"Unique participant IDs: {merged[ID_COLUMN].nunique():,}")
    print(f"Demographics matched: {audit['demographics_matched']:,}")
    print(f"Ferritin matched: {audit['ferritin_matched']:,}")
    print(f"Participants in all three: {audit['all_three_matched']:,}")
    print(f"\nCSV: {output_csv}")
    print(f"Parquet: {output_parquet}")
    print(f"Audit: {audit_json}")


if __name__ == "__main__":
    main()
