
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "nhanes_anomaly_features.parquet"

ID_COLUMN = "SEQN"


def main():
    df = pd.read_parquet(INPUT_FILE)
    X = df.drop(columns=[ID_COLUMN])

    duplicate_mask = X.duplicated(keep=False)
    duplicated_rows = X.loc[duplicate_mask].copy()

    # How often each feature vector occurs.
    group_sizes = X.value_counts(dropna=False)

    print("Total participants:", len(X))
    print("Unique feature rows:", len(X.drop_duplicates()))
    print("Rows belonging to a repeated feature vector:", int(duplicate_mask.sum()))
    print("Number of repeated feature vectors:", int((group_sizes > 1).sum()))
    print("Largest repeated group:", int(group_sizes.max()))
    print("\nMost common feature-vector frequencies:")
    print(group_sizes[group_sizes > 1].value_counts().sort_index().to_string())

    # Examine missingness indicators among repeated vectors.
    indicator_columns = [
        c for c in X.columns if c.endswith("_was_missing")
    ]

    if indicator_columns:
        print("\nMissingness-indicator totals:")
        print(X[indicator_columns].sum().sort_values(ascending=False).to_string())

        print(
            "\nRows with all selected biomarkers originally observed:",
            int((X[indicator_columns].sum(axis=1) == 0).sum()),
        )

    # Export only aggregate diagnostics; don't expose participant IDs.
    report = pd.DataFrame({
        "metric": [
            "total_rows",
            "unique_feature_rows",
            "rows_in_repeated_groups",
            "repeated_feature_groups",
            "largest_repeated_group",
        ],
        "value": [
            len(X),
            int(X.drop_duplicates().shape[0]),
            int(duplicate_mask.sum()),
            int((group_sizes > 1).sum()),
            int(group_sizes.max()),
        ],
    })

    output_file = ROOT / "outputs" / "metrics" / "feature_duplicate_audit.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_file, index=False)

    print(f"\nAggregate audit saved: {output_file}")


if __name__ == "__main__":
    main()
