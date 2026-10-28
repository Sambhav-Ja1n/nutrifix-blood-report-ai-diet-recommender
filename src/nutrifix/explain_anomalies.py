
from pathlib import Path

import numpy as np
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
OUTPUT_FILE = (
    ROOT / "outputs" / "metrics"
    / "anomaly_explanations.csv"
)

ID_COLUMN = "SEQN"

CBC_FEATURES = [
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
    predictions = pd.read_csv(PREDICTIONS_FILE)
    data = pd.read_parquet(DATA_FILE)

    predictions[ID_COLUMN] = predictions[ID_COLUMN].astype("int64")
    data[ID_COLUMN] = data[ID_COLUMN].astype("int64")

    available_features = [
        col for col in CBC_FEATURES if col in data.columns
    ]
    missing_columns = sorted(set(CBC_FEATURES) - set(available_features))

    if missing_columns:
        raise ValueError(
            f"Original dataset is missing CBC columns: {missing_columns}"
        )

    # Preserve only the participant ID and original CBC measurements.
    original = data[[ID_COLUMN] + available_features].copy()

    # Prevent duplicate IDs from multiplying rows during the merge.
    if original[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate participant IDs found in original data.")

    df = predictions.merge(
        original,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if (df["_merge"] != "both").any():
        unmatched = int((df["_merge"] != "both").sum())
        raise ValueError(
            f"{unmatched} prediction IDs did not match the original dataset."
        )

    df = df.drop(columns="_merge")

    # Use observed original measurements for reference statistics.
    reference = original[available_features]
    means = reference.mean()
    stds = reference.std(ddof=0).replace(0, np.nan)

    candidates = df[df["is_anomaly_candidate"].eq(True)]
    rows = []

    for _, participant in candidates.iterrows():
        for feature in available_features:
            value = participant[feature]

            # Missing original values are not explanations.
            if pd.isna(value) or pd.isna(stds[feature]):
                continue

            z_score = (value - means[feature]) / stds[feature]

            rows.append({
                ID_COLUMN: int(participant[ID_COLUMN]),
                "anomaly_score": float(participant["anomaly_score"]),
                "biomarker_feature": feature,
                "observed_value": float(value),
                "cohort_mean": float(means[feature]),
                "cohort_std": float(stds[feature]),
                "cohort_relative_z_score": float(z_score),
                "absolute_z_score": abs(float(z_score)),
            })

    columns = [
        ID_COLUMN,
        "anomaly_score",
        "biomarker_feature",
        "observed_value",
        "cohort_mean",
        "cohort_std",
        "cohort_relative_z_score",
        "absolute_z_score",
        "feature_rank",
    ]

    explanations = pd.DataFrame(rows)

    if not explanations.empty:
        explanations = explanations.sort_values(
            [ID_COLUMN, "absolute_z_score"],
            ascending=[True, False],
        )
        explanations["feature_rank"] = (
            explanations.groupby(ID_COLUMN).cumcount() + 1
        )
    else:
        explanations = pd.DataFrame(columns=columns)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    explanations.to_csv(OUTPUT_FILE, index=False)

    print("Anomaly explanation generation completed.")
    print("Anomaly candidates:", len(candidates))
    print("Candidates with at least one observed CBC feature:",
          explanations[ID_COLUMN].nunique())
    print("Explanation rows:", len(explanations))
    print("Output:", OUTPUT_FILE)

    if not explanations.empty:
        print("\nTop explanations:")
        print(
            explanations.groupby(ID_COLUMN)
            .head(3)
            .head(15)
            .to_string(index=False)
        )

    print(
        "\nCaution: cohort-relative z-scores describe statistical "
        "unusualness, not clinical severity, causation, or diagnosis."
    )


if __name__ == "__main__":
    main()
