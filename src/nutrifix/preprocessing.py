
from pathlib import Path
import json

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "nhanes_2017_2018_merged.parquet"
OUTPUT_DIR = ROOT / "data" / "processed"
AUDIT_DIR = ROOT / "outputs" / "dataset_audits"

ID_COLUMN = "SEQN"

# Initial feature set: CBC measurements with distinct interpretations.
# Absolute differential counts are omitted because they are derived from
# total WBC and differential percentages.
FEATURE_COLUMNS = [
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
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    required = [ID_COLUMN, *FEATURE_COLUMNS]
    missing_columns = [col for col in required if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")

    if df[ID_COLUMN].isna().any():
        raise ValueError("Participant IDs contain missing values")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Participant IDs contain duplicates")

    # Convert selected features to numeric; unexpected text becomes missing
    # and is counted in the audit report.
    X_raw = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    unexpected_non_numeric = {
        col: int((df[col].notna() & X_raw[col].isna()).sum())
        for col in FEATURE_COLUMNS
    }

    # Remove features that contain no usable observations.
    all_missing = [
        col for col in FEATURE_COLUMNS if X_raw[col].notna().sum() == 0
    ]
    if all_missing:
        raise ValueError(f"Features entirely missing: {all_missing}")

    # Preserve the original cohort and record missingness before imputation.
    missing_indicators = X_raw.isna().astype("int8")
    missing_indicators.columns = [
        f"{col}_was_missing" for col in missing_indicators.columns
    ]

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_raw),
        columns=FEATURE_COLUMNS,
        index=df.index,
    )

    # Isolation Forest does not require scaling, but scaling makes the
    # exported feature matrix easier to reuse with other candidate models.
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=FEATURE_COLUMNS,
        index=df.index,
    )

    X_ready = pd.concat([X_scaled, missing_indicators], axis=1)

    # Keep IDs alongside features for traceability, never as model inputs.
    prepared = pd.concat(
        [df[[ID_COLUMN]].reset_index(drop=True),
         X_ready.reset_index(drop=True)],
        axis=1,
    )

    output_file = OUTPUT_DIR / "nhanes_anomaly_features.parquet"
    prepared.to_parquet(output_file, index=False)

    audit = {
        "rows": int(len(prepared)),
        "participant_id_column": ID_COLUMN,
        "unique_participant_ids": int(df[ID_COLUMN].nunique()),
        "biomarker_features": FEATURE_COLUMNS,
        "biomarker_feature_count": len(FEATURE_COLUMNS),
        "missingness_indicator_count": len(FEATURE_COLUMNS),
        "model_input_feature_count": int(X_ready.shape[1]),
        "missing_values_before_imputation": {
            col: int(X_raw[col].isna().sum()) for col in FEATURE_COLUMNS
        },
        "missingness_percent_before_imputation": {
            col: round(float(X_raw[col].isna().mean() * 100), 2)
            for col in FEATURE_COLUMNS
        },
        "unexpected_non_numeric_values": unexpected_non_numeric,
        "imputation": "median",
        "scaling": "StandardScaler",
        "id_excluded_from_model_features": True,
        "ferritin_excluded": True,
        "note": (
            "Exploratory preprocessing only. Imputer and scaler are fitted "
            "on this cohort for the baseline. For prospective evaluation, "
            "fit preprocessing on training data only."
        ),
    }

    audit_file = AUDIT_DIR / "anomaly_feature_preparation.json"
    with audit_file.open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)

    print("NutriFix anomaly feature preparation completed.")
    print(f"Participants retained: {len(prepared):,}")
    print(f"Biomarker features: {len(FEATURE_COLUMNS)}")
    print(f"Missingness indicators: {len(FEATURE_COLUMNS)}")
    print(f"Total model input features: {X_ready.shape[1]}")
    print(f"Missing values remaining in model matrix: {int(X_ready.isna().sum().sum())}")
    print(f"Output: {output_file}")
    print(f"Audit: {audit_file}")


if __name__ == "__main__":
    main()
