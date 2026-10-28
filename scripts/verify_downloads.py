
from pathlib import Path
import hashlib
import json
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nhanes" / "2017_2018"
OUT = ROOT / "outputs" / "dataset_audits"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED = {
    "CBC_J.xpt": [
        "SEQN", "LBXWBCSI", "LBXRBCSI",
        "LBXHGB", "LBXMCVSI", "LBXPLTSI"
    ],
    "DEMO_J.xpt": ["SEQN"],
    "FERTIN_J.xpt": ["SEQN", "LBXFER"],
}

report = {"files": {}, "errors": []}

for filename, required in EXPECTED.items():
    path = RAW / filename

    if not path.is_file() or path.stat().st_size == 0:
        report["errors"].append(f"Missing or empty file: {filename}")
        continue

    try:
        df, meta = pyreadstat.read_xport(str(path))
    except Exception as exc:
        report["errors"].append(f"Could not read {filename}: {exc}")
        continue

    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        report["errors"].append(
            f"{filename}: missing expected columns {missing_columns}"
        )

    ids = df["SEQN"] if "SEQN" in df.columns else None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    report["files"][filename] = {
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "rows": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "missing_counts": {
            str(k): int(v) for k, v in df.isna().sum().items()
        },
        "missing_percent": {
            str(k): round(float(v), 2)
            for k, v in (df.isna().mean() * 100).items()
        },
        "duplicate_SEQN": (
            int(ids.duplicated().sum()) if ids is not None else None
        ),
        "null_SEQN": (
            int(ids.isna().sum()) if ids is not None else None
        ),
        "missing_required_columns": missing_columns,
    }

    print(f"\n=== {filename} ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Size: {path.stat().st_size:,} bytes")
    print(f"SHA-256: {digest}")
    print("Column names:", ", ".join(df.columns))
    print("\nMissing values by column:")
    print(df.isna().sum().to_string())

report_path = OUT / "download_verification.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(f"\nAudit report saved to: {report_path}")

if report["errors"]:
    print("\nISSUES FOUND:")
    for error in report["errors"]:
        print("-", error)
else:
    print("\nAll files passed the basic readability and schema checks.")

print("Review missingness and participant IDs before merging.")
