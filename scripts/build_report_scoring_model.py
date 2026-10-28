"""Train a local reference-only model bundle for complete reviewed CBC profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nutrifix.anomaly_model_comparison import INPUT_FILE, PRIMARY_SEED, SCHEMA_FILE, TEST_SIZE
from src.nutrifix.report_scoring import (
    DEFAULT_MODEL_DIR,
    fit_report_scoring_bundle,
    save_report_scoring_bundle,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--schema", type=Path, default=SCHEMA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    args = parser.parse_args(argv)

    if args.output_dir.exists():
        parser.error(
            f"output directory exists; refusing to overwrite: {args.output_dir}. "
            "Choose a new path with --output-dir."
        )
    if not args.input.is_file():
        parser.error(f"reference dataset does not exist: {args.input}")
    if not args.schema.is_file():
        parser.error(f"canonical schema does not exist: {args.schema}")

    frame = pd.read_parquet(args.input)
    with args.schema.open(encoding="utf-8") as file:
        schema = json.load(file)
    bundle = fit_report_scoring_bundle(
        frame,
        schema,
        random_seed=args.seed,
        test_size=args.test_size,
    )
    bundle["metadata"]["input_dataset"] = str(args.input)
    artifact, metadata = save_report_scoring_bundle(bundle, args.output_dir)
    print(json.dumps({
        "reference_training_participants": bundle["metadata"]["reference_training_participants"],
        "held_out_calibration_participants": bundle["metadata"]["held_out_calibration_participants"],
        "feature_count": len(bundle["feature_columns"]),
        "models": list(bundle["models"]),
        "artifact": str(artifact),
        "metadata": str(metadata),
        "purpose": bundle["metadata"]["artifact_purpose"],
    }, indent=2))
    print("Scores are exploratory statistical signals, not medical diagnoses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
