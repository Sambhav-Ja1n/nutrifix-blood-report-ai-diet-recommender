
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "canonical_biomarkers.json"

def main():
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")

    with SCHEMA_PATH.open(encoding="utf-8") as file:
        schema = json.load(file)

    assert schema["schema_version"]
    assert schema["identifier"]["canonical_name"] == "participant_id"

    biomarkers = schema["biomarkers"]
    assert biomarkers, "No biomarker definitions found"

    for name, definition in biomarkers.items():
        required = ["display_name", "source_column", "canonical_unit", "data_type"]
        missing = [field for field in required if not definition.get(field)]

        if missing:
            raise ValueError(f"{name}: missing fields {missing}")

    print("NutriFix schema validation passed.")
    print(f"Schema version: {schema['schema_version']}")
    print(f"Biomarker definitions: {len(biomarkers)}")
    print("\nMappings:")

    for name, definition in biomarkers.items():
        print(
            f"- {name}: {definition['source_column']} "
            f"({definition['canonical_unit']})"
        )

    flagged = [
        name for name, definition in biomarkers.items()
        if definition.get("verification_required")
    ]

    if flagged:
        print("\nMappings requiring codebook verification:")
        for name in flagged:
            print(f"- {name}")

if __name__ == "__main__":
    main()
