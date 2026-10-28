"""Validation and cautious analysis for explicitly user-reviewed lab values."""

from __future__ import annotations

import json
from functools import lru_cache
import math
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from src.nutrifix.report_extraction import compare_to_supplied_range


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "schemas" / "canonical_biomarkers.json"
RangeStatus = Literal[
    "within_supplied_range",
    "below_supplied_range",
    "above_supplied_range",
    "below_supplied_threshold",
    "at_or_above_supplied_threshold",
    "needs_review",
]


@lru_cache(maxsize=1)
def _load_biomarker_definitions() -> dict[str, dict]:
    with SCHEMA_PATH.open(encoding="utf-8") as file:
        schema = json.load(file)
    return {
        entry["source_column"]: {
            "canonical_name": canonical_name,
            "display_name": entry["display_name"],
            "canonical_unit": entry["canonical_unit"],
        }
        for canonical_name, entry in schema["biomarkers"].items()
    }


def _unit_key(value: str) -> str:
    normalized = re.sub(r"\s+", "", value).casefold().replace("µ", "u").replace("μ", "u")
    return re.sub(r"(?<=\d)cells(?=/|$)", "", normalized)


class ReviewedBiomarker(BaseModel):
    """A field the user explicitly checked and confirmed in the review step."""

    model_config = ConfigDict(extra="ignore")

    canonical_biomarker: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=100)
    value: FiniteFloat
    unit: str = Field(min_length=1, max_length=40)
    reference_range: str = Field(default="", max_length=100)
    reviewed_by_user: Literal[True]
    # Any value provided by a client is overwritten from the supplied interval.
    range_status: RangeStatus = "needs_review"

    @model_validator(mode="after")
    def validate_and_derive_status(self):
        definitions = _load_biomarker_definitions()
        definition = definitions.get(self.canonical_biomarker)
        if definition is None:
            raise ValueError("Biomarker is not present in the canonical schema")
        if _unit_key(self.unit) != _unit_key(definition["canonical_unit"]):
            raise ValueError(
                "Unit must match the canonical unit; automatic unit conversion is unavailable"
            )
        self.name = definition["display_name"]
        self.range_status = compare_to_supplied_range(self.value, self.reference_range)
        return self


class ReviewedReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    biomarkers: list[ReviewedBiomarker] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def reject_duplicate_canonical_markers(self):
        columns = [marker.canonical_biomarker for marker in self.biomarkers]
        if len(columns) != len(set(columns)):
            raise ValueError("Confirm only one value per canonical biomarker")
        return self


def analyze_reviewed_report(request: ReviewedReportRequest) -> dict:
    """Compare confirmed results only with ranges supplied in that report."""
    results = []
    explanations = []
    for marker in request.biomarkers:
        results.append({
            "canonical_biomarker": marker.canonical_biomarker,
            "name": marker.name,
            "value": marker.value,
            "unit": marker.unit,
            "reference_range": marker.reference_range or None,
            "range_status": marker.range_status,
            "validation_status": "user_confirmed_syntactically_validated",
        })
        if marker.range_status == "within_supplied_range":
            detail = "falls within the interval supplied in the report"
        elif marker.range_status == "below_supplied_range":
            detail = "falls below the interval supplied in the report"
        elif marker.range_status == "above_supplied_range":
            detail = "falls above the interval supplied in the report"
        elif marker.range_status == "below_supplied_threshold":
            detail = "is below the threshold supplied in the report"
        elif marker.range_status == "at_or_above_supplied_threshold":
            detail = "is at or above the threshold supplied in the report"
        else:
            detail = "could not be compared with a clear supplied interval"
        interval_text = (
            f" The report supplied: {marker.reference_range}."
            if marker.reference_range
            else " No usable interval was supplied."
        )
        explanations.append({
            "canonical_biomarker": marker.canonical_biomarker,
            "text": (
                f"{marker.name} ({marker.value:g} {marker.unit}) {detail}."
                f"{interval_text} This is a comparison with report-provided text, not a diagnosis."
            ),
        })

    return {
        "status": "completed_with_reviewed_inputs",
        "data_quality": {
            "submitted_biomarkers": len(request.biomarkers),
            "user_confirmed_biomarkers": len(request.biomarkers),
            "values_finite": all(math.isfinite(marker.value) for marker in request.biomarkers),
            "units_match_canonical_schema": True,
            "unit_conversion_performed": False,
        },
        "reference_range_comparisons": results,
        "explanations": explanations,
        "statistical_anomaly": {
            "status": "not_evaluated",
            "reason": (
                "The five-model NHANES comparison is an exploratory cohort experiment; "
                "no individual-report scoring model has been deployed or clinically validated."
            ),
        },
        "limitations": [
            "Reference comparisons use only intervals supplied in the report and confirmed by the user.",
            "OCR confidence and user confirmation do not establish clinical correctness.",
            "An out-of-range result is not a diagnosis; review results with a qualified clinician.",
            "Statistical anomaly scores are exploratory reference-cohort comparisons, not probabilities or diagnoses.",
        ],
    }
