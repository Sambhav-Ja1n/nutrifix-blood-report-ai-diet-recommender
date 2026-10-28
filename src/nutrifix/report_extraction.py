"""Local report text/OCR extraction with explicit provenance and review flags.

This module extracts candidate fields only. It does not validate laboratory
results, supply missing units or ranges, or make medical interpretations.
"""

from __future__ import annotations

import csv
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time
import unicodedata

import pymupdf
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_FILE = ROOT / "schemas" / "canonical_biomarkers.json"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_OCR_PAGES = 30
MAX_IMAGE_PIXELS = 30_000_000
OCR_TIMEOUT_SECONDS = 30
MAX_REPORT_OCR_SECONDS = 120
MIN_SELECTABLE_TEXT_CHARACTERS = 12
LOW_OCR_CONFIDENCE = 70.0
LEGACY_UNCANONICAL_LABELS = {"fasting glucose", "total cholesterol"}

NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])([<>≤≥]=?\s*)?(-?\d+(?:[.,]\d+)?)(?![A-Za-z])")
REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<range>(?:<=|>=|<|>|≤|≥)?\s*"
    r"-?\d+(?:\.\d+)?(?:\s*(?:-|–|—|to)\s*-?\d+(?:\.\d+)?)?)\s*$",
    flags=re.IGNORECASE,
)


class ReportExtractionError(ValueError):
    """The report could not be safely processed."""


class UnsupportedReportError(ReportExtractionError):
    """The report format is unsupported or does not match its declared type."""


class OCRUnavailableError(RuntimeError):
    """The local OCR executable is unavailable or failed."""


def _normalise_label(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("%", " percent ")
    value = value.replace("μ", "u").replace("µ", "u")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def _schema_index(schema: dict) -> tuple[dict, dict]:
    definitions = {}
    alias_index = {}
    for canonical_name, definition in schema.get("biomarkers", {}).items():
        column = definition.get("source_column")
        if not column:
            continue
        definitions[column] = {
            "canonical_name": canonical_name,
            "display_name": definition.get("display_name", canonical_name),
            "canonical_unit": definition.get("canonical_unit"),
        }
        aliases = [
            canonical_name,
            definition.get("display_name", ""),
            column,
            *schema.get("normalization_aliases", {}).get(column, []),
        ]
        for alias in aliases:
            normalized = _normalise_label(alias)
            if normalized:
                # Ambiguous aliases are removed rather than assigned arbitrarily.
                previous = alias_index.get(normalized)
                if previous is not None and previous != column:
                    alias_index[normalized] = None
                elif normalized not in alias_index:
                    alias_index[normalized] = column
    return definitions, alias_index


def _tesseract_binary() -> str:
    executable = shutil.which("tesseract")
    if not executable:
        raise OCRUnavailableError(
            "Tesseract OCR is not installed or is not available on PATH."
        )
    return executable


def _tesseract_call(
    executable: str,
    image_bytes: bytes,
    *args: str,
    timeout: float = OCR_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [executable, "stdin", "stdout", *args],
            input=image_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise OCRUnavailableError("Tesseract exceeded the per-page processing limit.") from exc
    except OSError as exc:
        raise OCRUnavailableError(f"Could not start Tesseract: {exc}") from exc


def _image_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _remaining_ocr_timeout(deadline: float | None) -> float:
    if deadline is None:
        return OCR_TIMEOUT_SECONDS
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise OCRUnavailableError("The report exceeded the total OCR processing limit.")
    return min(OCR_TIMEOUT_SECONDS, remaining)


def _orientation_degrees(
    executable: str,
    image_bytes: bytes,
    deadline: float | None = None,
) -> int:
    result = _tesseract_call(
        executable,
        image_bytes,
        "-l",
        "osd",
        "--psm",
        "0",
        timeout=_remaining_ocr_timeout(deadline),
    )
    if result.returncode != 0:
        return 0
    match = re.search(rb"Rotate:\s*(0|90|180|270)", result.stdout)
    return int(match.group(1)) if match else 0


def _ocr_image(image: Image.Image, deadline: float | None = None) -> dict:
    executable = _tesseract_binary()
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ReportExtractionError(
            f"Image dimensions exceed the {MAX_IMAGE_PIXELS:,}-pixel processing limit."
        )

    image = image.convert("RGB")
    initial_bytes = _image_bytes(image)
    rotation = _orientation_degrees(executable, initial_bytes, deadline)
    if rotation:
        image = image.rotate(-rotation, expand=True, fillcolor="white")

    # Conservative contrast/noise cleanup; no geometric deskew is attempted.
    prepared = ImageOps.grayscale(image)
    prepared = ImageOps.autocontrast(prepared, cutoff=1)
    prepared = ImageEnhance.Contrast(prepared).enhance(1.35)
    prepared = prepared.filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.SHARPEN)
    image_bytes = _image_bytes(prepared)
    result = _tesseract_call(
        executable,
        image_bytes,
        "-l",
        "eng",
        "--psm",
        "6",
        "tsv",
        timeout=_remaining_ocr_timeout(deadline),
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise OCRUnavailableError(
            "Tesseract could not process the report page"
            + (f": {detail[:300]}" if detail else ".")
        )

    text_stream = StringIO(result.stdout.decode("utf-8", "replace"))
    lines: dict[tuple[str, str, str], dict] = {}
    try:
        for row in csv.DictReader(text_stream, delimiter="\t"):
            word = (row.get("text") or "").strip()
            if not word:
                continue
            key = (row.get("block_num", "0"), row.get("par_num", "0"), row.get("line_num", "0"))
            record = lines.setdefault(key, {"words": [], "confidences": []})
            record["words"].append(word)
            try:
                confidence = float(row.get("conf", "-1"))
            except (TypeError, ValueError):
                confidence = -1.0
            if confidence >= 0:
                record["confidences"].append(confidence)
    except csv.Error as exc:
        raise OCRUnavailableError(f"Tesseract returned invalid TSV output: {exc}") from exc

    output_lines = []
    for line_number, record in enumerate(lines.values(), start=1):
        line_text = " ".join(record["words"])
        confidences = record["confidences"]
        confidence = sum(confidences) / len(confidences) if confidences else None
        output_lines.append({
            "line_number": line_number,
            "text": line_text,
            "confidence": confidence,
        })
    all_confidences = [line["confidence"] for line in output_lines if line["confidence"] is not None]
    page_confidence = (
        sum(all_confidences) / len(all_confidences) if all_confidences else None
    )
    return {
        "text": "\n".join(line["text"] for line in output_lines),
        "lines": output_lines,
        "confidence": page_confidence,
        "orientation_correction_degrees": rotation,
    }


def _page_text_record(page_number: int, method: str, text: str, lines=None, confidence=None) -> dict:
    if lines is None:
        lines = [
            {"line_number": index, "text": line, "confidence": None}
            for index, line in enumerate(text.splitlines(), start=1)
        ]
    return {
        "page_number": page_number,
        "extraction_method": method,
        "text": text,
        "confidence": confidence,
        "lines": lines,
    }


def _extract_pdf(content: bytes) -> list[dict]:
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ReportExtractionError("The PDF could not be opened.") from exc
    try:
        if document.needs_pass:
            raise ReportExtractionError("Password-protected PDFs are not supported.")
        if len(document) == 0:
            raise UnsupportedReportError("The PDF contains no pages.")
        if len(document) > MAX_PDF_PAGES:
            raise ReportExtractionError(
                f"The PDF exceeds the {MAX_PDF_PAGES}-page limit."
            )

        pages = []
        ocr_pages = 0
        ocr_deadline = time.monotonic() + MAX_REPORT_OCR_SECONDS
        for index, page in enumerate(document, start=1):
            selectable_text = page.get_text("text") or ""
            compact_text = re.sub(r"\s+", "", selectable_text)
            if len(compact_text) >= MIN_SELECTABLE_TEXT_CHARACTERS:
                pages.append(_page_text_record(index, "pdf_text", selectable_text))
                continue

            ocr_pages += 1
            if ocr_pages > MAX_OCR_PAGES:
                raise ReportExtractionError(
                    f"OCR is limited to {MAX_OCR_PAGES} scanned pages per report."
                )
            page_rect = page.rect
            estimated_width = math.ceil(page_rect.width * 2)
            estimated_height = math.ceil(page_rect.height * 2)
            if estimated_width * estimated_height > MAX_IMAGE_PIXELS:
                raise ReportExtractionError(
                    f"Rendered page {index} exceeds the OCR pixel processing limit."
                )
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            image = Image.open(BytesIO(pixmap.tobytes("png")))
            ocr = _ocr_image(image, deadline=ocr_deadline)
            pages.append(_page_text_record(
                index,
                "tesseract_ocr",
                ocr["text"],
                ocr["lines"],
                ocr["confidence"],
            ) | {"orientation_correction_degrees": ocr["orientation_correction_degrees"]})
        return pages
    finally:
        document.close()


def _extract_image(content: bytes, media_type: str) -> list[dict]:
    try:
        with Image.open(BytesIO(content)) as probe:
            image_format = probe.format
            width, height = probe.size
            probe.verify()
        expected_format = {"image/jpeg": "JPEG", "image/png": "PNG"}.get(media_type)
        if image_format not in {"JPEG", "PNG"} or image_format != expected_format:
            raise ReportExtractionError("The image content does not match JPEG or PNG.")
        if width * height > MAX_IMAGE_PIXELS:
            raise ReportExtractionError(
                f"Image dimensions exceed the {MAX_IMAGE_PIXELS:,}-pixel processing limit."
            )
        with Image.open(BytesIO(content)) as image:
            ocr = _ocr_image(image, deadline=time.monotonic() + MAX_REPORT_OCR_SECONDS)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ReportExtractionError("The image could not be decoded safely.") from exc
    return [_page_text_record(
        1,
        "tesseract_ocr",
        ocr["text"],
        ocr["lines"],
        ocr["confidence"],
    ) | {"orientation_correction_degrees": ocr["orientation_correction_degrees"]}]


def _unit_status(observed: str | None, canonical: str | None) -> str:
    if not observed:
        return "missing"
    if not canonical:
        return "unmapped"
    compact = lambda value: re.sub(r"\s+", "", value).casefold().replace("µ", "u").replace("μ", "u")
    return "exact" if compact(observed) == compact(canonical) else "not_converted"


def _parse_numeric_value(operator: str | None, number_text: str) -> tuple[float | None, str | None]:
    if "," in number_text:
        return None, "decimal_separator_ambiguous"
    try:
        value = float(number_text)
    except ValueError:
        return None, "numeric_value_unreadable"
    if not math.isfinite(value):
        return None, "numeric_value_non_finite"
    if operator:
        return None, "reported_as_threshold_not_exact_value"
    return value, None


def compare_to_supplied_range(value: float | None, reference_range: str | None) -> str:
    if value is None or not reference_range:
        return "needs_review"
    simple = re.fullmatch(
        r"\s*(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(-?\d+(?:\.\d+)?)\s*",
        reference_range,
        flags=re.IGNORECASE,
    )
    if simple:
        lower, upper = float(simple.group(1)), float(simple.group(2))
        if lower > upper:
            return "needs_review"
        if value < lower:
            return "below_supplied_range"
        if value > upper:
            return "above_supplied_range"
        return "within_supplied_range"
    threshold = re.fullmatch(r"\s*(<=|>=|<|>|≤|≥)\s*(-?\d+(?:\.\d+)?)\s*", reference_range)
    if threshold:
        operator, bound_text = threshold.groups()
        bound = float(bound_text)
        if operator in {"<", "<=", "≤"}:
            return "below_supplied_threshold" if value < bound else "at_or_above_supplied_threshold"
        return "below_supplied_range" if value <= bound else "within_supplied_range"
    return "needs_review"


# Keep the historic internal name available for extraction parsing.
_range_status = compare_to_supplied_range


def parse_extracted_pages(pages: list[dict], schema: dict) -> dict:
    """Parse rows conservatively and retain source text and page provenance."""
    definitions, alias_index = _schema_index(schema)
    candidates = []
    unresolved_lines = []

    for page in pages:
        for line in page.get("lines", []):
            source_line = (line.get("text") or "").strip()
            if not source_line:
                continue
            match = NUMBER_RE.search(source_line)
            if not match:
                continue
            raw_label = source_line[:match.start()].strip(" :\t-•|\u2022")
            if not raw_label:
                continue
            normalized_label = _normalise_label(raw_label)
            canonical_column = alias_index.get(normalized_label)
            legacy_unmapped = (
                canonical_column is None
                and normalized_label in LEGACY_UNCANONICAL_LABELS
            )
            if canonical_column is None and not legacy_unmapped:
                unresolved_lines.append({
                    "page_number": page["page_number"],
                    "line_number": line.get("line_number"),
                    "text": source_line,
                    "reason": "biomarker_name_unmapped_or_ambiguous",
                    "confidence": line.get("confidence"),
                })
                continue

            operator = (match.group(1) or "").strip() or None
            number_text = match.group(2)
            value, numeric_issue = _parse_numeric_value(operator, number_text)
            tail = source_line[match.end():].strip(" \t:;()[]")
            reference_match = REFERENCE_RE.search(tail)
            reference_range = reference_match.group("range").strip() if reference_match else None
            observed_unit = tail[:reference_match.start()].strip(" \t:;()[]") if reference_match else tail
            # Remove common result flags only; do not use them to infer a value or range.
            if observed_unit.casefold() in {"h", "l", "high", "low", "abnormal"}:
                observed_unit = ""
            definition = definitions.get(canonical_column, {})
            confidence = line.get("confidence")
            reasons = ["not_medically_verified", "user_review_required"]
            if page.get("extraction_method") == "tesseract_ocr":
                reasons.append("ocr_extracted")
            if confidence is not None and confidence < LOW_OCR_CONFIDENCE:
                reasons.append("low_ocr_confidence")
            if numeric_issue:
                reasons.append(numeric_issue)
            if not observed_unit:
                reasons.append("unit_missing")
            if not reference_range:
                reasons.append("reference_range_missing")
            if legacy_unmapped:
                reasons.append("canonical_schema_mapping_unavailable")

            candidates.append({
                "name": raw_label,
                "canonical_biomarker": canonical_column,
                "canonical_name": definition.get("canonical_name"),
                "display_name": definition.get("display_name", raw_label),
                "normalization_status": "unmapped_to_canonical_schema" if legacy_unmapped else "matched",
                "reported_value": ((operator or "") + number_text).strip(),
                "value": value,
                "unit": observed_unit or None,
                "canonical_unit": definition.get("canonical_unit"),
                "unit_normalization_status": _unit_status(observed_unit, definition.get("canonical_unit")),
                "reference_range": reference_range,
                "range_status": _range_status(value, reference_range),
                "confidence": confidence,
                "review_required": True,
                "review_reasons": reasons,
                "validation_status": "unverified_extraction",
                "provenance": {
                    "page_number": page["page_number"],
                    "line_number": line.get("line_number"),
                    "extraction_method": page.get("extraction_method"),
                    "source_line": source_line,
                },
            })

    # Repeated canonical results that disagree are surfaced, never overwritten.
    by_column: dict[str, list[dict]] = {}
    for record in candidates:
        if record["canonical_biomarker"]:
            by_column.setdefault(record["canonical_biomarker"], []).append(record)
    for records in by_column.values():
        values = {record["reported_value"] for record in records}
        if len(values) > 1:
            for record in records:
                record["review_reasons"].append("conflicting_duplicate_values")
                record["range_status"] = "needs_review"

    return {"biomarkers": candidates, "unresolved_numeric_lines": unresolved_lines[:100]}


def parse_text(text: str, schema_path: Path = DEFAULT_SCHEMA_FILE) -> list[dict]:
    """Compatibility helper for callers that already have extracted text."""
    with Path(schema_path).open(encoding="utf-8") as file:
        schema = json.load(file)
    page = _page_text_record(1, "provided_text", text)
    return parse_extracted_pages([page], schema)["biomarkers"]


def extract_report(content: bytes, media_type: str, schema_path: Path = DEFAULT_SCHEMA_FILE) -> dict:
    """Extract page text (PDF text first, then local OCR) and candidate fields."""
    if not content:
        raise ReportExtractionError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ReportExtractionError("File exceeds the 15 MB limit.")
    if media_type not in {"application/pdf", "image/jpeg", "image/png"}:
        raise UnsupportedReportError("Upload a PDF, JPG, or PNG file.")
    if media_type == "application/pdf":
        if not content.startswith(b"%PDF-"):
            raise ReportExtractionError("The file is not a valid PDF.")
        pages = _extract_pdf(content)
    else:
        pages = _extract_image(content, media_type)

    with Path(schema_path).open(encoding="utf-8") as file:
        schema = json.load(file)
    parsed = parse_extracted_pages(pages, schema)
    combined_text = "\n\n".join(page["text"] for page in pages)
    used_ocr = any(page["extraction_method"] == "tesseract_ocr" for page in pages)
    used_text = any(page["extraction_method"] == "pdf_text" for page in pages)
    methods = {page["extraction_method"] for page in pages}
    if used_ocr and used_text:
        status = "mixed_text_and_ocr"
    elif used_ocr:
        status = "ocr_extracted"
    elif parsed["biomarkers"]:
        status = "text_extracted"
    else:
        status = "text_extracted_no_supported_biomarkers"
    return {
        "status": status,
        "extracted_text": combined_text[:100_000],
        "pages": [
            {key: value for key, value in page.items() if key != "lines"}
            | {"lines": page["lines"][:1000]}
            for page in pages
        ],
        "page_count": len(pages),
        "extraction_methods": sorted(methods),
        "biomarkers": parsed["biomarkers"],
        "unresolved_numeric_lines": parsed["unresolved_numeric_lines"],
        "review_required": True,
        "validation_status": "unverified_extraction",
        "message": (
            "Extracted fields are unverified. Review each value, unit, and supplied range before use."
        ),
    }
