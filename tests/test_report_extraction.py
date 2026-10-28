from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from src.nutrifix.report_extraction import (
    ReportExtractionError,
    UnsupportedReportError,
    extract_report,
    parse_text,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas" / "canonical_biomarkers.json").read_text())
FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
]
FONT_PATH = next((path for path in FONT_CANDIDATES if path.exists()), None)
OCR_AVAILABLE = bool(shutil.which("tesseract") and FONT_PATH)


def make_scan(text: str, image_format: str = "PNG") -> bytes:
    image = Image.new("RGB", (1800, 360), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT_PATH), 54)
    draw.text((45, 75), text, font=font, fill="black")
    stream = BytesIO()
    image.save(stream, format=image_format)
    return stream.getvalue()


def make_digital_pdf(page_lines: list[list[str]]) -> bytes:
    document = pymupdf.open()
    for lines in page_lines:
        page = document.new_page(width=612, height=792)
        for index, line in enumerate(lines):
            page.insert_text((45, 70 + index * 28), line, fontsize=15)
    result = document.tobytes()
    document.close()
    return result


def make_scanned_pdf(text: str) -> bytes:
    image_bytes = make_scan(text)
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=image_bytes)
    result = document.tobytes()
    document.close()
    return result


class ReportExtractionTests(unittest.TestCase):
    def test_digital_pdf_text_first_and_expected_canonical_fields(self):
        pdf = make_digital_pdf([[
            "Hemoglobin 13.5 g/dL 12.0 - 15.5",
            "WBC 6.1 10^3 cells/uL 4.0 - 10.0",
        ]])
        result = extract_report(pdf, "application/pdf")
        self.assertEqual(result["status"], "text_extracted")
        self.assertEqual(result["extraction_methods"], ["pdf_text"])
        self.assertIn("Hemoglobin 13.5", result["extracted_text"])
        self.assertEqual(
            [row["canonical_biomarker"] for row in result["biomarkers"]],
            ["LBXHGB", "LBXWBCSI"],
        )
        self.assertEqual(result["biomarkers"][0]["value"], 13.5)
        self.assertEqual(result["biomarkers"][0]["reference_range"], "12.0 - 15.5")
        self.assertEqual(result["biomarkers"][0]["provenance"]["page_number"], 1)
        self.assertTrue(all(row["review_required"] for row in result["biomarkers"]))

    @unittest.skipUnless(OCR_AVAILABLE, "Tesseract and a readable test font are required")
    def test_scanned_pdf_uses_tesseract_with_field_confidence(self):
        result = extract_report(
            make_scanned_pdf("Hemoglobin 13.5 g/dL 12.0 - 15.5"),
            "application/pdf",
        )
        self.assertEqual(result["status"], "ocr_extracted")
        self.assertEqual(result["pages"][0]["extraction_method"], "tesseract_ocr")
        self.assertIn("Hemoglobin", result["extracted_text"])
        self.assertEqual(result["biomarkers"][0]["canonical_biomarker"], "LBXHGB")
        self.assertIsNotNone(result["biomarkers"][0]["confidence"])
        self.assertEqual(
            result["biomarkers"][0]["provenance"]["extraction_method"],
            "tesseract_ocr",
        )

    @unittest.skipUnless(OCR_AVAILABLE, "Tesseract and a readable test font are required")
    def test_tesseract_timeout_is_reported(self):
        with patch(
            "src.nutrifix.report_extraction.subprocess.run",
            side_effect=subprocess.TimeoutExpired("tesseract", timeout=1),
        ):
            with self.assertRaisesRegex(RuntimeError, "processing limit"):
                extract_report(
                    make_scan("Hemoglobin 13.5 g/dL 12.0 - 15.5"), "image/png"
                )

    @unittest.skipUnless(OCR_AVAILABLE, "Tesseract and a readable test font are required")
    def test_png_and_jpeg_images_are_supported(self):
        for image_format, media_type in [("PNG", "image/png"), ("JPEG", "image/jpeg")]:
            with self.subTest(image_format=image_format):
                result = extract_report(
                    make_scan("Platelet count 250 10^3 cells/uL 150 - 400", image_format),
                    media_type,
                )
                self.assertEqual(result["page_count"], 1)
                self.assertEqual(result["status"], "ocr_extracted")
                self.assertEqual(result["biomarkers"][0]["canonical_biomarker"], "LBXPLTSI")
                self.assertEqual(result["biomarkers"][0]["value"], 250.0)

    @unittest.skipUnless(OCR_AVAILABLE, "Tesseract and a readable test font are required")
    def test_multi_page_pdf_keeps_per_page_methods_and_provenance(self):
        document = pymupdf.open()
        first = document.new_page(width=612, height=792)
        first.insert_text((45, 70), "Hemoglobin 13.5 g/dL 12.0 - 15.5", fontsize=15)
        second = document.new_page(width=612, height=792)
        second.insert_image(second.rect, stream=make_scan("WBC 6.1 10^3 cells/uL 4.0 - 10.0"))
        pdf = document.tobytes()
        document.close()

        result = extract_report(pdf, "application/pdf")
        self.assertEqual(result["page_count"], 2)
        self.assertEqual(
            [page["extraction_method"] for page in result["pages"]],
            ["pdf_text", "tesseract_ocr"],
        )
        self.assertEqual(
            [row["provenance"]["page_number"] for row in result["biomarkers"]],
            [1, 2],
        )

    def test_similar_names_are_distinct_and_ambiguous_names_stay_unmapped(self):
        rows = parse_text(
            "RBC 4.7 10^6 cells/uL 4.0 - 5.5\n"
            "RDW 13.2 % 11.5 - 14.5\n"
            "Neutrophils 4.1 10^3/uL 2.0 - 7.0"
        )
        self.assertEqual(
            [row["canonical_biomarker"] for row in rows],
            ["LBXRBCSI", "LBXRDW"],
        )

    def test_decimal_comma_conflict_and_missing_fields_require_review(self):
        rows = parse_text(
            "Hemoglobin 13,5 g/dL 12.0 - 15.5\n"
            "Hemoglobin 14.0 g/dL 12.0 - 15.5\n"
            "Hemoglobin 13.5 12.0 - 15.5"
        )
        self.assertIsNone(rows[0]["value"])
        self.assertIn("decimal_separator_ambiguous", rows[0]["review_reasons"])
        self.assertIn("conflicting_duplicate_values", rows[0]["review_reasons"])
        self.assertIn("unit_missing", rows[2]["review_reasons"])
        self.assertTrue(all(row["range_status"] == "needs_review" for row in rows))

    @unittest.skipUnless(OCR_AVAILABLE, "Tesseract and a readable test font are required")
    def test_blank_low_information_image_returns_no_invented_fields(self):
        image = Image.new("RGB", (900, 400), "white")
        stream = BytesIO()
        image.save(stream, format="PNG")
        result = extract_report(stream.getvalue(), "image/png")
        self.assertEqual(result["biomarkers"], [])
        self.assertEqual(result["unresolved_numeric_lines"], [])
        self.assertTrue(result["review_required"])

    def test_empty_corrupt_and_unsupported_uploads_fail_explicitly(self):
        with self.assertRaisesRegex(ReportExtractionError, "empty"):
            extract_report(b"", "application/pdf")
        with self.assertRaises(ReportExtractionError):
            extract_report(b"not a pdf", "application/pdf")
        with self.assertRaises(UnsupportedReportError):
            extract_report(b"data", "application/octet-stream")
        with self.assertRaises(ReportExtractionError):
            extract_report(b"not an image", "image/png")
        jpeg_stream = BytesIO()
        Image.new("RGB", (200, 120), "white").save(jpeg_stream, format="JPEG")
        with self.assertRaisesRegex(ReportExtractionError, "does not match"):
            extract_report(jpeg_stream.getvalue(), "image/png")

    def test_pdf_page_and_render_limits_are_checked_before_ocr(self):
        document = pymupdf.open()
        for _ in range(101):
            document.new_page()
        too_many_pages = document.tobytes()
        document.close()
        with self.assertRaisesRegex(ReportExtractionError, "100-page limit"):
            extract_report(too_many_pages, "application/pdf")

        document = pymupdf.open()
        document.new_page(width=3000, height=3000)
        oversized_page = document.tobytes()
        document.close()
        with self.assertRaisesRegex(ReportExtractionError, "pixel processing limit"):
            extract_report(oversized_page, "application/pdf")

    def test_negative_numeric_text_is_preserved_without_clinical_claim(self):
        rows = parse_text("Hemoglobin -1.2 g/dL -3.0 - 3.0")
        self.assertEqual(rows[0]["value"], -1.2)
        self.assertEqual(rows[0]["range_status"], "within_supplied_range")
        self.assertTrue(rows[0]["review_required"])


if __name__ == "__main__":
    unittest.main()
