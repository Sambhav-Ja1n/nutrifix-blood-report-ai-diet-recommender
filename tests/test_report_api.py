import asyncio
import unittest

import httpx
import pymupdf

from api import app


def digital_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((45, 70), "Hemoglobin 13.5 g/dL 12.0 - 15.5", fontsize=15)
    content = document.tobytes()
    document.close()
    return content


class ReportApiTests(unittest.TestCase):
    def request(self, method: str, path: str, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def test_health_endpoint(self):
        response = self.request("GET", "/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_digital_pdf_upload_preserves_fields_and_marks_review(self):
        response = self.request(
            "POST",
            "/api/reports/extract",
            files={"file": ("private-name.pdf", digital_pdf(), "application/pdf")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["filename"], "private-name.pdf")
        self.assertEqual(payload["status"], "text_extracted")
        self.assertEqual(payload["biomarkers"][0]["canonical_biomarker"], "LBXHGB")
        self.assertTrue(payload["biomarkers"][0]["review_required"])
        self.assertEqual(payload["validation_status"], "unverified_extraction")

    def test_invalid_uploads_have_explicit_client_errors(self):
        unsupported = self.request(
            "POST",
            "/api/reports/extract",
            files={"file": ("report.txt", b"data", "text/plain")},
        )
        empty = self.request(
            "POST",
            "/api/reports/extract",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        corrupt = self.request(
            "POST",
            "/api/reports/extract",
            files={"file": ("corrupt.pdf", b"not a PDF", "application/pdf")},
        )
        self.assertEqual(unsupported.status_code, 415)
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(corrupt.status_code, 400)

    def test_malformed_recommendation_payload_is_rejected_before_external_call(self):
        response = self.request(
            "POST",
            "/api/reports/recommendations",
            json={"diet": "vegetarian", "biomarkers": []},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
