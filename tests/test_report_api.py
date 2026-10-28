import logging
import unittest

import httpx
import pymupdf
from unittest.mock import AsyncMock, patch

from api import ProcessingTimeoutError, _report_worker_environment, app
from src.nutrifix.report_extraction import OCRUnavailableError


def digital_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((45, 40), "Patient: Synthetic Person ID: TEST-0001", fontsize=10)
    page.insert_text((45, 70), "Hemoglobin 13.5 g/dL 12.0 - 15.5", fontsize=15)
    content = document.tobytes()
    document.close()
    return content


class ReportApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(
            transport=self.transport, base_url="http://testserver"
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def request(self, method: str, path: str, **kwargs):
        return await self.client.request(method, path, **kwargs)

    async def test_health_endpoint(self):
        response = await self.request("GET", "/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    async def test_digital_pdf_upload_preserves_fields_and_marks_review(self):
        response = await self.request(
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

    async def test_upload_analysis_recommendation_workflow_uses_explicit_review(self):
        uploaded = await self.request(
            "POST",
            "/api/reports/extract",
            files={"file": ("synthetic.pdf", digital_pdf(), "application/pdf")},
        )
        self.assertEqual(uploaded.status_code, 200)
        extracted = uploaded.json()["biomarkers"][0]
        self.assertTrue(extracted["review_required"])

        confirmed = {
            "canonical_biomarker": extracted["canonical_biomarker"],
            "name": extracted["name"],
            "value": extracted["value"],
            "unit": extracted["unit"],
            "reference_range": extracted["reference_range"],
            "reviewed_by_user": True,
        }
        report = {"biomarkers": [confirmed]}
        analysis = await self.request("POST", "/api/reports/analyze", json=report)
        self.assertEqual(analysis.status_code, 200, analysis.text)
        self.assertEqual(
            analysis.json()["reference_range_comparisons"][0]["range_status"],
            "within_supplied_range",
        )
        self.assertEqual(analysis.json()["statistical_anomaly"]["status"], "insufficient_data")

        with patch(
            "api.generate_food_recommendations",
            new_callable=AsyncMock,
            return_value={"recommendations": []},
        ):
            recommendations = await self.request(
                "POST",
                "/api/reports/recommendations",
                json={"diet": "vegetarian", **report},
            )
        self.assertEqual(recommendations.status_code, 200, recommendations.text)

    async def test_report_content_is_not_written_to_api_logs(self):
        logger = logging.getLogger("nutrifix.api")
        with self.assertNoLogs(logger, level="INFO"):
            response = await self.request(
                "POST",
                "/api/reports/extract",
                files={"file": ("synthetic.pdf", digital_pdf(), "application/pdf")},
            )
        self.assertEqual(response.status_code, 200)

    async def test_full_report_text_is_opt_in_and_unrelated_identifiers_are_omitted(self):
        default_response = await self.request(
            "POST",
            "/api/reports/extract",
            files={"file": ("synthetic.pdf", digital_pdf(), "application/pdf")},
        )
        self.assertEqual(default_response.status_code, 200)
        self.assertNotIn("Synthetic Person", default_response.text)
        self.assertNotIn("TEST-0001", default_response.text)
        self.assertNotIn("extracted_text", default_response.json())
        self.assertNotIn("lines", default_response.json()["pages"][0])

        audit_response = await self.request(
            "POST",
            "/api/reports/extract",
            data={"include_source_text": "true"},
            files={"file": ("synthetic.pdf", digital_pdf(), "application/pdf")},
        )
        self.assertEqual(audit_response.status_code, 200)
        self.assertIn("Patient: Synthetic Person", audit_response.json()["extracted_text"])

    async def test_full_report_text_is_opt_in_and_unrelated_identifiers_are_omitted(self):
        default_response = await self.request(
            "POST",
            "/api/reports/extract",
            files={"file": ("synthetic.pdf", digital_pdf(), "application/pdf")},
        )
        self.assertEqual(default_response.status_code, 200)
        self.assertNotIn("Synthetic Person", default_response.text)
        self.assertNotIn("TEST-0001", default_response.text)
        self.assertNotIn("extracted_text", default_response.json())
        self.assertNotIn("lines", default_response.json()["pages"][0])

        audit_response = await self.request(
            "POST",
            "/api/reports/extract",
            data={"include_source_text": "true"},
            files={"file": ("synthetic.pdf", digital_pdf(), "application/pdf")},
        )
        self.assertEqual(audit_response.status_code, 200)
        self.assertIn("Patient: Synthetic Person", audit_response.json()["extracted_text"])

    async def test_invalid_uploads_have_explicit_client_errors(self):
        unsupported = await self.request(
            "POST",
            "/api/reports/extract",
            files={"file": ("report.txt", b"data", "text/plain")},
        )
        empty = await self.request(
            "POST",
            "/api/reports/extract",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        corrupt = await self.request(
            "POST",
            "/api/reports/extract",
            files={"file": ("corrupt.pdf", b"not a PDF", "application/pdf")},
        )
        self.assertEqual(unsupported.status_code, 415)
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(corrupt.status_code, 400)

    async def test_malformed_recommendation_payload_is_rejected_before_external_call(self):
        response = await self.request(
            "POST",
            "/api/reports/recommendations",
            json={"diet": "vegetarian", "biomarkers": []},
        )
        self.assertEqual(response.status_code, 422)

    async def test_reviewed_analysis_recomputes_report_range_status(self):
        response = await self.request(
            "POST",
            "/api/reports/analyze",
            json={"biomarkers": [{
                "canonical_biomarker": "LBXHGB",
                "name": "Untrusted client label",
                "value": 17.0,
                "unit": "g/dL",
                "reference_range": "12.0 - 15.5",
                "range_status": "within_supplied_range",
                "reviewed_by_user": True,
            }]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(
            payload["reference_range_comparisons"][0]["range_status"],
            "above_supplied_range",
        )
        self.assertEqual(
            payload["reference_range_comparisons"][0]["name"], "Hemoglobin"
        )
        self.assertEqual(payload["statistical_anomaly"]["status"], "insufficient_data")
        self.assertIn("not a diagnosis", payload["explanations"][0]["text"])

    async def test_complete_reviewed_profile_is_passed_to_exploratory_scorer(self):
        from src.nutrifix.anomaly_model_comparison import FEATURE_COLUMNS

        unit_by_column = {
            "LBXWBCSI": "10^3 cells/uL", "LBXLYPCT": "%", "LBXMOPCT": "%",
            "LBXNEPCT": "%", "LBXEOPCT": "%", "LBXBAPCT": "%",
            "LBXRBCSI": "10^6 cells/uL", "LBXHGB": "g/dL", "LBXHCT": "%",
            "LBXMCVSI": "fL", "LBXMCHSI": "pg", "LBXMC": "g/dL",
            "LBXRDW": "%", "LBXPLTSI": "10^3 cells/uL", "LBXMPSI": "fL",
        }
        markers = []
        for index, column in enumerate(FEATURE_COLUMNS):
            markers.append({
                "canonical_biomarker": column,
                "name": column,
                "value": 1.0 + index,
                "unit": unit_by_column[column],
                "reference_range": "0 - 100",
                "reviewed_by_user": True,
            })
        expected = {
            "status": "scored",
            "models": {},
            "limitations": ["exploratory only"],
        }
        with patch("api.score_reviewed_profile", return_value=expected) as scorer:
            response = await self.request("POST", "/api/reports/analyze", json={"biomarkers": markers})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["statistical_anomaly"], expected)
        received = scorer.call_args.args[0]
        self.assertEqual(set(received), set(FEATURE_COLUMNS))
        self.assertNotIn("SEQN", received)

    async def test_unavailable_reference_bundle_has_safe_status(self):
        from src.nutrifix.anomaly_model_comparison import FEATURE_COLUMNS

        # Reuse the complete-profile unit construction above via canonical names;
        # units are checked by the request schema before scoring is attempted.
        unit_by_column = {
            "LBXWBCSI": "10^3 cells/uL", "LBXLYPCT": "%", "LBXMOPCT": "%",
            "LBXNEPCT": "%", "LBXEOPCT": "%", "LBXBAPCT": "%",
            "LBXRBCSI": "10^6 cells/uL", "LBXHGB": "g/dL", "LBXHCT": "%",
            "LBXMCVSI": "fL", "LBXMCHSI": "pg", "LBXMC": "g/dL",
            "LBXRDW": "%", "LBXPLTSI": "10^3 cells/uL", "LBXMPSI": "fL",
        }
        markers = [{
            "canonical_biomarker": column, "name": column, "value": 1.0,
            "unit": unit_by_column[column], "reference_range": "0 - 100",
            "reviewed_by_user": True,
        } for column in FEATURE_COLUMNS]
        with patch("api.score_reviewed_profile", side_effect=__import__(
            "src.nutrifix.report_scoring", fromlist=["ScoringArtifactUnavailable"]
        ).ScoringArtifactUnavailable("missing")):
            response = await self.request("POST", "/api/reports/analyze", json={"biomarkers": markers})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["statistical_anomaly"]["status"], "model_unavailable")

    async def test_unreviewed_and_unsupported_units_cannot_reach_analysis(self):
        unreviewed = await self.request(
            "POST",
            "/api/reports/analyze",
            json={"biomarkers": [{
                "canonical_biomarker": "LBXHGB",
                "name": "Hemoglobin",
                "value": 13.5,
                "unit": "g/dL",
                "reference_range": "12.0 - 15.5",
                "reviewed_by_user": False,
            }]},
        )
        unsupported_unit = await self.request(
            "POST",
            "/api/reports/analyze",
            json={"biomarkers": [{
                "canonical_biomarker": "LBXHGB",
                "name": "Hemoglobin",
                "value": 13.5,
                "unit": "mg/L",
                "reference_range": "12.0 - 15.5",
                "reviewed_by_user": True,
            }]},
        )
        self.assertEqual(unreviewed.status_code, 422)
        self.assertEqual(unsupported_unit.status_code, 422)

    async def test_recommendations_require_a_usable_reviewed_range_and_recompute_status(self):
        payload = {
            "diet": "vegetarian",
            "biomarkers": [{
                "canonical_biomarker": "LBXHGB",
                "name": "Hemoglobin",
                "value": 13.5,
                "unit": "g/dL",
                "reference_range": "12.0 - 15.5",
                "range_status": "above_supplied_range",
                "reviewed_by_user": True,
            }],
        }
        with patch(
            "api.generate_food_recommendations",
            new_callable=AsyncMock,
            return_value={"ok": True},
        ) as mocked:
            response = await self.request(
                "POST", "/api/reports/recommendations", json=payload
            )
        self.assertEqual(response.status_code, 200, response.text)
        forwarded = mocked.call_args.args[0]
        self.assertEqual(forwarded.biomarkers[0].range_status, "within_supplied_range")

        payload["biomarkers"][0]["reference_range"] = ""
        with patch("api.generate_food_recommendations") as mocked:
            rejected = await self.request(
                "POST", "/api/reports/recommendations", json=payload
            )
        self.assertEqual(rejected.status_code, 422)
        mocked.assert_not_called()

    async def test_upload_size_limit_is_enforced(self):
        with patch("api.MAX_UPLOAD_BYTES", 32):
            response = await self.request(
                "POST",
                "/api/reports/extract",
                files={"file": ("large.pdf", b"%PDF-" + b"x" * 40, "application/pdf")},
            )
        self.assertEqual(response.status_code, 413)

    async def test_busy_local_processor_returns_retryable_status(self):
        with patch("api._PROCESSING_SLOTS.acquire", return_value=False):
            response = await self.request(
                "POST",
                "/api/reports/extract",
                files={"file": ("report.pdf", digital_pdf(), "application/pdf")},
            )
        self.assertEqual(response.status_code, 503)

    async def test_report_worker_timeout_returns_gateway_timeout_without_details(self):
        with patch(
            "api._extract_report_isolated",
            new_callable=AsyncMock,
            side_effect=ProcessingTimeoutError("sensitive worker detail"),
        ):
            response = await self.request(
                "POST",
                "/api/reports/extract",
                files={"file": ("report.pdf", digital_pdf(), "application/pdf")},
            )
        self.assertEqual(response.status_code, 504)
        self.assertNotIn("sensitive worker detail", response.text)

    async def test_extraction_worker_receives_no_application_credentials(self):
        environment = _report_worker_environment()
        self.assertIn("PATH", environment)
        self.assertNotIn("GEMINI_API_KEY", environment)
        self.assertNotIn("GOOGLE_API_KEY", environment)
        self.assertNotIn("GH_TOKEN", environment)

    async def test_ocr_service_failure_returns_safe_retryable_error(self):
        with patch(
            "api._extract_report_isolated",
            new_callable=AsyncMock,
            side_effect=OCRUnavailableError("private detail"),
        ):
            response = await self.request(
                "POST",
                "/api/reports/extract",
                files={"file": ("scan.pdf", digital_pdf(), "application/pdf")},
            )
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private detail", response.text)
        self.assertNotIn("/run/media", response.text)


if __name__ == "__main__":
    unittest.main()
