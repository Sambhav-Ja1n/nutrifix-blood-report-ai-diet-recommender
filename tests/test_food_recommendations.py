import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from food_recommendations import (
    FoodRecommendationRequest,
    generate_food_recommendations,
)


def reviewed_request(reference_range="12.0 - 15.5"):
    return FoodRecommendationRequest.model_validate({
        "diet": "vegetarian",
        "biomarkers": [{
            "canonical_biomarker": "LBXHGB",
            "name": "Hemoglobin",
            "value": 13.5,
            "unit": "g/dL",
            "reference_range": reference_range,
            "reviewed_by_user": True,
        }],
    })


class FoodRecommendationTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_cannot_change_finding_or_range_status(self):
        fake_response = SimpleNamespace(text=json.dumps({
            "summary": "General nutrition education.",
            "recommendations": [{
                "finding": "hemoglobin",
                "finding_status": "above_supplied_range",
                "why_it_matters": "Discuss report results with a clinician.",
                "foods_to_consider": [],
                "foods_to_limit": [],
                "next_steps": "Review with a clinician.",
                "evidence_note": "General nutrition rationale only.",
            }],
            "safety_notes": [],
            "limitations": [],
        }))
        generate = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=generate),
                aclose=AsyncMock(),
            ),
            close=Mock(),
        )
        with patch("food_recommendations._client", return_value=fake_client):
            result = await generate_food_recommendations(reviewed_request())

        self.assertEqual(result["recommendations"][0]["finding"], "Hemoglobin")
        self.assertEqual(
            result["recommendations"][0]["finding_status"],
            "within_supplied_range",
        )
        self.assertTrue(result["limitations"])
        fake_client.aio.aclose.assert_awaited_once()
        fake_client.close.assert_called_once()

    async def test_recommendation_generation_rejects_missing_or_unclear_range(self):
        with patch("food_recommendations._client") as make_client:
            with self.assertRaisesRegex(ValueError, "usable report range"):
                await generate_food_recommendations(reviewed_request(reference_range=""))
        make_client.assert_not_called()

    async def test_unrecognized_generated_finding_is_rejected(self):
        fake_response = SimpleNamespace(text=json.dumps({
            "recommendations": [{"finding": "Unprovided biomarker"}],
            "safety_notes": [],
            "limitations": [],
        }))
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=AsyncMock(return_value=fake_response)),
                aclose=AsyncMock(),
            ),
            close=Mock(),
        )
        with patch("food_recommendations._client", return_value=fake_client):
            with self.assertRaisesRegex(RuntimeError, "finding that was not supplied"):
                await generate_food_recommendations(reviewed_request())


if __name__ == "__main__":
    unittest.main()
