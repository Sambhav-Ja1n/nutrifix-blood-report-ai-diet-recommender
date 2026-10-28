import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
from threading import BoundedSemaphore

from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware

from food_recommendations import (
    FoodRecommendationRequest,
    generate_food_recommendations,
)
from src.nutrifix.report_extraction import (
    MAX_UPLOAD_BYTES,
    OCRUnavailableError,
    ReportExtractionError,
    UnsupportedReportError,
    extract_report as extract_report_content,
    parse_text,
)
from src.nutrifix.report_workflow import (
    ReviewedReportRequest,
    analyze_reviewed_report,
)

logger = logging.getLogger("nutrifix.api")
_PROCESSING_WORKERS = 2
_PROCESSING_SLOTS = BoundedSemaphore(_PROCESSING_WORKERS)


class ProcessingBusyError(RuntimeError):
    """All bounded local processing slots are currently occupied."""


async def _run_bounded(func, *args):
    if not _PROCESSING_SLOTS.acquire(blocking=False):
        raise ProcessingBusyError("Local processing capacity is currently full")
    try:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nutrifix-processing",
        ) as executor:
            return await loop.run_in_executor(executor, partial(func, *args))
    finally:
        _PROCESSING_SLOTS.release()

app = FastAPI(
    title="NutriFix API",
    description="Blood report extraction and AI-assisted food guidance API",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"application/pdf", "image/jpeg", "image/png"}


def parse_biomarkers(text: str) -> list[dict]:
    """Compatibility wrapper for parsing text with canonical-name provenance."""
    return parse_text(text)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "NutriFix API"}


@app.post("/api/reports/extract")
async def extract_report(file: UploadFile = File(...)):
    """Extract selectable PDF text or use local Tesseract for scans/images."""
    if file.content_type not in ALLOWED_TYPES:
        await file.close()
        raise HTTPException(status_code=415, detail="Upload a PDF, JPG, or PNG file.")

    filename = file.filename or "uploaded-report"
    content_type = file.content_type
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()

    if not content:
        raise HTTPException(status_code=400, detail="The file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 15 MB limit.")

    try:
        result = await _run_bounded(extract_report_content, content, content_type)
        result["filename"] = filename
        return result
    except ProcessingBusyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Local report processing is busy. Try again shortly.",
        ) from exc
    except UnsupportedReportError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ReportExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OCRUnavailableError as exc:
        logger.warning("Local OCR service unavailable")
        raise HTTPException(status_code=503, detail="Local OCR is unavailable. Try a selectable-text PDF or contact the service owner.") from exc
    except Exception as exc:
        logger.exception("Report extraction failed")
        raise HTTPException(status_code=400, detail="Could not process this report.") from exc


@app.post("/api/reports/recommendations")
async def report_food_recommendations(request: FoodRecommendationRequest = Body(...)):
    """Generate guidance only from explicitly reviewed, validated report values."""
    try:
        if any(marker.range_status == "needs_review" for marker in request.biomarkers):
            raise HTTPException(
                status_code=422,
                detail="Recommendations require a user-confirmed value, canonical unit, and usable report range.",
            )
        data = await _run_bounded(generate_food_recommendations, request)
        return {
            "status": "success",
            "message": "Food guidance generated. Review the limitations and safety notes.",
            "data": data,
        }
    except HTTPException:
        raise
    except ProcessingBusyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Local service capacity is currently full. Try again shortly.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.warning("Food recommendation service unavailable: %s", str(exc))
        raise HTTPException(status_code=503, detail="The AI service is not configured or returned an unusable response.") from exc
    except Exception as exc:
        logger.exception("Food recommendation request failed")
        raise HTTPException(status_code=502, detail="Food recommendations could not be generated. Please try again later.") from exc


@app.post("/api/reports/analyze")
async def analyze_report(request: ReviewedReportRequest = Body(...)):
    """Compare user-confirmed values with only the ranges supplied in the report."""
    try:
        return analyze_reviewed_report(request)
    except Exception as exc:
        logger.exception("Reviewed report analysis failed")
        raise HTTPException(
            status_code=400,
            detail="The reviewed report could not be analyzed.",
        ) from exc
