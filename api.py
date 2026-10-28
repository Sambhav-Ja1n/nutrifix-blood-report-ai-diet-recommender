import asyncio
import json
import logging
import os
import sys
from threading import BoundedSemaphore

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware

from food_recommendations import (
    FoodRecommendationRequest,
    generate_food_recommendations,
)
from src.nutrifix.report_extraction import (
    MAX_UPLOAD_BYTES,
    MAX_REPORT_OCR_SECONDS,
    OCRUnavailableError,
    ReportExtractionError,
    UnsupportedReportError,
    parse_text,
)
from src.nutrifix.report_workflow import (
    ReviewedReportRequest,
    analyze_reviewed_report,
)
from src.nutrifix.report_scoring import (
    InsufficientReportProfile,
    ReportScoringError,
    ScoringArtifactUnavailable,
    score_reviewed_profile,
)

logger = logging.getLogger("nutrifix.api")
_PROCESSING_WORKERS = 2
_PROCESSING_SLOTS = BoundedSemaphore(_PROCESSING_WORKERS)
_REPORT_WORKER_ENV_KEYS = {
    "PATH",
    "LD_LIBRARY_PATH",
    "TESSDATA_PREFIX",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PYTHONPATH",
    "PYTHONHOME",
}


class ProcessingBusyError(RuntimeError):
    """All bounded local processing slots are currently occupied."""


class ProcessingTimeoutError(RuntimeError):
    """A bounded report worker exceeded its total wall-clock allowance."""


def _report_worker_environment() -> dict[str, str]:
    """Pass only runtime and OCR configuration, never application credentials."""
    return {
        name: value
        for name, value in os.environ.items()
        if name in _REPORT_WORKER_ENV_KEYS
    }


async def _extract_report_isolated(content: bytes, content_type: str) -> dict:
    if not _PROCESSING_SLOTS.acquire(blocking=False):
        raise ProcessingBusyError("Local processing capacity is currently full")
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "src.nutrifix.report_worker",
            content_type,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_report_worker_environment(),
        )
        try:
            stdout, _stderr = await asyncio.wait_for(
                process.communicate(content),
                timeout=MAX_REPORT_OCR_SECONDS + 10,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise ProcessingTimeoutError("Report processing exceeded its time limit.") from exc
        except asyncio.CancelledError:
            process.kill()
            await process.communicate()
            raise

        try:
            payload = json.loads(stdout)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ReportExtractionError("The report processor returned invalid output.") from exc
        status_code = payload.get("status_code")
        if status_code == 200 and isinstance(payload.get("result"), dict):
            return payload["result"]
        detail = payload.get("detail", "Could not process this report.")
        if status_code == 503:
            raise OCRUnavailableError(detail)
        if status_code == 400:
            raise ReportExtractionError(detail)
        raise ReportExtractionError("The report processor failed safely.")
    except FileNotFoundError as exc:
        raise ReportExtractionError("The local Python report worker could not start.") from exc
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.communicate()
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
async def extract_report(
    file: UploadFile = File(...),
    include_source_text: bool = Form(default=False),
):
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
        result = await _extract_report_isolated(content, content_type)
        result["filename"] = filename
        if not include_source_text:
            result.pop("extracted_text", None)
            for page in result.get("pages", []):
                page.pop("text", None)
                page.pop("lines", None)
            for line in result.get("unresolved_numeric_lines", []):
                line.pop("text", None)
        return result
    except ProcessingBusyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Local report processing is busy. Try again shortly.",
        ) from exc
    except ProcessingTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Report processing exceeded its time limit. Try a smaller report.",
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
        data = await generate_food_recommendations(request)
        return {
            "status": "success",
            "message": "Food guidance generated. Review the limitations and safety notes.",
            "data": data,
        }
    except HTTPException:
        raise
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
        result = analyze_reviewed_report(request)
        values_by_column = {
            marker.canonical_biomarker: marker.value
            for marker in request.biomarkers
        }
        try:
            anomaly_result = score_reviewed_profile(values_by_column)
        except InsufficientReportProfile as exc:
            anomaly_result = {
                "status": "insufficient_data",
                "required_feature_count": 15,
                "missing_biomarkers": exc.missing_features,
                "message": "A statistical score requires all 15 reviewed CBC model biomarkers.",
            }
        except ScoringArtifactUnavailable:
            anomaly_result = {
                "status": "model_unavailable",
                "message": "The local reference scoring model is not built or does not match the current schema.",
            }
        except ReportScoringError:
            anomaly_result = {
                "status": "scoring_failed",
                "message": "The reviewed profile could not be scored safely.",
            }
        except Exception:
            logger.exception("Reference model scoring failed")
            anomaly_result = {
                "status": "scoring_failed",
                "message": "The reviewed profile could not be scored safely.",
            }
        result["statistical_anomaly"] = anomaly_result
        return result
    except Exception as exc:
        logger.exception("Reviewed report analysis failed")
        raise HTTPException(
            status_code=400,
            detail="The reviewed report could not be analyzed.",
        ) from exc
