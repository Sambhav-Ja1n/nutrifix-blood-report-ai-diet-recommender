from io import BytesIO
import re
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from PIL import Image, UnidentifiedImageError

from food_recommendations import (
    FoodRecommendationRequest,
    generate_food_recommendations,
)

logger = logging.getLogger("nutrifix.api")

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

MAX_FILE_SIZE = 15 * 1024 * 1024
ALLOWED_TYPES = {"application/pdf", "image/jpeg", "image/png"}


def parse_biomarkers(text: str) -> list[dict]:
    """Parse supported sample-report formats; this is not a universal parser."""
    biomarkers = []
    patterns = [
        ("Hemoglobin", r"Hemoglobin\s+([\d.]+)\s+g/dL\s+([\d.]+\s*-\s*[\d.]+)", "g/dL"),
        ("White blood cell count", r"White blood cell count\s+([\d.]+)\s+10\^3/uL\s+([\d.]+\s*-\s*[\d.]+)", "10^3/uL"),
        ("Platelet count", r"Platelet count\s+([\d.]+)\s+10\^3/uL\s+([\d.]+\s*-\s*[\d.]+)", "10^3/uL"),
        ("Fasting glucose", r"Fasting glucose\s+([\d.]+)\s+mg/dL\s+([\d.]+\s*-\s*[\d.]+)", "mg/dL"),
        ("Total cholesterol", r"Total cholesterol\s+([\d.]+)\s+mg/dL\s+(Less than\s+[\d.]+|[\d.]+\s*-\s*[\d.]+)", "mg/dL"),
        ("Ferritin", r"Ferritin\s+([\d.]+)\s+ng/mL\s+(Example only;\s*varies by lab)", "ng/mL"),
    ]
    for name, pattern, unit in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        reference_range = match.group(2).strip()
        range_match = re.fullmatch(r"([\d.]+)\s*-\s*([\d.]+)", reference_range)
        if range_match:
            lower, upper = float(range_match.group(1)), float(range_match.group(2))
            if lower > upper:
                status = "needs_review"
            elif value < lower:
                status = "below_supplied_range"
            elif value > upper:
                status = "above_supplied_range"
            else:
                status = "within_supplied_range"
        else:
            threshold_match = re.fullmatch(r"Less than\s+([\d.]+)", reference_range, flags=re.IGNORECASE)
            if threshold_match:
                status = "below_supplied_threshold" if value < float(threshold_match.group(1)) else "at_or_above_supplied_threshold"
            else:
                status = "needs_review"
        biomarkers.append({
            "name": name,
            "value": value,
            "unit": unit,
            "reference_range": reference_range,
            "range_status": status,
        })
    return biomarkers


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "NutriFix API"}


@app.post("/api/reports/extract")
async def extract_report(file: UploadFile = File(...)):
    """Validate an upload and extract selectable PDF text or request OCR."""
    if file.content_type not in ALLOWED_TYPES:
        await file.close()
        raise HTTPException(status_code=415, detail="Upload a PDF, JPG, or PNG file.")

    filename = file.filename or "uploaded-report"
    content_type = file.content_type
    content = await file.read(MAX_FILE_SIZE + 1)
    await file.close()

    if not content:
        raise HTTPException(status_code=400, detail="The file is empty.")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds the 15 MB limit.")

    try:
        if content_type == "application/pdf":
            if not content.startswith(b"%PDF-"):
                raise HTTPException(status_code=400, detail="The file is not a valid PDF.")
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise HTTPException(status_code=400, detail="Password-protected PDFs are not supported yet.")
            if len(reader.pages) > 100:
                raise HTTPException(status_code=400, detail="The PDF exceeds the 100-page limit.")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if not text.strip():
                return {
                    "filename": filename,
                    "status": "needs_ocr",
                    "message": "No selectable text was found. This may be a scanned PDF; OCR is needed.",
                    "extracted_text": "",
                    "biomarkers": [],
                }
        else:
            with Image.open(BytesIO(content)) as image:
                image.verify()
            return {
                "filename": filename,
                "status": "needs_ocr",
                "message": "The image was accepted. OCR is needed to extract its blood report text.",
                "extracted_text": "",
                "biomarkers": [],
            }
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file could not be read.")
    except Exception as exc:
        logger.exception("Report extraction failed")
        raise HTTPException(status_code=400, detail="Could not process this report.") from exc

    biomarkers = parse_biomarkers(text)
    return {
        "filename": filename,
        "status": "text_extracted",
        "message": "Report text extracted. Supported biomarkers were checked against ranges supplied in the report.",
        "extracted_text": text[:100_000],
        "biomarkers": biomarkers,
    }


@app.post("/api/reports/recommendations")
def report_food_recommendations(request: FoodRecommendationRequest = Body(...)):
    """Generate AI-assisted food education for the supplied structured markers."""
    try:
        data = generate_food_recommendations(request)
        return {
            "status": "success",
            "message": "Food guidance generated. Review the limitations and safety notes.",
            "data": data,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.warning("Food recommendation service unavailable: %s", str(exc))
        raise HTTPException(status_code=503, detail="The AI service is not configured or returned an unusable response.") from exc
    except Exception as exc:
        logger.exception("Food recommendation request failed")
        raise HTTPException(status_code=502, detail="Food recommendations could not be generated. Please try again later.") from exc
