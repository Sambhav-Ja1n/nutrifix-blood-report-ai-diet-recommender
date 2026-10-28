# NutriFix Project Status Report

**Checkpoint date:** 28 October 2026  
**Repository:** [Sambhav-Ja1n/nutrifix-blood-report-ai-diet-recommender](https://github.com/Sambhav-Ja1n/nutrifix-blood-report-ai-diet-recommender)  
**Current branch:** `main`

## Executive summary

NutriFix is a local research prototype for blood-report extraction, explicit human review, report-supplied range comparison, exploratory anomaly scoring, and general nutrition education. The current repository is private on GitHub. Sensitive environment files, datasets, generated outputs, and local machine files remain excluded from version control.

The frontend now includes a responsive landing page, light/dark mode, an interactive report-review story section, upload/review/results flows, and responsive layouts tested at desktop, tablet, and mobile viewport sizes.

## Current functionality

- FastAPI health, extraction, reviewed-analysis, and nutrition-recommendation endpoints.
- Selectable-text PDF extraction using PyMuPDF.
- Local Tesseract OCR for scanned PDFs and JPEG/PNG uploads.
- Conservative canonical biomarker matching with provenance and review flags.
- Editable candidate review with explicit user confirmation.
- Report-range comparisons using only the ranges supplied in the uploaded report.
- Five-model exploratory report scoring when all required CBC fields are confirmed.
- Gemini-backed general nutrition education when configured.
- React/Vite frontend with upload, review, analysis, recommendation, dark mode, and responsive presentation.
- Streamlit NHANES cohort explorer and research scripts.

## Recent fixes and improvements

1. Reconstructed table rows split across separate PDF text lines.
2. Accepted equivalent report notation such as `10^3/uL` while retaining strict unit validation.
3. Replaced unreadable `[object Object]` API errors with structured validation messages.
4. Added dark mode with persisted browser preference.
5. Added interactive editorial sections, animated cards, and a principles divider.
6. Reworked fluid sizing and breakpoints to prevent horizontal overflow.
7. Replaced the visually heavy green ticker with a subtle neutral divider.
8. Removed confirmed-unused starter assets and local generated clutter.

## Testing and verification

### Automated tests

```text
Ran 44 tests in 12.977s
OK
```

The suite covers report extraction, OCR behavior, table-row reconstruction, API validation, reviewed analysis, recommendation contracts, report scoring, and safety limits.

### Frontend checks

- `npm run build`: passed.
- `npm run lint`: passed.
- `git diff --check`: passed.

### Browser checks

The frontend was opened through the Vite development server and checked at:

- 1440 × 900
- 1024 × 768
- 768 × 900
- 390 × 844

All checked sizes reported zero horizontal overflow. The dark-mode toggle and interactive story tabs were also exercised.

### Sample report check

The synthetic sample PDF was processed through the live Vite proxy and returned four candidates:

- Hemoglobin — `13.8 g/dL`
- White blood cell count — `7.2 10^3/uL`
- Platelet count — `250 10^3/uL`
- Ferritin — `45 ng/mL`

Ferritin remains review-required because the sample report does not provide a usable numeric interval.

### AI provider check

The local `.env` contains a configured Gemini key, but the key value is not included in this report or Git history. A synthetic recommendation request reached Gemini and returned a temporary `503 UNAVAILABLE` response due to model demand. This was not a missing-key response. The repository currently uses Gemini; Groq is not wired into the application.

## Cleanup completed

Removed from the repository:

- Unreferenced Vite starter asset: `frontend/src/assets/react.svg`
- Unreferenced Vite starter asset: `frontend/src/assets/vite.svg`
- Unreferenced icon bundle: `frontend/public/icons.svg`

Removed locally as generated or backup clutter:

- `api.py.save`
- `frontend/dist/`
- Python `__pycache__/` directories

Preserved intentionally:

- `docs/screenshots/` because the screenshots are used by the README and reporting deliverables.
- `data/` and `outputs/` locally, because they support research workflows; they remain ignored by Git.
- Source scripts, tests, schemas, API code, frontend code, and documentation.

## Safety and limitations

- `.env` is ignored and was not committed.
- Raw NHANES files, merged datasets, participant-level outputs, and generated model artifacts remain local and ignored.
- OCR and extraction results are unverified and require human confirmation.
- The project does not diagnose, treat, or replace clinical care.
- Exploratory anomaly scores are not clinical risk scores.
- Nutrition output is general education, not individualized medical treatment.
- No clinical validation or field-level OCR precision/recall study has been completed.

## Key files

- [API](../api.py)
- [Frontend app](../frontend/src/App.jsx)
- [Frontend styles](../frontend/src/App.css)
- [Extraction module](../src/nutrifix/report_extraction.py)
- [Reviewed workflow](../src/nutrifix/report_workflow.py)
- [Test suite](../tests/)
- [README](../README.md)

## Screenshots

The repository includes synthetic/demo screenshots used for documentation:

![Landing page](screenshots/01-landing-page.png)

![Extraction review](screenshots/03-extraction-review.png)

![Results dashboard](screenshots/04-results-dashboard.png)

![Mobile layout](screenshots/07-mobile-layout.png)

These are interface captures only and do not represent clinical validation or real patient data.
