# NutriFix v1.0.0 — Baseline Project Checkpoint

This release records the existing NutriFix research prototype and its current source workflows. It is a baseline checkpoint, not a claim of clinical readiness.

## Included

- FastAPI service for a health check, limited selectable-text PDF extraction, basic biomarker pattern parsing, and Gemini-backed general nutrition education.
- React/Vite upload interface and a separate Streamlit explorer for NHANES anomaly outputs.
- NHANES 2017–2018 CBC, demographics, and ferritin download-verification, merge, audit, and feature-preparation scripts.
- Isolation Forest anomaly-detection baseline and exploratory explanation/report utilities.
- A separate fixed-split comparison workflow for Isolation Forest, novelty Local Outlier Factor, and regularized robust Mahalanobis distance, with tests for score direction, alignment, missingness, reproducibility, and output integrity.
- A canonical biomarker schema and a root ignore policy for secrets, datasets, generated outputs, caches, and local build files.

## Not included or not implemented

- Raw or merged NHANES data, local participant-level reports, generated outputs, and API credentials are excluded from version control.
- OCR is not implemented. Scanned PDFs and images receive a `needs_ocr` response.
- Clinical reference-range rules have not been added because applicable authoritative intervals and required population metadata are not present in the project.
- No clinical validation, diagnostic accuracy evaluation, or prospective evaluation is claimed. The model comparison is exploratory and based on a single train/test split.
- No project license has been selected.

## Data source

The local data workflow is based on NHANES 2017–2018 public-use files. See the official [NHANES data page](https://wwwn.cdc.gov/nchs/nhanes/search/DataPage.aspx?Cycle=2017-2018) and the [CBC_J](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/CBC_J.htm), [DEMO_J](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.htm), and [FERTIN_J](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/FERTIN_J.htm) documentation.
