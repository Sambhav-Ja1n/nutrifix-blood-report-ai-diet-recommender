# NutriFix

### AI-Powered Blood Report Analysis & Personalized Diet Recommendations

NutriFix is a research prototype for extracting a small set of values from supported blood-report formats, exploring CBC data statistically, and returning general nutrition education from structured inputs. It is not a medical device or a clinically validated diagnostic system.

## What is implemented

- A FastAPI service with a health endpoint, limited selectable-text PDF extraction, basic pattern matching for a small set of biomarkers, and a structured food-guidance endpoint.
- A React/Vite interface for report upload and nutrition-guidance requests.
- A separate Streamlit explorer for the NHANES cohort and the existing Isolation Forest outputs.
- Scripts to verify downloaded files, merge NHANES 2017–2018 CBC, demographics, and ferritin data, audit data quality, prepare CBC features, and run an Isolation Forest baseline.
- Exploratory anomaly explanations, participant reports, and sensitivity checks.
- A held-out comparison script for Isolation Forest, novelty Local Outlier Factor, and regularized robust Mahalanobis distance. Its rankings are exploratory and have not been clinically validated.

### OCR status

OCR is **not implemented**. The API can extract selectable text from some PDFs and check a few known text patterns. Scanned PDFs and uploaded images receive a `needs_ocr` response; no image text is recognized. The parser is not a universal laboratory-report parser.

### Nutrition guidance status

The recommendation endpoint calls the Google Gemini API when configured. It is intended for general nutrition education. The project does not connect a curated medical evidence database, does not provide validated personalized treatment advice, and must not be used to diagnose or treat disease.

## Architecture and data flow

```mermaid
flowchart TD
  A[NHANES XPT files] --> B[Merge and audit scripts]
  B --> C[Merged local Parquet dataset]
  C --> D[Quality checks and CBC feature preparation]
  D --> E[Isolation Forest baseline]
  E --> F[Local metrics and reports]
  F --> G[Streamlit cohort explorer]
  D --> H[Held-out anomaly comparison]
  H --> I[Local comparison metrics]
  J[React/Vite upload interface] --> K[FastAPI]
  K --> L[Selectable PDF text and limited pattern parsing]
  K --> M[Gemini nutrition-education request]
```

The API/UI path and the NHANES cohort-analysis path are separate. The comparison script reads the merged local dataset directly and writes its outputs under `outputs/metrics/`; it does not replace the production API or frontend.

## Repository layout

```text
NutriFix/
├── api.py                         # FastAPI endpoints
├── app.py                         # Streamlit NHANES explorer
├── food_recommendations.py        # Gemini-backed education response
├── frontend/                      # React and Vite client
├── schemas/                       # Canonical biomarker definitions
├── scripts/                       # Download, merge, audit, and model commands
├── src/nutrifix/                  # Data and model implementation
├── tests/                          # Unit tests
├── data/                           # Local-only source and processed datasets
└── outputs/                        # Local-only audits, metrics, and reports
```

`data/` and `outputs/` are intentionally excluded from Git. The source code expects those paths to exist locally when running the data workflows.

## Technology

- Python 3.12, pandas, NumPy, scikit-learn, SciPy, and PyArrow
- FastAPI, Pydantic, Uvicorn, pypdf, and Pillow
- Google Gen AI SDK with an optional Gemini API key
- Streamlit for cohort exploration
- React, Vite, and npm for the upload interface

The Python versions in `requirements.txt` match the available project `mlenv` environment at checkpoint preparation. The frontend versions are locked in `frontend/package-lock.json`.

## Garuda Linux / Linux setup

Create a Python 3.12 virtual environment and install the recorded dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env` only if you intend to call the recommendation endpoint. Never commit `.env`. The API defaults to the model name configured in `GEMINI_MODEL` or `gemini-2.5-flash`.

For the frontend, use a Node version satisfying Vite's lockfile requirement (`^20.19.0` or `>=22.12.0`), then run:

```bash
cd frontend
npm ci
npm run dev
```

The client calls the local API at `http://127.0.0.1:8000`. In another terminal, from the project root with the Python environment activated, start the API:

```bash
uvicorn api:app --reload
```

The FastAPI health endpoint is `/api/health`; interactive API documentation is provided by FastAPI at `/docs` while the local service is running.

## NHANES data and cohort explorer

The project uses the 2017–2018 NHANES CBC (`CBC_J.xpt`), demographics (`DEMO_J.xpt`), and ferritin (`FERTIN_J.xpt`) public-use files. Download the files and their documentation from the official [NHANES 2017–2018 data page](https://wwwn.cdc.gov/nchs/nhanes/search/DataPage.aspx?Cycle=2017-2018), [CBC_J codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/CBC_J.htm), [DEMO_J codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.htm), and [FERTIN_J codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/FERTIN_J.htm). Review component-specific eligibility, missingness, and analytic notes before interpreting results.

Place the three files here:

```text
data/raw/nhanes/2017_2018/CBC_J.xpt
data/raw/nhanes/2017_2018/DEMO_J.xpt
data/raw/nhanes/2017_2018/FERTIN_J.xpt
```

Then run the existing workflow from the project root:

```bash
python scripts/verify_downloads.py
python scripts/merge_nhanes.py
python scripts/run_pipeline.py
python -m src.nutrifix.explain_anomalies
python -m src.nutrifix.participant_report
streamlit run app.py
```

`run_pipeline.py` performs quality checks, feature preparation, and Isolation Forest scoring; merging is a separate step. The Streamlit explorer also needs the explanation and participant-report files generated by the two module commands above.

To run the separate held-out comparison, use a fresh output directory each time. The script refuses to overwrite an existing experiment directory:

```bash
python scripts/compare_anomaly_models.py \
  --output-dir outputs/metrics/anomaly_model_comparison_run_01
```

## Tests

The comparison unit tests use Python's standard library test runner:

```bash
python -m unittest discover -s tests -v
```

This command was verified in the project `mlenv` environment during checkpoint preparation. `scripts/test_pipeline_outputs.py` is a separate output-check script and requires the generated baseline prediction, explanation, and report files to be present.

## Anomaly-detection limitations

The original Isolation Forest baseline uses CBC measurements and missingness indicators. Its earlier cohort-level preprocessing fitted medians and scaling on the full cohort; its outputs are exploratory and are not a held-out clinical evaluation. The separate comparison script fits preprocessing and models on a fixed reference partition and scores a held-out partition, but one split does not establish generalization. Agreement, ranking stability, or a selected fraction of candidates is not evidence of diagnostic accuracy. No validated clinical labels are included.

The comparison does not implement clinical reference-range rules. The canonical schema provides biomarker names and units, but the repository does not provide sourced, population-applicable reference intervals. Such a baseline needs authoritative interval sources and relevant age, sex, pregnancy, assay, and specimen context before implementation.

## Privacy and safe use

- Do not commit `.env`, API keys, health reports, patient data, raw datasets, merged datasets, or generated participant-level outputs. Root `.gitignore` excludes these paths and file types by default.
- NHANES is public-use survey data, but local merged data and reports contain participant-level biomarker and demographic fields and should remain local unless deliberately reviewed.
- The API reads uploaded content into memory for processing. If the recommendations endpoint is called, the supplied structured biomarkers and preferences are sent to Google's Gemini API; review the provider's terms and your privacy obligations before using real reports.
- Treat extracted values and supplied ranges as unverified inputs. The API's parser supports only a narrow set of text patterns and does not validate clinical correctness.
- NutriFix is a research prototype. It is not a substitute for a clinician, laboratory, or validated medical device. Do not use it for diagnosis, treatment, or urgent-care decisions.

## Roadmap

- Add OCR only after selecting and evaluating an appropriate OCR workflow.
- Source and review clinically applicable reference intervals before adding a clinical-rules baseline.
- Evaluate anomaly methods on appropriate labeled or clinician-reviewed data before making clinical claims.
- Improve report-format coverage, privacy controls, and documented validation before any real-world clinical use.
