import { useEffect, useRef, useState } from "react";

import "./App.css";

const API = "/api/reports";
const BIOMARKERS = [
  ["LBXWBCSI", "White blood cell count", "10^3 cells/uL"],
  ["LBXLYPCT", "Lymphocyte percentage", "%"],
  ["LBXMOPCT", "Monocyte percentage", "%"],
  ["LBXNEPCT", "Neutrophil percentage", "%"],
  ["LBXEOPCT", "Eosinophil percentage", "%"],
  ["LBXBAPCT", "Basophil percentage", "%"],
  ["LBDLYMNO", "Absolute lymphocyte count", "10^3 cells/uL"],
  ["LBDMONO", "Absolute monocyte count", "10^3 cells/uL"],
  ["LBDNENO", "Absolute neutrophil count", "10^3 cells/uL"],
  ["LBDEONO", "Absolute eosinophil count", "10^3 cells/uL"],
  ["LBDBANO", "Absolute basophil count", "10^3 cells/uL"],
  ["LBXRBCSI", "Red blood cell count", "10^6 cells/uL"],
  ["LBXHGB", "Hemoglobin", "g/dL"],
  ["LBXHCT", "Hematocrit", "%"],
  ["LBXMCVSI", "Mean corpuscular volume", "fL"],
  ["LBXMCHSI", "Mean corpuscular hemoglobin", "pg"],
  ["LBXMC", "Mean corpuscular hemoglobin concentration", "g/dL"],
  ["LBXRDW", "Red cell distribution width", "%"],
  ["LBXPLTSI", "Platelet count", "10^3 cells/uL"],
  ["LBXMPSI", "Mean platelet volume", "fL"],
  ["LBXNRBC", "Nucleated red blood cells", "cells per 100 WBC"],
  ["LBXFER", "Ferritin", "ng/mL"],
];

const initialReview = (candidate) => ({
  biomarker: candidate.canonical_biomarker || "",
  value: candidate.value ?? "",
  unit: candidate.unit || "",
  reference_range: candidate.reference_range || "",
  confirmed: false,
});

const errorMessage = (cause, fallback) => {
  if (typeof cause?.message === "string") return cause.message;
  if (typeof cause?.message?.detail === "string") return cause.message.detail;
  if (typeof cause?.detail === "string") return cause.detail;
  if (typeof cause === "string") return cause;
  if (cause && typeof cause === "object") {
    try {
      return JSON.stringify(cause);
    } catch {
      return fallback;
    }
  }
  return fallback;
};

const formatApiDetail = (detail, fallback) => {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => (typeof item?.msg === "string" ? item.msg : null))
      .filter(Boolean);
    if (messages.length) return messages.join(" ");
  }
  return errorMessage(detail, fallback);
};

const readJsonResponse = async (response, fallbackMessage) => {
  const body = await response.text();
  if (!body.trim()) {
    throw new Error(
      response.ok
        ? fallbackMessage
        : "The NutriFix API returned an empty response. Start the API with `uvicorn api:app --reload` and try again.",
    );
  }

  let data;
  try {
    data = JSON.parse(body);
  } catch {
    throw new Error(
      response.ok
        ? "The NutriFix API returned an invalid response."
        : "The NutriFix API returned an invalid response. Check that the API is running and try again.",
    );
  }

  if (!response.ok) {
    throw new Error(formatApiDetail(data.detail, fallbackMessage));
  }
  return data;
};

function App() {
  const [page, setPage] = useState("home");
  const [theme, setTheme] = useState(() => localStorage.getItem("nutrifix-theme") || "light");
  const [activeStory, setActiveStory] = useState(0);
  const [file, setFile] = useState(null);
  const [extraction, setExtraction] = useState(null);
  const [review, setReview] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [diet, setDiet] = useState("vegetarian");
  const [allergies, setAllergies] = useState("");
  const [preferences, setPreferences] = useState("Indian meals");
  const [goal, setGoal] = useState("general");
  const [budget, setBudget] = useState("moderate");
  const [allowEggs, setAllowEggs] = useState(false);
  const [age, setAge] = useState("");
  const [dietaryGuidance, setDietaryGuidance] = useState("");
  const fileInput = useRef(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("nutrifix-theme", theme);
  }, [theme]);

  const chooseFile = (nextFile) => {
    setError("");
    if (!nextFile) return;
    if (!/\.(pdf|jpe?g|png)$/i.test(nextFile.name)) {
      setError("Choose a PDF, JPG, or PNG file.");
      return;
    }
    if (nextFile.size > 15 * 1024 * 1024) {
      setError("The file must be 15 MB or smaller.");
      return;
    }
    setFile(nextFile);
  };

  const startReview = async () => {
    if (!file || busy) return;
    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch(`${API}/extract`, { method: "POST", body });
      const data = await readJsonResponse(response, "Report extraction failed.");
      setExtraction(data);
      setReview((data.biomarkers || []).map(initialReview));
      setPage("review");
    } catch (cause) {
      setError(errorMessage(cause, "Could not connect to the local NutriFix API."));
    } finally {
      setBusy(false);
    }
  };

  const updateReview = (index, field, value) => {
    setReview((current) => current.map((row, rowIndex) =>
      rowIndex === index ? { ...row, [field]: value, ...(field !== "confirmed" ? { confirmed: false } : {}) } : row,
    ));
  };

  const analyzeReviewed = async () => {
    const confirmed = review.filter((row) => row.confirmed && row.biomarker && row.value !== "");
    if (!confirmed.length) {
      setError("Confirm at least one complete biomarker row before analysis.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ biomarkers: confirmed.map((row) => ({
          canonical_biomarker: row.biomarker,
          name: BIOMARKERS.find(([code]) => code === row.biomarker)?.[1] || row.biomarker,
          value: Number(row.value),
          unit: row.unit,
          reference_range: row.reference_range,
          reviewed_by_user: true,
        })) }),
      });
      const data = await readJsonResponse(response, "Analysis failed.");
      setAnalysis(data);
      setRecommendations(null);
      setPage("results");
    } catch (cause) {
      setError(errorMessage(cause, "Could not analyze the reviewed values."));
    } finally {
      setBusy(false);
    }
  };

  const getRecommendations = async () => {
    const usable = (analysis?.reference_range_comparisons || []).filter((row) => row.range_status !== "needs_review");
    if (!usable.length) {
      setError("Recommendations need at least one reviewed result with a clear laboratory range.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API}/recommendations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          diet,
          allow_eggs: diet === "vegetarian" && allowEggs,
          allergies: allergies.split(",").map((value) => value.trim()).filter(Boolean),
          food_preferences: preferences.split(",").map((value) => value.trim()).filter(Boolean),
          goal,
          budget,
          age: age.trim() ? Number(age) : null,
          dietary_guidance: dietaryGuidance.trim(),
          biomarkers: usable.map((row) => ({ ...row, reviewed_by_user: true })),
        }),
      });
      const data = await readJsonResponse(response, "Could not prepare food guidance.");
      setRecommendations(data.data);
    } catch (cause) {
      setError(errorMessage(cause, "The recommendation service is unavailable."));
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    setPage("home"); setFile(null); setExtraction(null); setReview([]);
    setAnalysis(null); setRecommendations(null); setError("");
  };

  return <div className="app-shell">
    <header className="topbar">
      <button className="brand" onClick={reset} aria-label="NutriFix home"><span className="brand-symbol">N</span><span>nutri<span>fix</span></span></button>
      <nav aria-label="Main navigation">
        <button onClick={reset}>Home</button>
        <a href="#methodology">How it works</a>
        <button className="theme-toggle" aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`} onClick={() => setTheme((current) => current === "light" ? "dark" : "light")}><span>{theme === "light" ? "☾" : "☀"}</span><small>{theme === "light" ? "Dark" : "Light"}</small></button>
        <button className="nav-action" onClick={() => setPage("upload")}>Analyze a report <span>↗</span></button>
      </nav>
    </header>

    {page === "home" && <main>
      <section className="hero-wrap">
        <div className="hero-copy">
          <p className="eyebrow"><span /> A clearer look at your lab report</p>
          <h1>Understand your results, <em>one step at a time.</em></h1>
          <p className="hero-lede">Review extracted blood test values, compare them with the ranges printed by your laboratory, and explore cautious nutrition education.</p>
          <div className="hero-actions"><button className="button-primary" onClick={() => setPage("upload")}>Start with a report <span>→</span></button><a className="button-secondary" href="#methodology">See how it works</a></div>
          <div className="privacy-line"><span>◇</span> Local prototype · Review is required · No diagnosis</div>
        </div>
        <div className="hero-art" aria-label="Illustration of a report review card">
          <div className="art-orbit orbit-one" /><div className="art-orbit orbit-two" />
          <article className="sample-card"><div className="sample-card-head"><span className="mini-mark">N</span><div><small>REPORT REVIEW</small><strong>Values are yours to confirm</strong></div><span className="sample-pill">Example</span></div><div className="sample-row"><span className="sample-icon">Hb</span><div><strong>Hemoglobin</strong><small>Extracted candidate · review needed</small></div><b>— <small>g/dL</small></b></div><div className="sample-row"><span className="sample-icon pale">W</span><div><strong>White blood cells</strong><small>Lab supplied interval</small></div><b>— <small>10³/uL</small></b></div><div className="sample-note">Illustrative interface · no patient data</div></article>
          <div className="float-chip"><span>✓</span> You stay in control</div>
        </div>
      </section>
      <div className="signal-ticker" aria-label="NutriFix principles"><span>NO DIAGNOSIS</span><i>✦</i><span>HUMAN REVIEW FIRST</span><i>✦</i><span>REPORT-RANGE COMPARISONS</span><i>✦</i><span>GENERAL NUTRITION EDUCATION</span><i>✦</i><span>NO DIAGNOSIS</span></div>
      <section className="workflow-strip"><div><b>01</b><span><strong>Upload</strong><small>PDF, JPG, or PNG</small></span></div><i>→</i><div><b>02</b><span><strong>Review</strong><small>Correct and confirm</small></span></div><i>→</i><div><b>03</b><span><strong>Understand</strong><small>Explore your results</small></span></div></section>
      <section className="story-section">
        <div className="story-intro"><p className="eyebrow">A BETTER WAY TO READ A REPORT</p><h2>Less guessing.<br /><em>More context.</em></h2><p>NutriFix turns a dense report into a calm, reviewable conversation. Every step keeps you in control of what gets analyzed.</p><div className="story-tabs">{["Extract the signal", "Keep the context", "Choose your next step"].map((label, index) => <button key={label} className={activeStory === index ? "active" : ""} onClick={() => setActiveStory(index)}><span>0{index + 1}</span>{label}<b>↗</b></button>)}</div></div>
        <div className={`story-canvas story-${activeStory}`}><div className="story-orbit" /><div className="story-stat"><small>STEP {activeStory + 1} / 3</small><strong>{["Signal, not noise.", "Your report sets the range.", "Small steps, clearly explained."][activeStory]}</strong></div><div className="story-card"><span className="story-card-kicker">{["EXTRACTED CANDIDATE", "LAB-SUPPLIED RANGE", "GENERAL GUIDANCE"][activeStory]}</span><strong>{["Hemoglobin", "White blood cell count", "Balanced plate"][activeStory]}</strong><div className="story-card-line"><span>{["13.8", "7.2", "↗"][activeStory]}</span><small>{["g/dL · review needed", "10³/uL · 4.0–11.0", "food education"][activeStory]}</small></div><div className="story-card-foot">{["Review each field before analysis", "Only use ranges printed on the report", "Never a diagnosis or treatment plan"][activeStory]}</div></div><span className="story-dot dot-one" /><span className="story-dot dot-two" /></div>
      </section>
      <section className="method-section" id="methodology"><div><p className="eyebrow">THE NUTRIFIX APPROACH</p><h2>Built for clarity.<br /><em>Honest about limits.</em></h2><p>Extraction is a starting point, not a verified result. NutriFix keeps lab range comparisons, statistical outlier rankings, and nutrition guidance as separate pieces of information.</p></div><div className="method-cards"><article><span>01</span><h3>You review first</h3><p>Check every extracted biomarker, value, unit, and range before analysis.</p></article><article><span>02</span><h3>Your report sets the range</h3><p>Range comparisons use only the interval supplied with your report.</p></article><article><span>03</span><h3>Outliers are not diagnoses</h3><p>Exploratory model scores describe statistical similarity, not disease or risk.</p></article></div></section>
      <section className="numbers-section"><div><strong>01</strong><span>Report in<br />your hands</span></div><div><strong>03</strong><span>Review steps<br />before insight</span></div><div><strong>∞</strong><span>Room for<br />better questions</span></div><p>Designed for careful exploration, not instant conclusions.</p></section>
      <section className="notice"><strong>Research prototype</strong><p>NutriFix is not a medical device and cannot diagnose, treat, or replace care from a qualified health professional. Do not use it for urgent decisions.</p></section>
    </main>}

    {page === "upload" && <main className="workspace"><button className="back-button" onClick={reset}>← Home</button><div className="page-heading"><p className="eyebrow">STEP 1 OF 3 · UPLOAD</p><h1>Start with your report</h1><p>Choose a PDF or image. Your results will remain unverified until you review and confirm them.</p></div>
      <section className="panel upload-panel"><button className="dropzone" onClick={() => fileInput.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); chooseFile(event.dataTransfer.files?.[0]); }}><span className="upload-glyph">↑</span><strong>Drop your report here</strong><span>or browse files</span><small>PDF, JPG, PNG · up to 15 MB</small></button><input ref={fileInput} type="file" accept=".pdf,.jpg,.jpeg,.png,application/pdf,image/jpeg,image/png" hidden onChange={(event) => { chooseFile(event.target.files?.[0]); event.target.value = ""; }} />
        {file && <div className="file-selected"><span className="file-type">{file.name.toLowerCase().endsWith(".pdf") ? "PDF" : "IMG"}</span><span><strong>{file.name}</strong><small>{(file.size / 1048576).toFixed(2)} MB · selected</small></span><button onClick={() => setFile(null)}>Remove</button></div>}
        {error && <p className="inline-error" role="alert">{error}</p>}<button className="button-primary full-button" disabled={!file || busy} onClick={startReview}>{busy ? "Processing report…" : "Extract candidate values →"}</button>
        <p className="privacy-callout">The API omits full source text by default. Extracted fields can still contain sensitive health information; use synthetic reports for demonstrations.</p>
      </section>
    </main>}

    {page === "review" && <main className="workspace wide"><button className="back-button" onClick={() => setPage("upload")}>← Upload</button>{/synthetic/i.test(extraction?.filename || "") && <div className="synthetic-banner">SYNTHETIC DEMO DATA — NOT A REAL PATIENT REPORT</div>}<div className="page-heading"><p className="eyebrow">STEP 2 OF 3 · HUMAN REVIEW</p><h1>Check every candidate</h1><p>These values were extracted from <strong>{extraction?.filename}</strong>. Correct them against the original report and confirm each row you want to analyze.</p></div>
      {!review.length ? <section className="panel empty-state"><h2>No biomarker candidates found</h2><p>Try a clearer report. NutriFix will not invent missing values.</p></section> : <section className="panel review-panel"><div className="review-warning">Extraction is unverified. OCR confidence does not establish that a value is clinically correct.</div><div className="review-table-wrap"><table className="review-table"><thead><tr><th>Biomarker</th><th>Value</th><th>Unit</th><th>Lab range</th><th>Source</th><th>Confirm</th></tr></thead><tbody>{review.map((row, index) => { const candidate = extraction.biomarkers[index]; return <tr key={`${candidate.canonical_biomarker || "unknown"}-${index}`}><td><select aria-label={`Biomarker row ${index + 1}`} value={row.biomarker} onChange={(event) => updateReview(index, "biomarker", event.target.value)}><option value="">Unmapped / skip</option>{BIOMARKERS.map(([code, label]) => <option key={code} value={code}>{label}</option>)}</select><small>{candidate.name || candidate.canonical_biomarker || "Unresolved name"}</small>{candidate.confidence != null && <small>Extraction confidence: {Math.round(candidate.confidence * 100)}%</small>}{candidate.review_reasons?.length > 0 && <small className="row-warning">{candidate.review_reasons.join(", ")}</small>}</td><td><input aria-label={`Value row ${index + 1}`} inputMode="decimal" value={row.value} onChange={(event) => updateReview(index, "value", event.target.value)} /></td><td><input aria-label={`Unit row ${index + 1}`} value={row.unit} onChange={(event) => updateReview(index, "unit", event.target.value)} /></td><td><input aria-label={`Reference range row ${index + 1}`} value={row.reference_range} onChange={(event) => updateReview(index, "reference_range", event.target.value)} placeholder="Not supplied" /></td><td><small>{candidate.provenance?.page_number ? `Page ${candidate.provenance.page_number}` : "Page unavailable"}</small>{candidate.provenance?.extraction_method && <small>{candidate.provenance.extraction_method}</small>}</td><td><label className="confirm-box"><input type="checkbox" checked={row.confirmed} disabled={!row.biomarker || row.value === "" || !row.unit} onChange={(event) => updateReview(index, "confirmed", event.target.checked)} /><span>Checked</span></label></td></tr>; })}</tbody></table></div><div className="review-footer"><span>{review.filter((row) => row.confirmed).length} rows confirmed · {review.length} candidates</span><button className="button-primary" disabled={busy || !review.some((row) => row.confirmed)} onClick={analyzeReviewed}>{busy ? "Analyzing…" : "Analyze confirmed values →"}</button></div></section>}
      {error && <p className="inline-error" role="alert">{error}</p>}<p className="disclaimer">Only rows you explicitly confirm are sent for analysis. Units must match NutriFix’s canonical schema; automatic unit conversion is unavailable.</p>
    </main>}

    {page === "results" && <main className="workspace wide"><button className="back-button" onClick={() => setPage("review")}>← Review values</button>{/synthetic/i.test(extraction?.filename || "") && <div className="synthetic-banner">SYNTHETIC DEMO DATA — NOT A REAL PATIENT REPORT</div>}<div className="page-heading"><p className="eyebrow">STEP 3 OF 3 · RESULTS</p><h1>Reviewed report overview</h1><p>Comparisons below use only the ranges you confirmed from your report.</p></div>
      <section className="panel results-panel"><div className="result-summary"><div><span className="status-mark">✓</span><div><h2>Analysis complete</h2><p>{analysis.data_quality.user_confirmed_biomarkers} confirmed biomarker{analysis.data_quality.user_confirmed_biomarkers === 1 ? "" : "s"}</p></div></div><span className="status-tag">User reviewed</span></div><div className="result-table-wrap"><table className="result-table"><thead><tr><th>Biomarker</th><th>Result</th><th>Lab range</th><th>Comparison</th></tr></thead><tbody>{analysis.reference_range_comparisons.map((row) => <tr key={row.canonical_biomarker}><td>{row.name}</td><td>{row.value} {row.unit}</td><td>{row.reference_range || "Not supplied"}</td><td><span className={`range-status ${row.range_status}`}>{row.range_status.replaceAll("_", " ")}</span></td></tr>)}</tbody></table></div>
        <section className="anomaly-card"><div><p className="eyebrow">EXPLORATORY MODEL COMPARISON</p><h3>Statistical outlier ranking</h3></div>{analysis.statistical_anomaly.status === "scored" ? <><p>These scores compare the complete profile with an NHANES reference cohort. They are not probabilities or diagnoses.</p><div className="model-score-list">{Object.entries(analysis.statistical_anomaly.models).map(([name, item]) => <div key={name}><span>{name.replaceAll("_", " ")}</span><strong>{item.anomaly_score.toPrecision(4)}</strong><small>{item.empirical_reference_percentile.toFixed(1)}th reference rank</small></div>)}</div></> : <p>{analysis.statistical_anomaly.message || analysis.statistical_anomaly.reason || "A model score is not available for this profile."} Scores are exploratory and are not diagnoses.</p>}</section>
        <details className="limitations"><summary>Data quality and limitations</summary><ul>{analysis.limitations.map((item) => <li key={item}>{item}</li>)}</ul></details>
      </section>
        <section className="panel recommendations-panel"><p className="eyebrow">OPTIONAL · GENERAL NUTRITION EDUCATION</p><h2>Explore food guidance</h2><p>Guidance uses confirmed results with clear report ranges and your dietary preferences. It is not treatment.</p><div className="profile-grid"><label>Diet<select value={diet} onChange={(event) => setDiet(event.target.value)}><option value="vegan">Vegan</option><option value="vegetarian">Vegetarian</option><option value="non_vegetarian">Non-vegetarian</option></select></label><label>Goal<select value={goal} onChange={(event) => setGoal(event.target.value)}><option value="general">Overall nutrition</option><option value="weight_management">Weight management</option><option value="muscle_gain">Muscle gain</option><option value="focus_on_report">Focus on report findings</option></select></label><label>Budget<select value={budget} onChange={(event) => setBudget(event.target.value)}><option value="low">Budget-friendly</option><option value="moderate">Moderate</option><option value="flexible">Flexible</option></select></label><label>Food preferences<input value={preferences} onChange={(event) => setPreferences(event.target.value)} /></label><label className="span-two">Allergies (comma-separated)<input value={allergies} onChange={(event) => setAllergies(event.target.value)} placeholder="Leave blank if none" /></label><label>Age (optional)<input type="number" min="1" max="120" value={age} onChange={(event) => setAge(event.target.value)} placeholder="Years" /></label><label>Dietary guidance (optional)<input value={dietaryGuidance} onChange={(event) => setDietaryGuidance(event.target.value)} placeholder="Restrictions or clinician guidance" /></label></div>{diet === "vegetarian" && <label className="checkbox-line"><input type="checkbox" checked={allowEggs} onChange={(event) => setAllowEggs(event.target.checked)} /> I eat eggs</label>}
        <button className="button-primary" disabled={busy || !analysis.reference_range_comparisons.some((row) => row.range_status !== "needs_review")} onClick={getRecommendations}>{busy ? "Preparing guidance…" : "Get food guidance →"}</button>
        {error && <p className="inline-error" role="alert">{error}</p>}
        {recommendations && <div className="guidance-result"><h3>{recommendations.summary || "Food guidance"}</h3>{(recommendations.recommendations || []).map((item, index) => <article key={`${item.finding}-${index}`}><h4>{item.finding} · {item.finding_status?.replaceAll("_", " ")}</h4><p>{item.why_it_matters}</p>{item.foods_to_consider?.length > 0 && <p><strong>Consider:</strong> {item.foods_to_consider.map((food) => food.food).join(", ")}</p>}{item.next_steps && <p>{item.next_steps}</p>}</article>)}{recommendations.safety_notes?.length > 0 && <p className="disclaimer">{recommendations.safety_notes.join(" ")}</p>}<details><summary>Limitations</summary><ul>{(recommendations.limitations || []).map((item) => <li key={item}>{item}</li>)}</ul></details></div>}
      </section><section className="notice"><strong>Not a diagnosis</strong><p>An out-of-range result or statistical outlier is not a diagnosis. Discuss your report with a qualified health professional.</p></section><button className="button-secondary reset-button" onClick={reset}>Start a new report</button>
    </main>}

    <footer className="footer"><button className="brand" onClick={reset}><span className="brand-symbol">N</span><span>nutri<span>fix</span></span></button><p>Clearer information, with honest limits.</p><small>Research prototype · Not for diagnosis or treatment</small></footer>
  </div>;
}

export default App;
