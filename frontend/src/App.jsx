

import { useRef, useState } from "react";

import "./App.css";
import "./FoodRecommendations.css";



const features = [

  {

    number: "01",

    title: "Understand your results",

    description:

      "Explore blood test values with clear explanations and laboratory reference ranges.",

    icon: "⌁",

  },

  {

    number: "02",

    title: "Find what needs attention",

    description:

      "Identify results outside supplied reference ranges and understand what they may mean.",

    icon: "◎",

  },

  {

    number: "03",

    title: "Make informed choices",

    description:

      "Explore evidence-based nutrition guidance for healthier everyday decisions.",

    icon: "✳",

  },

];



function App() {

  const [page, setPage] = useState("home");

  const [menuOpen, setMenuOpen] = useState(false);

  const [file, setFile] = useState(null);

  const [error, setError] = useState("");

  const [loading, setLoading] = useState(false);

  const [result, setResult] = useState(null);
  const [diet, setDiet] = useState("vegetarian");
  const [allowEggs, setAllowEggs] = useState(false);
  const [allergies, setAllergies] = useState("");
  const [foodPreferences, setFoodPreferences] = useState("Indian meals");
  const [goal, setGoal] = useState("general");
  const [budget, setBudget] = useState("moderate");
  const [age, setAge] = useState("");
  const [dietaryGuidance, setDietaryGuidance] = useState("");
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const [recommendationError, setRecommendationError] = useState("");
  const [recommendations, setRecommendations] = useState(null);

  const fileInput = useRef(null);



  function openUpload() {

    setError("");
    setRecommendationError("");

    setPage("upload");

    window.scrollTo({ top: 0, behavior: "smooth" });

  }



  function goHome() {

    setPage("home");

    setError("");

    window.scrollTo({ top: 0, behavior: "smooth" });

  }



  function selectFile(selectedFile) {

    setError("");



    if (!selectedFile) return;



    const allowedTypes = [

      "application/pdf",

      "image/jpeg",

      "image/png",

    ];



    const extensionAllowed = /\.(pdf|jpe?g|png)$/i.test(selectedFile.name);



    if (

      !allowedTypes.includes(selectedFile.type) &&

      !extensionAllowed

    ) {

      setError("Please select a PDF, JPG, or PNG file.");

      return;

    }



    if (selectedFile.size > 15 * 1024 * 1024) {

      setError("The file must be smaller than 15 MB.");

      return;

    }



    setFile(selectedFile);

  }



  function handleFileChange(event) {
    selectFile(event.target.files?.[0]);
    event.target.value = "";
  }

  async function handleAnalyze() {
    if (!file || loading) return;
    setLoading(true);
    setError("");
    setRecommendationError("");
    setResult(null);
    setRecommendations(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch("http://127.0.0.1:8000/api/reports/extract", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to process this report.");
      setResult(data);
    } catch (err) {
      setError(err.message || "Could not connect to NutriFix. Check that the backend is running.");
    } finally {
      setLoading(false);
    }
  }

  async function handleGetRecommendations() {
    if (!result?.biomarkers?.length || recommendationLoading) return;
    setRecommendationLoading(true);
    setRecommendationError("");
    setRecommendations(null);
    try {
      const response = await fetch("http://127.0.0.1:8000/api/reports/recommendations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          diet,
          allow_eggs: diet === "vegetarian" ? allowEggs : false,
          allergies: allergies.split(",").map((item) => item.trim()).filter(Boolean),
          food_preferences: foodPreferences.split(",").map((item) => item.trim()).filter(Boolean),
          goal,
          budget,
          age: age.trim() ? Number(age) : null,
          dietary_guidance: dietaryGuidance.trim(),
          biomarkers: result.biomarkers,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not generate food recommendations.");
      setRecommendations(data.data);
    } catch (err) {
      setRecommendationError(err.message || "Could not connect to the AI recommendation service.");
    } finally {
      setRecommendationLoading(false);
    }
  }

  return (

    <div className="site-shell">

      <header className="navbar">

        <button className="brand brand-button" onClick={goHome}>

          <span className="brand-mark">N</span>

          <span>nutri<span className="brand-accent">fix</span></span>

        </button>



        <button

          className="menu-toggle"

          onClick={() => setMenuOpen(!menuOpen)}

          aria-label="Toggle navigation"

        >

          {menuOpen ? "✕" : "☰"}

        </button>



        <nav className={menuOpen ? "nav-links open" : "nav-links"}>

          <button onClick={() => { goHome(); setMenuOpen(false); }}>

            Home

          </button>

          <a href={page === "home" ? "#how-it-works" : "#upload"} onClick={() => setMenuOpen(false)}>

            How it works

          </a>

          <a href={page === "home" ? "#features" : "#upload"} onClick={() => setMenuOpen(false)}>

            Features

          </a>

          <button className="nav-cta" onClick={() => { openUpload(); setMenuOpen(false); }}>

            Get started <span>↗</span>

          </button>

        </nav>

      </header>



      {page === "home" ? (

        <main id="home">

          <section className="hero">

            <div className="hero-copy">

              <div className="eyebrow">

                <span className="status-dot" />

                YOUR HEALTH, MADE CLEARER

              </div>



              <h1>

                Your blood report.

                <br />

                <span>Finally, understood.</span>

              </h1>



              <p className="hero-description">

                Make sense of your blood test results with clear explanations

                and practical nutrition insights. Less confusion, more informed

                decisions about your health.

              </p>



              <div className="hero-actions">

                <button className="button-primary" onClick={openUpload}>

                  Analyze my report <span>→</span>

                </button>

                <a className="button-secondary" href="#features">

                  Discover features

                </a>

              </div>



              <div className="trust-note">

                <span className="trust-icon">✓</span>

                Built for clarity, designed around your health

              </div>

            </div>



            <div className="hero-visual">

              <div className="decor decor-one" />

              <div className="decor decor-two" />



              <div className="report-card">

                <div className="report-top">

                  <div>

                    <div className="report-label">NUTRIFIX INSIGHTS</div>

                    <h3>Blood report overview</h3>

                  </div>

                  <div className="report-icon">✳</div>

                </div>



                <div className="report-divider" />



                {[

                  ["Hb", "Hemoglobin", "Complete blood count", "g/dL"],

                  ["W", "White blood cells", "Immune system marker", "×10⁹/L"],

                  ["Fe", "Iron markers", "When included in your report", "Units vary"],

                ].map((item) => (

                  <div className="metric-row" key={item[1]}>

                    <div className="metric-icon mint">{item[0]}</div>

                    <div className="metric-info">

                      <strong>{item[1]}</strong>

                      <span>{item[2]}</span>

                    </div>

                    <div className="metric-value">

                      <strong>—</strong>

                      <span>{item[3]}</span>

                    </div>

                  </div>

                ))}



                <div className="report-footer">

                  <span className="footer-dot" />

                  Sample interface · No patient data

                </div>

              </div>



              <div className="floating-note">

                <span className="floating-check">✓</span>

                <div>

                  <strong>Clarity over complexity</strong>

                  <span>Health information, made approachable</span>

                </div>

              </div>



              <div className="hero-orbit">+</div>

            </div>

          </section>



          <section className="stats-strip">

            <div><strong>Understand</strong><span>Results in plain language</span></div>

            <div><strong>Explore</strong><span>Biomarkers and ranges</span></div>

            <div><strong>Discover</strong><span>Practical nutrition insights</span></div>

          </section>



          <section className="features-section" id="features">

            <div className="section-heading">

              <span className="section-kicker">WHAT NUTRIFIX DOES</span>

              <h2>From numbers to <span>understanding.</span></h2>

              <p>

                Blood reports can feel overwhelming. NutriFix aims to make the

                information easier to explore and understand.

              </p>

            </div>



            <div className="feature-grid">

              {features.map((feature) => (

                <article className="feature-card" key={feature.number}>

                  <div className="feature-card-top">

                    <span className="feature-number">{feature.number}</span>

                    <span className="feature-icon">{feature.icon}</span>

                  </div>

                  <h3>{feature.title}</h3>

                  <p>{feature.description}</p>

                  <button className="text-link" onClick={openUpload}>

                    Try NutriFix <span>↗</span>

                  </button>

                </article>

              ))}

            </div>

          </section>



          <section className="how-section" id="how-it-works">

            <div className="how-copy">

              <span className="section-kicker">A SIMPLER EXPERIENCE</span>

              <h2>Health information should feel <span>human.</span></h2>

              <p>

                Our goal is to connect laboratory results with understandable

                information, while keeping the limits of automated analysis clear.

              </p>

              <button className="button-primary" onClick={openUpload}>

                Start with a report <span>→</span>

              </button>

            </div>



            <div className="steps-list">

              {[

                ["01", "Bring your results", "Start with a blood report containing your laboratory measurements."],

                ["02", "Explore the details", "Review extracted values, units and the reference ranges supplied by the lab."],

                ["03", "Learn what comes next", "Explore relevant information and questions to discuss with a qualified professional."],

              ].map((step) => (

                <div className="step" key={step[0]}>

                  <span>{step[0]}</span>

                  <div><h3>{step[1]}</h3><p>{step[2]}</p></div>

                </div>

              ))}

            </div>

          </section>



          <section className="closing-cta" id="about">

            <div className="closing-symbol">✳</div>

            <h2>More understanding.<br />Better-informed choices.</h2>

            <p>

              NutriFix is being developed to make blood report information

              clearer and more accessible.

            </p>

            <button className="button-light" onClick={openUpload}>

              Analyze my report →

            </button>

          </section>

        </main>

      ) : (

        <main className="upload-page" id="upload">

          <button className="back-link" onClick={goHome}>← Back to home</button>



          <div className="upload-heading">

            <span className="section-kicker">YOUR HEALTH, MADE CLEARER</span>

            <h1>Let's understand <span>your report.</span></h1>

            <p>

              Upload a blood report to get started. You can review the selected

              file before analysis.

            </p>

          </div>



          <section className="upload-card">

            {!file ? (

              <button

                type="button"

                className="dropzone"

                onClick={() => fileInput.current?.click()}

                onDragOver={(event) => event.preventDefault()}

                onDrop={(event) => {

                  event.preventDefault();

                  selectFile(event.dataTransfer.files?.[0]);

                }}

              >

                <span className="upload-icon">↑</span>

                <strong>Drop your blood report here</strong>

                <span className="dropzone-subtitle">or click to browse your files</span>

                <span className="file-formats">PDF, JPG or PNG · Maximum 15 MB</span>

              </button>

            ) : (

              <div className="selected-file">

                <span className="selected-file-icon">

                  {file.type === "application/pdf" ? "PDF" : "IMG"}

                </span>

                <div className="selected-file-info">

                  <strong>{file.name}</strong>

                  <span>{(file.size / (1024 * 1024)).toFixed(2)} MB · Selected</span>

                </div>

                <button

                  className="remove-file"

                  onClick={() => {

                    setFile(null);

                    setError("");

                  }}

                >

                  Remove

                </button>

              </div>

            )}



            <input

              ref={fileInput}

              type="file"

              accept=".pdf,.jpg,.jpeg,.png,application/pdf,image/jpeg,image/png"

              hidden

              onChange={handleFileChange}

            />



            {error && <p className="upload-error" role="alert">{error}</p>}



            <div className="privacy-note">

              <span>♢</span>

              <p>

                <strong>Your health data matters.</strong>

                This is an early prototype. The selected file stays in this

                browser until you connect an analysis service.

              </p>

            </div>



            <button
              type="button"
              className="button-primary upload-continue"
              disabled={!file || loading}
              onClick={handleAnalyze}
            >
              {loading ? "Analyzing report..." : "Continue to analysis"}
              {!loading && <span> →</span>}
            </button>

          </section>

          {result && (
            <section className="upload-card results-card" aria-live="polite">
              <div className="upload-heading">
                <span className="section-kicker">REPORT ANALYZED</span>
                <h2>Your personalized guidance starts here</h2>
                <p>{result.filename}</p>
              </div>

              {result.status === "needs_ocr" ? (
                <>
                  <p>
                    This report appears to need text recognition (OCR), which is
                    not available in this version. Try a PDF with selectable text
                    or a clear supported report format.
                  </p>
                  <button
                    type="button"
                    className="button-primary"
                    onClick={() => { setResult(null); setFile(null); setError(""); }}
                  >
                    Choose another report
                  </button>
                </>
              ) : result.biomarkers?.length > 0 ? (
                <>
                  <p className="report-ready-message">
                    Your report has been processed. Complete your dietary profile
                    below to receive one combined food-guidance plan. NutriFix uses
                    the extracted results in the background; technical biomarker
                    details are not shown in this view.
                  </p>

                  <section className="food-recommendation-form">
                    <div className="section-heading">
                      <span className="section-kicker">YOUR DIETARY PROFILE</span>
                      <h2>Personalize <span>your food guidance</span></h2>
                      <p>Tell us what foods suit your routine so recommendations can consider your report and your needs together.</p>
                    </div>

                    <label className="profile-label" htmlFor="diet-preference">Dietary preference</label>
                    <select id="diet-preference" value={diet} onChange={(event) => setDiet(event.target.value)} className="profile-select">
                      <option value="vegan">Vegan</option>
                      <option value="vegetarian">Vegetarian</option>
                      <option value="non_vegetarian">Non-vegetarian</option>
                    </select>

                    {diet === "vegetarian" && (
                      <label className="profile-checkbox">
                        <input type="checkbox" checked={allowEggs} onChange={(event) => setAllowEggs(event.target.checked)} />
                        I eat eggs
                      </label>
                    )}

                    <label className="profile-label" htmlFor="allergies">Food allergies or intolerances (comma-separated)</label>
                    <input id="allergies" className="profile-input" value={allergies} onChange={(event) => setAllergies(event.target.value)} placeholder="e.g. peanuts, milk; leave blank if none" />

                    <label className="profile-label" htmlFor="food-preferences">Food preferences (comma-separated)</label>
                    <input id="food-preferences" className="profile-input" value={foodPreferences} onChange={(event) => setFoodPreferences(event.target.value)} placeholder="e.g. Indian meals, high-protein foods" />

                    <label className="profile-label" htmlFor="nutrition-goal">Main nutrition goal</label>
                    <select id="nutrition-goal" className="profile-select" value={goal} onChange={(event) => setGoal(event.target.value)}>
                      <option value="general">Improve overall nutrition</option>
                      <option value="weight_management">Weight management</option>
                      <option value="muscle_gain">Muscle gain and adequate protein</option>
                      <option value="focus_on_report">Focus on blood-report findings</option>
                    </select>

                    <label className="profile-label" htmlFor="food-budget">Food budget</label>
                    <select id="food-budget" className="profile-select" value={budget} onChange={(event) => setBudget(event.target.value)}>
                      <option value="low">Budget-friendly</option>
                      <option value="moderate">Moderate</option>
                      <option value="flexible">Flexible</option>
                    </select>

                    <label className="profile-label" htmlFor="profile-age">Age (optional)</label>
                    <input id="profile-age" className="profile-input" type="number" min="1" max="120" value={age} onChange={(event) => setAge(event.target.value)} placeholder="Age in years" />

                    <label className="profile-label" htmlFor="dietary-guidance">Dietary restrictions or clinician guidance (optional)</label>
                    <textarea id="dietary-guidance" className="profile-input" rows="3" value={dietaryGuidance} onChange={(event) => setDietaryGuidance(event.target.value)} placeholder="Only include guidance you want NutriFix to consider" />

                    {recommendationError && <p className="upload-error" role="alert">{recommendationError}</p>}
                    <button type="button" className="button-primary" onClick={handleGetRecommendations} disabled={recommendationLoading || !result.biomarkers?.length}>
                      {recommendationLoading ? "Preparing your food guidance..." : "Get personalized food guidance"}
                      {!recommendationLoading && <span> →</span>}
                    </button>
                  </section>

                  {recommendationLoading && (
                    <p role="status">Combining your report with your dietary profile. This can take a little while.</p>
                  )}

                  {recommendations && (() => {
                    const findings = Array.isArray(recommendations.recommendations) ? recommendations.recommendations : [];
                    const foodMap = new Map();
                    const limits = new Map();
                    findings.forEach((finding) => {
                      (finding.foods_to_consider || []).forEach((food) => {
                        const name = String(food.food || "").trim();
                        if (!name) return;
                        const key = name.toLocaleLowerCase();
                        if (!foodMap.has(key)) {
                          foodMap.set(key, { food: name, reasons: [], mealIdeas: [], findings: [] });
                        }
                        const entry = foodMap.get(key);
                        if (food.reason && !entry.reasons.includes(food.reason)) entry.reasons.push(food.reason);
                        if (food.meal_idea && !entry.mealIdeas.includes(food.meal_idea)) entry.mealIdeas.push(food.meal_idea);
                        if (finding.finding && !entry.findings.includes(finding.finding)) entry.findings.push(finding.finding);
                      });
                      (finding.foods_to_limit || []).forEach((item) => {
                        const text = typeof item === "string" ? item : item?.food || item?.name || "";
                        if (text.trim()) limits.set(text.trim().toLocaleLowerCase(), text.trim());
                      });
                    });

                    return (
                      <section className="food-results" aria-live="polite">
                        <span className="section-kicker">YOUR FOOD GUIDANCE</span>
                        <h2>Personalized food <span>guidance</span></h2>
                        {recommendations.summary && <p className="guidance-summary">{recommendations.summary}</p>}

                        {foodMap.size > 0 ? (
                          <>
                            <h3>Foods to consider</h3>
                            <div className="recommended-food-list">
                              {[...foodMap.values()].map((food) => (
                                <article className="recommended-food" key={food.food.toLocaleLowerCase()}>
                                  <h4>{food.food}</h4>
                                  {food.reasons.map((reason, index) => <p key={index}>{reason}</p>)}
                                  {food.mealIdeas.length > 0 && <p><strong>Meal ideas:</strong> {food.mealIdeas.join("; ")}</p>}
                                  {food.findings.length > 0 && <p className="evidence-note"><strong>Relevant report context:</strong> {food.findings.join(", ")}</p>}
                                </article>
                              ))}
                            </div>
                          </>
                        ) : (
                          <p>No suitable food suggestions were returned. Please try again or consult a qualified dietitian.</p>
                        )}

                        {findings.some((item) => item.why_it_matters || item.next_steps) && (
                          <div className="recommendation-notes">
                            <h3>Why this guidance fits your report</h3>
                            {findings.filter((item) => item.why_it_matters || item.next_steps).map((item, index) => (
                              <p key={`${item.finding || "finding"}-${index}`}>
                                {item.why_it_matters && <><strong>{item.finding || "Report context"}: </strong>{item.why_it_matters}</>}
                                {item.next_steps && <> {item.next_steps}</>}
                              </p>
                            ))}
                          </div>
                        )}

                        {limits.size > 0 && (
                          <div className="recommendation-notes">
                            <h3>Foods to discuss or moderate</h3>
                            <ul>{[...limits.values()].map((food) => <li key={food}>{food}</li>)}</ul>
                          </div>
                        )}

                        {recommendations.safety_notes?.length > 0 && (
                          <div className="recommendation-notes">
                            <h3>Safety notes</h3>
                            <ul>{recommendations.safety_notes.map((note, index) => <li key={index}>{note}</li>)}</ul>
                          </div>
                        )}
                        {recommendations.limitations?.length > 0 && (
                          <details className="recommendation-notes">
                            <summary>Limitations and evidence</summary>
                            <ul>{recommendations.limitations.map((note, index) => <li key={index}>{note}</li>)}</ul>
                          </details>
                        )}
                      </section>
                    );
                  })()}
                </>
              ) : (
                <p>
                  We could not recognize supported biomarkers in this report, so
                  NutriFix cannot safely personalize food guidance from it yet.
                  Try a report with selectable text and laboratory reference ranges.
                </p>
              )}

              <p className="upload-disclaimer">
                NutriFix provides educational information, not a diagnosis or
                treatment. An AI-generated food plan cannot replace advice from
                a qualified healthcare professional.
              </p>

              <button
                type="button"
                className="button-primary"
                onClick={() => {
                  setResult(null);
                  setRecommendations(null);
                  setRecommendationError("");
                  setError("");
                  setFile(null);
                }}
              >
                Upload another report
              </button>
            </section>
          )}

          <div className="upload-disclaimer">
            NutriFix is not a substitute for professional medical advice,

            diagnosis, or treatment.

          </div>

        </main>

      )}



      <footer className="footer">

        <button className="brand brand-button footer-brand" onClick={goHome}>

          <span className="brand-mark">N</span>

          <span>nutri<span className="brand-accent">fix</span></span>

        </button>

        <p>Making health information easier to understand.</p>

        <span className="footer-disclaimer">

          A developing project, not a substitute for professional medical advice.

        </span>

        <span className="copyright">© {new Date().getFullYear()} NutriFix</span>

      </footer>

    </div>

  );

}



export default App;
