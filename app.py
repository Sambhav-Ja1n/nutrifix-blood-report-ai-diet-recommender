
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

PREDICTIONS_FILE = ROOT / "outputs/metrics/isolation_forest_baseline_predictions.csv"
REPORT_FILE = ROOT / "outputs/metrics/participant_anomaly_report.csv"
EXPLANATIONS_FILE = ROOT / "outputs/metrics/anomaly_explanations.csv"

st.set_page_config(
    page_title="NutriFix | Blood Report Explorer",
    page_icon="🩸",
    layout="wide",
)

st.title("NutriFix")
st.caption("Blood report exploration and statistical anomaly screening")

st.warning(
    "Research prototype: anomaly candidates are statistical outliers, "
    "not diagnoses. Do not use this dashboard for medical decisions."
)


@st.cache_data
def load_data():
    predictions = pd.read_csv(PREDICTIONS_FILE)
    report = pd.read_csv(REPORT_FILE, low_memory=False)
    explanations = pd.read_csv(EXPLANATIONS_FILE)

    for frame in (predictions, report, explanations):
        if "SEQN" in frame.columns:
            frame["SEQN"] = pd.to_numeric(
                frame["SEQN"], errors="coerce"
            ).astype("Int64")

    return predictions, report, explanations


try:
    predictions, report, explanations = load_data()
except FileNotFoundError as exc:
    st.error(f"Required output file not found: {exc.filename}")
    st.info("Run the NutriFix pipeline and generate its reports first.")
    st.stop()

total_participants = len(predictions)
candidate_mask = predictions["is_anomaly_candidate"].eq(True)
candidate_count = int(candidate_mask.sum())
scored_count = int(predictions["scoring_status"].eq("scored").sum())

col1, col2, col3 = st.columns(3)
col1.metric("Participants", f"{total_participants:,}")
col2.metric("Scored participants", f"{scored_count:,}")
col3.metric("Anomaly candidates", f"{candidate_count:,}")

st.divider()

st.subheader("Explore participants")

show_candidates_only = st.checkbox(
    "Show anomaly candidates only",
    value=True,
)

search_id = st.text_input(
    "Search participant ID (SEQN)",
    placeholder="Enter a participant ID, e.g. 96216",
)

filtered = report.copy()

if show_candidates_only:
    filtered = filtered[filtered["is_anomaly_candidate"].eq(True)]

if search_id.strip():
    if not search_id.strip().isdigit():
        st.error("Enter a numeric participant ID.")
        st.stop()

    filtered = filtered[filtered["SEQN"] == int(search_id.strip())]

display_columns = [
    column for column in [
        "SEQN",
        "scoring_status",
        "is_anomaly_candidate",
        "anomaly_score",
        "decision_score",
        "top_biomarker",
        "top_observed_value",
        "top_cohort_z_score",
    ]
    if column in filtered.columns
]

st.caption(f"Showing {len(filtered):,} participant records")
st.dataframe(
    filtered[display_columns],
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.subheader("Participant details")

if not search_id.strip():
    st.info("Enter a participant ID above to view their original CBC measurements.")
else:
    participant_id = int(search_id.strip())
    participant_rows = report[report["SEQN"] == participant_id]

    if participant_rows.empty:
        st.warning("Participant ID not found in the report.")
    else:
        participant = participant_rows.iloc[0]
        is_candidate = bool(participant["is_anomaly_candidate"])

        if participant["scoring_status"] != "scored":
            st.warning(
                "This participant was not scored because biomarker data "
                "were insufficient."
            )
        elif is_candidate:
            st.warning(
                "Statistical anomaly candidate — this is not a diagnosis."
            )
        else:
            st.success("Not flagged as an anomaly candidate by this model.")

        a, b = st.columns(2)
        a.metric("Anomaly score", f"{participant['anomaly_score']:.4f}"
                 if pd.notna(participant["anomaly_score"]) else "N/A")
        b.metric("Decision score", f"{participant['decision_score']:.4f}"
                 if pd.notna(participant["decision_score"]) else "N/A")

        # Display the participant's original laboratory measurements.
        cbc_columns = [
            "LBXWBCSI", "LBXLYPCT", "LBXMOPCT", "LBXNEPCT",
            "LBXEOPCT", "LBXBAPCT", "LBXRBCSI", "LBXHGB",
            "LBXHCT", "LBXMCVSI", "LBXMCHSI", "LBXMC",
            "LBXRDW", "LBXPLTSI", "LBXMPSI",
        ]
        available_cbc = [
            column for column in cbc_columns
            if column in report.columns
        ]

        st.markdown("**Original CBC measurements**")
        measurements = pd.DataFrame({
            "Biomarker": available_cbc,
            "Observed value": [
                participant[column] for column in available_cbc
            ],
        })
        st.dataframe(measurements, use_container_width=True, hide_index=True)

        st.markdown("**Cohort-relative feature comparisons**")
        participant_explanations = explanations[
            explanations["SEQN"] == participant_id
        ].copy()

        if participant_explanations.empty:
            st.info("No observed biomarker comparisons are available.")
        else:
            columns = [
                column for column in [
                    "biomarker_feature",
                    "observed_value",
                    "cohort_mean",
                    "cohort_relative_z_score",
                    "absolute_z_score",
                ]
                if column in participant_explanations.columns
            ]
            participant_explanations = participant_explanations.sort_values(
                "absolute_z_score", ascending=False
            )
            st.dataframe(
                participant_explanations[columns],
                use_container_width=True,
                hide_index=True,
            )

st.divider()
st.caption(
    "Cohort-relative comparisons describe statistical unusualness. "
    "They are not clinical reference ranges or explanations of causation."
)
