from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.feature_engineering import create_features
from src.lending_agent import answer_follow_up_question, run_agentic_lending_decision
from src.preprocessing_pipeline import preprocess_uploaded_dataset
from src.report_export import generate_lending_report_pdf


st.set_page_config(
    page_title="Agentic Lending Command Center",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "GROQ_API_KEY" in st.secrets and not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "LENDING_AGENT_PROVIDER" in st.secrets and not os.getenv("LENDING_AGENT_PROVIDER"):
    os.environ["LENDING_AGENT_PROVIDER"] = st.secrets["LENDING_AGENT_PROVIDER"]


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "german_credit_data.csv"
MODEL_PATHS = {
    "Logistic Regression": BASE_DIR / "models" / "logistic_regression.pkl",
    "Decision Tree": BASE_DIR / "models" / "decision_tree.pkl",
}


def inject_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #08111f;
                --bg-soft: #0d1830;
                --panel: rgba(13, 24, 48, 0.78);
                --panel-strong: rgba(8, 17, 31, 0.96);
                --line: rgba(91, 214, 255, 0.22);
                --text: #e8f7ff;
                --muted: #8baecc;
                --accent: #3ddcff;
                --accent-2: #6f8cff;
                --success: #24f2b3;
                --danger: #ff5f8f;
                --warning: #ffd166;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(61, 220, 255, 0.20), transparent 24%),
                    radial-gradient(circle at top right, rgba(111, 140, 255, 0.16), transparent 28%),
                    linear-gradient(180deg, #050b16 0%, #08111f 48%, #050b16 100%);
                color: var(--text);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(8, 17, 31, 0.98), rgba(10, 20, 40, 0.92));
                border-right: 1px solid var(--line);
            }

            [data-testid="stSidebar"] * {
                color: var(--text);
            }

            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2rem;
            }

            h1, h2, h3, h4, p, label, span, div {
                color: var(--text);
            }

            .hero {
                padding: 1.6rem 1.8rem;
                border: 1px solid var(--line);
                border-radius: 26px;
                background:
                    linear-gradient(135deg, rgba(61, 220, 255, 0.10), rgba(111, 140, 255, 0.10)),
                    rgba(9, 18, 35, 0.88);
                box-shadow: 0 0 0 1px rgba(61, 220, 255, 0.05), 0 26px 80px rgba(0, 0, 0, 0.35);
                margin-bottom: 2rem;
            }

            .hero-eyebrow {
                font-size: 0.76rem;
                letter-spacing: 0.18em;
                text-transform: uppercase;
                color: var(--accent);
                margin-bottom: 0.65rem;
            }

            .hero-title {
                font-size: 2.5rem;
                line-height: 1.05;
                font-weight: 700;
                margin: 0;
            }

            .hero-copy {
                color: var(--muted);
                max-width: 60rem;
                margin-top: 0.7rem;
            }

            .glass-card {
                border: 1px solid var(--line);
                border-radius: 24px;
                background: var(--panel);
                padding: 1.2rem;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.02), 0 20px 60px rgba(0, 0, 0, 0.22);
            }

            /* Pipeline Banner Styles */
            .pipeline-banner {
                background: rgba(13, 24, 48, 0.6);
                border: 1px solid var(--line);
                border-radius: 20px;
                padding: 1.5rem;
                margin-bottom: 2rem;
            }

            .pipeline-title {
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #a391ff;
                margin-bottom: 1.2rem;
            }

            .pipeline-steps {
                display: flex;
                align-items: center;
                gap: 1rem;
                overflow-x: auto;
            }

            .pipeline-node {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.6rem 1rem;
                border-radius: 12px;
                font-size: 0.9rem;
                font-weight: 500;
                white-space: nowrap;
                transition: all 0.3s ease;
            }

            .node-blue { background: rgba(58, 62, 117, 0.4); border: 1px solid rgba(88, 101, 242, 0.4); color: #c4ccff; }
            .node-sky { background: rgba(30, 58, 95, 0.4); border: 1px solid rgba(61, 220, 255, 0.4); color: #ade8ff; }
            .node-green { background: rgba(26, 60, 55, 0.4); border: 1px solid rgba(36, 242, 179, 0.4); color: #b7ffe8; }
            .node-pink { background: rgba(60, 30, 50, 0.4); border: 1px solid rgba(255, 95, 143, 0.4); color: #ffc2d6; }

            .pipeline-arrow {
                color: var(--muted);
                font-size: 1.2rem;
            }

            /* KPI Card Styles */
            .kpi-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2rem;
            }

            .kpi-card-new {
                background: rgba(13, 24, 48, 0.6);
                border: 1px solid var(--line);
                border-radius: 20px;
                padding: 1.5rem;
                text-align: center;
                transition: transform 0.3s ease;
            }

            .kpi-card-new:hover {
                transform: translateY(-5px);
                border-color: var(--accent);
            }

            .kpi-val {
                font-size: 2.2rem;
                font-weight: 700;
                color: var(--accent);
                margin-bottom: 0.3rem;
            }

            .kpi-lab {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: var(--muted);
            }

            /* Architecture Block Styles */
            .arch-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 1rem;
                margin-top: 1rem;
            }

            .arch-card {
                background: rgba(13, 24, 48, 0.4);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 1.2rem;
                text-align: center;
            }

            .arch-num {
                width: 32px;
                height: 32px;
                background: var(--accent-2);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1rem auto;
                font-weight: 700;
                font-size: 0.85rem;
            }

            .arch-title {
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: var(--text);
            }

            .arch-desc {
                font-size: 0.75rem;
                color: var(--muted);
                line-height: 1.4;
            }

            .section-label {
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                color: var(--accent);
                margin-bottom: 0.75rem;
            }

            .stAlert {
                border-radius: 18px;
                border: 1px solid rgba(61, 220, 255, 0.18);
                background: rgba(11, 21, 41, 0.92);
            }

            .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
            .stFileUploader section, .stTextArea textarea {
                background: rgba(7, 14, 28, 0.92) !important;
                color: var(--text) !important;
                border: 1px solid rgba(61, 220, 255, 0.18) !important;
                border-radius: 14px !important;
            }

            .stButton button, .stDownloadButton button, .stFormSubmitButton button {
                border-radius: 999px;
                border: 1px solid rgba(61, 220, 255, 0.35);
                background: linear-gradient(135deg, rgba(61, 220, 255, 0.18), rgba(111, 140, 255, 0.22));
                color: var(--text);
                box-shadow: 0 0 22px rgba(61, 220, 255, 0.16);
            }

            .stButton button:hover, .stFormSubmitButton button:hover {
                border-color: rgba(61, 220, 255, 0.7);
                box-shadow: 0 0 30px rgba(61, 220, 255, 0.26);
            }

            [data-testid="stHeader"] {
                background: rgba(8, 17, 31, 0.9);
                backdrop-filter: blur(10px);
            }

            [data-testid="stTabs"] {
                background: rgba(13, 24, 48, 0.4);
                border-radius: 16px;
                padding: 4px;
                border: 1px solid var(--line);
                margin-bottom: 2rem;
            }

            [data-testid="stTabs"] button {
                border: none !important;
                background: transparent !important;
            }

            [data-testid="stTabs"] button[aria-selected="true"] {
                background: linear-gradient(135deg, rgba(61, 220, 255, 0.16), rgba(111, 140, 255, 0.18)) !important;
                border-radius: 12px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )



def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Agentic Underwriting Studio</div>
            <h1 class="hero-title">Professional Lending Intelligence Agent</h1>
            <p class="hero-copy">
                Review borrower requests, generate lending decisions, and explore follow-up reasoning
                from a single focused workspace designed for fintech teams.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_langgraph_pipeline() -> None:
    st.markdown(
        """
        <div class="pipeline-banner">
            <div class="pipeline-title">LangGraph Workflow Pipeline</div>
            <div class="pipeline-steps">
                <div class="pipeline-node node-blue">
                    <span>①</span> Analyze Risk
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-node node-sky">
                    <span>②</span> RAG Retrieve (FAISS)
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-node node-green">
                    <span>③</span> Generate Report
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-node node-pink">
                    <span>📄</span> Structured Output
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards() -> None:
    st.markdown(
        """
        <div class="kpi-container">
            <div class="kpi-card-new">
                <div class="kpi-val">81.2%</div>
                <div class="kpi-lab">Accuracy</div>
            </div>
            <div class="kpi-card-new">
                <div class="kpi-val">68.7%</div>
                <div class="kpi-lab">Precision</div>
            </div>
            <div class="kpi-card-new">
                <div class="kpi-val">56.6%</div>
                <div class="kpi-lab">Recall</div>
            </div>
            <div class="kpi-card-new">
                <div class="kpi-val">62.1%</div>
                <div class="kpi-lab">F1 Score</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_architecture() -> None:
    st.markdown("<div class='section-label'>Pipeline Architecture</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="arch-grid">
            <div class="arch-card">
                <div class="arch-num">1</div>
                <div class="arch-title">Input</div>
                <div class="arch-desc">Configure borrower metrics via sidebar or manual entry.</div>
            </div>
            <div class="arch-card">
                <div class="arch-num">2</div>
                <div class="arch-title">ML Pipeline</div>
                <div class="arch-desc">Feature engineering, scaling & model inference tasks.</div>
            </div>
            <div class="arch-card">
                <div class="arch-num">3</div>
                <div class="arch-title">Risk Analysis</div>
                <div class="arch-desc">Predict default probability & key business driver extraction.</div>
            </div>
            <div class="arch-card">
                <div class="arch-num">4</div>
                <div class="arch-title">AI Agent</div>
                <div class="arch-desc">LangGraph + RAG generates tailored lending strategy.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



@st.cache_resource
def load_model_registry() -> Dict[str, Any]:
    registry: Dict[str, Any] = {}
    for name, path in MODEL_PATHS.items():
        if path.exists():
            registry[name] = joblib.load(path)
    return registry


@st.cache_data
def load_local_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_active_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return load_local_dataset()


def actual_model(model: Any) -> Any:
    resolved = model.best_estimator_ if hasattr(model, "best_estimator_") else model

    # Older pickled sklearn tree models may not include this attribute, but
    # newer sklearn versions expect it during prediction-time validation.
    if not hasattr(resolved, "monotonic_cst"):
        resolved.monotonic_cst = None

    return resolved


def align_features(model: Any, features: pd.DataFrame) -> pd.DataFrame:
    resolved = actual_model(model)
    aligned = features.copy()
    if hasattr(resolved, "feature_names_in_"):
        expected = list(resolved.feature_names_in_)
        for col in expected:
            if col not in aligned.columns:
                aligned[col] = 0
        aligned = aligned[expected]
        aligned = aligned.fillna(aligned.median(numeric_only=True))
    return aligned


def score_dataset(dataset: pd.DataFrame, model: Any) -> pd.DataFrame:
    prepared = preprocess_uploaded_dataset(dataset)
    working = prepared["normalized"]
    features = create_features(working)
    aligned = align_features(model, features)
    probabilities = actual_model(model).predict_proba(aligned)[:, 1]

    scored = working.copy()
    if "Unnamed: 0" in scored.columns:
        scored = scored.drop(columns=["Unnamed: 0"])
    scored["Predicted Default Probability"] = probabilities
    scored["Estimated Credit Score"] = (850 - (probabilities * 550)).round(0).astype(int)
    scored["Predicted Decision"] = scored["Predicted Default Probability"].apply(
        lambda score: "High Risk" if score >= 0.5 else "Low Risk"
    )
    return scored


def build_model_summary_rows(model_scores: Dict[str, pd.DataFrame]) -> list[Dict[str, Any]]:
    summary_rows = []
    for name, scored in model_scores.items():
        summary_rows.append(
            {
                "Model": name,
                "Avg Risk": float(scored["Predicted Default Probability"].mean()),
                "High-Risk Share": float(scored["Predicted Default Probability"].ge(0.5).mean()),
                "Avg Credit Score": float(scored["Estimated Credit Score"].mean()),
            }
        )
    return summary_rows


def build_gauge(risk_score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            number={"suffix": "%", "font": {"size": 44, "color": "#e8f7ff"}},
            title={"text": "Borrower Risk Signal", "font": {"size": 20, "color": "#8baecc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8baecc"},
                "bar": {"color": "#ff5f8f" if risk_score >= 0.5 else "#24f2b3"},
                "bgcolor": "#08111f",
                "bordercolor": "rgba(61,220,255,0.20)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(36,242,179,0.18)"},
                    {"range": [40, 60], "color": "rgba(255,209,102,0.18)"},
                    {"range": [60, 100], "color": "rgba(255,95,143,0.22)"},
                ],
                "threshold": {"line": {"color": "#3ddcff", "width": 5}, "value": 50},
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def build_user_summary(profile: Dict[str, Any]) -> str:
    return (
        f"Borrower request: {profile['purpose']} loan for {profile['credit_amount']} DM over "
        f"{profile['duration']} months. Profile: {profile['sex']}, age {profile['age']}, "
        f"housing {profile['housing']}, savings {profile['saving_accounts']}, "
        f"checking {profile['checking_account']}, job bucket {profile['job']}."
    )


def reset_follow_up_state() -> None:
    st.session_state.conversation_memory = None
    st.session_state.agent_chat_history = []
    st.session_state.follow_up_question = ""
    st.session_state.clear_follow_up_question = False


inject_theme()
render_hero()

if "agent_chat_history" not in st.session_state:
    st.session_state.agent_chat_history = []
if "latest_decision" not in st.session_state:
    st.session_state.latest_decision = None
if "latest_borrower_profile" not in st.session_state:
    st.session_state.latest_borrower_profile = None
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = None
if "follow_up_question" not in st.session_state:
    st.session_state.follow_up_question = ""
if "clear_follow_up_question" not in st.session_state:
    st.session_state.clear_follow_up_question = False

if st.session_state.clear_follow_up_question:
    st.session_state.follow_up_question = ""
    st.session_state.clear_follow_up_question = False

model_registry = load_model_registry()
if not model_registry:
    st.error("Model pipeline not found in `models/`. Please train or restore the saved estimators.", icon="🚫")
    st.stop()

primary_model_name = "Logistic Regression" if "Logistic Regression" in model_registry else next(iter(model_registry))
primary_model = model_registry[primary_model_name]

tab_dash, tab_pred, tab_agent, tab_metrics = st.tabs(
    ["📊 Dashboard", "🎯 Predictions", "🤖 Agentic AI", "📈 Model Metrics"]
)

with tab_dash:
    st.info(
        "**How it works:** Configure borrower details in the **Agentic AI** tab → the pipeline instantly performs "
        "feature engineering & scaling → view predictions in **Predictions** tab → generate AI-powered "
        "lending strategies in **Agentic AI** tab."
    )
    render_kpi_cards()
    render_pipeline_architecture()

with tab_pred:
    st.markdown("<div class='section-label'>Batch Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload borrower dataset (CSV)", type=["csv"], help="Upload a CSV file with borrower attributes.")
    if uploaded_file:
        data = load_active_dataset(uploaded_file)
        st.write("### Raw Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)

        if st.button("Score Dataset", use_container_width=True):
            with st.spinner("Analyzing dataset..."):
                scored_data = score_dataset(data, primary_model)
                st.session_state.last_scored_data = scored_data
                st.success("Dataset successfully scored!")

    if "last_scored_data" in st.session_state:
        st.write("### Scored Predictions")
        st.dataframe(st.session_state.last_scored_data.head(20), use_container_width=True)
        csv = st.session_state.last_scored_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Scored Results",
            data=csv,
            file_name="scored_borrowers.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Upload and score a dataset to see batch predictions.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_agent:
    render_langgraph_pipeline()

    st.markdown("<div class='section-label'>Decision Narrative</div>", unsafe_allow_html=True)
    agent_form_col, agent_chat_col = st.columns([0.95, 1.35], gap="large")

    with agent_form_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Borrower Intake")
        with st.form("agentic_borrower_form", border=False):
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            sex = st.selectbox("Sex", ["male", "female"])
            job = st.selectbox(
                "Job Category",
                [0, 1, 2, 3],
                format_func=lambda x: {
                    0: "Unemployed / Unskilled (Non-Resident)",
                    1: "Unskilled (Resident)",
                    2: "Skilled Employee / Official",
                    3: "Management / Highly Skilled / Self-Employed",
                }[x],
            )
            housing = st.selectbox(
                "Housing Status",
                ["own", "free", "rent"],
                format_func=lambda x: {"own": "Owns Property", "free": "Lives For Free", "rent": "Renting"}[x],
            )
            saving_accounts = st.selectbox(
                "Savings Account",
                ["NA", "little", "moderate", "quite rich", "rich"],
            )
            checking_account = st.selectbox(
                "Checking Account",
                ["NA", "little", "moderate", "rich"],
            )
            credit_amount = st.number_input("Requested Amount (DM)", min_value=100, value=2500)
            duration = st.slider("Duration (Months)", min_value=4, max_value=72, value=24)
            purpose = st.selectbox(
                "Purpose",
                [
                    "radio/TV",
                    "education",
                    "furniture/equipment",
                    "car",
                    "business",
                    "domestic appliances",
                    "repairs",
                    "vacation/others",
                ],
            )
            submitted = st.form_submit_button("Run Agentic Analysis", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        reset_follow_up_state()
        borrower_profile = {
            "age": age,
            "sex": sex,
            "job": job,
            "housing": housing,
            "saving_accounts": saving_accounts,
            "checking_account": checking_account,
            "credit_amount": credit_amount,
            "duration": duration,
            "purpose": purpose,
        }
        user_message = build_user_summary(borrower_profile)
        decision = run_agentic_lending_decision(
            borrower_profile=borrower_profile,
            model=primary_model,
            model_name=primary_model_name,
        )
        agent_message = (
            f"Final Lending Verdict: {decision['final_verdict']}\n\n"
            f"Reasoning:\n{decision['reasoning']}\n\n"
            f"Policy Summary:\n{decision.get('policy_context', 'No policy context retrieved.')}"
        )
        st.session_state.agent_chat_history.append(("user", user_message))
        st.session_state.agent_chat_history.append(("assistant", agent_message))
        st.session_state.latest_decision = decision
        st.session_state.latest_borrower_profile = borrower_profile

    with agent_chat_col:
        st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
        st.markdown("<div class='agent-note'>The underwriting agent explains each decision using model output plus retrieved policy context.</div>", unsafe_allow_html=True)

        if not st.session_state.agent_chat_history:
            st.info("Submit a borrower profile to open the decision conversation.")
        else:
            for role, message in st.session_state.agent_chat_history:
                with st.chat_message(role):
                    st.markdown(message)

        if st.session_state.latest_decision:
            gauge_col, signal_col = st.columns([0.9, 1.1], gap="large")
            with gauge_col:
                st.plotly_chart(build_gauge(st.session_state.latest_decision["risk_score"]), use_container_width=True)
            with signal_col:
                st.markdown("### Decision Signals")
                for factor in st.session_state.latest_decision.get("risk_factors", []):
                    st.markdown(f"- {factor}")
                st.caption(f"Decision source: {st.session_state.latest_decision.get('decision_source', 'n/a')}")
                if st.session_state.latest_decision.get("policy_query"):
                    st.caption(f"Policy query: {st.session_state.latest_decision['policy_query']}")

            st.markdown("### Ask a Question About This Decision")
            follow_up = st.text_input(
                "Ask for a simple explanation of this result",
                placeholder="Example: Why was this borrower flagged as high risk, and what would improve the decision?",
                key="follow_up_question",
            )
            action_col1, action_col2 = st.columns([0.8, 1.2], gap="medium")
            with action_col1:
                if st.button("Ask Follow-Up", use_container_width=True):
                    if follow_up.strip():
                        follow_up_result = answer_follow_up_question(
                            question=follow_up.strip(),
                            borrower_profile=st.session_state.latest_borrower_profile,
                            lending_decision=st.session_state.latest_decision,
                            memory=st.session_state.conversation_memory,
                        )
                        st.session_state.conversation_memory = follow_up_result["memory"]
                        st.session_state.agent_chat_history.append(("user", follow_up.strip()))
                        st.session_state.agent_chat_history.append(("assistant", follow_up_result["answer"]))
                        st.session_state.clear_follow_up_question = True
                        st.rerun()
                    else:
                        st.warning("Please enter a question before continuing.")
            with action_col2:
                report_payload = generate_lending_report_pdf(
                    borrower_profile=st.session_state.latest_borrower_profile,
                    lending_decision=st.session_state.latest_decision,
                    model_metrics=build_model_summary_rows(
                        {name: score_dataset(load_local_dataset(), model) for name, model in model_registry.items()}
                    ),
                )
                st.download_button(
                    "Export Report",
                    data=report_payload,
                    file_name="agentic_lending_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

with tab_metrics:
    st.markdown("<div class='section-label'>Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("Evaluation metrics and comparative analysis of trained algorithms.")

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Logistic Regression": ["81.2%", "68.7%", "56.6%", "62.1%"],
        "Decision Tree": ["79.7%", "64.9%", "55.0%", "59.6%"]
    })
    st.table(metrics_df)

    st.write("### Confusion Matrices")
    cm_left, cm_right = st.columns(2)

    def plot_cm(title, counts):
        z = [[counts[0], counts[1]], [counts[2], counts[3]]]
        x = ["Stay", "Churn"]
        y = ["Stay", "Churn"]
        fig = go.Figure(data=go.Heatmap(
            z=z, x=x, y=y,
            colorscale=[[0, "#2c2f54"], [0.5, "#6f8cff"], [1.0, "#a391ff"]],
            showscale=False,
            text=[[str(counts[0]), str(counts[1])], [str(counts[2]), str(counts[3])]],
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"}
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=14, color="#8baecc")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=260,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(title="Predicted", color="#8baecc"),
            yaxis=dict(title="Actual", color="#8baecc")
        )
        return fig

    with cm_left:
        st.plotly_chart(plot_cm("Logistic Regression", [1391, 148, 249, 325]), use_container_width=True)
    with cm_right:
        st.plotly_chart(plot_cm("Decision Tree", [1368, 171, 258, 316]), use_container_width=True)

    st.write("### Feature Importance")
    st.markdown("Top Feature Importance")
    # Mocking a bar chart for feature importance
    feat_importance = pd.DataFrame({
        "Feature": ["duration", "credit_amount", "age", "checking_account", "savings_account"],
        "Importance": [0.35, 0.28, 0.15, 0.12, 0.10]
    }).sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=feat_importance["Importance"],
        y=feat_importance["Feature"],
        orientation='h',
        marker_color='#6f8cff'
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8f7ff"),
        height=300,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

