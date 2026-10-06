from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(18, 115, 92, .18), transparent 42%),
            linear-gradient(160deg, #0b1220 0%, #101a2b 55%, #0b1220 100%);
    }
    .block-container {
        max-width: 1050px;
        padding-top: 3rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.5rem 0 1rem;
    }
    .eyebrow {
        color: #55d6a5;
        font-size: .8rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: .45rem 0 .55rem;
        color: #f4f7fb;
        font-size: clamp(2rem, 5vw, 3.2rem);
        letter-spacing: -.045em;
        line-height: 1.08;
    }
    .hero p, .muted {
        color: #9aa9bd;
        font-size: 1.02rem;
        line-height: 1.65;
    }
    .section-title {
        color: #f4f7fb;
        font-size: 1.2rem;
        font-weight: 700;
        margin: .2rem 0 .15rem;
    }
    .section-caption {
        color: #9aa9bd;
        font-size: .92rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(21, 34, 52, .8);
        border: 1px solid rgba(148, 163, 184, .16);
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #9aa9bd;
    }
    div[data-testid="stMetricValue"] {
        color: #f4f7fb;
    }
    div[data-testid="stForm"] {
        background: rgba(16, 27, 43, .78);
        border: 1px solid rgba(148, 163, 184, .18);
        border-radius: 18px;
        padding: 1.35rem 1.5rem 1.15rem;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] > div > div {
        border-radius: 10px;
    }
    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 3rem;
        border: 0;
        border-radius: 10px;
        background: linear-gradient(90deg, #10a879, #16b98b);
        color: #071710;
        font-weight: 700;
        transition: filter .2s ease, transform .2s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
    }
    .result-card {
        margin-top: 1.3rem;
        padding: 1.25rem 1.4rem;
        border-radius: 14px;
        border: 1px solid rgba(85, 214, 165, .35);
        background: linear-gradient(110deg, rgba(16, 168, 121, .18), rgba(16, 27, 43, .65));
    }
    .result-card.not-placed {
        border-color: rgba(251, 146, 60, .38);
        background: linear-gradient(110deg, rgba(194, 92, 35, .15), rgba(16, 27, 43, .65));
    }
    .result-label {
        color: #55d6a5;
        font-size: 1.2rem;
        font-weight: 750;
    }
    .not-placed .result-label {
        color: #ffb36b;
    }
    .result-detail {
        color: #b6c3d3;
        margin-top: .35rem;
    }
    .footer-note {
        color: #8190a5;
        font-size: .82rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(148, 163, 184, .14);
        margin-top: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

df = pd.read_csv(Path(__file__).with_name("converted_data.csv"))
feature_columns = [
    "CGPA",
    "Attendance",
    "Internship",
    "Coding",
    "Aptitude",
    "Communication",
]
X = df[feature_columns]
y = df["Placement"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Career readiness • Student insights</div>
        <h1>Student Placement Predictor</h1>
        <p>Explore how academic performance and key skills may shape a placement outcome.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

accuracy_col, records_col, features_col = st.columns(3)
accuracy_col.metric("Model test accuracy", f"{accuracy:.1%}")
records_col.metric("Student records", f"{len(df):,}")
features_col.metric("Assessment areas", len(feature_columns))

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Build a student profile</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Enter scores on the scales shown. All fields are required.</div>',
    unsafe_allow_html=True,
)

with st.form("student_profile"):
    academic_col, skills_col = st.columns(2, gap="large")

    with academic_col:
        st.markdown("#### Academics & experience")
        cgpa = st.number_input(
            "CGPA (out of 10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1
        )
        attendance = st.number_input(
            "Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0
        )
        internship = st.selectbox("Completed an internship?", ["No", "Yes"])

    with skills_col:
        st.markdown("#### Skills assessment")
        coding = st.number_input(
            "Coding score (out of 100)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0,
        )
        aptitude = st.number_input(
            "Aptitude score (out of 100)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0,
        )
        communication = st.number_input(
            "Communication score (out of 100)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0,
        )

    submitted = st.form_submit_button("✨  Predict placement")

if submitted:
    student = pd.DataFrame(
        [[cgpa, attendance, int(internship == "Yes"), coding, aptitude, communication]],
        columns=feature_columns,
    )
    prediction = model.predict(student)[0]
    st.session_state["placement_prediction"] = int(prediction)

if "placement_prediction" in st.session_state:
    if st.session_state["placement_prediction"] == 1:
        st.markdown(
            """
            <div class="result-card">
                <div class="result-label">✓  Placement predicted</div>
                <div class="result-detail">The profile is classified as placed by the decision-tree model.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-card not-placed">
                <div class="result-label">↗  Placement not predicted</div>
                <div class="result-detail">The profile is classified as not placed by the decision-tree model.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="footer-note">
        This result is an estimate from a decision-tree model trained on the included sample dataset.
        It is for learning and exploration only, not a guarantee of a real-world outcome.
    </div>
    """,
    unsafe_allow_html=True,
)
