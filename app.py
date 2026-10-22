
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Loan_default.csv"
MODEL_DIR = ROOT / "models"
META_PATH = MODEL_DIR / "metadata.json"
RF_PATH = MODEL_DIR / "random_forest_pipeline.joblib"
LR_PATH = MODEL_DIR / "logistic_regression_pipeline.joblib"

st.set_page_config(
    page_title="LoanGuard AI | Loan Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM THEME
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --background: #07111f;
    --surface: #101f32;
    --surface-light: #172b42;
    --border: #38536e;
    --text: #f8fafc;
    --muted: #cbd5e1;
    --accent: #5eead4;
    --teal: #14b8a6;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #07111f 0%,
        #0a1628 55%,
        #0b1d31 100%
    );
    color: var(--text);
}

/* General text */
.stApp p,
.stApp li,
.stApp span {
    color: inherit;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0b1727;
    border-right: 1px solid #29445f;
}

[data-testid="stSidebar"] * {
    color: #f1f5f9;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #cbd5e1;
}

/* Headings */
h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif;
    color: #f8fafc !important;
}

/* Hero section */
.hero {
    padding: 28px;
    border: 1px solid #38536e;
    border-radius: 22px;
    background: linear-gradient(
        120deg,
        rgba(20, 52, 82, 0.98),
        rgba(13, 29, 49, 0.98)
    );
    margin-bottom: 24px;
}

.eyebrow {
    color: #5eead4;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 12px;
    font-weight: 800;
}

.hero-title {
    font-family: 'Manrope', sans-serif;
    font-size: 34px;
    line-height: 1.25;
    font-weight: 800;
    margin: 10px 0;
    color: #ffffff;
}

.hero-copy {
    color: #d4e0ed;
    font-size: 15px;
    max-width: 780px;
    line-height: 1.7;
}

/* Metric cards */
.metric-card {
    background: #101f32;
    border: 1px solid #38536e;
    border-radius: 16px;
    padding: 18px;
    min-height: 110px;
}

.metric-label {
    color: #cbd5e1;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: 800;
    margin-top: 6px;
}

.metric-note {
    color: #5eead4;
    font-size: 12px;
    margin-top: 4px;
}

/* Forms */
div[data-testid="stForm"] {
    background: #0d1b2d;
    padding: 22px;
    border: 1px solid #38536e;
    border-radius: 18px;
}

/* Widget labels */
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp label {
    color: #f8fafc !important;
    font-weight: 600 !important;
}

/* Number input and text input containers */
.stApp [data-testid="stNumberInput"] div[data-baseweb="input"],
.stApp [data-testid="stTextInput"] div[data-baseweb="input"] {
    background-color: #172b42 !important;
    border: 1px solid #55718e !important;
    border-radius: 9px !important;
}

/* Actual typed values */
.stApp input,
.stApp textarea {
    background-color: #172b42 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #5eead4 !important;
    border-color: #55718e !important;
}

/* Number input buttons */
.stApp [data-testid="stNumberInput"] button {
    background-color: #223c56 !important;
    color: #ffffff !important;
    border-color: #55718e !important;
}

.stApp [data-testid="stNumberInput"] button:hover {
    background-color: #315574 !important;
}

/* Select boxes */
.stApp [data-baseweb="select"] > div {
    background-color: #172b42 !important;
    border-color: #55718e !important;
    color: #ffffff !important;
}

/* Selected dropdown value */
.stApp [data-baseweb="select"] span,
.stApp [data-baseweb="select"] div,
.stApp [data-baseweb="select"] input {
    color: #ffffff !important;
}

/* Dropdown arrow */
.stApp [data-baseweb="select"] svg {
    fill: #ffffff !important;
}

/* Dropdown menu */
[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="list"] {
    background-color: #172b42 !important;
    color: #ffffff !important;
}

[role="option"] {
    background-color: #172b42 !important;
    color: #ffffff !important;
}

[role="option"]:hover {
    background-color: #315574 !important;
    color: #ffffff !important;
}

/* Placeholder text */
.stApp input::placeholder,
.stApp textarea::placeholder {
    color: #cbd5e1 !important;
    -webkit-text-fill-color: #cbd5e1 !important;
    opacity: 1 !important;
}

/* Captions and helper text */
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p,
.stApp .small-muted {
    color: #bdcddd !important;
}

/* Radio buttons */
.stApp [data-testid="stRadio"] label {
    color: #f1f5f9 !important;
}

/* Checkboxes */
.stApp [data-testid="stCheckbox"] label {
    color: #f1f5f9 !important;
}

/* Buttons */
.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: #14b8a6 !important;
    color: #06201f !important;
    border: 1px solid #5eead4 !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
    padding: 10px 20px !important;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #5eead4 !important;
    color: #06201f !important;
}

/* Alerts */
.stApp [data-testid="stAlert"] {
    border-radius: 12px;
}

/* Dataframes */
.stApp [data-testid="stDataFrame"] {
    border: 1px solid #38536e;
    border-radius: 10px;
    overflow: hidden;
}

/* Dividers */
hr {
    border-color: #29445f !important;
}

/* Responsive hero title */
@media (max-width: 768px) {
    .hero {
        padding: 20px;
    }

    .hero-title {
        font-size: 26px;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metadata():
    if not META_PATH.exists():
        return None
    with open(META_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_models():
    if not RF_PATH.exists() or not LR_PATH.exists():
        return None, None

    # Only load model artifacts created by your own training script.
    return joblib.load(RF_PATH), joblib.load(LR_PATH)


df = load_data()
metadata = load_metadata()
rf_model, lr_model = load_models()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 🏦 LoanGuard AI")
    st.caption("Loan default risk analytics")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Predict Risk",
            "Dataset Analytics",
            "Model Performance",
            "About"
        ]
    )

    st.divider()

    st.markdown("**Project stack**")
    st.caption("Python · Pandas · Scikit-learn · Streamlit")

    st.markdown(
        '<p class="small-muted">'
        'Academic prototype — not a lending decision system.'
        '</p>',
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">Machine Learning • Financial Analytics</div>
    <div class="hero-title">
        Understand loan risk.<br>
        Make data-informed assessments.
    </div>
    <div class="hero-copy">
        Explore borrower patterns, compare classification models,
        and test a sample applicant using your Loan Default
        Prediction project.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if df is None:
    st.error(
        "Loan_default.csv is missing. "
        "Place it in the same folder as app.py."
    )
    st.stop()

if metadata is None or rf_model is None or lr_model is None:
    st.warning(
        "Model files are not available yet. "
        "Run train_model.py before launching the app."
    )
    st.code(
        "python train_model.py\nstreamlit run app.py",
        language="bash"
    )
    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    total = metadata.get("row_count", len(df))
    default_rate = metadata.get("default_rate", 0)

    rf_acc = metadata["metrics"]["Random Forest"]["accuracy"]
    rf_recall = metadata["metrics"]["Random Forest"]["recall"]

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (
            c1,
            "DATASET RECORDS",
            f"{total:,}",
            "Rows used in training"
        ),
        (
            c2,
            "DEFAULT RATE",
            f"{default_rate:.1%}",
            "Target class proportion"
        ),
        (
            c3,
            "RF TEST ACCURACY",
            f"{rf_acc:.1%}",
            "Held-out test set"
        ),
        (
            c4,
            "RF DEFAULT RECALL",
            f"{rf_recall:.1%}",
            "Defaults identified"
        )
    ]

    for col, label, value, note in cards:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Portfolio snapshot")

    left, right = st.columns([1.1, 1])

    with left:
        if "Default" in df.columns:
            counts = (
                df["Default"]
                .value_counts(dropna=False)
                .rename_axis("Default")
                .reset_index(name="Count")
            )

            counts["Label"] = (
                counts["Default"]
                .map({0: "No Default", 1: "Default"})
                .fillna(counts["Default"].astype(str))
            )

            fig = px.pie(
                counts,
                values="Count",
                names="Label",
                hole=0.62,
                title="Target class distribution"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8eef7",
                legend_title_text=""
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("The dataset does not contain a Default column.")

    with right:
        st.markdown("#### What this dashboard offers")

        st.markdown("""
        - **Risk prediction:** Test applicant details with the trained Random Forest pipeline.
        - **Dataset analytics:** Explore default distribution and numeric patterns.
        - **Model comparison:** Review accuracy, precision, recall, F1 and confusion matrices.
        - **Reproducible workflow:** Missing-value handling and categorical encoding are inside the model pipeline.
        """)

        st.info(
            "For imbalanced datasets, do not judge a model by accuracy alone. "
            "Review default-class recall and precision too."
        )


# ============================================================
# PREDICT RISK
# ============================================================

elif page == "Predict Risk":

    st.markdown("## Applicant risk assessment")

    st.write(
        "Enter applicant details below. "
        "The labels and values are styled for better visibility."
    )

    feature_cols = metadata["feature_columns"]
    categorical_cols = set(metadata["categorical_columns"])
    numeric_cols = set(metadata["numeric_columns"])

    form_values = {}

    with st.form("prediction_form"):

        st.markdown("#### Applicant profile")

        columns = st.columns(2)

        for idx, feature in enumerate(feature_cols):

            col = columns[idx % 2]

            with col:

                if feature == "loan_to_income_ratio":
                    st.caption(
                        "Loan-to-income ratio is calculated automatically "
                        "from LoanAmount and Income."
                    )
                    continue

                display_name = feature.replace("_", " ").title()

                if feature in categorical_cols:

                    raw_series = df[feature].dropna()

                    choices = sorted(
                        raw_series.astype(str).unique().tolist()
                    )

                    if not choices:
                        choices = ["Unknown"]

                    form_values[feature] = st.selectbox(
                        display_name,
                        choices,
                        key=f"cat_{feature}",
                        help=f"Select the applicant's {display_name.lower()}."
                    )

                else:

                    series = pd.to_numeric(
                        df[feature],
                        errors="coerce"
                    ).dropna()

                    median = (
                        float(series.median())
                        if len(series)
                        else 0.0
                    )

                    min_value = (
                        float(series.min())
                        if len(series)
                        else 0.0
                    )

                    max_value = (
                        float(series.max())
                        if len(series)
                        else max(1.0, median + 1)
                    )

                    if feature in ("Income", "LoanAmount"):
                        step = 1000.0
                    elif feature == "CreditScore":
                        step = 1.0
                    else:
                        step = max(
                            (max_value - min_value) / 100,
                            0.01
                        )

                    safe_max = (
                        max_value
                        if max_value > min_value
                        else min_value + 1.0
                    )

                    default_value = min(
                        max(median, min_value),
                        safe_max
                    )

                    form_values[feature] = st.number_input(
                        display_name,
                        min_value=min_value,
                        max_value=safe_max,
                        value=default_value,
                        step=float(step),
                        key=f"num_{feature}",
                        help=f"Enter the applicant's {display_name.lower()}."
                    )

        submitted = st.form_submit_button(
            "Analyze default risk",
            use_container_width=True
        )

    if submitted:

        row = {}

        for feature in feature_cols:
            if feature == "loan_to_income_ratio":
                continue

            row[feature] = form_values.get(feature, np.nan)

        if {"LoanAmount", "Income"}.issubset(row):

            income = float(row["Income"])

            row["loan_to_income_ratio"] = (
                float(row["LoanAmount"]) / (income + 1)
            )

        input_df = pd.DataFrame([row])

        # Restore feature order expected by the fitted pipeline.
        input_df = input_df.reindex(columns=feature_cols)

        try:

            prediction = int(rf_model.predict(input_df)[0])

            probability = None

            if hasattr(rf_model, "predict_proba"):

                if hasattr(rf_model, "named_steps"):
                    classes = list(
                        rf_model.named_steps["model"].classes_
                    )
                else:
                    classes = list(rf_model.classes_)

                if 1 in classes:
                    probability = float(
                        rf_model.predict_proba(input_df)[0][
                            classes.index(1)
                        ]
                    )

            st.divider()
            st.markdown("### Assessment result")

            result_col, prob_col = st.columns([1, 1])

            with result_col:

                if prediction == 1:
                    st.error("Model prediction: DEFAULT")

                    st.write(
                        "The model classifies this application "
                        "as belonging to the default class."
                    )

                else:
                    st.success("Model prediction: NO DEFAULT")

                    st.write(
                        "The model classifies this application "
                        "as belonging to the non-default class."
                    )

            with prob_col:

                if probability is not None:

                    st.metric(
                        "Estimated model default probability",
                        f"{probability:.1%}"
                    )

                    st.progress(
                        min(max(probability, 0.0), 1.0)
                    )

                st.caption(
                    "This is a model estimate, not a calibrated "
                    "guarantee or a credit decision."
                )

            st.warning(
                "Educational demonstration only. Do not use this "
                "prototype to approve, reject, or price real loans."
            )

        except Exception as error:
            st.error("The prediction could not be completed.")
            st.exception(error)


# ============================================================
# DATASET ANALYTICS
# ============================================================

elif page == "Dataset Analytics":

    st.markdown("## Dataset analytics")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("#### Default distribution")

        if "Default" in df.columns:

            counts = (
                df["Default"]
                .value_counts()
                .rename_axis("Default")
                .reset_index(name="Count")
            )

            counts["Class"] = (
                counts["Default"]
                .map({0: "No Default", 1: "Default"})
                .fillna(counts["Default"].astype(str))
            )

            fig = px.bar(
                counts,
                x="Class",
                y="Count",
                color="Class",
                title="Borrowers by target class"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                font_color="#e8eef7"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("The dataset does not contain a Default column.")

    with c2:

        numeric = (
            df.select_dtypes(include=np.number)
            .drop(columns=["Default"], errors="ignore")
        )

        if not numeric.empty:

            selected = st.selectbox(
                "Choose a numeric feature",
                numeric.columns.tolist()
            )

            fig = px.histogram(
                df,
                x=selected,
                color="Default" if "Default" in df.columns else None,
                marginal="box",
                barmode="overlay",
                opacity=0.72,
                title=f"{selected} distribution"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8eef7"
            )

            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Dataset preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.caption(
        "Preview limited to the first 100 rows for responsiveness."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown("## Model performance")

    rows = []

    for name, metric in metadata["metrics"].items():

        rows.append({
            "Model": name,
            "Accuracy": metric["accuracy"],
            "Precision (Default)": metric["precision"],
            "Recall (Default)": metric["recall"],
            "F1 (Default)": metric["f1"],
            "ROC-AUC": metric.get("roc_auc")
        })

    comparison = pd.DataFrame(rows)

    st.dataframe(
        comparison.style.format({
            c: "{:.3f}"
            for c in comparison.columns
            if c != "Model"
        }),
        use_container_width=True
    )

    metric_columns = [
        "Accuracy",
        "Precision (Default)",
        "Recall (Default)",
        "F1 (Default)"
    ]

    fig = px.bar(
        comparison.melt(
            id_vars="Model",
            value_vars=metric_columns,
            var_name="Metric",
            value_name="Score"
        ),
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Model comparison"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eef7",
        yaxis_range=[0, 1]
    )

    st.plotly_chart(fig, use_container_width=True)

    chosen = st.selectbox(
        "Confusion matrix for",
        list(metadata["metrics"].keys())
    )

    matrix = np.array(
        metadata["metrics"][chosen]["confusion_matrix"]
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        x=["Predicted No Default", "Predicted Default"],
        y=["Actual No Default", "Actual Default"],
        color_continuous_scale="Teal",
        title=f"{chosen}: confusion matrix",
        aspect="auto"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eef7"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Precision answers how many predicted defaulters were actual "
        "defaulters. Recall answers how many actual defaulters "
        "the model found."
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown("## About LoanGuard AI")

    st.write(
        "LoanGuard AI is a student machine-learning demonstration "
        "built around the Loan Default Prediction dataset and "
        "classification workflow."
    )

    st.markdown("""
    **Workflow**

    1. Load the dataset and inspect missing values.
    2. Remove identifier columns and columns with more than 50% missing data.
    3. Engineer a loan-to-income ratio where the source columns are available.
    4. Impute missing values, scale numeric features, and one-hot encode categorical features.
    5. Train and compare Logistic Regression and Random Forest on a stratified hold-out set.
    6. Use the fitted Random Forest pipeline for interactive predictions.
    """)

    st.markdown(
        "**Repository:** "
        "[rohit20020/IBM_PROJECT](https://github.com/rohit20020/ibm_project)"
    )

    st.caption(
        "Model outputs are experimental and may reflect bias, "
        "dataset limitations, and class imbalance. This is not "
        "financial advice or a production credit-scoring system."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<p class="small-muted">'
    'LoanGuard AI · Student project prototype · '
    'Built with Streamlit and scikit-learn'
    '</p>',
    unsafe_allow_html=True
)