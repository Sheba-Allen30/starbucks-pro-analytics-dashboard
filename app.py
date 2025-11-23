# =============================================================================
# Starbucks Pro Analytics & ML Dashboard — Emerald Edition
# With: GitHub-hosted logos + model saving (.pkl) + full EDA & ML
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import tempfile
import json
import joblib
import requests
from io import BytesIO
from pathlib import Path

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# CONFIG — GitHub Image URLs (Replace these with your RAW file URLs!)
# =============================================================================
RCSS_URL = "https://raw.githubusercontent.com/Sheba-Allen30/starbucks-pro-analytics-dashboard/main/rajagiri.png"
GRANT_URL = "https://raw.githubusercontent.com/Sheba-Allen30/starbucks-pro-analytics-dashboard/main/grand.jpeg"
STARBUCKS_LOGO_URL = "https://raw.githubusercontent.com/Sheba-Allen30/starbucks-pro-analytics-dashboard/main/starbucks.png"


# =============================================================================
# Helper: Load image from GitHub URL
# =============================================================================
def load_image_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return base64.b64encode(r.content).decode()
    except:
        return None


# =============================================================================
# UI Styling — Emerald Theme
# =============================================================================
def inject_emerald_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #06301f 0%, #0b2e22 60%, #071a13 100%) !important;
        color: #eafff5 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .card {
        background: rgba(255,255,255,0.04);
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    }
    .logo-box {
        background: rgba(255,255,255,0.06);
        padding: 12px;
        width: 170px;
        height: 180px;
        border-radius: 12px;
        display:flex;
        align-items:center;
        justify-content:center;
        box-shadow:0 4px 16px rgba(0,0,0,0.4);
    }
    h1, h2, h3, h4, h5 {
        color: #eafff5 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# Data helpers
# =============================================================================
def clean_numeric_like_columns(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            temp = df[col].astype(str).str.replace(",", "").str.replace("%", "")
            temp = temp.str.replace(r"[^0-9\.\-]", "", regex=True)
            coerced = pd.to_numeric(temp, errors="ignore")
            df[col] = coerced
    return df


def auto_encode_features(X):
    X = X.copy()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
    return pd.get_dummies(X, columns=cat_cols, drop_first=True)


def regression_metrics(y, p):
    mse = mean_squared_error(y, p)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, p)
    return {"MSE": mse, "RMSE": rmse, "R2": r2}


def classification_metrics(y, p):
    return {
        "Accuracy": accuracy_score(y, p),
        "Precision": precision_score(y, p, average="weighted", zero_division=0),
        "Recall": recall_score(y, p, average="weighted", zero_division=0),
        "F1": f1_score(y, p, average="weighted", zero_division=0)
    }


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.set_page_config(page_title="Starbucks Pro — Emerald", layout="wide")
    inject_emerald_css()

    st.sidebar.header("📂 Upload CSV")
    file = st.sidebar.file_uploader("Upload Starbucks CSV", type=["csv"])

    st.sidebar.header("📌 Navigation")
    page = st.sidebar.radio("Go to:", [
        "Overview",
        "EDA (All Charts)",
        "Modeling & Comparison",
        "Prediction Playground",
        "Export Report",
        "About / Team"
    ])

    # Load dataset
    if file is None:
        st.warning("Upload a CSV to continue.")
        return

    df = pd.read_csv(file)
    df = clean_numeric_like_columns(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # -------------------------------------------------------------------------
    # PAGE 1 — Overview
    # -------------------------------------------------------------------------
    if page == "Overview":
        st.title("📊 Dataset Overview")
        st.write(df.head())
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        st.write("Missing values:")
        st.write(df.isnull().sum())

    # -------------------------------------------------------------------------
    # PAGE 2 — EDA
    # -------------------------------------------------------------------------
    elif page == "EDA (All Charts)":
        st.title("📈 Full EDA Suite")

        # Histogram
        st.subheader("Histogram")
        col = st.selectbox("Select numeric column:", numeric_cols)
        fig, ax = plt.subplots()
        ax.hist(df[col], color="#0b6b48")
        st.pyplot(fig)

        # Scatter
        if len(numeric_cols) >= 2:
            st.subheader("Scatter Plot")
            x = st.selectbox("X-axis:", numeric_cols)
            y = st.selectbox("Y-axis:", numeric_cols)
            fig2, ax2 = plt.subplots()
            ax2.scatter(df[x], df[y], color="#ffd59e")
            st.pyplot(fig2)

        # Correlation
        st.subheader("Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="Greens", ax=ax3)
        st.pyplot(fig3)

    # -------------------------------------------------------------------------
    # PAGE 3 — Modeling
    # -------------------------------------------------------------------------
    elif page == "Modeling & Comparison":
        st.title("🧠 Model Training & Comparison")

        target = st.selectbox("Select target column:", df.columns)
        features = st.multiselect("Select features:", [c for c in df.columns if c != target])

        if st.button("Train Models") and features:
            X = auto_encode_features(df[features])
            y = df[target]

            if y.dtype == object:
                problem = "classification"
                y = LabelEncoder().fit_transform(y)
            else:
                problem = "regression"

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            results = {}
            models = {}

            # Regression models
            if problem == "regression":
                lr = LinearRegression().fit(X_train, y_train)
                rf = RandomForestRegressor().fit(X_train, y_train)

                models = {"Linear Regression": lr, "Random Forest": rf}

            # Classification models
            else:
                log = LogisticRegression(max_iter=2000).fit(X_train, y_train)
                rfc = RandomForestClassifier().fit(X_train, y_train)

                models = {"Logistic Regression": log, "Random Forest": rfc}

            for name, model in models.items():
                preds = model.predict(X_test)
                metrics = regression_metrics(y_test, preds) if problem == "regression" else classification_metrics(y_test, preds)
                results[name] = metrics

                # AUTO-SAVE MODEL
                out_path = f"{name.replace(' ', '_').lower()}.pkl"
                joblib.dump(model, out_path)

            st.session_state["trained_models"] = models
            st.session_state["X_cols"] = X.columns.tolist()
            st.session_state["problem"] = problem
            st.session_state["target"] = target

            st.subheader("Model Comparison")
            st.write(pd.DataFrame(results))

            st.success("Models trained and saved successfully (.pkl files created).")

    # -------------------------------------------------------------------------
    # PAGE 4 — Prediction Playground
    # -------------------------------------------------------------------------
    elif page == "Prediction Playground":
        st.title("🎯 Prediction Playground")

        if "trained_models" not in st.session_state:
            st.error("Train models first.")
            return

        models = st.session_state["trained_models"]
        choice = st.selectbox("Choose model:", list(models.keys()))

        model = models[choice]

        # User input fields
        st.subheader("Enter feature values:")
        inputs = {}
        for col in st.session_state["X_cols"]:
            inputs[col] = st.number_input(col, 0.0)

        if st.button("Predict"):
            Xnew = pd.DataFrame([inputs])
            pred = model.predict(Xnew)[0]
            st.success(f"Prediction: {pred}")

            # DOWNLOAD SELECTED MODEL (OPTION C)
            buf = BytesIO()
            joblib.dump(model, buf)
            st.download_button(
                label="Download This Model (.pkl)",
                data=buf.getvalue(),
                file_name=f"{choice.replace(' ', '_').lower()}.pkl",
                mime="application/octet-stream"
            )

    # -------------------------------------------------------------------------
    # PAGE 5 — PDF Export
    # -------------------------------------------------------------------------
    elif page == "Export Report":
        st.title("📄 Export Report (EDA + Models)")

        st.info("PDF exports coming soon — safe ASCII version.")

    # -------------------------------------------------------------------------
    # PAGE 6 — ABOUT PAGE
    # -------------------------------------------------------------------------
    elif page == "About / Team":
        st.title("About & Team — Rajagiri (Grant Thornton Add-On)")

        col1, col2 = st.columns([1, 2])

        with col1:
            rcss = load_image_url(RCSS_URL)
            grant = load_image_url(GRANT_URL)

            st.markdown("### Partner Institutions")

            st.markdown("<div class='logo-box'>", unsafe_allow_html=True)
            if rcss:
                st.image(rcss, width=140, use_column_width=False)
            else:
                st.write("RCSS Logo not found")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='logo-box'>", unsafe_allow_html=True)
            if grant:
                st.image(grant, width=140, use_column_width=False)
            else:
                st.write("Grant Thornton Logo not found")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.subheader("Project: Starbucks Pro — Analytics & Machine Learning Dashboard")
            st.write("""
                Capstone-style project as part of the Grant Thornton add-on program 
                at Rajagiri College of Social Sciences. 
                Focus: Analytics, Machine Learning, Industry Skills.
            """)

        st.markdown("---")

        st.subheader("Contributors")
        st.write("""
        **Sheba Allen Lalu** — Team Lead  
        LinkedIn: https://www.linkedin.com/in/sheba-lalu-b59b3a398  
        """)


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    main()
