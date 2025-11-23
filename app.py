# app.py
"""
Starbucks Pro — Analytics & ML (Emerald Theme) — Full (Save ALL models)
- Minimal sidebar: CSV upload + page nav
- About page: side-by-side logos (from repo root or /mnt/data fallbacks)
- Saves ALL trained models as separate .pkl files (lr.pkl, rf.pkl, log.pkl, rfc.pkl)
- Also saves label encoder (if classification) and features list for alignment
- Provides download buttons for saved model files
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import tempfile
import json
import pickle
from pathlib import Path
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder

# joblib is optional; we'll use pickle for portability
import joblib

# Optional Lottie support
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except Exception:
    LOTTIE_AVAILABLE = False

# Optional PDF (fpdf) - unchanged (used elsewhere)
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# -------------------------
# Paths & logos (repo-root friendly)
# -------------------------
# Prefer repo root files (these are the files you uploaded to GitHub / Streamlit)
RCSS_REPO = Path("rajagiri.png")
GRANT_REPO = Path("grand.jpeg")  # or grand.jpg / grand.png if you used different name

# Fallbacks (if app environment provided them in /mnt/data)
RCSS_FALLBACK = Path("/mnt/data/fbed8862-e4ff-4218-8008-8ad9bbed415e.png")
GRANT_FALLBACK = Path("/mnt/data/e0e0713b-c281-4a62-b35a-623830e23dcf.png")

# Default CSV/logo in repo (if present)
DEFAULT_CSV = Path("starbucks.csv")
DEFAULT_LOGO = Path("starbucks.png")

# Where to write model files (tmp dir within environment)
MODEL_DIR = Path(tempfile.gettempdir()) / "starbucks_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Styling (emerald)
# -------------------------
def inject_emerald_css():
    css = r"""
    <style>
    :root{ --emerald-1:#0b6b48; --cream:#fbfdfb; }
    .stApp { background: linear-gradient(180deg,#06301f 0%,#0b2e22 60%,#071a13 100%) !important; color:var(--cream); }
    .card { background: rgba(255,255,255,0.02); border-radius:12px; padding:16px; margin-bottom:16px; border:1px solid rgba(255,255,255,0.03); }
    .header-container{display:flex;align-items:center;gap:12px}
    .app-title{font-size:28px;font-weight:700}
    .subtitle{font-size:13px;color:rgba(235,248,240,0.88)}
    .about-logos-row{display:flex;gap:28px;align-items:center;flex-wrap:wrap}
    .logo-card{background:rgba(255,255,255,0.03);border-radius:12px;padding:12px;display:flex;align-items:center;justify-content:center;width:160px;height:160px;box-shadow:0 8px 20px rgba(0,0,0,0.45)}
    .logo-caption{text-align:center;color:rgba(235,248,240,0.85);margin-top:6px;font-size:13px}
    .team-card{background:linear-gradient(180deg, rgba(255,255,255,0.016), rgba(255,255,255,0.01));border-radius:10px;padding:12px;margin:6px;border:1px solid rgba(255,255,255,0.03)}
    .team-name{font-weight:700;color:var(--cream)}
    .team-role{color:rgba(235,248,240,0.8);font-size:13px;margin-top:4px}
    .small-muted{font-size:13px;color:rgba(235,248,240,0.78)}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -------------------------
# Helpers: data, metrics, IO
# -------------------------
@st.cache_data(ttl=3600)
def load_csv_cached(path_like):
    return pd.read_csv(path_like)

def load_data(uploaded):
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded)
        except Exception as e:
            st.error("Error reading uploaded CSV: " + str(e))
            return None
    if DEFAULT_CSV.exists():
        try:
            return pd.read_csv(DEFAULT_CSV)
        except Exception:
            pass
    st.warning(f"No dataset found. Upload a CSV using the sidebar or push starbucks.csv to the repo root.")
    return None

def clean_numeric_like_columns(df: pd.DataFrame):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            s = s.str.replace(r"[^\d\.\-]", "", regex=True)
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().sum() >= 0.5 * len(coerced):
                df[c] = coerced
    return df

def auto_encode_features(X: pd.DataFrame):
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": float(mse), "RMSE": float(rmse), "R2": float(r2)}

def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {"Accuracy": float(acc), "Precision": float(prec), "Recall": float(rec), "F1": float(f1)}

def save_pickle(obj, out_path: Path):
    try:
        with open(out_path, "wb") as f:
            pickle.dump(obj, f)
        return True
    except Exception as e:
        st.error(f"Failed to save {out_path.name}: {e}")
        return False

# -------------------------
# Image loader - safe bytes
# -------------------------
def get_logo_bytes(preferred_paths):
    """
    preferred_paths: list of Path objects to try in order.
    returns (bytes, mime) or (None, None)
    """
    for p in preferred_paths:
        try:
            if p and p.exists():
                data = p.read_bytes()
                suffix = p.suffix.lower()
                mime = "image/png" if suffix.endswith("png") else ("image/jpeg" if suffix.endswith(("jpg","jpeg")) else "image/png")
                return data, mime
        except Exception:
            continue
    return None, None

# -------------------------
# App
# -------------------------
def main():
    st.set_page_config(page_title="Starbucks Pro — Emerald", layout="wide", initial_sidebar_state="expanded")
    inject_emerald_css()

    # Sidebar: only CSV upload & page navigation (minimal/professional)
    st.sidebar.title("")  # compact
    uploaded_file = st.sidebar.file_uploader("Upload Starbucks CSV", type=["csv"], key="csv_upload")
    page = st.sidebar.radio("", ["Overview", "EDA (All Charts)", "Modeling & Comparison", "Prediction Playground", "Export Report", "About / Team"])

    # Header
    col1, col2 = st.columns([0.78, 0.22])
    with col1:
        st.markdown("<div class='header-container'>", unsafe_allow_html=True)
        st.markdown("<div><h1 class='app-title'>☕ Starbucks Pro — Analytics & ML</h1></div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-left:6px'><div class='subtitle'>Emerald theme • EDA • Modeling • Save ALL models</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        # small decorative animation or svg fallback
        if LOTTIE_AVAILABLE:
            try:
                l = "https://assets2.lottiefiles.com/packages/lf20_0yfsb3a1.json"
                j = None
                try:
                    j = get_logo_bytes([Path("small_lottie.json")])  # not likely, skip
                except Exception:
                    j = None
                st.markdown("", unsafe_allow_html=True)
            except Exception:
                pass

    st.markdown("---")

    # Load data
    df = load_data(uploaded_file)
    if df is None:
        st.stop()

    # Clean numeric-like
    df = clean_numeric_like_columns(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # ---------- Overview ----------
    if page == "Overview":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Dataset Overview")
        st.dataframe(df.head(8))
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Numeric cols", len(numeric_cols))
        st.write("Missing values (descending):")
        st.dataframe(df.isnull().sum().sort_values(ascending=False).to_frame("missing"))
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- EDA ----------
    elif page == "EDA (All Charts)":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Exploratory Data Analysis — All Charts")
        st.write("Choose columns and inspect plots + interpretation.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Distribution & Boxplot
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Distribution & Outlier Analysis")
        if numeric_cols:
            sel_col = st.selectbox("Numeric column", numeric_cols, index=0)
            fig, ax = plt.subplots(figsize=(9,3))
            ax.hist(df[sel_col].dropna(), bins=28, color="#0b6b48", edgecolor="#083", alpha=0.95)
            ax.set_xlabel(sel_col); ax.set_ylabel("Frequency"); ax.set_title(f"Distribution of {sel_col}")
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(9,2.6))
            sns.boxplot(x=df[sel_col].dropna(), ax=ax2, color="#cdeccf")
            ax2.set_xlabel(sel_col); ax2.set_title(f"Boxplot of {sel_col}")
            st.pyplot(fig2)
        else:
            st.info("No numeric columns available.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Scatter
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Scatter Plot")
        if len(numeric_cols) >= 2:
            sc_x = st.selectbox("X-axis", numeric_cols, index=0, key="scx")
            sc_y = st.selectbox("Y-axis", numeric_cols, index=1, key="scy")
            fig_s, ax_s = plt.subplots(figsize=(8,4))
            ax_s.scatter(df[sc_x], df[sc_y], s=48, alpha=0.75, edgecolor="#fff", linewidth=0.4, color="#ffd59e")
            ax_s.set_xlabel(sc_x); ax_s.set_ylabel(sc_y); ax_s.set_title(f"{sc_y} vs {sc_x}")
            st.pyplot(fig_s)
        else:
            st.info("Need at least two numeric columns.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Categorical pie & bar
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Categorical Distribution")
        if categorical_cols:
            cat_sel = st.selectbox("Categorical column", categorical_cols, key="catselect")
            counts = df[cat_sel].value_counts()
            top = counts.head(8)
            fig_p, ax_p = plt.subplots(figsize=(6,4))
            ax_p.pie(top, labels=top.index, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("Greens"))
            ax_p.set_title(f"Top categories — {cat_sel}")
            st.pyplot(fig_p)

            fig_b, ax_b = plt.subplots(figsize=(8,3))
            top.plot.bar(ax=ax_b, color="#0b6b48")
            ax_b.set_xlabel(cat_sel); ax_b.set_ylabel("Count"); ax_b.set_title(f"Counts — {cat_sel}")
            st.pyplot(fig_b)
        else:
            st.info("No categorical columns detected.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Correlation heatmap
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Correlation Heatmap")
        if numeric_cols and len(numeric_cols) > 1:
            fig_h, ax_h = plt.subplots(figsize=(10,6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="YlGn", fmt=".2f", ax=ax_h, linewidths=0.4)
            ax_h.set_title("Correlation Heatmap")
            st.pyplot(fig_h)
        else:
            st.info("Need at least 2 numeric columns.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Modeling & Comparison ----------
    elif page == "Modeling & Comparison":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model Training & Comparison (fixed 80/20 split)")
        st.write("Categorical features are auto-encoded. After training, all models are saved as .pkl files and available to download.")
        st.markdown("</div>", unsafe_allow_html=True)

        default_target = "Calories" if "Calories" in df.columns else (numeric_cols[0] if numeric_cols else df.columns[0])
        target = st.selectbox("Target column (what to predict):", options=list(df.columns), index=list(df.columns).index(default_target) if default_target in df.columns else 0)
        problem = "classification" if df[target].dtype == object or df[target].dtype.name == "category" else "regression"
        st.info("Detected problem type: " + problem)

        feature_options = [c for c in df.columns if c != target]
        features = st.multiselect("Select features (can include categorical):", options=feature_options, default=feature_options[:6])

        if not features:
            st.warning("Please select at least 1 feature.")
        else:
            if st.button("Train & Compare Models"):
                X = df[features].copy()
                y = df[target].copy()
                X_enc = auto_encode_features(X)

                # Save feature names for later alignment
                features_encoded = X_enc.columns.tolist()
                with open(MODEL_DIR / "features_encoded.json", "w", encoding="utf-8") as f:
                    json.dump(features_encoded, f)

                # Encode y if classification
                label_encoder = None
                if problem == "classification":
                    label_encoder = LabelEncoder()
                    y_enc = label_encoder.fit_transform(y.astype(str))
                    # save encoder
                    save_pickle(label_encoder, MODEL_DIR / "label_encoder.pkl")
                else:
                    y_enc = y

                # Fixed split
                X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.2, random_state=42, stratify=None if problem=="regression" else y_enc)

                trained = {}

                # Train models and save each model to MODEL_DIR
                if problem == "regression":
                    # Linear Regression
                    lr = LinearRegression(); lr.fit(X_train, y_train)
                    trained["linear_regression"] = {"label": "Linear Regression", "model": lr, "pred": lr.predict(X_test)}
                    joblib.dump(lr, MODEL_DIR / "linear_regression.pkl")

                    # Random Forest Regressor
                    rf = RandomForestRegressor(n_estimators=200, random_state=42); rf.fit(X_train, y_train)
                    trained["random_forest_regressor"] = {"label": "Random Forest", "model": rf, "pred": rf.predict(X_test)}
                    joblib.dump(rf, MODEL_DIR / "random_forest_regressor.pkl")

                else:
                    # Logistic Regression
                    log = LogisticRegression(max_iter=2000); log.fit(X_train, y_train)
                    trained["logistic_regression"] = {"label": "Logistic Regression", "model": log, "pred": log.predict(X_test)}
                    joblib.dump(log, MODEL_DIR / "logistic_regression.pkl")

                    # Random Forest Classifier
                    rfc = RandomForestClassifier(n_estimators=200, random_state=42); rfc.fit(X_train, y_train)
                    trained["random_forest_classifier"] = {"label": "Random Forest", "model": rfc, "pred": rfc.predict(X_test)}
                    joblib.dump(rfc, MODEL_DIR / "random_forest_classifier.pkl")

                # Save a metadata JSON describing saved models
                saved_models = [str(p.name) for p in MODEL_DIR.glob("*.pkl")]
                with open(MODEL_DIR / "saved_models.json", "w", encoding="utf-8") as f:
                    json.dump(saved_models, f)

                # Metrics table
                rows = []
                for key, info in trained.items():
                    preds = info["pred"]
                    if problem == "regression":
                        mets = regression_metrics(y_test, preds)
                    else:
                        mets = classification_metrics(y_test, preds)
                    rows.append({"Model": info["label"], **mets})

                    # Keep model in session for Prediction Playground
                    st.session_state[f"model_{key}"] = info["model"]

                metrics_df = pd.DataFrame(rows).set_index("Model")
                st.subheader("Model Comparison")
                st.dataframe(metrics_df.style.format("{:.4f}"))

                # Visuals & model insights
                st.markdown("---")
                for key, info in trained.items():
                    st.markdown(f"#### {info['label']}")
                    preds = info["pred"]
                    if problem == "regression":
                        fig_r, ax_r = plt.subplots(figsize=(7,4))
                        ax_r.scatter(y_test, preds, s=60, alpha=0.75, edgecolor="#ffffff", color="#ffd59e")
                        mn = min(min(y_test), min(preds)); mx = max(max(y_test), max(preds))
                        ax_r.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
                        ax_r.set_xlabel("Actual"); ax_r.set_ylabel("Predicted")
                        st.pyplot(fig_r)
                        model_obj = info["model"]
                        if hasattr(model_obj, "feature_importances_"):
                            fi = model_obj.feature_importances_
                            feat_names = X_enc.columns.tolist()
                            fi_df = pd.DataFrame({"feature": feat_names, "importance": fi}).sort_values("importance", ascending=False)
                            st.write("Top feature importances (Random Forest):")
                            st.table(fi_df.head(10).reset_index(drop=True))
                    else:
                        cm = confusion_matrix(y_test, preds)
                        fig_c, ax_c = plt.subplots(figsize=(6,4))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax_c)
                        ax_c.set_xlabel("Predicted"); ax_c.set_ylabel("Actual")
                        st.pyplot(fig_c)
                        st.text(classification_report(y_test, preds, zero_division=0))

                # Provide model download buttons right after training
                st.markdown("---")
                st.write("Saved model files (download links):")
                for p in sorted(MODEL_DIR.glob("*.pkl")):
                    try:
                        with open(p, "rb") as f:
                            data = f.read()
                        st.download_button(label=f"Download {p.name}", data=data, file_name=p.name, mime="application/octet-stream")
                    except Exception as e:
                        st.error(f"Could not prepare download for {p.name}: {e}")

    # ---------- Prediction Playground ----------
    elif page == "Prediction Playground":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Playground")
        if "last_train" not in st.session_state and not any(MODEL_DIR.glob("*.pkl")):
            st.warning("Train models first in Modeling & Comparison page (this saves models to disk).")
        else:
            # Collect saved models names
            model_files = sorted(MODEL_DIR.glob("*.pkl"))
            model_names = [p.stem for p in model_files]
            if not model_names:
                st.info("No saved models found in environment. Train and save models first.")
            else:
                choice = st.selectbox("Choose model to load & predict", model_names)
                # Load features list
                features_encoded = []
                fpath = MODEL_DIR / "features_encoded.json"
                if fpath.exists():
                    try:
                        features_encoded = json.load(open(fpath, "r", encoding="utf-8"))
                    except Exception:
                        features_encoded = []
                # Build input form from features (we'll fall back to dataframe columns if not encoded list)
                input_features = features_encoded if features_encoded else (df.columns.drop([df.columns[0]])[:6].tolist() if df.shape[1] > 1 else df.columns.tolist())
                st.write("Enter values for features (numeric medians used if available):")
                cols = st.columns(2)
                user_vals = {}
                for i, f in enumerate(input_features):
                    default = float(df[f].median()) if f in df.columns and pd.api.types.is_numeric_dtype(df[f]) else 0.0
                    user_vals[f] = cols[i % 2].number_input(f, value=default, format="%.4f")

                if st.button("Predict"):
                    model_path = MODEL_DIR / f"{choice}.pkl"
                    if not model_path.exists():
                        st.error("Model file not found on disk.")
                    else:
                        mdl = joblib.load(model_path)
                        Xnew = pd.DataFrame([user_vals])
                        Xnew_enc = auto_encode_features(Xnew)
                        # Align columns
                        if features_encoded:
                            for col in features_encoded:
                                if col not in Xnew_enc.columns:
                                    Xnew_enc[col] = 0
                            Xnew_enc = Xnew_enc[features_encoded]
                        pred = mdl.predict(Xnew_enc)[0]
                        st.success(f"Predicted (raw): {pred}")
                        if hasattr(mdl, "predict_proba"):
                            proba = mdl.predict_proba(Xnew_enc)[0]
                            classes = getattr(mdl, "classes_", list(range(len(proba))))
                            st.table(pd.DataFrame({"class": classes, "probability": proba}).sort_values("probability", ascending=False).reset_index(drop=True))
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Export Report ----------
    elif page == "Export Report":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Export ASCII-safe PDF Report & Model Downloads")
        st.write("This page summarizes EDA snapshots and lists saved model files for download.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Simple EDA snapshot
        tmp_dir = Path(tempfile.gettempdir()) / "starbucks_report_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        images = []
        if numeric_cols:
            col0 = numeric_cols[0]
            fig, ax = plt.subplots(figsize=(7,3))
            ax.hist(df[col0].dropna(), bins=25, color="#0b6b48")
            ax.set_xlabel(col0); ax.set_ylabel("Frequency"); ax.set_title(f"Distribution of {col0}")
            p = tmp_dir / "hist.png"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            images.append((p, f"Histogram: {col0}"))

        st.write("Saved model files in environment:")
        for p in sorted(MODEL_DIR.glob("*.pkl")):
            try:
                with open(p, "rb") as f:
                    data = f.read()
                st.download_button(label=f"Download {p.name}", data=data, file_name=p.name, mime="application/octet-stream")
            except Exception as e:
                st.error(f"Could not prepare download for {p.name}: {e}")

        st.markdown("If you want a combined report (PDF), install `fpdf` in your runtime and use the Generate PDF button (not included by default here).")

    # ---------- About / Team ----------
    elif page == "About / Team":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("About & Team — Rajagiri (Grand Thornton Add-On)")

        colA, colB = st.columns([0.48, 0.52])
        with colA:
            st.markdown('<div class="about-logos-row">', unsafe_allow_html=True)

            # Prepare lists of potential paths to try for each logo
            rcss_candidates = [RCSS_REPO, RCSS_FALLBACK]
            grant_candidates = [GRANT_REPO, GRANT_FALLBACK]

            rcss_bytes, rcss_mime = get_logo_bytes(rcss_candidates)
            grant_bytes, grant_mime = get_logo_bytes(grant_candidates)

            if rcss_bytes:
                b64 = base64.b64encode(rcss_bytes).decode()
                st.markdown(f'<div class="logo-card"><img src="data:{rcss_mime};base64,{b64}" width="140"/></div>', unsafe_allow_html=True)
                st.markdown('<div class="logo-caption">Rajagiri College of Social Sciences</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="logo-card"><div class="small-muted">RCSS logo not found</div></div>', unsafe_allow_html=True)

            if grant_bytes:
                b64g = base64.b64encode(grant_bytes).decode()
                st.markdown(f'<div class="logo-card"><img src="data:{grant_mime};base64,{b64g}" width="140"/></div>', unsafe_allow_html=True)
                st.markdown('<div class="logo-caption">Grant Thornton (Add-On Partner)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="logo-card"><div class="small-muted">Grant Thornton logo not found</div></div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with colB:
            # Lottie or fallback cup
            if LOTTIE_AVAILABLE:
                lottie_urls = ["https://assets2.lottiefiles.com/packages/lf20_0yfsb3a1.json", "https://assets7.lottiefiles.com/packages/lf20_tutvdkg0.json"]
                loaded = None
                for u in lottie_urls:
                    try:
                        loaded = st_lottie  # no-op; we'll attempt to call directly below
                        break
                    except Exception:
                        loaded = None
                # to avoid complexity, display static SVG fallback here:
                st.markdown("<div style='padding:8px 0 12px 0'><img src='https://img.icons8.com/ios-filled/480/ffffff/developer.png' width='300' style='opacity:0.95'></div>", unsafe_allow_html=True)

            st.markdown("<div style='padding-top:6px'><strong>Project:</strong> Starbucks Pro — Analytics & Machine Learning Dashboard</div>", unsafe_allow_html=True)
            st.markdown("<div class='small-muted'>Capstone-style project as part of Grant Thornton add-on at Rajagiri College of Social Sciences. Focus: practical analytics & ML skills, industry alignment.</div>", unsafe_allow_html=True)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='team-card'>", unsafe_allow_html=True)
            st.markdown("<div class='team-name'>Rajagiri College of Social Sciences</div>", unsafe_allow_html=True)
            st.markdown("<div class='team-role'>Kalamassery, Kerala — Social Sciences & Professional Programs</div>", unsafe_allow_html=True)
            st.markdown("<div class='small-muted' style='margin-top:8px'>RCSS emphasizes research-led learning and industry collaborations with a focus on holistic student development.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='team-card'>", unsafe_allow_html=True)
            st.markdown("<div class='team-name'>Grand Thornton Add-On</div>", unsafe_allow_html=True)
            st.markdown("<div class='team-role'>Industry-focused analytics & ML program</div>", unsafe_allow_html=True)
            st.markdown("<div class='small-muted' style='margin-top:8px'>Hands-on modules on data pipelines, visualization, modeling, and business interpretation.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c3:
            st.markdown("<div class='team-card'>", unsafe_allow_html=True)
            st.markdown("<div class='team-name'>Faculty Guide</div>", unsafe_allow_html=True)
            st.markdown("<div class='team-role'>Butchi Babu Muvva</div>", unsafe_allow_html=True)
            st.markdown("<div class='small-muted' style='margin-top:8px'>Mentor for the add-on course and project guide.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Project Contributors")
        members = [
            ("Sheba Alien Lalu", "Team Lead / Data & ML", "https://www.linkedin.com/in/sheba-lalu-b59b3a398"),
            ("Shanum Rabia", "Data Engineer", None),
            ("Abhinav S Kumar", "Modeling & Backend", None),
            ("Johnathan Joy", "Frontend & Visualization", None),
            ("Chackochan Siju", "Research & Documentation", None),
            ("Dhyanjith P", "QA & Deployment", None)
        ]
        cols = st.columns(3)
        for i, (name, role, linkedin) in enumerate(members):
            col = cols[i % 3]
            with col:
                st.markdown("<div class='team-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='team-name'>{name}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='team-role'>{role}</div>", unsafe_allow_html=True)
                if linkedin:
                    st.markdown(f"<div style='margin-top:8px'><a href='{linkedin}' target='_blank' style='color:#dbeee2;'>LinkedIn</a></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Credits & Contact**")
        st.markdown("- Built by the contributors listed above as part of the Grand Thornton Add-On at Rajagiri College of Social Sciences.")
        st.markdown("<div class='small-muted'>Built with collaboration, learning, and lots of coffee ☕</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer watermark (optional)
    try:
        if DEFAULT_LOGO.exists():
            b = base64.b64encode(DEFAULT_LOGO.read_bytes()).decode()
            st.markdown(f'<img src="data:image/png;base64,{b}" style="position:fixed;right:18px;bottom:18px;opacity:0.12;width:150px;">', unsafe_allow_html=True)
    except Exception:
        pass

if __name__ == "__main__":
    main()
