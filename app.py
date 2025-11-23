# app.py
"""
Starbucks Pro — Analytics & ML (Emerald Theme) — About UI enhanced (Side-by-side logos)
- Sidebar reduced to only: CSV upload & page navigation (tabs)
- About page: Rajagiri + Grant Thornton logos side-by-side (140px each)
- LinkedIn added under Sheba's name
- Uses local logo paths:
    C:/Users/SHEBA/Documents/rajagiri.png
    C:/Users/SHEBA/Documents/grand.jpeg
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import tempfile
import json
from pathlib import Path
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder

# Optional libs
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except Exception:
    LOTTIE_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# -------------------------
# Default user paths (from conversation)
# -------------------------
DEFAULT_CSV = Path("C:/Users/SHEBA/Documents/starbucks.csv")
DEFAULT_LOGO = Path("C:/Users/SHEBA/Documents/starbucks.png")

# User-provided logos (use the exact paths you requested)
RCSS_LOCAL = Path(r"C:/Users/SHEBA/Documents/rajagiri.png")
GRANT_LOCAL = Path(r"C:/Users/SHEBA/Documents/grand.jpeg")

# Fallback container images (if exist in /mnt/data)
RCSS_FALLBACK = Path("/mnt/data/fbed8862-e4ff-4218-8008-8ad9bbed415e.png")
GRANT_FALLBACK = Path("/mnt/data/e0e0713b-c281-4a62-b35a-623830e23dcf.png")

# -------------------------
# Styling: Emerald Theme (CSS)
# -------------------------
def inject_emerald_css():
    css = r"""
    <style>
    :root{
      --emerald-1: #0b6b48;
      --emerald-2: #0e8a5b;
      --cream: #fbfdfb;
    }
    .stApp {
      background: linear-gradient(180deg, #06301f 0%, #0b2e22 60%, #071a13 100%) !important;
      color: var(--cream);
      min-height:100vh;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .card {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border-radius: 12px;
      padding: 16px;
      margin-bottom:16px;
      border: 1px solid rgba(255,255,255,0.04);
      box-shadow: 0 6px 18px rgba(3,14,8,0.35);
    }
    .header-container { display:flex; align-items:center; gap:14px; margin-bottom:10px; }
    .app-title { font-size:28px !important; font-weight:700 !important; color: var(--cream) !important; }
    .subtitle { font-size:13px; color: rgba(235,248,240,0.88); }

    /* About logos side-by-side */
    .about-logos-row {
      display:flex;
      align-items:center;
      gap:28px;
      justify-content:flex-start;
      flex-wrap:wrap;
      margin-bottom:8px;
    }
    .logo-card {
      background: rgba(255,255,255,0.03);
      border-radius:12px;
      padding:12px;
      display:flex;
      align-items:center;
      justify-content:center;
      width:160px;
      height:160px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.45);
    }
    .logo-caption { text-align:center; color: rgba(235,248,240,0.85); margin-top:8px; font-size:13px; }
    .team-card {
      background: linear-gradient(180deg, rgba(255,255,255,0.016), rgba(255,255,255,0.01));
      border-radius: 10px; padding: 12px; margin:6px; border:1px solid rgba(255,255,255,0.03);
    }
    .team-name { font-weight:700; color:var(--cream); }
    .team-role { color: rgba(235,248,240,0.8); font-size:13px; margin-top:4px; }
    .small-muted { font-size:13px; color: rgba(235,248,240,0.78); }
    /* Minimal sidebar style (clean) */
    .block-container .sidebar .element-container { padding-top:6px; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# -------------------------
# Animated cup fallback (SVG + CSS keyframes)
# -------------------------
CUP_SVG_FALLBACK = """
<div style="display:flex;align-items:center;justify-content:center;padding:6px 0 10px 0">
  <svg width="140" height="120" viewBox="0 0 120 110" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="g1" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="#9be9c8"/>
        <stop offset="100%" stop-color="#0b6b48"/>
      </linearGradient>
    </defs>
    <g transform="translate(8,8)">
      <path d="M10 70 Q10 30 40 30 H60 Q90 30 90 70 Q90 90 40 90 H25 Q10 90 10 70 Z" fill="url(#g1)" stroke="#073" stroke-width="1.2"/>
      <ellipse cx="50" cy="28" rx="40" ry="6" fill="#0b6b48" opacity="0.12"/>
      <g id="steam" fill="#ffffff" opacity="0.85">
        <path d="M25 10 C30 5 30 25 35 10" stroke="rgba(255,255,255,0.8)" stroke-width="1.4" fill="none"/>
        <path d="M45 8 C50 2 50 22 55 8" stroke="rgba(255,255,255,0.8)" stroke-width="1.4" fill="none"/>
      </g>
    </g>
  </svg>
</div>
<style>
@keyframes bob { 0% { transform: translateY(0px); } 50% { transform: translateY(-6px); } 100% { transform: translateY(0px); } }
svg { animation: bob 2.8s ease-in-out infinite; }
</style>
"""

# -------------------------
# Helpers (data, pdf, encoding)
# -------------------------
@st.cache_data(ttl=3600)
def load_csv_cached(path_like):
    return pd.read_csv(path_like)

def load_data(uploaded):
    # priority: uploaded -> default path -> warn
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded)
        except Exception as e:
            st.error("Error reading uploaded file: " + str(e))
            return None
    if DEFAULT_CSV.exists():
        try:
            return pd.read_csv(DEFAULT_CSV)
        except Exception:
            pass
    st.warning(f"No dataset found. Upload a CSV using the sidebar or place it at: {DEFAULT_CSV}")
    return None

def sanitize_ascii(text: str) -> str:
    if text is None:
        return ""
    return str(text).encode("ascii", errors="replace").decode("ascii")

def build_pdf_ascii(out_path: Path, title: str, df_info: dict, images: list, model_results: dict):
    if not FPDF_AVAILABLE:
        raise RuntimeError("fpdf not available")
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, sanitize_ascii(title), ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, sanitize_ascii(f"Rows: {df_info.get('rows')}, Columns: {df_info.get('cols')}"), ln=True)
    pdf.ln(6)
    for img_path, caption in images:
        try:
            if Path(img_path).exists():
                pdf.image(str(img_path), w=170)
                pdf.ln(3)
                pdf.set_font("Arial", "", 10)
                pdf.multi_cell(0, 5, sanitize_ascii(caption))
                pdf.ln(4)
        except Exception:
            continue
    if model_results:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Model Metrics", ln=True)
        pdf.ln(4)
        pdf.set_font("Arial", "", 10)
        for name, mets in model_results.items():
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 6, sanitize_ascii(name), ln=True)
            pdf.set_font("Arial", "", 10)
            for k, v in mets.items():
                pdf.cell(0, 5, sanitize_ascii(f"{k}: {v:.4f}"), ln=True)
            pdf.ln(3)
    pdf.output(str(out_path))
    return out_path

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

# -------------------------
# Lottie helper
# -------------------------
def try_load_lottie(path_or_url: str):
    try:
        p = Path(path_or_url)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        import requests
        r = requests.get(path_or_url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

# -------------------------
# App
# -------------------------
def main():
    st.set_page_config(page_title="Starbucks Pro — Emerald", layout="wide", initial_sidebar_state="expanded")

    # Inject styling
    inject_emerald_css()

    # -------------------------
    # Sidebar: minimal & professional
    # Only: CSV upload + Page navigation radio (tabs)
    # -------------------------
    st.sidebar.title("")  # keep sidebar compact
    uploaded_file = st.sidebar.file_uploader("Upload Starbucks CSV", type=["csv"], key="csv_upload")
    # Navigation lives on the main layout but we also provide it in the sidebar as required by Streamlit layout expectation:
    page = st.sidebar.radio("", ["Overview", "EDA (All Charts)", "Modeling & Comparison", "Prediction Playground", "Export Report", "About / Team"])

    # Header (main)
    col1, col2 = st.columns([0.78, 0.22])
    with col1:
        st.markdown("<div class='header-container'>", unsafe_allow_html=True)
        st.markdown("<div><h1 class='app-title'>☕ Starbucks Pro — Analytics & ML</h1></div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-left:6px'><div class='subtitle'>Professional Emerald theme • EDA • Modeling • Predictions</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        # show small decorative cup (svg or lottie) at header right
        if LOTTIE_AVAILABLE:
            lottie_url = "https://assets2.lottiefiles.com/packages/lf20_0yfsb3a1.json"
            lottie_json = try_load_lottie(lottie_url)
            if lottie_json:
                try:
                    st_lottie(lottie_json, height=80, key="header_lottie")
                except Exception:
                    st.markdown(CUP_SVG_FALLBACK, unsafe_allow_html=True)
            else:
                st.markdown(CUP_SVG_FALLBACK, unsafe_allow_html=True)
        else:
            st.markdown(CUP_SVG_FALLBACK, unsafe_allow_html=True)

    st.markdown("---")

    # Load dataset
    df = load_data(uploaded_file)
    if df is None:
        st.stop()

    # initial cleaning
    df = clean_numeric_like_columns(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # ---------- Pages ----------
    if page == "Overview":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Dataset Overview")
        st.write("Quick preview and dataset summary.")
        st.dataframe(df.head(8))
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Numeric cols", len(numeric_cols))
        st.write("Missing values (descending):")
        st.dataframe(df.isnull().sum().sort_values(ascending=False).to_frame("missing"))
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "EDA (All Charts)":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Exploratory Data Analysis — All Charts")
        st.write("Choose columns and inspect plots + interpretation.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Distribution & boxplot
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 1) Distribution & Outlier Analysis")
        if numeric_cols:
            sel_col = st.selectbox("Numeric column for distribution & outlier analysis", numeric_cols, index=0)
            fig, ax = plt.subplots(figsize=(9,3))
            ax.hist(df[sel_col].dropna(), bins=28, color="#0b6b48", edgecolor="#083", linewidth=0.6, alpha=0.95)
            ax.set_title(f"Distribution of {sel_col}", fontsize=12)
            ax.set_xlabel(sel_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(9,2.6))
            sns.boxplot(x=df[sel_col].dropna(), ax=ax2, color="#cdeccf")
            ax2.set_title(f"Boxplot of {sel_col}")
            ax2.set_xlabel(sel_col)
            st.pyplot(fig2)

            st.write("**Explanation:** Histogram shows common ranges and skewness. Boxplot shows median, IQR and outliers.")
        else:
            st.info("No numeric columns available.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Scatter
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 2) Scatter Plot — Relationship Between Two Numerics")
        if len(numeric_cols) >= 2:
            sc_x = st.selectbox("X-axis", numeric_cols, index=0, key="scx")
            sc_y = st.selectbox("Y-axis", numeric_cols, index=1, key="scy")
            fig_s, ax_s = plt.subplots(figsize=(8,4))
            ax_s.scatter(df[sc_x], df[sc_y], s=48, alpha=0.75, edgecolor="#fff", linewidth=0.4, color="#ffd59e")
            ax_s.set_xlabel(sc_x)
            ax_s.set_ylabel(sc_y)
            ax_s.set_title(f"{sc_y} vs {sc_x}")
            st.pyplot(fig_s)
            st.write("**Explanation:** Useful to see linear/non-linear relations and clusters.")
        else:
            st.info("Need at least two numeric columns.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Categorical pie & bar
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 3) Categorical Distribution (Pie + Bar)")
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
            ax_b.set_xlabel(cat_sel)
            ax_b.set_ylabel("Count")
            ax_b.set_title(f"Counts — top values of {cat_sel}")
            st.pyplot(fig_b)

            st.write("**Explanation:** Identify dominant classes and potential imbalances before classification modeling.")
        else:
            st.info("No categorical columns detected.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Correlation heatmap
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 4) Correlation Heatmap")
        if numeric_cols and len(numeric_cols) > 1:
            fig_h, ax_h = plt.subplots(figsize=(10,6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="YlGn", fmt=".2f", ax=ax_h, linewidths=0.4)
            ax_h.set_title("Correlation Heatmap")
            ax_h.set_xlabel("Features")
            ax_h.set_ylabel("Features")
            st.pyplot(fig_h)
            st.write("**Explanation:** Look for strongly correlated features.")
        else:
            st.info("Need at least 2 numeric columns.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Modeling & Comparison":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model Training & Comparison (fixed 80/20 split)")
        st.write("Select target and features. Categorical features are auto-encoded.")
        st.markdown("</div>", unsafe_allow_html=True)

        default_target = "Calories" if "Calories" in df.columns else (numeric_cols[0] if numeric_cols else df.columns[0])
        target = st.selectbox("Target column (what to predict):", options=list(df.columns), index=list(df.columns).index(default_target) if default_target in df.columns else 0)
        problem = "classification" if df[target].dtype == object or df[target].dtype.name == "category" else "regression"
        st.info("Detected problem type: " + problem)

        feature_options = [c for c in df.columns if c != target]
        features = st.multiselect("Select features (can include categorical):", options=feature_options, default=feature_options[:6])

        if not features:
            st.warning("Please select at least 1 feature to proceed.")
        else:
            if st.button("Train & Compare Models"):
                X = df[features].copy()
                y = df[target].copy()

                X_enc = auto_encode_features(X)

                if problem == "classification":
                    lbl = LabelEncoder()
                    y_enc = lbl.fit_transform(y.astype(str))
                else:
                    y_enc = y

                X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.2, random_state=42, stratify=None if problem=="regression" else y_enc)

                trained = {}
                if problem == "regression":
                    lr = LinearRegression(); lr.fit(X_train, y_train); trained["Linear Regression"] = {"model": lr, "pred": lr.predict(X_test)}
                    rf = RandomForestRegressor(n_estimators=200, random_state=42); rf.fit(X_train, y_train); trained["Random Forest"] = {"model": rf, "pred": rf.predict(X_test)}
                else:
                    log = LogisticRegression(max_iter=2000); log.fit(X_train, y_train); trained["Logistic Regression"] = {"model": log, "pred": log.predict(X_test)}
                    rfc = RandomForestClassifier(n_estimators=200, random_state=42); rfc.fit(X_train, y_train); trained["Random Forest"] = {"model": rfc, "pred": rfc.predict(X_test)}

                rows = []
                for name, info in trained.items():
                    preds = info["pred"]
                    mets = regression_metrics(y_test, preds) if problem=="regression" else classification_metrics(y_test, preds)
                    rows.append({"Model": name, **mets})
                    st.session_state[f"model_{name}"] = info["model"]

                metrics_df = pd.DataFrame(rows).set_index("Model")
                st.subheader("Model Comparison")
                st.dataframe(metrics_df.style.format("{:.4f}"))

                st.markdown("---")
                st.subheader("Model Visuals & Interpretations")
                for name, info in trained.items():
                    st.markdown(f"#### {name}")
                    preds = info["pred"]
                    if problem == "regression":
                        fig_r, ax_r = plt.subplots(figsize=(7,4))
                        ax_r.scatter(y_test, preds, s=60, alpha=0.75, edgecolor="#ffffff", color="#ffd59e")
                        mn = min(min(y_test), min(preds)); mx = max(max(y_test), max(preds))
                        ax_r.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
                        ax_r.set_xlabel("Actual"); ax_r.set_ylabel("Predicted"); ax_r.set_title(f"{name} — Actual vs Predicted")
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
                        ax_c.set_xlabel("Predicted"); ax_c.set_ylabel("Actual"); ax_c.set_title(f"{name} — Confusion Matrix")
                        st.pyplot(fig_c)
                        st.write("Classification Report:")
                        st.text(classification_report(y_test, preds, zero_division=0))

                st.session_state["last_train"] = {
                    "features": features,
                    "features_encoded": X_enc.columns.tolist(),
                    "target": target,
                    "problem": problem,
                    "X_test": X_test,
                    "y_test": y_test,
                    "models": list(trained.keys())
                }

    elif page == "Prediction Playground":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Playground")
        if "last_train" not in st.session_state:
            st.warning("Train models first in Modeling & Comparison page.")
        else:
            meta = st.session_state["last_train"]
            models = meta["models"]
            choice = st.selectbox("Choose model for prediction", models)
            st.write("Enter feature values (numeric medians used as default where available).")
            cols = st.columns(2)
            user_vals = {}
            for i, f in enumerate(meta["features"]):
                default = float(df[f].median()) if f in df.columns and pd.api.types.is_numeric_dtype(df[f]) else 0.0
                user_vals[f] = cols[i % 2].number_input(f, value=default, format="%.3f")

            if st.button("Predict"):
                mdl = st.session_state.get(f"model_{choice}")
                if mdl is None:
                    st.error("Model not found. Re-train models.")
                else:
                    Xnew = pd.DataFrame([user_vals])
                    Xnew_enc = auto_encode_features(Xnew)
                    # align columns
                    for col in meta["features_encoded"]:
                        if col not in Xnew_enc.columns:
                            Xnew_enc[col] = 0
                    Xnew_enc = Xnew_enc[meta["features_encoded"]]
                    pred = mdl.predict(Xnew_enc)[0]
                    if meta["problem"] == "regression":
                        st.success(f"Predicted {meta['target']}: {pred:.3f}")
                        fig_pr, ax_pr = plt.subplots(figsize=(4,3))
                        ax_pr.bar([f"Predicted {meta['target']}"], [pred], color="#0b6b48")
                        ax_pr.set_ylabel(meta['target'])
                        st.pyplot(fig_pr)
                    else:
                        st.success(f"Predicted class (encoded): {pred}")
                        if hasattr(mdl, "predict_proba"):
                            probs = mdl.predict_proba(Xnew_enc)[0]
                            classes = getattr(mdl, "classes_", list(range(len(probs))))
                            proba_df = pd.DataFrame({"class": classes, "probability": probs})
                            st.table(proba_df.sort_values("probability", ascending=False).reset_index(drop=True))
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Export Report":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Export ASCII-safe PDF Report")
        st.write("Generates a simple PDF with EDA snapshots and model metrics (requires `fpdf`).")
        st.markdown("</div>", unsafe_allow_html=True)

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
            images.append((p, f"Histogram: {col0} distribution"))

        model_results = {}
        if "last_train" in st.session_state:
            meta = st.session_state["last_train"]
            for name in meta["models"]:
                mdl = st.session_state.get(f"model_{name}")
                if mdl is not None:
                    preds = mdl.predict(meta["X_test"])
                    model_results[name] = regression_metrics(meta["y_test"], preds) if meta["problem"]=="regression" else classification_metrics(meta["y_test"], preds)

        if st.button("Generate PDF"):
            if not FPDF_AVAILABLE:
                st.error("Install fpdf to enable PDF export: python -m pip install fpdf")
            else:
                try:
                    out_path = tmp_dir / "starbucks_report_ascii.pdf"
                    df_info = {"rows": df.shape[0], "cols": df.shape[1]}
                    build_pdf_ascii(out_path, "Starbucks - EDA & Model Report (ASCII)", df_info, images, model_results)
                    with open(out_path, "rb") as f:
                        data = f.read()
                    st.download_button("Download PDF (ASCII)", data=data, file_name="starbucks_report_ascii.pdf", mime="application/pdf")
                    st.success("PDF generated (ASCII-safe).")
                except Exception as e:
                    st.error("Failed to build PDF: " + sanitize_ascii(str(e)))

    elif page == "About / Team":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("About & Team — Rajagiri (Grand Thornton Add-On)")

        # Top row: logos (side-by-side) + blurb
        colA, colB = st.columns([0.48, 0.52])
        with colA:
            st.markdown('<div class="about-logos-row">', unsafe_allow_html=True)

            # RCSS logo logic: prefer user's local path, then fallback container
            rcss_to_show = RCSS_LOCAL if RCSS_LOCAL.exists() else (RCSS_FALLBACK if RCSS_FALLBACK.exists() else None)
            grant_to_show = GRANT_LOCAL if GRANT_LOCAL.exists() else (GRANT_FALLBACK if GRANT_FALLBACK.exists() else None)

            if rcss_to_show:
                st.markdown(f'<div class="logo-card"><img src="data:image/png;base64,{base64.b64encode(rcss_to_show.read_bytes()).decode()}" width="140" /></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="logo-caption">Rajagiri College of Social Sciences</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="logo-card"><div class="small-muted">RCSS logo not found</div></div>', unsafe_allow_html=True)

            if grant_to_show:
                # grant could be jpeg; read bytes and base64 encode but keep src data URI generic
                grant_b64 = base64.b64encode(grant_to_show.read_bytes()).decode()
                # determine MIME (png/jpeg)
                mime = "image/png" if grant_to_show.suffix.lower().endswith(".png") else "image/jpeg"
                st.markdown(f'<div class="logo-card"><img src="data:{mime};base64,{grant_b64}" width="140" /></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="logo-caption">Grant Thornton (Add-On Partner)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="logo-card"><div class="small-muted">Grant Thornton logo not found</div></div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with colB:
            # Lottie or fallback
            if LOTTIE_AVAILABLE:
                lottie_urls = [
                    "https://assets2.lottiefiles.com/packages/lf20_0yfsb3a1.json",
                    "https://assets7.lottiefiles.com/packages/lf20_tutvdkg0.json"
                ]
                loaded = None
                for u in lottie_urls:
                    loaded = try_load_lottie(u)
                    if loaded:
                        break
                if loaded:
                    try:
                        st_lottie(loaded, height=180, key="about_top_lottie")
                    except Exception:
                        st.markdown(CUP_SVG_FALLBACK, unsafe_allow_html=True)
                else:
                    st.markdown(CUP_SVG_FALLBACK, unsafe_allow_html=True)
            else:
                st.markdown(CUP_SVG_FALLBACK, unsafe_allow_html=True)

            st.markdown("<div style='padding-top:6px'><strong>Project:</strong> Starbucks Pro — Analytics & Machine Learning Dashboard</div>", unsafe_allow_html=True)
            st.markdown("<div class='small-muted'>Capstone-style project as part of Grant Thornton add-on at Rajagiri College of Social Sciences. Focus: practical analytics & ML skills, industry alignment.</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Three info columns
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='team-card'>", unsafe_allow_html=True)
            st.markdown("<div class='team-name'>Rajagiri College of Social Sciences</div>", unsafe_allow_html=True)
            st.markdown("<div class='team-role'>Kalamassery, Kerala — Social Sciences & Professional Programs</div>", unsafe_allow_html=True)
            st.markdown("<div class='small-muted' style='margin-top:8px'>RCSS emphasizes research-led learning and industry collaborations with a focus on holistic student development.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='team-card'>", unsafe_allow_html=True)
            st.markdown("<div class='team-name'>Grant Thornton Add-On</div>", unsafe_allow_html=True)
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
                    # show LinkedIn as clickable link
                    st.markdown(f"<div style='margin-top:8px'><a href='{linkedin}' target='_blank' style='color:#dbeee2;'>LinkedIn</a></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Credits & Contact**")
        st.markdown("- Built by the contributors listed above as part of the Grand Thornton Add-On at Rajagiri College of Social Sciences.")
        st.markdown("<div class='small-muted'>Built with collaboration, learning, and lots of coffee ☕</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer watermark (optional Starbucks logo if present)
    try:
        if DEFAULT_LOGO.exists():
            b = base64.b64encode(DEFAULT_LOGO.read_bytes()).decode()
            st.markdown(f'<img src="data:image/png;base64,{b}" style="position:fixed;right:18px;bottom:18px;opacity:0.12;width:150px;">', unsafe_allow_html=True)
    except Exception:
        pass

if __name__ == "__main__":
    main()
