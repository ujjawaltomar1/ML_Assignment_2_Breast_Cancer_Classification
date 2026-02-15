import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Prefer pickle for .pkl; use joblib for .joblib if available
try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None

# -----------------------------------------------------
# Page meta
# -----------------------------------------------------
st.set_page_config(page_title="Breast Cancer — Streamlit App", layout="wide")
st.title("Breast Cancer (Wisconsin) — Model Evaluation")

# -----------------------------------------------------
# Look for test.csv at the below path
# -----------------------------------------------------
TEST_FILE = Path("test.csv")

def load_canonical_test_bytes() -> tuple[bytes | None, str]:
    if TEST_FILE.exists() and TEST_FILE.is_file():
        return TEST_FILE.read_bytes(), TEST_FILE.name
    return None, TEST_FILE.as_posix()

TARGET_FIELDS = ["diagnosis", "target", "label", "class", "y"]
LABEL_MAP = {"M": 1, "B": 0, "malignant": 1, "benign": 0}

def coerce_labels(y: pd.Series) -> np.ndarray:
    out = y.copy()
    if out.dtype == "object":
        out = out.astype(str).str.strip().map(lambda v: LABEL_MAP.get(v, v))
    if not np.issubdtype(pd.Series(out).dtype, np.number):
        out, _ = pd.factorize(out)
    return np.asarray(out)

def summarize_scores(y_true: np.ndarray, y_pred: np.ndarray, y_prob=None) -> dict:
    avg = "binary" if len(np.unique(y_true)) == 2 else "macro"
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": None,
    }
    try:
        if y_prob is not None:
            if getattr(y_prob, 'ndim', 1) == 1:
                scores['AUC'] = roc_auc_score(y_true, np.asarray(y_prob).ravel())
            else:
                scores['AUC'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        scores['AUC'] = None
    return scores

# -----------------------------------------------------
# Load trained models from ./model
# -----------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_model_registry() -> dict:
    models = {}
    model_dir = Path("model")
    if not model_dir.exists():
        return models
    files = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.joblib"))
    for fp in sorted(files):
        label = fp.stem.replace("_", " ").title()
        est = None
        try:
            if fp.suffix == ".pkl":
                with open(fp, "rb") as f:
                    est = pickle.load(f)
            else:
                if joblib_load is not None:
                    est = joblib_load(fp)
        except Exception as ex:
            st.warning(f"Could not load {fp.name}: {ex}")
            est = None
        if est is not None:
            models[label] = est
    return models

# -----------------------------------------------------
# Download the test.csv from data/test.csv
# -----------------------------------------------------
st.subheader("Step 1 — Download the assignment test.csv")
fixed_bytes, fixed_path = load_canonical_test_bytes()
if fixed_bytes is None:
    st.error(f"Canonical file not found at {fixed_path}. Please add your test CSV there in the repo.")
else:
    st.download_button(
        label="Download test.csv",
        data=fixed_bytes,
        file_name="test.csv",  # always served as 'test.csv'
        mime="text/csv",
        use_container_width=True,
    )

# -----------------------------------------------------
# Choose model
# -----------------------------------------------------
st.subheader("Step 2 — Choose model")
registry = build_model_registry()
if not registry:
    st.warning("No models found in ./model. Add your trained .pkl (preferred) or .joblib files.")
model_label = st.selectbox("Model", options=sorted(registry.keys()) if registry else [], index=0 if registry else None)

# -----------------------------------------------------
# Upload CSV
# -----------------------------------------------------
st.subheader("Step 3 — Upload CSV for evaluation/prediction")
uploaded_csv = st.file_uploader("Upload CSV (labeled file preferred for metrics)", type=["csv"])

frame = None
if uploaded_csv is not None:
    try:
        frame = pd.read_csv(uploaded_csv)
        st.dataframe(frame.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

# -----------------------------------------------------
# Run evaluation / predictions
# -----------------------------------------------------
if frame is not None and model_label:
    # Detect target
    target_col = next((c for c in TARGET_FIELDS if c in frame.columns), None)
    X_eval = frame.copy()
    y_eval = None
    if target_col is not None:
        y_eval = coerce_labels(X_eval.pop(target_col))

    # Predict
    est = registry[model_label]
    try:
        y_hat = est.predict(X_eval)
    except Exception as e:
        st.error("Prediction failed. Ensure input columns match training schema. Error: " + str(e))
        st.stop()

    # Probability for AUC
    y_hat_proba = None
    try:
        probs = est.predict_proba(X_eval)
        if probs is not None and getattr(probs, 'ndim', 1) == 2 and probs.shape[1] == 2:
            y_hat_proba = probs[:, 1]
        else:
            y_hat_proba = probs
    except Exception:
        y_hat_proba = None

    # Metrics if ground truth exists
    if y_eval is not None:
        st.subheader("Evaluation Metrics")
        m = summarize_scores(y_eval, y_hat, y_hat_proba)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{m['Accuracy']:.4f}")
        c2.metric("AUC", f"{m['AUC']:.4f}" if m["AUC"] is not None else "NA")
        c3.metric("Precision", f"{m['Precision']:.4f}")
        c4.metric("Recall", f"{m['Recall']:.4f}")
        c5.metric("F1", f"{m['F1']:.4f}")
        c6.metric("MCC", f"{m['MCC']:.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_eval, y_hat)
        labels = np.unique(y_eval)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_label}")
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.code(classification_report(y_eval, y_hat, zero_division=0))
    else:
        st.info("No target column detected — showing predictions only (metrics need ground truth).")
