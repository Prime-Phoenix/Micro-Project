"""
Phase 6 -- Loan Officer Dashboard (Streamlit)
=========================================================
Micro Project: Fraud and Repayment-Risk Detection in Microfinance Transactions
Dataset: Home Credit Default Risk (Kaggle)
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ----------------------------------------------------------------------
# CLI args (Streamlit-compatible: anything after a literal "--")
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="micro_project/")
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    return parser.parse_args(argv)


ARGS = parse_args()
DATA_DIR = Path(ARGS.data_dir)

# Mapping generic user-friendly labels to underlying dataset feature names
INPUT_FIELDS = [
    {"label": "Total Annual Income", "key": "AMT_INCOME_TOTAL", "step": 5000.0},
    {"label": "Total Loan Amount", "key": "AMT_CREDIT", "step": 5000.0},
    {"label": "Monthly Payment", "key": "AMT_ANNUITY", "step": 500.0},
    {"label": "Item / Goods Price", "key": "Item Price", "raw_key": "AMT_GOODS_PRICE", "step": 5000.0},
]


# ----------------------------------------------------------------------
# Cached loading -- runs once per session, not on every interaction
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts(data_dir: str):
    data_dir = Path(data_dir)
    required = [
        "preprocessor.joblib", "isolation_forest.joblib", "autoencoder.joblib",
        "best_model.joblib", "metrics.json", "feature_names_with_anomaly.json",
        "engineered_features.csv",
    ]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required Phase 2-5 outputs in {data_dir}: {missing}. "
            "Run phase2 through phase5 first."
        )

    preprocessor = joblib.load(data_dir / "preprocessor.joblib")
    iso_model = joblib.load(data_dir / "isolation_forest.joblib")
    ae_model = joblib.load(data_dir / "autoencoder.joblib")
    best_pipeline = joblib.load(data_dir / "best_model.joblib")
    metrics = json.loads((data_dir / "metrics.json").read_text(encoding="utf-8"))
    feature_names = json.loads((data_dir / "feature_names_with_anomaly.json").read_text(encoding="utf-8"))
    threshold = metrics.get("chosen_threshold", 0.5)

    clf = best_pipeline.named_steps["clf"] if hasattr(best_pipeline, "named_steps") else best_pipeline
    explainer = shap.TreeExplainer(clf)

    numeric_cols = preprocessor.transformers_[0][2]
    categorical_cols = preprocessor.transformers_[1][2]

    engineered_df = pd.read_csv(data_dir / "engineered_features.csv")

    return {
        "preprocessor": preprocessor,
        "iso_model": iso_model,
        "ae_model": ae_model,
        "best_pipeline": best_pipeline,
        "clf": clf,
        "explainer": explainer,
        "threshold": threshold,
        "feature_names": feature_names,
        "numeric_cols": list(numeric_cols),
        "categorical_cols": list(categorical_cols),
        "engineered_df": engineered_df,
        "metrics": metrics,
    }

# ----------------------------------------------------------------------
# Helper: Translate feature technical names to plain English labels
# ----------------------------------------------------------------------
FEATURE_LABEL_MAP = {
    "EXT_SOURCE_MEAN": "External Credit Score (Avg)",
    "EXT_SOURCE_MISSING_CNT": "Missing Credit Data Count",
    "ANNUITY_CREDIT_RATIO": "Payment-to-Loan Ratio",
    "CREDIT_INCOME_RATIO": "Debt-to-Income Ratio",
    "ANNUITY_INCOME_RATIO": "Payment-to-Income Ratio",
    "CREDIT_GOODS_RATIO": "Loan-to-Goods Price Ratio",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Credit Bureau Checks (Past Quarter)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Credit Bureau Checks (Past Year)",
    "AMT_INCOME_TOTAL": "Total Annual Income",
    "AMT_CREDIT": "Total Loan Amount",
    "AMT_ANNUITY": "Monthly Payment",
    "AMT_GOODS_PRICE": "Item Price",
    "NAME_EDUCATION_TYPE_Secondary / secondary special": "Education: High School / Vocational",
    "NAME_EDUCATION_TYPE_Higher education": "Education: Higher / University",
    "CODE_GENDER_F": "Gender: Female",
    "CODE_GENDER_M": "Gender: Male",
    "iso_anomaly_score": "Isolation Forest Anomaly Score",
    "ae_anomaly_score": "Autoencoder Anomaly Score",
}


def clean_feature_name(name: str) -> str:
    """Translates raw dataset column names into readable UI labels."""
    if name in FEATURE_LABEL_MAP:
        return FEATURE_LABEL_MAP[name]
    
    clean = name.replace("_", " ").title()
    clean = clean.replace("Amt ", "Amount ").replace("Cnt ", "Count ")
    return clean

# ----------------------------------------------------------------------
# Feature engineering re-application for edited fields (mirrors Phase 2)
# ----------------------------------------------------------------------
def recompute_ratio_features(row: dict) -> dict:
    row = dict(row)
    income = row.get("AMT_INCOME_TOTAL") or np.nan
    credit = row.get("AMT_CREDIT") or np.nan
    annuity = row.get("AMT_ANNUITY") or np.nan
    goods = row.get("AMT_GOODS_PRICE") or np.nan

    row["CREDIT_INCOME_RATIO"] = credit / income if income else np.nan
    row["ANNUITY_INCOME_RATIO"] = annuity / income if income else np.nan
    row["ANNUITY_CREDIT_RATIO"] = annuity / credit if credit else np.nan
    row["CREDIT_GOODS_RATIO"] = credit / goods if goods else np.nan
    return row


# ----------------------------------------------------------------------
# Scoring pipeline -- mirrors Phase 3 + Phase 4 exactly, at inference time
# ----------------------------------------------------------------------
def score_borrower(artifacts: dict, row: dict) -> dict:
    numeric_cols = artifacts["numeric_cols"]
    categorical_cols = artifacts["categorical_cols"]
    row_df = pd.DataFrame([row])[numeric_cols + categorical_cols]

    X_base = artifacts["preprocessor"].transform(row_df)

    iso_score = -artifacts["iso_model"].decision_function(X_base)
    ae_pred = artifacts["ae_model"].predict(X_base)
    ae_score = np.mean((X_base - ae_pred) ** 2, axis=1)

    X_full = np.hstack([X_base, np.column_stack([iso_score, ae_score])])

    proba = artifacts["best_pipeline"].predict_proba(X_full)[0, 1]
    threshold = artifacts["threshold"]

    sv = artifacts["explainer"](X_full)
    row_values = sv.values[0]
    order = np.argsort(np.abs(row_values))[::-1][:5]
    top_factors = [
        {
            "feature": artifacts["feature_names"][i],
            "friendly_feature": clean_feature_name(artifacts["feature_names"][i]),
            "shap_value": float(row_values[i]),
            "direction": "increases risk" if row_values[i] > 0 else "decreases risk",
        }
        for i in order
    ]

    if proba >= threshold:
        action = "Decline / manual review"
    elif proba >= threshold * 0.6:
        action = "Manual review recommended"
    else:
        action = "Approve"

    top_increasing = [f["friendly_feature"] for f in top_factors if f["shap_value"] > 0]
    top_decreasing = [f["friendly_feature"] for f in top_factors if f["shap_value"] < 0]

    if action == "Decline / manual review":
        if top_increasing:
            reasons = ", ".join(top_increasing[:2])
            action_explanation = f"Risk score ({proba:.3f}) exceeds cutoff ({threshold:.3f}) primarily due to high-risk factors: {reasons}."
        else:
            action_explanation = f"Risk score ({proba:.3f}) exceeds decision threshold ({threshold:.3f})."
    elif action == "Manual review recommended":
        if top_increasing:
            reasons = ", ".join(top_increasing[:2])
            action_explanation = f"Moderate risk score ({proba:.3f}). Elevated risk signals detected in {reasons}."
        else:
            action_explanation = f"Moderate risk score ({proba:.3f}) near evaluation threshold ({threshold:.3f})."
    else:
        if top_decreasing:
            reasons = ", ".join(top_decreasing[:2])
            action_explanation = f"Low risk score ({proba:.3f}). Driven by favorable factors: {reasons}."
        else:
            action_explanation = f"Low risk score ({proba:.3f}) safely below threshold ({threshold:.3f})."

    return {
        "risk_score": float(proba),
        "threshold": float(threshold),
        "action": action,
        "action_explanation": action_explanation,
        "top_factors": top_factors,
        "shap_explanation": sv,
        "iso_anomaly_score": float(iso_score[0]),
        "ae_anomaly_score": float(ae_score[0]),
    }


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Microfinance Risk Dashboard", layout="wide")
    
    # CSS override to prevent truncation across all metric labels & text cards
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] {
            white-space: normal !important;
            word-wrap: break-word !important;
            font-size: 1.5rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Microfinance Fraud & Repayment-Risk Dashboard")
    st.caption(
        "Explainable risk scoring for loan officers -- anomaly detection + "
        "SMOTE-corrected XGBoost/LightGBM + SHAP, built on the Home Credit "
        "Default Risk dataset."
    )

    try:
        artifacts = load_artifacts(str(DATA_DIR))
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    df = artifacts["engineered_df"]
    threshold = artifacts["threshold"]

    st.sidebar.header("Select Borrower")
    borrower_ids = df["SK_ID_CURR"].tolist()
    borrower_id = st.sidebar.selectbox(
        "Borrower / Application ID",
        options=borrower_ids,
        format_func=lambda x: f"Applicant #{x}"
    )
    base_row = df[df["SK_ID_CURR"] == borrower_id].iloc[0].to_dict()

    st.sidebar.markdown("---")
    st.sidebar.header("What-if: Adjust Loan Terms")
    st.sidebar.caption("Edits recompute this borrower's financial ratio features automatically.")
    
    edited_row = dict(base_row)
    
    edited_row["AMT_INCOME_TOTAL"] = st.sidebar.number_input(
        "Total Annual Income", 
        value=float(base_row.get("AMT_INCOME_TOTAL", 0.0) or 0.0), 
        step=5000.0, 
        format="%.2f"
    )
    edited_row["AMT_CREDIT"] = st.sidebar.number_input(
        "Total Loan Amount", 
        value=float(base_row.get("AMT_CREDIT", 0.0) or 0.0), 
        step=5000.0, 
        format="%.2f"
    )
    edited_row["AMT_ANNUITY"] = st.sidebar.number_input(
        "Monthly Payment", 
        value=float(base_row.get("AMT_ANNUITY", 0.0) or 0.0), 
        step=500.0, 
        format="%.2f"
    )
    edited_row["AMT_GOODS_PRICE"] = st.sidebar.number_input(
        "Item / Goods Price", 
        value=float(base_row.get("AMT_GOODS_PRICE", 0.0) or 0.0), 
        step=5000.0, 
        format="%.2f"
    )

    edited_row = recompute_ratio_features(edited_row)

    result = score_borrower(artifacts, edited_row)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Risk Score", f"{result['risk_score']:.3f}", help=f"Decision threshold: {threshold:.3f}")
    
    # Render recommended action using HTML container to avoid truncation completely
    with col2:
        st.caption("Recommended Action")
        st.markdown(f"### **{result['action']}**")

    true_target = base_row.get("TARGET")
    col3.metric(
        "True Outcome (if known)",
        "Payment Difficulty" if true_target == 1 else "Repaid" if true_target == 0 else "N/A",
    )

    if result["action"] == "Decline / manual review":
        st.error(f"**Reason for Decision:** {result['action_explanation']}")
    elif result["action"] == "Manual review recommended":
        st.warning(f"**Reason for Decision:** {result['action_explanation']}")
    else:
        st.success(f"**Reason for Decision:** {result['action_explanation']}")

    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Key Factors Driving This Score")
        for f in result["top_factors"]:
            arrow = "⬆️" if f["shap_value"] > 0 else "⬇️"
            st.write(f"{arrow} **{f['friendly_feature']}** {f['direction']}  (SHAP = {f['shap_value']:+.4f})")

        st.markdown("**Anomaly Detection Signals**")
        st.write(f"Isolation Forest score: `{result['iso_anomaly_score']:.4f}`")
        st.write(f"Autoencoder error rate: `{result['ae_anomaly_score']:.4f}`")

    with right:
        st.subheader("SHAP Impact Breakdown")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7, 5.5))
        shap.plots.waterfall(result["shap_explanation"][0], show=False, max_display=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Raw Borrower Details (Adjusted Values)"):
        st.dataframe(pd.DataFrame([edited_row]))

    with st.expander("Model Metrics Summary"):
        m = artifacts["metrics"]
        st.write(f"Selected Model: **{m.get('best_model')}**")
        st.write(f"Validation ROC-AUC: {m.get('lightgbm', {}).get('val_auc') if m.get('best_model') == 'LightGBM' else m.get('xgboost', {}).get('val_auc'):.4f}")
        st.write(f"Test ROC-AUC: {m.get('test_auc'):.4f}")
        st.write(f"Decision Threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
