"""
Phase 5 -- Explainability Layer (SHAP)
=========================================================
Micro Project: Fraud and Repayment-Risk Detection in Microfinance Transactions
Dataset: Home Credit Default Risk (Kaggle)

What this script does
----------------------
1. Loads best_model.joblib (the winning SMOTE+classifier pipeline from
   Phase 4c), feature_names_with_anomaly.json, splits_with_anomaly_features.npz,
   and metrics.json (for the chosen decision threshold).
2. Pulls the raw tree classifier out of the pipeline (SMOTE only ever runs
   during .fit(), never during .predict()/.predict_proba(), so it's not
   part of what we need to explain) and wraps it in a shap.TreeExplainer --
   works for both XGBoost and LightGBM without changing any code.
3. Computes SHAP values on the TEST split (the same split Phase 4's final
   numbers were reported on, so the explanations match the model you'd
   actually be reporting).
4. Produces:
     - a GLOBAL feature-importance view (beeswarm + bar plot) -- "what does
       the model rely on overall"
     - THREE per-borrower waterfall plots -- highest-risk, borderline
       (closest to the chosen operating threshold), and confidently
       low-risk -- "why THIS borrower got THIS score"
5. Saves everything a reader needs for the report, plus a small reusable
   `explain_borrower()` function (in shap_explainer.py) that Phase 6's
   dashboard will import directly to explain a borrower on demand.

Setup
-----
    pip install pandas numpy scikit-learn shap joblib matplotlib

Usage
-----
    python phase5_shap_explainability.py
    python phase5_shap_explainability.py --data-dir ./data/processed --out-dir ./data/processed
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

MAX_SHAP_ROWS = 2000  # cap for speed on the full ~300k-row real dataset


# ----------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------
def load_everything(data_dir: Path):
    model_path = data_dir / "best_model.joblib"
    names_path = data_dir / "feature_names_with_anomaly.json"
    splits_path = data_dir / "splits_with_anomaly_features.npz"
    metrics_path = data_dir / "metrics.json"

    for p in [model_path, names_path, splits_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run phase4c_select_and_evaluate.py first."
            )

    pipeline = joblib.load(model_path)
    feature_names = json.loads(names_path.read_text(encoding="utf-8"))
    data = dict(np.load(splits_path))
    threshold = 0.5
    if metrics_path.exists():
        threshold = json.loads(metrics_path.read_text(encoding="utf-8")).get("chosen_threshold", 0.5)

    return pipeline, feature_names, data, threshold


def get_raw_classifier(pipeline):
    """
    Pulls the actual tree model out of the imblearn Pipeline(SMOTE -> clf).
    SMOTE never runs at prediction time, so shap.TreeExplainer only needs
    (and only accepts) the classifier step itself.
    """
    if hasattr(pipeline, "named_steps") and "clf" in pipeline.named_steps:
        return pipeline.named_steps["clf"]
    return pipeline  # already a bare classifier


# ----------------------------------------------------------------------
# 2. SHAP computation
# ----------------------------------------------------------------------
def compute_shap_values(clf, X: np.ndarray, random_state: int):
    """Subsamples to MAX_SHAP_ROWS for speed, then runs TreeExplainer."""
    if X.shape[0] > MAX_SHAP_ROWS:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=MAX_SHAP_ROWS, replace=False)
        idx.sort()
    else:
        idx = np.arange(X.shape[0])

    explainer = shap.TreeExplainer(clf)
    sv = explainer(X[idx])  # shap.Explanation object
    return explainer, sv, idx


# ----------------------------------------------------------------------
# 3. Global plots
# ----------------------------------------------------------------------
def plot_global_beeswarm(sv, out_dir: Path) -> None:
    fig = plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(sv, show=False, max_display=15)
    plt.title("SHAP summary -- feature impact across borrowers (test split)")
    plt.tight_layout()
    plt.savefig(out_dir / "01_shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_global_bar(sv, out_dir: Path) -> None:
    fig = plt.figure(figsize=(8, 7))
    shap.plots.bar(sv, show=False, max_display=15)
    plt.title("Mean |SHAP value| -- global feature importance (test split)")
    plt.tight_layout()
    plt.savefig(out_dir / "02_shap_feature_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def top_features_text(sv, feature_names: list, top_n: int = 15) -> list:
    mean_abs = np.abs(sv.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    return [(feature_names[i], float(mean_abs[i])) for i in order]


# ----------------------------------------------------------------------
# 4. Per-borrower explanations
# ----------------------------------------------------------------------
def pick_example_borrowers(proba: np.ndarray, threshold: float) -> dict:
    """Indices (into the SHAP-sampled subset) of three illustrative cases."""
    high_risk_idx = int(np.argmax(proba))
    low_risk_idx = int(np.argmin(proba))
    borderline_idx = int(np.argmin(np.abs(proba - threshold)))
    return {"high_risk": high_risk_idx, "borderline": borderline_idx, "low_risk": low_risk_idx}


def plot_waterfall(sv, local_idx: int, title: str, filename: str, out_dir: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(sv[local_idx], show=False, max_display=12)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def borrower_explanation_text(sv, local_idx: int, feature_names: list, proba_value: float,
                                borrower_id, threshold: float, top_n: int = 5) -> str:
    row_values = sv.values[local_idx]
    order = np.argsort(np.abs(row_values))[::-1][:top_n]
    decision = "FLAG for review" if proba_value >= threshold else "no flag"
    lines = [
        f"Borrower SK_ID_CURR={borrower_id}  |  predicted risk={proba_value:.3f}  |  "
        f"threshold={threshold:.3f}  |  decision: {decision}",
        "  Top contributing factors:",
    ]
    for i in order:
        direction = "increases" if row_values[i] > 0 else "decreases"
        lines.append(f"    - {feature_names[i]:35s} {direction} risk  (SHAP={row_values[i]:+.4f})")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 5. Reusable explainer module for the Phase 6 dashboard
# ----------------------------------------------------------------------
EXPLAINER_MODULE_TEMPLATE = '''"""
Reusable borrower-explanation function, generated by Phase 5.
Import this directly from the Phase 6 Streamlit dashboard.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import shap

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


def load_explainer(data_dir):
    data_dir = Path(data_dir)
    pipeline = joblib.load(data_dir / "best_model.joblib")
    clf = pipeline.named_steps["clf"] if hasattr(pipeline, "named_steps") else pipeline
    feature_names = json.loads((data_dir / "feature_names_with_anomaly.json").read_text(encoding="utf-8"))
    threshold = 0.5
    metrics_path = data_dir / "metrics.json"
    if metrics_path.exists():
        threshold = json.loads(metrics_path.read_text(encoding="utf-8")).get("chosen_threshold", 0.5)
    explainer = shap.TreeExplainer(clf)
    return pipeline, explainer, feature_names, threshold


def explain_borrower(pipeline, explainer, feature_names, threshold, x_row, top_n=5):
    """
    x_row: a single borrower's feature vector, already preprocessed the
    SAME way as training (use the saved preprocessor.joblib from Phase 2
    for a brand-new borrower's raw fields).
    Returns: dict with risk_score, decision, action_explanation, and top contributing factors.
    """
    x_row = np.asarray(x_row).reshape(1, -1)
    proba = pipeline.predict_proba(x_row)[0, 1]
    sv = explainer(x_row)
    row_values = sv.values[0]
    order = np.argsort(np.abs(row_values))[::-1][:top_n]
    
    factors = [
        {
            "raw_feature": feature_names[i],
            "feature": clean_feature_name(feature_names[i]),
            "shap_value": float(row_values[i]),
            "direction": "increases risk" if row_values[i] > 0 else "decreases risk",
        }
        for i in order
    ]

    # Action Determination
    if proba >= threshold:
        action = "Decline / manual review"
    elif proba >= threshold * 0.6:
        action = "Manual review recommended"
    else:
        action = "Approve"

    # Dynamic explanation generation based on top risk drivers
    top_increasing = [f["feature"] for f in factors if f["shap_value"] > 0]
    top_decreasing = [f["feature"] for f in factors if f["shap_value"] < 0]

    if action == "Decline / manual review":
        if top_increasing:
            reasons = ", ".join(top_increasing[:2])
            explanation = f"Risk score ({proba:.3f}) exceeds threshold ({threshold:.3f}) primarily due to: {reasons}."
        else:
            explanation = f"Risk score ({proba:.3f}) exceeds threshold ({threshold:.3f})."
    elif action == "Manual review recommended":
        if top_increasing:
            reasons = ", ".join(top_increasing[:2])
            explanation = f"Moderate risk score ({proba:.3f}). Driven higher by {reasons}, but mitigated by other financial factors."
        else:
            explanation = f"Moderate risk score ({proba:.3f}) near evaluation threshold ({threshold:.3f})."
    else:
        if top_decreasing:
            reasons = ", ".join(top_decreasing[:2])
            explanation = f"Low risk score ({proba:.3f}). Strongly supported by favorable factors: {reasons}."
        else:
            explanation = f"Low risk score ({proba:.3f}) below decision threshold ({threshold:.3f})."

    return {
        "risk_score": float(proba),
        "action": action,
        "action_explanation": explanation,
        "decision": "flag for review" if proba >= threshold else "no flag",
        "threshold": float(threshold),
        "top_factors": factors,
    }
'''


# ----------------------------------------------------------------------
# 6. Orchestration
# ----------------------------------------------------------------------
def run_phase5(data_dir: str, out_dir: str, random_state: int) -> None:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Phase 4 outputs from {data_dir} ...")
    pipeline, feature_names, data, threshold = load_everything(data_dir)
    X_test, id_test = data["X_test"], data["id_test"]

    clf = get_raw_classifier(pipeline)

    print(f"Computing SHAP values on the test split (up to {MAX_SHAP_ROWS} rows) ...")
    explainer, sv, sample_idx = compute_shap_values(clf, X_test, random_state)
    sv.feature_names = feature_names

    proba_sample = pipeline.predict_proba(X_test[sample_idx])[:, 1]
    id_sample = id_test[sample_idx]

    print("Generating global explanation plots ...")
    plot_global_beeswarm(sv, out_dir)
    plot_global_bar(sv, out_dir)
    top_feats = top_features_text(sv, feature_names, top_n=15)

    print(f"Selecting example borrowers around threshold={threshold:.3f} ...")
    examples = pick_example_borrowers(proba_sample, threshold)

    example_texts = []
    for label, local_idx in examples.items():
        title = {
            "high_risk": "Highest predicted-risk borrower",
            "borderline": f"Borderline case (closest to threshold {threshold:.3f})",
            "low_risk": "Confidently low-risk borrower",
        }[label]
        filename = f"0{3 + list(examples).index(label)}_waterfall_{label}.png"
        plot_waterfall(sv, local_idx, title, filename, out_dir)
        text = borrower_explanation_text(
            sv, local_idx, feature_names, proba_sample[local_idx], id_sample[local_idx], threshold
        )
        example_texts.append(f"-- {title} --\n{text}")

    print("Saving SHAP values and reusable explainer module ...")
    np.savez_compressed(
        out_dir / "shap_values_test_sample.npz",
        values=sv.values,
        base_values=sv.base_values,
        sample_idx=sample_idx,
        id_sample=id_sample,
        proba_sample=proba_sample,
    )
    (out_dir / "shap_explainer.py").write_text(EXPLAINER_MODULE_TEMPLATE, encoding="utf-8")

    report_lines = [
        "PHASE 5 -- EXPLAINABILITY LAYER (SHAP) REPORT",
        "=" * 70,
        f"SHAP values computed on {len(sample_idx):,} test-split borrowers "
        f"(of {X_test.shape[0]:,} total; subsampled for speed).",
        f"Decision threshold used (from Phase 4c): {threshold:.3f}",
        "",
        "Top 15 features by mean |SHAP value| (global importance):",
        *[f"  {i+1:>2}. {name:35s} {val:.4f}" for i, (name, val) in enumerate(top_feats)],
        "",
        "=" * 70,
        "EXAMPLE PER-BORROWER EXPLANATIONS",
        "=" * 70,
        "",
        *[t + "\n" for t in example_texts],
    ]
    report_text = "\n".join(report_lines)
    (out_dir / "phase5_report.txt").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"\nAll Phase 5 outputs saved to: {out_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 5 -- SHAP explainability for the Home Credit risk model"
    )
    parser.add_argument("--data-dir", default="C:/Documents/Programms/Data_Science/micro_project",
                         help="Folder containing Phase 4c outputs (best_model.joblib etc.)")
    parser.add_argument("--out-dir", default="C:/Documents/Programms/Data_Science/micro_project")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    run_phase5(args.data_dir, args.out_dir, args.random_state)


if __name__ == "__main__":
    main()