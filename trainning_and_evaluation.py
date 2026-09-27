"""
Phase 4c -- Select Best Model & Final Evaluation
=========================================================
Micro Project: Fraud and Repayment-Risk Detection in Microfinance Transactions
Dataset: Home Credit Default Risk (Kaggle)

Run this AFTER both phase4a_train_xgboost.py and phase4b_train_lightgbm.py
have finished. This script does no heavy training -- it only loads the two
already-fitted models and runs inference (predict_proba), so it is fast and
does not run into the Windows worker-pool resource issue the tuning stages
did.

What this script does
----------------------
1. Loads xgboost_model.joblib + xgboost_metrics.json and
   lightgbm_model.joblib + lightgbm_metrics.json.
2. Loads splits_with_anomaly_features.npz (Phase 3 output) for X_val/X_test.
3. Re-scores both models on the validation split (cheap inference, not
   retraining) so the ROC comparison plot is built from live predictions.
4. Picks the better model by validation ROC-AUC.
5. Sweeps precision/recall vs. threshold on validation and picks the
   F1-optimal operating point (instead of the default 0.5 cut-off).
6. Evaluates the winning model on the TEST split EXACTLY ONCE, at that
   chosen threshold.
7. Saves:
     - best_model.joblib
     - metrics.json
     - 01_roc_curves.png
     - 02_precision_recall_vs_threshold.png
     - 03_confusion_matrix.png
     - phase4_report.txt

Setup
-----
    pip install pandas numpy scikit-learn xgboost lightgbm joblib matplotlib seaborn

Usage
-----
    python phase4c_select_and_evaluate.py
    python phase4c_select_and_evaluate.py --data-dir ./data/processed --out-dir ./data/processed
"""

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import roc_auc_score

from common import (
    best_f1_threshold,
    evaluate_at_threshold,
    load_data,
    load_json,
    plot_confusion_matrix,
    plot_precision_recall_vs_threshold,
    plot_roc_curves,
    save_json,
)


def run(data_dir: str, out_dir: str) -> None:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xgb_model_path = data_dir / "xgboost_model.joblib"
    lgbm_model_path = data_dir / "lightgbm_model.joblib"
    xgb_metrics_path = data_dir / "xgboost_metrics.json"
    lgbm_metrics_path = data_dir / "lightgbm_metrics.json"
    for p in [xgb_model_path, lgbm_model_path, xgb_metrics_path, lgbm_metrics_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run phase4a_train_xgboost.py and phase4b_train_lightgbm.py first "
                "(each as its own separate `python` invocation)."
            )

    print("Loading fitted models ...")
    xgb_pipeline = joblib.load(xgb_model_path)
    lgbm_pipeline = joblib.load(lgbm_model_path)
    xgb_saved_metrics = load_json(xgb_metrics_path)
    lgbm_saved_metrics = load_json(lgbm_metrics_path)

    print("Loading Phase 3 data splits ...")
    data = load_data(data_dir)
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print("Scoring both models on the validation split (inference only, no retraining) ...")
    results = {}
    for name, pipeline, saved in [
        ("XGBoost", xgb_pipeline, xgb_saved_metrics),
        ("LightGBM", lgbm_pipeline, lgbm_saved_metrics),
    ]:
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        results[name] = {
            "pipeline": pipeline,
            "cv_best_auc": saved.get("cv_best_auc"),
            "best_params": saved.get("best_params"),
            "y_val": y_val,
            "val_proba": val_proba,
            "val_auc": val_auc,
        }
        print(f"  {name}: CV ROC-AUC={saved.get('cv_best_auc'):.4f} | Val ROC-AUC={val_auc:.4f}")

    best_name = max(results, key=lambda k: results[k]["val_auc"])
    best_pipeline = results[best_name]["pipeline"]
    print(f"\nWinner: {best_name} (val AUC={results[best_name]['val_auc']:.4f})")

    print("Generating comparison plots ...")
    plot_roc_curves(results, out_dir)
    plot_precision_recall_vs_threshold(y_val, results[best_name]["val_proba"], best_name, out_dir)

    default_metrics = evaluate_at_threshold(y_val, results[best_name]["val_proba"], 0.5)
    tuned = best_f1_threshold(y_val, results[best_name]["val_proba"])
    chosen_threshold = tuned["threshold"]

    print(f"Evaluating {best_name} on the TEST split (touched once, at the end) ...")
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    test_at_default = evaluate_at_threshold(y_test, test_proba, 0.5)
    test_at_tuned = evaluate_at_threshold(y_test, test_proba, chosen_threshold)
    test_pred_tuned = (test_proba >= chosen_threshold).astype(int)

    plot_confusion_matrix(y_test, test_pred_tuned, best_name, chosen_threshold, out_dir)

    print("Saving winning model and metrics ...")
    joblib.dump(best_pipeline, out_dir / "best_model.joblib")

    metrics_out = {
        "xgboost": {
            "cv_best_auc": results["XGBoost"]["cv_best_auc"],
            "best_params": results["XGBoost"]["best_params"],
            "val_auc": results["XGBoost"]["val_auc"],
        },
        "lightgbm": {
            "cv_best_auc": results["LightGBM"]["cv_best_auc"],
            "best_params": results["LightGBM"]["best_params"],
            "val_auc": results["LightGBM"]["val_auc"],
        },
        "best_model": best_name,
        "validation_at_default_threshold_0.5": default_metrics,
        "validation_best_f1_threshold": tuned,
        "test_auc": test_auc,
        "test_at_default_threshold_0.5": test_at_default,
        "test_at_chosen_threshold": test_at_tuned,
        "chosen_threshold": chosen_threshold,
    }
    save_json(metrics_out, out_dir / "metrics.json")

    report_lines = [
        "PHASE 4 -- CLASS-IMBALANCE CORRECTION & SUPERVISED RISK SCORING",
        "(trained across phase4a + phase4b as separate processes; selected here)",
        "=" * 70,
        f"XGBoost  : CV ROC-AUC={results['XGBoost']['cv_best_auc']:.4f} | Val ROC-AUC={results['XGBoost']['val_auc']:.4f}",
        f"LightGBM : CV ROC-AUC={results['LightGBM']['cv_best_auc']:.4f} | Val ROC-AUC={results['LightGBM']['val_auc']:.4f}",
        f"Selected model: {best_name}",
        "",
        f"-- Validation split, default 0.5 threshold --",
        f"  Precision: {default_metrics['precision']:.3f}  Recall: {default_metrics['recall']:.3f}  F1: {default_metrics['f1']:.3f}",
        "",
        f"-- Validation split, F1-optimal threshold ({chosen_threshold:.3f}) --",
        f"  Precision: {tuned['precision']:.3f}  Recall: {tuned['recall']:.3f}  F1: {tuned['f1']:.3f}",
        "",
        f"-- TEST split (held out, touched once) --",
        f"  ROC-AUC: {test_auc:.4f}",
        f"  @ default 0.5     -> Precision: {test_at_default['precision']:.3f}  Recall: {test_at_default['recall']:.3f}  F1: {test_at_default['f1']:.3f}",
        f"  @ chosen {chosen_threshold:.3f}   -> Precision: {test_at_tuned['precision']:.3f}  Recall: {test_at_tuned['recall']:.3f}  F1: {test_at_tuned['f1']:.3f}",
        "",
        "Target ranges from the synopsis: AUC-ROC >= 0.80, minority-class F1 >= 0.65.",
    ]
    report_text = "\n".join(report_lines)
    (out_dir / "phase4_report.txt").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"\nAll Phase 4 outputs saved to: {out_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4c -- Select best of XGBoost/LightGBM and run final evaluation"
    )
    parser.add_argument("--data-dir", default="C:/Documents/Programms/Data_Science/micro_project",
                         help="Folder containing Phase 3 outputs AND the phase4a/phase4b outputs")
    parser.add_argument("--out-dir", default="C:/Documents/Programms/Data_Science/micro_project")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.out_dir)