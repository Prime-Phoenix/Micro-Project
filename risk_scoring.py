"""
Phase 4 -- Class-Imbalance Correction & Supervised Risk Scoring
=========================================================
Micro Project: Fraud and Repayment-Risk Detection in Microfinance Transactions
Dataset: Home Credit Default Risk (Kaggle)

What this script does
----------------------
1. Loads splits_with_anomaly_features.npz + feature_names_with_anomaly.json
   from Phase 3 (Phase 2 features + ISO_ANOMALY_SCORE + AE_ANOMALY_SCORE).
2. Builds an imblearn Pipeline of SMOTE -> classifier for BOTH XGBoost and
   LightGBM. SMOTE lives INSIDE the pipeline, not applied once up front --
   this matters because if you SMOTE the training set once and then
   cross-validate on top of that, synthetic neighbours of a validation-fold
   point can leak into the training folds and inflate your CV score. Putting
   SMOTE inside the pipeline means it is re-fit fresh on only the real
   training rows of each CV fold.
3. Tunes both pipelines with RandomizedSearchCV (StratifiedKFold, scored on
   ROC-AUC, matching the target metric in the synopsis).
4. Refits each best pipeline on the FULL real training set (SMOTE still only
   ever sees training data -- val/test are never touched by it).
5. Evaluates both models on the untouched validation split: Precision,
   Recall, F1 (minority class), AUC-ROC, plus a precision-recall-vs-threshold
   sweep so you can pick an operating point deliberately instead of using
   the default 0.5 cut-off.
6. Picks the better of the two models by validation AUC-ROC, reports its
   final numbers on the held-out TEST split (touched only once, at the end).
7. Saves:
     - best_model.joblib             (winning fitted pipeline: SMOTE + classifier)
     - xgboost_model.joblib / lightgbm_model.joblib (both fitted pipelines)
     - metrics.json                  (all metrics, both models, all thresholds tried)
     - 01_roc_curves.png
     - 02_precision_recall_vs_threshold.png
     - 03_confusion_matrix.png       (best model, chosen threshold, test split)
     - phase4_report.txt

Setup
-----
    pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm joblib matplotlib seaborn
    (venv activation on Windows: venv\\Scripts\\Activate)

Usage
-----
    python phase4_risk_scoring.py
    python phase4_risk_scoring.py --data-dir ./data/processed --out-dir ./data/processed

Notes on parallelism (Windows)
-------------------------------
RandomizedSearchCV is parallelized across folds/candidates via n_jobs=-1.
XGBClassifier and LGBMClassifier are each ALSO capable of spinning up a
thread pool across every core. Nesting "every core" inside "every core"
oversubscribes threads/processes; on Windows this reliably blows through
available OS handles and RandomizedSearchCV.fit() dies with
`OSError: [WinError 1450] Insufficient system resources exist to complete
the requested service` -- not a bug in the modeling logic, just too many
workers fighting over too few handles. Fix: let the outer search own the
parallelism (n_jobs=-1 there) and pin each inner classifier to a single
thread (n_jobs=1) so you don't multiply core-count by core-count.
"""

import argparse
import json
from pathlib import Path

import joblib
from joblib.externals.loky import get_reusable_executor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------
def load_data(data_dir: Path) -> tuple[dict, list]:
    npz_path = data_dir / "splits_with_anomaly_features.npz"
    names_path = data_dir / "feature_names_with_anomaly.json"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run phase3_anomaly_detection.py first."
        )
    data = dict(np.load(npz_path))
    feature_names = json.loads(names_path.read_text(encoding="utf-8")) if names_path.exists() else None
    return data, feature_names


# ----------------------------------------------------------------------
# 2. Model pipelines + tuning
# ----------------------------------------------------------------------
def build_search(estimator_name: str, random_state: int, n_iter: int, cv_folds: int, search_n_jobs: int = -1):
    """
    Returns a RandomizedSearchCV over an imblearn Pipeline(SMOTE -> classifier).
    SMOTE is refit inside every CV fold on that fold's training rows only.

    IMPORTANT (Windows resource-exhaustion fix): the classifiers are pinned
    to n_jobs=1. RandomizedSearchCV already parallelizes across CV
    folds/candidates via `search_n_jobs`; if the classifiers ALSO try to
    parallelize internally, you oversubscribe threads/processes by
    core-count-squared, which on Windows exhausts OS handles and raises
    `OSError: [WinError 1450]`. Only one layer should own the parallelism.
    """
    if estimator_name == "xgboost":
        clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=random_state,
            n_jobs=1,  # was -1: let RandomizedSearchCV own the parallelism instead
        )
        param_dist = {
            "clf__n_estimators": [200, 300, 400],
            "clf__max_depth": [3, 4, 5, 6],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__subsample": [0.7, 0.85, 1.0],
            "clf__colsample_bytree": [0.7, 0.85, 1.0],
            "smote__k_neighbors": [3, 5],
        }
    elif estimator_name == "lightgbm":
        clf = LGBMClassifier(
            objective="binary",
            random_state=random_state,
            n_jobs=1,  # was -1: same oversubscription fix
            verbosity=-1,
        )
        param_dist = {
            "clf__n_estimators": [200, 300, 400],
            "clf__max_depth": [-1, 5, 8],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__num_leaves": [15, 31, 63],
            "clf__subsample": [0.7, 0.85, 1.0],
            "smote__k_neighbors": [3, 5],
        }
    else:
        raise ValueError(estimator_name)

    pipeline = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=random_state)),
        ("clf", clf),
    ])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=search_n_jobs,
        refit=True,
    )
    return search


# ----------------------------------------------------------------------
# 3. Evaluation
# ----------------------------------------------------------------------
def evaluate_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )
    # precision_recall_curve returns one more point than thresholds
    best_idx = np.argmax(f1s[:-1]) if len(thresholds) > 0 else 0
    if len(thresholds) == 0:
        return {"threshold": 0.5, "precision": precisions[0], "recall": recalls[0], "f1": f1s[0]}
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": float(f1s[best_idx]),
    }


def plot_roc_curves(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["y_val"], r["val_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['val_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves -- validation split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "01_roc_curves.png", dpi=150)
    plt.close(fig)


def plot_precision_recall_vs_threshold(y_true: np.ndarray, y_proba: np.ndarray, model_name: str, out_dir: Path) -> None:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, precisions[:-1], label="Precision")
    ax.plot(thresholds, recalls[:-1], label="Recall")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision / Recall vs. threshold -- {model_name} (validation split)")
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=1, label="Default 0.5 cut-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "02_precision_recall_vs_threshold.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, threshold: float, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Repaid (0)", "Default (1)"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"{model_name} -- test split @ threshold={threshold:.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / "03_confusion_matrix.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 4. Orchestration
# ----------------------------------------------------------------------
def run_phase4(data_dir: str, out_dir: str, n_iter: int, cv_folds: int, random_state: int, search_n_jobs: int = -1) -> None:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Phase 3 outputs from {data_dir} ...")
    data, feature_names = load_data(data_dir)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    results = {}
    fitted_pipelines = {}

    for name, key in [("XGBoost", "xgboost"), ("LightGBM", "lightgbm")]:
        print(f"\nTuning {name} (SMOTE refit inside every CV fold, {cv_folds}-fold, scoring=ROC-AUC) ...")
        search = build_search(key, random_state, n_iter, cv_folds, search_n_jobs=search_n_jobs)
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        fitted_pipelines[name] = best_pipeline
        print(f"  best CV ROC-AUC: {search.best_score_:.4f}")
        print(f"  best params: {search.best_params_}")

        val_proba = best_pipeline.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        results[name] = {
            "cv_best_auc": search.best_score_,
            "best_params": search.best_params_,
            "y_val": y_val,
            "val_proba": val_proba,
            "val_auc": val_auc,
        }
        print(f"  validation ROC-AUC: {val_auc:.4f}")

        # Windows/loky fix: fully tear down this search's worker-process pool
        # before starting the next model's RandomizedSearchCV. Without this,
        # loky can leave the previous search's workers alive in the background;
        # the next search then opens a fresh batch on top of them and Windows
        # runs out of process/thread handles (WinError 1450), even though each
        # individual search would have been fine on its own.
        del search
        get_reusable_executor().shutdown(wait=True)

    print("\nComparing models on validation ROC-AUC ...")
    best_name = max(results, key=lambda k: results[k]["val_auc"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"  winner: {best_name} (val AUC={results[best_name]['val_auc']:.4f})")

    print("Generating diagnostic plots ...")
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

    print("Saving models and metrics ...")
    joblib.dump(fitted_pipelines["XGBoost"], out_dir / "xgboost_model.joblib")
    joblib.dump(fitted_pipelines["LightGBM"], out_dir / "lightgbm_model.joblib")
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
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    report_lines = [
        "PHASE 4 -- CLASS-IMBALANCE CORRECTION & SUPERVISED RISK SCORING",
        "=" * 70,
        "SMOTE was applied INSIDE the training pipeline only -- refit fresh on",
        "each CV fold's real training rows, and never applied to validation or",
        "test data at any point.",
        "",
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
        "Compare the TEST numbers above against these -- and remember results on a",
        "small or synthetic dataset will not hit these targets; they're meaningful",
        "once run on the full real Home Credit data.",
    ]
    report_text = "\n".join(report_lines)
    (out_dir / "phase4_report.txt").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"\nAll Phase 4 outputs saved to: {out_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4 -- SMOTE + XGBoost/LightGBM risk scoring for Home Credit Default Risk"
    )
    parser.add_argument("--data-dir", default="C:/Documents/Programms/Data_Science/micro_project",
                         help="Folder containing Phase 3 outputs (splits_with_anomaly_features.npz etc.)")
    parser.add_argument("--out-dir", default="C:/Documents/Programms/Data_Science/micro_project",
                         help="Folder to write Phase 4 outputs to")
    parser.add_argument("--n-iter", type=int, default=15,
                         help="Number of RandomizedSearchCV parameter combinations to try, per model")
    parser.add_argument("--cv-folds", type=int, default=3,
                         help="Number of stratified CV folds used during tuning")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--search-n-jobs", type=int, default=-1,
                         help="n_jobs for RandomizedSearchCV. If you still hit WinError 1450 "
                              "with n_jobs=-1, try a smaller fixed number (e.g. 4).")
    return parser.parse_args()


def main():
    args = parse_args()
    run_phase4(args.data_dir, args.out_dir, args.n_iter, args.cv_folds, args.random_state, args.search_n_jobs)


if __name__ == "__main__":
    main()