"""
Phase 4b -- Train & Tune LightGBM ONLY
=========================================================
Micro Project: Fraud and Repayment-Risk Detection in Microfinance Transactions
Dataset: Home Credit Default Risk (Kaggle)

Run this SEPARATELY from phase4a_train_xgboost.py -- as its own `python`
invocation, after phase4a has finished and exited. See the header comment
in phase4a_train_xgboost.py for why (Windows WinError 1450).

What this script does
----------------------
1. Loads splits_with_anomaly_features.npz from Phase 3.
2. Builds an imblearn Pipeline(SMOTE -> LGBMClassifier), same leakage-safe
   SMOTE-inside-the-pipeline setup as phase4a.
3. Tunes it with RandomizedSearchCV (StratifiedKFold, scored on ROC-AUC).
4. Refits the best pipeline on the full real training set.
5. Evaluates on the validation split (Precision/Recall/F1 @ 0.5, AUC-ROC).
6. Saves:
     - lightgbm_model.joblib
     - lightgbm_metrics.json

Setup
-----
    pip install pandas numpy scikit-learn imbalanced-learn lightgbm joblib

Usage
-----
    python phase4b_train_lightgbm.py
    python phase4b_train_lightgbm.py --data-dir ./data/processed --out-dir ./data/processed

If you hit WinError 1450 with the default settings, try:
    python phase4b_train_lightgbm.py --search-n-jobs 2
"""

import argparse
from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from common import evaluate_at_threshold, fit_search_safely, load_data, save_json


def build_search(random_state: int, n_iter: int, cv_folds: int, search_n_jobs: int) -> RandomizedSearchCV:
    clf = LGBMClassifier(
        objective="binary",
        random_state=random_state,
        n_jobs=1,  # pinned to 1: RandomizedSearchCV owns the parallelism, not the classifier
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
    pipeline = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=random_state)),
        ("clf", clf),
    ])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=search_n_jobs,
        refit=True,
    )


def run(data_dir: str, out_dir: str, n_iter: int, cv_folds: int, random_state: int, search_n_jobs: int, backend: str) -> None:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Phase 3 outputs from {data_dir} ...")
    data = load_data(data_dir)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    print(f"Tuning LightGBM ({cv_folds}-fold CV, {n_iter} candidates, scoring=ROC-AUC, "
          f"search_n_jobs={search_n_jobs}, backend={backend}) ...")
    search = build_search(random_state, n_iter, cv_folds, search_n_jobs)
    fit_search_safely(search, X_train, y_train, backend)
    best_pipeline = search.best_estimator_

    print(f"  best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"  best params: {search.best_params_}")

    val_proba = best_pipeline.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    default_metrics = evaluate_at_threshold(y_val, val_proba, 0.5)
    print(f"  validation ROC-AUC: {val_auc:.4f}")
    print(f"  validation @0.5 -> Precision: {default_metrics['precision']:.3f}  "
          f"Recall: {default_metrics['recall']:.3f}  F1: {default_metrics['f1']:.3f}")

    model_path = out_dir / "lightgbm_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"Saved {model_path}")

    metrics = {
        "model": "LightGBM",
        "cv_best_auc": search.best_score_,
        "best_params": search.best_params_,
        "val_auc": val_auc,
        "val_at_default_threshold_0.5": default_metrics,
    }
    metrics_path = out_dir / "lightgbm_metrics.json"
    save_json(metrics, metrics_path)
    print(f"Saved {metrics_path}")
    print("\nDone. If you haven't already, run phase4a_train_xgboost.py separately too,")
    print("then phase4c_select_and_evaluate.py to pick the winner and get test-set numbers.")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4b -- Train & tune LightGBM only")
    parser.add_argument("--data-dir", default="C:/Documents/Programms/Data_Science/micro_project")
    parser.add_argument("--out-dir", default="C:/Documents/Programms/Data_Science/micro_project")
    parser.add_argument("--n-iter", type=int, default=15)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--search-n-jobs", type=int, default=-1,
                         help="n_jobs for RandomizedSearchCV.")
    parser.add_argument("--backend", default="threading", choices=["threading", "loky", "sequential"],
                         help="joblib backend. 'threading' (default) avoids spawning new OS "
                              "processes, which is what fixes WinError 1450 on Windows. "
                              "'loky' is joblib's normal process-based backend (what caused the "
                              "error). 'sequential' disables parallelism entirely as a last resort.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.out_dir, args.n_iter, args.cv_folds, args.random_state,
        args.search_n_jobs, args.backend)