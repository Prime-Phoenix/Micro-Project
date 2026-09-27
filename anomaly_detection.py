"""
Phase 3 -- Unsupervised Anomaly Detection
=========================================================
Micro Project: Fraud and Repayment-Risk Detection in Microfinance Transactions
Dataset: Home Credit Default Risk (Kaggle)

What this script does
----------------------
1. Loads processed_splits.npz + feature_names.json from Phase 2.
2. Fits an Isolation Forest on X_train ONLY (label-free, TARGET is never
   used for fitting -- that's the entire point of this stage).
3. Fits an Autoencoder baseline on X_train, implemented as a
   scikit-learn MLPRegressor trained to reconstruct its own input
   (bottlenecked hidden layer forces it to learn a compressed
   representation of "normal" transactions). Reconstruction error
   (MSE) becomes the anomaly score. This avoids a heavy TensorFlow/
   PyTorch dependency while still giving a genuinely different,
   comparative anomaly-detection method, as the synopsis specifies.
4. Scores every split with both methods.
5. Validates the scores against the TRUE TARGET label (validation-set
   only, never used for fitting) to sanity-check that anomalous
   transactions are actually enriched for real payment difficulty --
   this becomes a figure/table in your report.
6. Appends both anomaly scores as two new engineered features onto
   every split's feature matrix (this is the "anomaly score as an
   engineered feature" step Phase 4 / XGBoost will consume).
7. Saves:
     - anomaly_scores.csv                (id, split, true target, iso score, ae score)
     - splits_with_anomaly_features.npz  (X_* with 2 extra columns + everything from Phase 2)
     - feature_names_with_anomaly.json
     - isolation_forest.joblib / autoencoder.joblib
     - 01_score_distribution_by_target.png
     - 02_score_correlation.png
     - phase3_report.txt

Setup
-----
    pip install pandas numpy scikit-learn joblib matplotlib seaborn

Usage
-----
    python phase3_anomaly_detection.py
    python phase3_anomaly_detection.py --data-dir ./data/processed --out-dir ./data/processed
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------
def load_processed(data_dir: Path) -> tuple[dict, list]:
    npz_path = data_dir / "processed_splits.npz"
    names_path = data_dir / "feature_names.json"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run phase2_preprocessing_feature_engineering.py first."
        )
    data = dict(np.load(npz_path))
    feature_names = json.loads(names_path.read_text(encoding="utf-8")) if names_path.exists() else None
    return data, feature_names


# ----------------------------------------------------------------------
# 2. Isolation Forest
# ----------------------------------------------------------------------
def fit_isolation_forest(X_train: np.ndarray, contamination, random_state: int) -> IsolationForest:
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def isolation_forest_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    model.decision_function: HIGHER = more normal.
    We flip the sign so, consistently across this whole project,
    HIGHER anomaly_score = MORE anomalous.
    """
    return -model.decision_function(X)


# ----------------------------------------------------------------------
# 3. Autoencoder (MLPRegressor reconstructing its own input)
# ----------------------------------------------------------------------
def fit_autoencoder(X_train: np.ndarray, hidden_layers: tuple, random_state: int) -> MLPRegressor:
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=random_state,
    )
    model.fit(X_train, X_train)  # target == input: this is what makes it an autoencoder
    return model


def autoencoder_scores(model: MLPRegressor, X: np.ndarray) -> np.ndarray:
    """Per-sample reconstruction error (mean squared error). Higher = more anomalous."""
    X_hat = model.predict(X)
    return np.mean((X - X_hat) ** 2, axis=1)


# ----------------------------------------------------------------------
# 4. Validation against the true label (diagnostic only -- never used for fitting)
# ----------------------------------------------------------------------
def top_k_lift(scores: np.ndarray, y: np.ndarray, k_fraction: float = 0.10) -> dict:
    n = len(scores)
    k = max(1, int(n * k_fraction))
    top_idx = np.argsort(scores)[-k:]
    base_rate = y.mean()
    top_rate = y[top_idx].mean()
    return {
        "k_fraction": k_fraction,
        "k_count": k,
        "base_rate_pct": base_rate * 100,
        "top_rate_pct": top_rate * 100,
        "lift": (top_rate / base_rate) if base_rate > 0 else float("nan"),
    }


def plot_score_distribution_by_target(iso_scores: np.ndarray, ae_scores: np.ndarray, y: np.ndarray, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, scores, title in zip(axes, [iso_scores, ae_scores], ["Isolation Forest anomaly score", "Autoencoder reconstruction error"]):
        df = pd.DataFrame({"score": scores, "TARGET": y})
        sns.kdeplot(data=df, x="score", hue="TARGET", common_norm=False, fill=True, alpha=0.3, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("anomaly score (higher = more anomalous)")
    fig.suptitle("Anomaly score distributions split by TRUE label (validation split)", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "01_score_distribution_by_target.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_correlation(iso_scores: np.ndarray, ae_scores: np.ndarray, y: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sc = ax.scatter(iso_scores, ae_scores, c=y, cmap="coolwarm", alpha=0.5, s=12)
    ax.set_xlabel("Isolation Forest anomaly score")
    ax.set_ylabel("Autoencoder reconstruction error")
    ax.set_title("Agreement between the two anomaly-detection methods\n(colour = true TARGET)")
    fig.colorbar(sc, ax=ax, label="TARGET")
    fig.tight_layout()
    fig.savefig(out_dir / "02_score_correlation.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 5. Orchestration
# ----------------------------------------------------------------------
def run_phase3(data_dir: str, out_dir: str, contamination, ae_hidden_layers: tuple, random_state: int) -> None:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Phase 2 outputs from {data_dir} ...")
    data, feature_names = load_processed(data_dir)
    X_train, y_train, id_train = data["X_train"], data["y_train"], data["id_train"]
    X_val, y_val, id_val = data["X_val"], data["y_val"], data["id_val"]
    X_test, y_test, id_test = data["X_test"], data["y_test"], data["id_test"]

    print(f"Fitting Isolation Forest on X_train ({X_train.shape[0]:,} rows, TARGET not used) ...")
    iso_model = fit_isolation_forest(X_train, contamination, random_state)

    print(f"Fitting Autoencoder (MLPRegressor {ae_hidden_layers}) on X_train (TARGET not used) ...")
    ae_model = fit_autoencoder(X_train, ae_hidden_layers, random_state)

    print("Scoring all splits with both methods ...")
    splits = {
        "train": (X_train, y_train, id_train),
        "val": (X_val, y_val, id_val),
        "test": (X_test, y_test, id_test),
    }
    iso_scores, ae_scores = {}, {}
    for name, (X, y, ids) in splits.items():
        iso_scores[name] = isolation_forest_scores(iso_model, X)
        ae_scores[name] = autoencoder_scores(ae_model, X)

    print("Validating scores against the true label (validation split; label was never used for fitting) ...")
    plot_score_distribution_by_target(iso_scores["val"], ae_scores["val"], y_val, out_dir)
    plot_score_correlation(iso_scores["val"], ae_scores["val"], y_val, out_dir)

    iso_lift = top_k_lift(iso_scores["val"], y_val, k_fraction=0.10)
    ae_lift = top_k_lift(ae_scores["val"], y_val, k_fraction=0.10)

    print("Saving per-borrower score table ...")
    rows = []
    for name, (X, y, ids) in splits.items():
        rows.append(pd.DataFrame({
            "SK_ID_CURR": ids,
            "split": name,
            "TARGET": y,
            "iso_anomaly_score": iso_scores[name],
            "ae_anomaly_score": ae_scores[name],
        }))
    scores_df = pd.concat(rows, ignore_index=True)
    scores_df.to_csv(out_dir / "anomaly_scores.csv", index=False)

    print("Appending anomaly scores as engineered features and saving augmented splits ...")
    augmented = {}
    for name, (X, y, ids) in splits.items():
        extra = np.column_stack([iso_scores[name], ae_scores[name]])
        augmented[f"X_{name}"] = np.hstack([X, extra])
        augmented[f"y_{name}"] = y
        augmented[f"id_{name}"] = ids
    np.savez_compressed(out_dir / "splits_with_anomaly_features.npz", **augmented)

    if feature_names is not None:
        new_feature_names = feature_names + ["ISO_ANOMALY_SCORE", "AE_ANOMALY_SCORE"]
        (out_dir / "feature_names_with_anomaly.json").write_text(
            json.dumps(new_feature_names, indent=2), encoding="utf-8"
        )

    joblib.dump(iso_model, out_dir / "isolation_forest.joblib")
    joblib.dump(ae_model, out_dir / "autoencoder.joblib")

    report_lines = [
        "PHASE 3 -- UNSUPERVISED ANOMALY DETECTION REPORT",
        "=" * 70,
        f"Isolation Forest: n_estimators=300, contamination={contamination}, fit on {X_train.shape[0]:,} train rows",
        f"Autoencoder     : MLPRegressor hidden_layers={ae_hidden_layers}, fit on {X_train.shape[0]:,} train rows",
        "",
        "Both models were fit WITHOUT using TARGET. The lift figures below use",
        "TARGET only to VALIDATE the scores after the fact, on the validation split.",
        "",
        f"-- Isolation Forest top-{int(iso_lift['k_fraction']*100)}% lift --",
        f"  Base default rate in validation split : {iso_lift['base_rate_pct']:.2f}%",
        f"  Default rate among top {iso_lift['k_count']} most-anomalous  : {iso_lift['top_rate_pct']:.2f}%",
        f"  Lift                                   : {iso_lift['lift']:.2f}x",
        "",
        f"-- Autoencoder top-{int(ae_lift['k_fraction']*100)}% lift --",
        f"  Base default rate in validation split : {ae_lift['base_rate_pct']:.2f}%",
        f"  Default rate among top {ae_lift['k_count']} most-anomalous  : {ae_lift['top_rate_pct']:.2f}%",
        f"  Lift                                   : {ae_lift['lift']:.2f}x",
        "",
        "A lift > 1.0x means anomalous transactions are enriched for real payment",
        "difficulty relative to the base rate -- evidence the unsupervised layer is",
        "catching a genuinely useful signal, not just noise, before Phase 4 ever",
        "sees a label.",
        "",
        f"Output feature matrix width: {X_train.shape[1]} (Phase 2) + 2 (iso + ae scores) "
        f"= {X_train.shape[1] + 2}",
    ]
    report_text = "\n".join(report_lines)
    (out_dir / "phase3_report.txt").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"\nAll Phase 3 outputs saved to: {out_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3 -- Unsupervised Anomaly Detection for Home Credit Default Risk"
    )
    parser.add_argument("--data-dir", default="C:/Documents/Programms/Data_Science/micro_project",
                         help="Folder containing Phase 2 outputs (processed_splits.npz etc.)")
    parser.add_argument("--out-dir", default="C:/Documents/Programms/Data_Science/micro_project",
                         help="Folder to write Phase 3 outputs to")
    parser.add_argument("--contamination", default="auto",
                         help="Isolation Forest contamination: 'auto' or a float e.g. 0.08")
    parser.add_argument("--ae-hidden-layers", default="32,16,32",
                         help="Comma-separated autoencoder hidden layer sizes, e.g. '32,16,32'")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    contamination = args.contamination
    if contamination != "auto":
        contamination = float(contamination)
    ae_hidden_layers = tuple(int(x) for x in args.ae_hidden_layers.split(","))
    run_phase3(args.data_dir, args.out_dir, contamination, ae_hidden_layers, args.random_state)


if __name__ == "__main__":
    main()