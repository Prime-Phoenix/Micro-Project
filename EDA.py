"""
Phase 1 -- Data Acquisition & Exploratory Data Analysis
=========================================================
Micro Project: Fraud and Repayment-Risk Detection in Microfinance Transactions
Dataset: Home Credit Default Risk (Kaggle)
https://www.kaggle.com/c/home-credit-default-risk

What this script does
----------------------
1. Loads all 8 Home Credit CSV files from a local folder (any that are
   missing are skipped with a warning, so this also works if you only
   have a subset downloaded).
2. Cleans the known DAYS_EMPLOYED == 365243 anomaly in application_train.
3. Prints/saves shape, dtype and missing-value summaries for every file.
4. Generates and saves the core EDA plots into --out-dir.
5. Writes a plain-text EDA summary report alongside the plots.

Setup
-----
    pip install pandas numpy matplotlib seaborn

Before running, place the extracted Home Credit CSVs in a folder, e.g.
    ./data/home-credit-default-risk/application_train.csv
    ./data/home-credit-default-risk/application_test.csv
    ./data/home-credit-default-risk/bureau.csv
    ./data/home-credit-default-risk/bureau_balance.csv
    ./data/home-credit-default-risk/previous_application.csv
    ./data/home-credit-default-risk/POS_CASH_balance.csv
    ./data/home-credit-default-risk/credit_card_balance.csv
    ./data/home-credit-default-risk/installments_payments.csv

Usage
-----
    # Default: reads from ./data/home-credit-default-risk/, writes to ./reports/phase1/
    python phase1_data_acquisition_eda.py

    # Custom locations
    python phase1_data_acquisition_eda.py --data-dir /path/to/csvs --out-dir /path/to/reports
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe backend, no display needed
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------
# File registry -- every table in the Home Credit release
# ----------------------------------------------------------------------
DATA_FILES = {
    "app_train": "application_train.csv",
    "app_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
}

ANOMALY_DAYS_EMPLOYED = 365243


# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------
def load_data(data_dir: Path, files: dict | None = None) -> dict:
    """
    Loads the requested CSV files (default: all 8) into a dict of
    DataFrames, keyed the same way as DATA_FILES. Missing files are
    skipped with a warning rather than raising, so this still runs
    on a partial download (e.g. just application_train.csv).
    """
    files = files or DATA_FILES
    frames = {}
    for key, filename in files.items():
        path = data_dir / filename
        if not path.exists():
            print(f"  [skip] {filename} not found in {data_dir}")
            continue
        print(f"  [load] {filename} ...")
        frames[key] = pd.read_csv(path)

    if "app_train" not in frames:
        raise FileNotFoundError(
            f"application_train.csv is required and was not found in {data_dir}. "
            "Check --data-dir points to the folder containing your extracted CSVs."
        )
    return frames


# ----------------------------------------------------------------------
# 2. Cleaning
# ----------------------------------------------------------------------
def clean_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Home Credit encodes 'not currently employed' as DAYS_EMPLOYED == 365243
    instead of a real negative day-count. Left as-is, this single value
    (about 18% of rows) distorts any numeric use of the column. We flag
    it in a new boolean column and null out the original value.
    """
    if "DAYS_EMPLOYED" not in df.columns:
        return df
    df = df.copy()
    df["DAYS_EMPLOYED_ANOM"] = df["DAYS_EMPLOYED"] == ANOMALY_DAYS_EMPLOYED
    df.loc[df["DAYS_EMPLOYED_ANOM"], "DAYS_EMPLOYED"] = np.nan
    return df


# ----------------------------------------------------------------------
# 3. Text summaries
# ----------------------------------------------------------------------
def summarize_dataframe(df: pd.DataFrame, name: str) -> str:
    lines = [f"\n{'=' * 70}", f"{name}  --  shape: {df.shape[0]:,} rows x {df.shape[1]} columns", "=" * 70]

    dtype_counts = df.dtypes.value_counts()
    lines.append("Dtype breakdown:")
    for dtype, count in dtype_counts.items():
        lines.append(f"  {str(dtype):10s}: {count}")

    missing = df.isnull().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0]
    lines.append(f"\nColumns with missing values: {len(missing)} / {df.shape[1]}")
    if len(missing) > 0:
        lines.append("Top 15 by missing %:")
        for col, pct in missing.head(15).items():
            lines.append(f"  {col:45s}: {pct:5.1f}%")

    return "\n".join(lines)


def target_balance_report(app_train: pd.DataFrame) -> str:
    counts = app_train["TARGET"].value_counts()
    pct = app_train["TARGET"].value_counts(normalize=True) * 100
    n0, n1 = counts.get(0, 0), counts.get(1, 0)
    lines = [
        "\nTARGET class balance (application_train.csv):",
        f"  0 (repaid)             : {n0:>8,}  ({pct.get(0, 0):5.2f}%)",
        f"  1 (payment difficulty) : {n1:>8,}  ({pct.get(1, 0):5.2f}%)",
        f"  Imbalance ratio        : ~1 : {n0 / max(n1, 1):.1f}",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 4. Plots
# ----------------------------------------------------------------------
def plot_target_distribution(app_train: pd.DataFrame, out_dir: Path) -> None:
    counts = app_train["TARGET"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_xticklabels(["Repaid (0)", "Payment difficulty (1)"], rotation=0)
    ax.set_ylabel("Number of applicants")
    ax.set_title("TARGET class distribution")
    for i, v in enumerate(counts):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "01_target_distribution.png", dpi=150)
    plt.close(fig)


def plot_missing_values(df: pd.DataFrame, name: str, out_dir: Path, top_n: int = 20) -> None:
    missing = df.isnull().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0].head(top_n)
    if missing.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(missing))))
    missing.sort_values().plot(kind="barh", ax=ax, color="#8172B2")
    ax.set_xlabel("% missing")
    ax.set_title(f"Top {len(missing)} columns by missing % -- {name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"02_missing_values_{name}.png", dpi=150)
    plt.close(fig)


def plot_numeric_by_target(app_train: pd.DataFrame, out_dir: Path) -> None:
    cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH",
            "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    cols = [c for c in cols if c in app_train.columns]
    if not cols:
        return
    ncols = 2
    nrows = (len(cols) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        data = app_train[[col, "TARGET"]].dropna()
        if col == "DAYS_BIRTH":
            data = data.assign(**{col: -data[col] / 365.25})
            ax.set_xlabel("Age (years)")
        elif col == "DAYS_EMPLOYED":
            data = data.assign(**{col: -data[col] / 365.25})
            ax.set_xlabel("Years employed")
        else:
            ax.set_xlabel(col)
        sns.kdeplot(data=data, x=col, hue="TARGET", common_norm=False, ax=ax, fill=True, alpha=0.3)
        ax.set_title(col)
        ax.set_ylabel("")

    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Feature distributions split by TARGET", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "03_numeric_distributions_by_target.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_categorical_default_rate(app_train: pd.DataFrame, out_dir: Path) -> None:
    cols = ["NAME_CONTRACT_TYPE", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE",
            "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "CODE_GENDER"]
    cols = [c for c in cols if c in app_train.columns]
    if not cols:
        return
    ncols = 2
    nrows = (len(cols) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes = np.array(axes).flatten()

    overall_rate = app_train["TARGET"].mean() * 100
    for i, col in enumerate(cols):
        ax = axes[i]
        rate = (app_train.groupby(col)["TARGET"].mean() * 100).sort_values(ascending=False)
        rate.plot(kind="bar", ax=ax, color="#55A868")
        ax.axhline(overall_rate, color="red", linestyle="--", linewidth=1, label="Overall rate")
        ax.set_ylabel("Default rate (%)")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)

    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(out_dir / "04_categorical_default_rate.png", dpi=150)
    plt.close(fig)


def plot_correlation_with_target(app_train: pd.DataFrame, out_dir: Path, top_n: int = 15) -> None:
    numeric_df = app_train.select_dtypes(include=[np.number])
    if "TARGET" not in numeric_df.columns:
        return
    corr = numeric_df.corr()["TARGET"].drop("TARGET").dropna().sort_values()
    if corr.empty:
        return
    top = pd.concat([corr.head(top_n), corr.tail(top_n)]).drop_duplicates()
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(top))))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top.values]
    top.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Correlation with TARGET")
    ax.set_title(f"Top {len(top)} positive & negative correlations with TARGET")
    fig.tight_layout()
    fig.savefig(out_dir / "05_correlation_with_target.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 5. Orchestration
# ----------------------------------------------------------------------
def run_eda(data_dir: str, out_dir: str) -> None:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir} ...")
    frames = load_data(data_dir)

    app_train = clean_days_employed(frames["app_train"])
    frames["app_train"] = app_train

    report_lines = ["HOME CREDIT DEFAULT RISK -- PHASE 1 EDA REPORT", "=" * 70]

    print("Summarizing tables ...")
    for name, df in frames.items():
        report_lines.append(summarize_dataframe(df, name))
        plot_missing_values(df, name, out_dir)

    report_lines.append(target_balance_report(app_train))

    print("Generating plots ...")
    plot_target_distribution(app_train, out_dir)
    plot_numeric_by_target(app_train, out_dir)
    plot_categorical_default_rate(app_train, out_dir)
    plot_correlation_with_target(app_train, out_dir)

    report_text = "\n".join(report_lines)
    report_path = out_dir / "phase1_eda_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nSaved plots and report to: {out_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1 -- Data Acquisition & EDA for Home Credit Default Risk"
    )
    parser.add_argument(
        "--data-dir", default="C:/Documents/Programms/Data_Science/micro_project",
        help="Folder containing the extracted Home Credit CSV files",
    )
    parser.add_argument(
        "--out-dir", default="C:/Documents/Programms/Data_Science/micro_project/Plots",
        help="Folder to write EDA plots and the summary report to",
    )
    return parser.parse_args()


args = parse_args()
run_eda(args.data_dir, args.out_dir)
