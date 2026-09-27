"""
Phase 2 -- Preprocessing & Feature Engineering
=========================================================
Micro Project: Fraud and Repayment-Risk Detection in Microfinance Transactions
Dataset: Home Credit Default Risk (Kaggle)

What this script does
----------------------
1. Loads application_train.csv + bureau.csv from a local folder.
2. Cleans the known DAYS_EMPLOYED == 365243 anomaly.
3. Aggregates bureau.csv (the client's OTHER loans, from other banks)
   into one row per SK_ID_CURR and merges it onto the main table.
4. Engineers the behavioural/ratio features promised in the synopsis:
     - credit-to-income, annuity-to-income, annuity-to-credit,
       credit-to-goods-price ratios
     - age in years, years employed, employment-to-age ratio
     - EXT_SOURCE mean / count-missing
     - total address-mismatch flag count
     - bureau-derived: number of previous credits, number active,
       total debt, total overdue amount/count, overdue rate,
       distinct credit types, debt-to-credit ratio
5. Splits into stratified train / validation / test sets (fit-only
   on train to avoid leakage).
6. Builds and fits a scikit-learn ColumnTransformer (median-impute +
   scale for numeric, most-frequent-impute + one-hot for categorical)
   on the TRAIN split only, then transforms all three splits.
7. Saves:
     - engineered_features.csv        (full table, pre-split, human-readable)
     - processed_splits.npz           (X_train/y_train/X_val/y_val/X_test/y_test as arrays)
     - preprocessor.joblib            (fitted ColumnTransformer, reusable in Phase 3/4/dashboard)
     - feature_names.json             (column names for the transformed matrix, in order)
     - phase2_report.txt              (shapes, class balance per split, feature counts)

Setup
-----
    pip install pandas numpy scikit-learn joblib

Usage
-----
    python phase2_preprocessing_feature_engineering.py
    python phase2_preprocessing_feature_engineering.py --data-dir /path/to/csvs --out-dir ./data/processed
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ANOMALY_DAYS_EMPLOYED = 365243
ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

# Columns that should never enter the model as features (IDs / label / raw
# day-counts that we've already converted into cleaner derived features).
DROP_COLS = [ID_COL, TARGET_COL, "DAYS_BIRTH", "DAYS_EMPLOYED"]


# ----------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------
def load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    app_path = data_dir / "application_train.csv"
    if not app_path.exists():
        raise FileNotFoundError(
            f"application_train.csv not found in {data_dir}. "
            "Check --data-dir points to the folder containing your extracted CSVs."
        )
    print(f"  [load] application_train.csv ...")
    app = pd.read_csv(app_path)

    bureau_path = data_dir / "bureau.csv"
    bureau = None
    if bureau_path.exists():
        print(f"  [load] bureau.csv ...")
        bureau = pd.read_csv(bureau_path)
    else:
        print("  [skip] bureau.csv not found -- bureau-derived features will be skipped.")

    return app, bureau


# ----------------------------------------------------------------------
# 2. Cleaning
# ----------------------------------------------------------------------
def clean_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """See Phase 1: DAYS_EMPLOYED == 365243 means 'not currently employed'."""
    if "DAYS_EMPLOYED" not in df.columns:
        return df
    df = df.copy()
    df["DAYS_EMPLOYED_ANOM"] = df["DAYS_EMPLOYED"] == ANOMALY_DAYS_EMPLOYED
    df.loc[df["DAYS_EMPLOYED_ANOM"], "DAYS_EMPLOYED"] = np.nan
    return df


# ----------------------------------------------------------------------
# 3. Bureau aggregation (one row per SK_ID_CURR)
# ----------------------------------------------------------------------
def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses bureau.csv (many rows per client, one per external credit
    line) into a single row per SK_ID_CURR summarising that client's
    credit history with OTHER institutions.
    """
    b = bureau.copy()

    is_active = (b["CREDIT_ACTIVE"] == "Active") if "CREDIT_ACTIVE" in b.columns else pd.Series(False, index=b.index)
    has_overdue = (b["AMT_CREDIT_SUM_OVERDUE"] > 0) if "AMT_CREDIT_SUM_OVERDUE" in b.columns else pd.Series(False, index=b.index)

    b["_IS_ACTIVE"] = is_active.astype(int)
    b["_HAS_OVERDUE"] = has_overdue.astype(int)

    agg_spec = {ID_COL: "size"}
    rename_map = {ID_COL: "BUREAU_CNT"}

    if "_IS_ACTIVE" in b.columns:
        agg_spec["_IS_ACTIVE"] = "sum"
    if "_HAS_OVERDUE" in b.columns:
        agg_spec["_HAS_OVERDUE"] = "sum"
    if "AMT_CREDIT_SUM" in b.columns:
        agg_spec["AMT_CREDIT_SUM"] = "sum"
    if "AMT_CREDIT_SUM_DEBT" in b.columns:
        agg_spec["AMT_CREDIT_SUM_DEBT"] = "sum"
    if "AMT_CREDIT_SUM_OVERDUE" in b.columns:
        agg_spec["AMT_CREDIT_SUM_OVERDUE"] = "sum"
    if "CREDIT_TYPE" in b.columns:
        agg_spec["CREDIT_TYPE"] = "nunique"

    grouped = b.groupby(ID_COL).agg(agg_spec).rename(columns={
        ID_COL: "BUREAU_CNT",
        "_IS_ACTIVE": "BUREAU_ACTIVE_CNT",
        "_HAS_OVERDUE": "BUREAU_OVERDUE_CNT",
        "AMT_CREDIT_SUM": "BUREAU_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT": "BUREAU_DEBT_SUM",
        "AMT_CREDIT_SUM_OVERDUE": "BUREAU_OVERDUE_SUM",
        "CREDIT_TYPE": "BUREAU_CREDIT_TYPES_NUNIQUE",
    })

    # BUREAU_CNT needs its own true count (agg_spec[ID_COL]='size' already gives
    # that via groupby's implicit index handling above); recompute explicitly
    # to be safe regardless of which optional columns existed.
    grouped["BUREAU_CNT"] = b.groupby(ID_COL).size()

    if "BUREAU_OVERDUE_CNT" in grouped.columns:
        grouped["BUREAU_OVERDUE_RATE"] = grouped["BUREAU_OVERDUE_CNT"] / grouped["BUREAU_CNT"].replace(0, np.nan)
    if {"BUREAU_DEBT_SUM", "BUREAU_CREDIT_SUM"}.issubset(grouped.columns):
        grouped["BUREAU_DEBT_CREDIT_RATIO"] = grouped["BUREAU_DEBT_SUM"] / grouped["BUREAU_CREDIT_SUM"].replace(0, np.nan)

    grouped = grouped.reset_index()
    return grouped


# ----------------------------------------------------------------------
# 4. Feature engineering
# ----------------------------------------------------------------------
def engineer_features(app: pd.DataFrame, bureau_agg: pd.DataFrame | None) -> pd.DataFrame:
    df = app.copy()

    # -- Ratio features --------------------------------------------------
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)
    if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(df.columns):
        df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)

    # -- Age / employment --------------------------------------------------
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    if "DAYS_EMPLOYED" in df.columns:
        df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365.25
    if {"AGE_YEARS", "YEARS_EMPLOYED"}.issubset(df.columns):
        df["EMPLOYED_AGE_RATIO"] = df["YEARS_EMPLOYED"] / df["AGE_YEARS"].replace(0, np.nan)

    # -- External credit scores --------------------------------------------
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_MISSING_CNT"] = df[ext_cols].isnull().sum(axis=1)

    # -- Address-mismatch flag count ---------------------------------------
    mismatch_cols = [c for c in [
        "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
    ] if c in df.columns]
    if mismatch_cols:
        df["ADDRESS_MISMATCH_CNT"] = df[mismatch_cols].sum(axis=1)

    # -- Merge bureau aggregates --------------------------------------------
    if bureau_agg is not None:
        df = df.merge(bureau_agg, on=ID_COL, how="left")
        bureau_feature_cols = [c for c in bureau_agg.columns if c != ID_COL]
        # a client absent from bureau.csv genuinely has zero external credit
        # history, not a missing measurement -- fill those with 0.
        df[bureau_feature_cols] = df[bureau_feature_cols].fillna(0)

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list, list]:
    """Auto-detects numeric vs categorical feature columns, excluding IDs/target/raw-days."""
    feature_df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    numeric_cols = feature_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    # exclude (not include=["object"]) so this works the same whether pandas
    # represents text columns as "object" (pandas <3) or "str" (pandas >=3).
    categorical_cols = feature_df.drop(columns=numeric_cols).columns.tolist()
    return numeric_cols, categorical_cols


# ----------------------------------------------------------------------
# 5. Split
# ----------------------------------------------------------------------
def stratified_split(df: pd.DataFrame, test_size: float, val_size: float, random_state: int):
    """
    Two-stage stratified split: first carve off the test set, then split
    the remainder into train/val. val_size is expressed as a fraction of
    the ORIGINAL dataset (matching common convention), not of the remainder.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COL], random_state=random_state
    )
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val_size, stratify=train_val[TARGET_COL], random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ----------------------------------------------------------------------
# 6. Preprocessor
# ----------------------------------------------------------------------
def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])


def get_output_feature_names(preprocessor: ColumnTransformer, numeric_cols: list, categorical_cols: list) -> list:
    names = list(numeric_cols)
    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        names += list(ohe.get_feature_names_out(categorical_cols))
    return names


# ----------------------------------------------------------------------
# 7. Orchestration
# ----------------------------------------------------------------------
def run_phase2(data_dir: str, out_dir: str, test_size: float, val_size: float, random_state: int) -> None:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir} ...")
    app, bureau = load_raw(data_dir)

    print("Cleaning DAYS_EMPLOYED anomaly ...")
    app = clean_days_employed(app)

    bureau_agg = None
    if bureau is not None:
        print("Aggregating bureau.csv per client ...")
        bureau_agg = aggregate_bureau(bureau)

    print("Engineering features ...")
    df = engineer_features(app, bureau_agg)

    engineered_path = out_dir / "engineered_features.csv"
    df.to_csv(engineered_path, index=False)
    print(f"  saved {engineered_path} ({df.shape[0]:,} rows x {df.shape[1]} columns)")

    print("Splitting into train / val / test (stratified on TARGET) ...")
    train_df, val_df, test_df = stratified_split(df, test_size, val_size, random_state)

    numeric_cols, categorical_cols = get_feature_columns(train_df)
    print(f"  {len(numeric_cols)} numeric features, {len(categorical_cols)} categorical features")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    print("Fitting preprocessor on TRAIN split only, transforming all splits ...")
    X_train = preprocessor.fit_transform(train_df[numeric_cols + categorical_cols])
    X_val = preprocessor.transform(val_df[numeric_cols + categorical_cols])
    X_test = preprocessor.transform(test_df[numeric_cols + categorical_cols])

    y_train = train_df[TARGET_COL].to_numpy()
    y_val = val_df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    id_train = train_df[ID_COL].to_numpy()
    id_val = val_df[ID_COL].to_numpy()
    id_test = test_df[ID_COL].to_numpy()

    feature_names = get_output_feature_names(preprocessor, numeric_cols, categorical_cols)

    npz_path = out_dir / "processed_splits.npz"
    np.savez_compressed(
        npz_path,
        X_train=X_train, y_train=y_train, id_train=id_train,
        X_val=X_val, y_val=y_val, id_val=id_val,
        X_test=X_test, y_test=y_test, id_test=id_test,
    )
    print(f"  saved {npz_path}")

    preproc_path = out_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preproc_path)
    print(f"  saved {preproc_path}")

    feature_names_path = out_dir / "feature_names.json"
    feature_names_path.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    print(f"  saved {feature_names_path}")

    def class_balance(y):
        n = len(y)
        n1 = int(y.sum())
        return f"{n:,} rows | TARGET=1: {n1:,} ({100 * n1 / n:.2f}%)"

    report_lines = [
        "PHASE 2 -- PREPROCESSING & FEATURE ENGINEERING REPORT",
        "=" * 70,
        f"Engineered table (pre-split): {df.shape[0]:,} rows x {df.shape[1]} columns",
        f"Numeric features used   : {len(numeric_cols)}",
        f"Categorical features used: {len(categorical_cols)}",
        f"Final transformed width : {X_train.shape[1]} columns (after one-hot encoding)",
        "",
        f"Train split : {class_balance(y_train)}",
        f"Val split   : {class_balance(y_val)}",
        f"Test split  : {class_balance(y_test)}",
        "",
        "Numeric features:",
        *[f"  - {c}" for c in numeric_cols],
        "",
        "Categorical features (one-hot encoded):",
        *[f"  - {c}" for c in categorical_cols],
    ]
    report_text = "\n".join(report_lines)
    (out_dir / "phase2_report.txt").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"\nAll Phase 2 outputs saved to: {out_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2 -- Preprocessing & Feature Engineering for Home Credit Default Risk"
    )
    parser.add_argument("--data-dir", default="C:/Documents/Programms/Data_Science/micro_project",
                         help="Folder containing the extracted Home Credit CSV files")
    parser.add_argument("--out-dir", default="C:/Documents/Programms/Data_Science/micro_project",
                         help="Folder to write engineered/processed outputs to")
    parser.add_argument("--test-size", type=float, default=0.15,
                         help="Fraction of the full dataset held out as the test split")
    parser.add_argument("--val-size", type=float, default=0.15,
                         help="Fraction of the full dataset held out as the validation split")
    parser.add_argument("--random-state", type=int, default=42,
                         help="Random seed for reproducible splits")
    return parser.parse_args()


def main():
    args = parse_args()
    run_phase2(args.data_dir, args.out_dir, args.test_size, args.val_size, args.random_state)


if __name__ == "__main__":
    main()