"""
train_demo.py -- Ecommerce Retention ML Pipeline demo bundle trainer.
Uses the real ecommerce_sales_data.csv dataset, treating each
(Product Name x Region) combination as a synthetic customer to generate
RFM-style features, then trains the retention classifier + spend regressor.
Saves models/demo_bundle.pkl for the Streamlit dashboard.
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    mean_squared_error, classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from src.features.build_features import build_customer_features

BUNDLE_PATH = Path("models/demo_bundle.pkl")
CSV_PATH = Path("ecommerce_sales_data.csv")

# ── Load & prepare raw data ───────────────────────────────────────
print("Loading data...")
raw = pd.read_csv(CSV_PATH)
raw.columns = raw.columns.str.strip().str.lower().str.replace(" ", "_")
# canonical column names: order_date, product_name, category, region, quantity, sales, profit

raw["order_date"] = pd.to_datetime(raw["order_date"], errors="coerce")
raw = raw.dropna(subset=["order_date"])

# Create synthetic customer_id: product_name x region x month (monthly cohorts)
# Each unique (product, region, year-month) group = one synthetic customer
# This gives ~1440 possible cohorts but pruned to those with real orders
raw["order_id"] = raw.index.astype(str)
raw["amount"] = pd.to_numeric(raw["sales"], errors="coerce")
raw["profit"] = pd.to_numeric(raw["profit"], errors="coerce")
raw["ym"] = raw["order_date"].dt.to_period("M").astype(str)
raw["customer_id"] = (raw["product_name"].str.strip() + " / "
                      + raw["region"].str.strip() + " / "
                      + raw["ym"])

transactions = raw[["customer_id", "order_id", "order_date", "amount", "profit",
                     "category", "region", "product_name", "quantity"]].copy()

print(f"  {len(transactions)} transactions | {transactions['customer_id'].nunique()} synthetic customers")
print(f"  Date range: {transactions['order_date'].min().date()} to {transactions['order_date'].max().date()}")

# ── Build RFM-style customer features ────────────────────────────
print("\nBuilding customer features (history=90d, horizon=60d)...")
features = build_customer_features(
    transactions[["customer_id", "order_id", "order_date", "amount"]],
    history_window_days=90,
    prediction_horizon_days=60,
)
print(f"  Feature matrix: {features.shape}")
print(f"  Positive rate (repeat purchase): {features['label_repeat_purchase'].mean():.2%}")
print(f"  Columns: {list(features.columns)}")

feature_cols = [c for c in features.columns
                if c not in {"customer_id", "label_repeat_purchase", "future_spend"}]
X = features[feature_cols]
y_cls = features["label_repeat_purchase"]
y_spend = features["future_spend"]

# ── Build pipelines ───────────────────────────────────────────────
classifier = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=2,
        random_state=42, n_jobs=-1, class_weight="balanced_subsample",
    )),
])

regressor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )),
])

# ── CV evaluation ──────────────────────────────────────────────────
print("\nRunning cross-validation...")
min_class = int(y_cls.value_counts().min()) if y_cls.nunique() > 1 else 0
n_splits = min(5, min_class) if min_class >= 2 else 0

if n_splits >= 2:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_proba = cross_val_predict(classifier, X, y_cls, cv=cv, method="predict_proba")
    cv_pred = cv_proba[:, 1] if cv_proba.shape[1] > 1 else np.zeros(len(X))
    roc_auc_cv = float(roc_auc_score(y_cls, cv_pred)) if y_cls.nunique() > 1 else 0.5
    pr_auc_cv = float(average_precision_score(y_cls, cv_pred)) if y_cls.sum() > 0 else 0.0
else:
    classifier.fit(X, y_cls)
    proba = classifier.predict_proba(X)
    cv_pred = proba[:, 1] if proba.shape[1] > 1 else np.ones(len(X), dtype=float)
    roc_auc_cv = float(roc_auc_score(y_cls, cv_pred)) if y_cls.nunique() > 1 else 0.5
    pr_auc_cv = float(average_precision_score(y_cls, cv_pred)) if y_cls.sum() > 0 else 0.0

print(f"  CV ROC-AUC={roc_auc_cv:.4f}  PR-AUC={pr_auc_cv:.4f}")

# ── Train/test split ───────────────────────────────────────────────
stratify = y_cls if (y_cls.nunique() > 1 and min_class >= 2) else None
X_train, X_test, y_train, y_test, spend_train, spend_test = train_test_split(
    X, y_cls, y_spend, test_size=0.25, stratify=stratify, random_state=42
)

classifier.fit(X_train, y_train)
prob_test = classifier.predict_proba(X_test)[:, 1]
roc_auc_test = float(roc_auc_score(y_test, prob_test)) if y_test.nunique() > 1 else 0.5

# Regressor on positive training samples
positive_mask = y_train == 1
if positive_mask.sum() >= 5:
    regressor.fit(X_train[positive_mask], spend_train[positive_mask])
    cond_spend_pred = np.clip(regressor.predict(X_test), a_min=0, a_max=None)
    expected_spend_pred = prob_test * cond_spend_pred
else:
    regressor = None
    expected_spend_pred = prob_test * float(spend_train.mean())

rmse = float(np.sqrt(mean_squared_error(spend_test, expected_spend_pred)))

print(f"  Test ROC-AUC={roc_auc_test:.4f}  RMSE Expected Spend=${rmse:.0f}")
print("\nClassification report (test):")
print(classification_report(y_test, (prob_test >= 0.5).astype(int), zero_division=0))

# ── Feature importances ────────────────────────────────────────────
clf_imp = dict(zip(feature_cols, classifier.named_steps["model"].feature_importances_.tolist()))
reg_imp = {}
if regressor is not None:
    reg_imp = dict(zip(feature_cols, regressor.named_steps["model"].feature_importances_.tolist()))

# ── Raw data analytics ─────────────────────────────────────────────
# Category revenue/profit
cat_stats = (raw.groupby("category")
             .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"),
                  order_count=("order_id", "count"))
             .reset_index().to_dict(orient="records"))

# Region revenue
region_stats = (raw.groupby("region")
                .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"),
                     order_count=("order_id", "count"))
                .reset_index().to_dict(orient="records"))

# Product performance
product_stats = (raw.groupby("product_name")
                 .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"),
                      avg_quantity=("quantity", "mean"), order_count=("order_id", "count"))
                 .reset_index().sort_values("total_sales", ascending=False)
                 .to_dict(orient="records"))

# Monthly revenue
raw["year_month"] = raw["order_date"].dt.to_period("M").astype(str)
monthly = (raw.groupby("year_month")
           .agg(sales=("sales", "sum"), orders=("order_id", "count"))
           .reset_index().to_dict(orient="records"))

# Profit margin by category
raw["profit_pct"] = raw["profit"] / raw["sales"] * 100
profit_margin = (raw.groupby("category")["profit_pct"].mean().reset_index()
                 .rename(columns={"profit_pct": "avg_margin_pct"})
                 .to_dict(orient="records"))

# Scatter: actual vs predicted spend
scatter_data = {
    "actual": spend_test.tolist(),
    "predicted": expected_spend_pred.tolist(),
    "prob": prob_test.tolist(),
    "label": y_test.tolist(),
}

# Calibration data (decile lift)
df_cal = pd.DataFrame({"prob": prob_test, "label": y_test})
df_cal = df_cal.sort_values("prob", ascending=False).reset_index(drop=True)
n = len(df_cal)
decile_size = max(1, n // 10)
lift_data = []
for i in range(0, n, decile_size):
    chunk = df_cal.iloc[i:i + decile_size]
    lift_data.append({
        "decile": i // decile_size + 1,
        "avg_prob": float(chunk["prob"].mean()),
        "conversion_rate": float(chunk["label"].mean()),
    })

# Dataset stats
dataset_stats = {
    "n_transactions": int(len(raw)),
    "n_customers_synthetic": int(transactions["customer_id"].nunique()),
    "n_products": int(raw["product_name"].nunique()),
    "n_regions": int(raw["region"].nunique()),
    "n_categories": int(raw["category"].nunique()),
    "date_min": str(raw["order_date"].min().date()),
    "date_max": str(raw["order_date"].max().date()),
    "total_revenue": float(raw["sales"].sum()),
    "avg_order_value": float(raw["sales"].mean()),
    "avg_profit": float(raw["profit"].mean()),
    "overall_profit_margin": float(raw["profit"].sum() / raw["sales"].sum() * 100),
}

metrics = {
    "roc_auc_cv": roc_auc_cv,
    "pr_auc_cv": pr_auc_cv,
    "roc_auc_test": roc_auc_test,
    "rmse_expected_spend": rmse,
    "n_customers": int(len(features)),
    "positive_rate": float(y_cls.mean()),
}

# ── Save bundle ────────────────────────────────────────────────────
BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
bundle = {
    "classifier": classifier,
    "regressor": regressor,
    "feature_cols": feature_cols,
    "metrics": metrics,
    "clf_importance": clf_imp,
    "reg_importance": reg_imp,
    "scatter_data": scatter_data,
    "lift_data": lift_data,
    "cat_stats": cat_stats,
    "region_stats": region_stats,
    "product_stats": product_stats,
    "monthly": monthly,
    "profit_margin": profit_margin,
    "dataset_stats": dataset_stats,
    "customer_features": features.to_dict(orient="records"),
    "feature_stats": {
        col: {"mean": float(features[col].mean()), "min": float(features[col].min()),
              "max": float(features[col].max()), "std": float(features[col].std())}
        for col in feature_cols
    },
    "categories": sorted(raw["category"].unique().tolist()),
    "regions": sorted(raw["region"].unique().tolist()),
    "products": sorted(raw["product_name"].unique().tolist()),
}

joblib.dump(bundle, BUNDLE_PATH)
print(f"\nBundle saved -> {BUNDLE_PATH}")
print(f"Keys: {list(bundle.keys())}")

# Also save metrics JSON
Path("artifacts/models").mkdir(parents=True, exist_ok=True)
with open("artifacts/models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics: {metrics}")
