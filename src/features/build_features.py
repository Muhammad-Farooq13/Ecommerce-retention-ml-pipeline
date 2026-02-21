from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config


def _safe_mean_gap(series: pd.Series) -> float:
    ordered = series.sort_values().drop_duplicates()
    if len(ordered) < 2:
        return 0.0
    gaps = ordered.diff().dropna().dt.days
    return float(gaps.mean()) if len(gaps) else 0.0


def build_customer_features(
    transactions: pd.DataFrame,
    history_window_days: int,
    prediction_horizon_days: int,
) -> pd.DataFrame:
    df = transactions.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])

    max_date = df["order_date"].max()
    cutoff_date = max_date - pd.Timedelta(days=prediction_horizon_days)
    history_start = cutoff_date - pd.Timedelta(days=history_window_days)

    history = df[(df["order_date"] >= history_start) & (df["order_date"] <= cutoff_date)].copy()
    future = df[df["order_date"] > cutoff_date].copy()

    grouped = history.groupby("customer_id", as_index=False)
    features = grouped.agg(
        frequency_orders=("order_id", "nunique"),
        monetary_sum=("amount", "sum"),
        monetary_mean=("amount", "mean"),
        active_days=("order_date", lambda x: x.dt.date.nunique()),
        first_purchase=("order_date", "min"),
        last_purchase=("order_date", "max"),
    )

    features["recency_days"] = (cutoff_date - features["last_purchase"]).dt.days
    features["customer_tenure_days"] = (features["last_purchase"] - features["first_purchase"]).dt.days
    gap_map = history.groupby("customer_id")["order_date"].apply(_safe_mean_gap)
    features["avg_order_gap_days"] = features["customer_id"].map(gap_map).fillna(0.0)

    future_label = future.groupby("customer_id", as_index=False).agg(
        future_orders=("order_id", "nunique"),
        future_spend=("amount", "sum"),
    )
    future_label["label_repeat_purchase"] = (future_label["future_orders"] > 0).astype(int)

    dataset = features.merge(
        future_label[["customer_id", "label_repeat_purchase", "future_spend"]],
        on="customer_id",
        how="left",
    )

    dataset["label_repeat_purchase"] = dataset["label_repeat_purchase"].fillna(0).astype(int)
    dataset["future_spend"] = dataset["future_spend"].fillna(0.0)

    dataset = dataset.drop(columns=["first_purchase", "last_purchase"])
    numeric_columns = [
        "frequency_orders",
        "monetary_sum",
        "monetary_mean",
        "active_days",
        "recency_days",
        "customer_tenure_days",
        "avg_order_gap_days",
        "future_spend",
    ]
    dataset[numeric_columns] = dataset[numeric_columns].replace([np.inf, -np.inf], 0).fillna(0)

    return dataset


def run_build_features(config_path: str = "configs/base.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    transactions = pd.read_parquet(config.data.transactions_path)
    features = build_customer_features(
        transactions,
        history_window_days=config.features.history_window_days,
        prediction_horizon_days=config.features.prediction_horizon_days,
    )
    output_path = Path(config.data.features_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return features


if __name__ == "__main__":
    run_build_features()
