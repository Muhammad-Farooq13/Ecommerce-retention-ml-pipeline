from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.utils.io import write_json


def lift_at_k(y_true: pd.Series, y_score: pd.Series, k: float = 0.1) -> float:
    df = pd.DataFrame({"y_true": y_true, "y_score": y_score}).sort_values("y_score", ascending=False)
    top_n = max(1, int(len(df) * k))
    top_rate = df.head(top_n)["y_true"].mean()
    base_rate = df["y_true"].mean()
    return float(top_rate / base_rate) if base_rate > 0 else 0.0


def evaluate_predictions(config_path: str = "configs/base.yaml") -> dict[str, float]:
    config = load_config(config_path)
    pred_df = pd.read_parquet(config.data.validation_predictions_path)

    metrics = {
        "lift_at_10pct": lift_at_k(pred_df["y_true"], pred_df["pred_prob"], 0.1),
        "lift_at_20pct": lift_at_k(pred_df["y_true"], pred_df["pred_prob"], 0.2),
        "avg_expected_spend_top10pct": float(
            pred_df.sort_values("pred_prob", ascending=False)
            .head(max(1, int(len(pred_df) * 0.1)))["future_spend_pred"]
            .mean()
        ),
    }

    write_json(metrics, "artifacts/models/business_metrics.json")
    return metrics


if __name__ == "__main__":
    evaluate_predictions()
