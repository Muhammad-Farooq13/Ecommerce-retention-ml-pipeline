import pandas as pd

from src.features.build_features import build_customer_features


def test_build_customer_features_returns_expected_columns() -> None:
    transactions = pd.DataFrame(
        {
            "customer_id": ["A", "A", "B", "B", "B"],
            "order_id": ["1", "2", "3", "4", "5"],
            "order_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-01-10", "2024-03-01", "2024-04-10"]
            ),
            "amount": [100.0, 150.0, 40.0, 60.0, 80.0],
        }
    )

    output = build_customer_features(
        transactions, history_window_days=120, prediction_horizon_days=30
    )

    expected_cols = {
        "customer_id",
        "frequency_orders",
        "monetary_sum",
        "monetary_mean",
        "active_days",
        "recency_days",
        "customer_tenure_days",
        "avg_order_gap_days",
        "label_repeat_purchase",
        "future_spend",
    }
    assert expected_cols.issubset(set(output.columns))
    assert len(output) > 0
