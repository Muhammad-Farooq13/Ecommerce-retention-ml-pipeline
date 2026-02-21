from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config


CANDIDATE_DATE_COLUMNS = ["order_date", "invoice_date", "date", "timestamp", "created_at"]
CANDIDATE_CUSTOMER_COLUMNS = ["customer_id", "customerid", "user_id", "client_id"]
CANDIDATE_FALLBACK_ENTITY_COLUMNS = ["region", "segment", "product_name", "category"]
CANDIDATE_AMOUNT_COLUMNS = ["amount", "total_amount", "sales", "revenue", "total"]
CANDIDATE_QUANTITY_COLUMNS = ["quantity", "qty", "units"]
CANDIDATE_UNIT_PRICE_COLUMNS = ["unit_price", "price", "item_price"]
CANDIDATE_ORDER_COLUMNS = ["order_id", "invoice_no", "transaction_id", "invoice"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def _infer_column(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_and_clean_transactions(input_csv: str) -> pd.DataFrame:
    raw_df = pd.read_csv(input_csv)
    df = _normalize_columns(raw_df)

    columns = list(df.columns)
    date_col = _infer_column(columns, CANDIDATE_DATE_COLUMNS)
    customer_col = _infer_column(columns, CANDIDATE_CUSTOMER_COLUMNS)
    amount_col = _infer_column(columns, CANDIDATE_AMOUNT_COLUMNS)
    quantity_col = _infer_column(columns, CANDIDATE_QUANTITY_COLUMNS)
    unit_price_col = _infer_column(columns, CANDIDATE_UNIT_PRICE_COLUMNS)
    order_col = _infer_column(columns, CANDIDATE_ORDER_COLUMNS)

    if date_col is None:
        raise ValueError(f"Could not infer date column. Columns found: {columns}")
    fallback_entity_col = _infer_column(columns, CANDIDATE_FALLBACK_ENTITY_COLUMNS)
    if customer_col is None and fallback_entity_col is None:
        raise ValueError(
            "Could not infer customer column. Provide one of customer_id/user_id/client_id, "
            f"or one fallback entity column from {CANDIDATE_FALLBACK_ENTITY_COLUMNS}. "
            f"Columns found: {columns}"
        )

    canonical = pd.DataFrame()
    if customer_col is not None:
        canonical["customer_id"] = df[customer_col].astype(str).str.strip()
    else:
        canonical["customer_id"] = (
            "fallback_" + fallback_entity_col + "_" + df[fallback_entity_col].astype(str).str.strip()
        )
    canonical["order_date"] = pd.to_datetime(df[date_col], errors="coerce")

    if amount_col is not None:
        canonical["amount"] = pd.to_numeric(df[amount_col], errors="coerce")
    elif quantity_col is not None and unit_price_col is not None:
        quantity = pd.to_numeric(df[quantity_col], errors="coerce").fillna(0)
        price = pd.to_numeric(df[unit_price_col], errors="coerce").fillna(0)
        canonical["amount"] = quantity * price
    else:
        raise ValueError(
            "Could not infer amount. Provide one of amount/total/sales/revenue or quantity+unit_price."
        )

    if order_col is not None:
        canonical["order_id"] = df[order_col].astype(str).str.strip()
    else:
        canonical["order_id"] = canonical.index.astype(str)

    canonical = canonical.dropna(subset=["customer_id", "order_date", "amount"])
    canonical = canonical[canonical["amount"] >= 0]
    canonical = canonical.sort_values("order_date").drop_duplicates(
        subset=["customer_id", "order_id", "order_date", "amount"]
    )

    return canonical.reset_index(drop=True)


def run_make_dataset(config_path: str = "configs/base.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    transactions = load_and_clean_transactions(config.data.input_csv)
    output_path = Path(config.data.transactions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transactions.to_parquet(output_path, index=False)
    return transactions


if __name__ == "__main__":
    run_make_dataset()
