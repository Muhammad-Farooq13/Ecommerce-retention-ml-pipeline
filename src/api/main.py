from __future__ import annotations

import os
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.predict import score_customers


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1)


app = FastAPI(title="Ecommerce Retention Scoring API", version="0.1.0")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/models/model_bundle.joblib")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, list[dict[str, Any]]]:
    try:
        rows_df = pd.DataFrame(request.rows)
        scored = score_customers(MODEL_PATH, rows_df)
        result = scored[["prob_repeat_purchase", "expected_spend"]].to_dict(orient="records")
        return {"predictions": result}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc
