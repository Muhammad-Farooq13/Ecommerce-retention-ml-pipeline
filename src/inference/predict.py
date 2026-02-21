from __future__ import annotations

import joblib
import numpy as np
import pandas as pd


def score_customers(model_bundle_path: str, features: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(model_bundle_path)
    classifier = bundle["classifier"]
    regressor = bundle["regressor"]
    feature_columns = bundle["feature_columns"]

    missing_cols = [col for col in feature_columns if col not in features.columns]
    if missing_cols:
        raise ValueError(f"Missing required features for scoring: {missing_cols}")

    model_input = features[feature_columns].copy()
    prob = classifier.predict_proba(model_input)[:, 1]

    if regressor is not None:
        cond_spend = np.clip(regressor.predict(model_input), a_min=0, a_max=None)
    else:
        cond_spend = np.zeros(len(model_input))

    output = features.copy()
    output["prob_repeat_purchase"] = prob
    output["expected_spend"] = prob * cond_spend
    return output
