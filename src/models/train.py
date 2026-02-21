from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.utils.io import write_json


def _positive_probability(model: Pipeline, x_data: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x_data)
    classes = model.named_steps["model"].classes_
    if 1 in classes:
        class_index = int(np.where(classes == 1)[0][0])
        return proba[:, class_index]
    return np.zeros(len(x_data), dtype=float)


def train_models(config_path: str = "configs/base.yaml") -> dict[str, float]:
    config = load_config(config_path)
    df = pd.read_parquet(config.data.features_path)

    feature_cols = [
        col
        for col in df.columns
        if col not in {"customer_id", "label_repeat_purchase", "future_spend"}
    ]

    X = df[feature_cols]
    y_cls = df["label_repeat_purchase"]
    y_spend = df["future_spend"]

    classifier = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=config.model.n_estimators,
                    max_depth=config.model.max_depth,
                    min_samples_leaf=config.model.min_samples_leaf,
                    random_state=config.model.random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    min_class_count = int(y_cls.value_counts().min()) if y_cls.nunique() > 1 else 0
    n_splits = min(config.model.cv_folds, len(X), min_class_count) if min_class_count else 0

    if n_splits >= 2:
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.model.random_state,
        )
        cv_pred = cross_val_predict(classifier, X, y_cls, cv=cv, method="predict_proba")
        cv_pred = cv_pred[:, 1] if cv_pred.shape[1] > 1 else np.zeros(len(X), dtype=float)
        roc_auc = float(roc_auc_score(y_cls, cv_pred)) if y_cls.nunique() > 1 else 0.5
        pr_auc = float(average_precision_score(y_cls, cv_pred)) if y_cls.sum() > 0 else 0.0
    else:
        classifier.fit(X, y_cls)
        cv_pred = _positive_probability(classifier, X)
        roc_auc = float(roc_auc_score(y_cls, cv_pred)) if y_cls.nunique() > 1 else 0.5
        pr_auc = float(average_precision_score(y_cls, cv_pred)) if y_cls.sum() > 0 else 0.0

    stratify_target = y_cls if (y_cls.nunique() > 1 and min_class_count >= 2 and len(X) >= 6) else None
    X_train, X_valid, y_train, y_valid, spend_train, spend_valid = train_test_split(
        X,
        y_cls,
        y_spend,
        test_size=0.2 if len(X) >= 10 else 0.25,
        stratify=stratify_target,
        random_state=config.model.random_state,
    )

    classifier.fit(X_train, y_train)
    valid_prob = _positive_probability(classifier, X_valid)

    positive_mask = y_train == 1
    regressor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=max(200, config.model.n_estimators // 2),
                    max_depth=config.model.max_depth,
                    min_samples_leaf=config.model.min_samples_leaf,
                    random_state=config.model.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    if positive_mask.sum() >= 10:
        regressor.fit(X_train[positive_mask], spend_train[positive_mask])
        cond_spend_pred = np.clip(regressor.predict(X_valid), a_min=0, a_max=None)
        expected_spend_pred = valid_prob * cond_spend_pred
        rmse = float(np.sqrt(mean_squared_error(spend_valid, expected_spend_pred)))
    else:
        regressor = None
        expected_spend_pred = valid_prob * spend_train.mean()
        rmse = float(np.sqrt(mean_squared_error(spend_valid, expected_spend_pred)))

    model_bundle = {
        "classifier": classifier,
        "regressor": regressor,
        "feature_columns": feature_cols,
    }
    output_path = Path(config.outputs.model_bundle_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, output_path)

    valid_out = X_valid.copy()
    valid_out["y_true"] = y_valid.values
    valid_out["pred_prob"] = valid_prob
    valid_out["future_spend_true"] = spend_valid.values
    valid_out["future_spend_pred"] = expected_spend_pred
    valid_out.to_parquet(config.data.validation_predictions_path, index=False)

    metrics = {
        "roc_auc_cv": roc_auc,
        "pr_auc_cv": pr_auc,
        "rmse_expected_spend": rmse,
        "n_customers": int(len(df)),
        "positive_rate": float(y_cls.mean()),
    }
    write_json(metrics, config.outputs.metrics_path)
    return metrics


if __name__ == "__main__":
    train_models()
