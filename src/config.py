from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    input_csv: str
    transactions_path: str
    features_path: str
    validation_predictions_path: str


@dataclass
class FeatureConfig:
    history_window_days: int
    prediction_horizon_days: int


@dataclass
class ModelConfig:
    random_state: int
    cv_folds: int
    n_estimators: int
    max_depth: int
    min_samples_leaf: int


@dataclass
class OutputConfig:
    model_bundle_path: str
    metrics_path: str


@dataclass
class AppConfig:
    project_name: str
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    outputs: OutputConfig


def load_config(config_path: str | Path = "configs/base.yaml") -> AppConfig:
    with Path(config_path).open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    return AppConfig(
        project_name=raw["project_name"],
        data=DataConfig(**raw["data"]),
        features=FeatureConfig(**raw["features"]),
        model=ModelConfig(**raw["model"]),
        outputs=OutputConfig(**raw["outputs"]),
    )
