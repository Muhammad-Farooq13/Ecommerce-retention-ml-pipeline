from __future__ import annotations

from src.data.make_dataset import run_make_dataset
from src.features.build_features import run_build_features
from src.models.evaluate import evaluate_predictions
from src.models.train import train_models


def main() -> None:
    run_make_dataset()
    run_build_features()
    metrics = train_models()
    business_metrics = evaluate_predictions()
    print("Model metrics:", metrics)
    print("Business metrics:", business_metrics)


if __name__ == "__main__":
    main()
