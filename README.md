# Ecommerce Retention ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange.svg)](https://scikit-learn.org/)
[![CI](https://github.com/Muhammad-Farooq13/Ecommerce-retention-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Farooq13/Ecommerce-retention-ml-pipeline/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end ML pipeline for **customer retention prediction** in e-commerce. Predicts which customers will repeat-purchase and estimates their expected revenue using a two-stage Random Forest architecture trained on 3,500 real transactions.

---

## Features

| Area | Details |
|---|---|
| **Task** | Binary classification (repeat purchase) + regression (expected spend) |
| **Dataset** | 3,500 real e-commerce transactions · 3 categories · 4 regions · 10 products |
| **Features** | RFM: frequency, monetary (sum/mean), recency, tenure, active days, avg order gap |
| **Model** | Stage 1: RandomForestClassifier · Stage 2: RandomForestRegressor (positive-only) |
| **Output** | P(repeat purchase) + Expected Spend = P × Conditional Spend |
| **Evaluation** | ROC-AUC, PR-AUC (classifier) · RMSE (spend) · Decile Lift |
| **API** | FastAPI REST (`/health`, `/predict`, `/predict/batch`) |
| **Dashboard** | Streamlit 5-tab interactive app with live RFM prediction |
| **CI/CD** | GitHub Actions — Python 3.11/3.12, ruff lint, pytest + coverage, Docker |

---

## Quick Start

```bash
git clone https://github.com/Muhammad-Farooq13/Ecommerce-retention-ml-pipeline.git
cd Ecommerce-retention-ml-pipeline
pip install -r requirements.txt
pip install -e .

# Generate demo bundle (trains models, saves artefacts)
python train_demo.py

# Launch Streamlit dashboard
streamlit run streamlit_app.py

# OR start FastAPI server
uvicorn src.api.main:app --reload
```

---

## Model Results (Demo Bundle)

| Metric | Value |
|---|---|
| CV ROC-AUC | 1.0000 |
| CV PR-AUC | 1.0000 |
| RMSE Expected Spend | ~$193 |
| Training Customers | 108 |
| Positive Rate | ~3.7% |

*Demo uses monthly product×region cohorts as synthetic customers — ROC-AUC=1.0 reflects clean separability of this synthetic setup. Production metrics on real individual customer data typically reach 0.70–0.85.*

---

## Two-Stage Architecture

```
Input: RFM features
  │
  ├─ Stage 1: RandomForestClassifier (balanced)
  │    → P(repeat purchase)  [0..1]
  │
  └─ Stage 2: RandomForestRegressor (trained on positive-label customers only)
       → Conditional Spend ($)

Output: Expected Spend = P(repeat) × Conditional Spend
```

---

## Dataset

**`ecommerce_sales_data.csv`** — 3,500 rows:
- **Order Date** — Jan 2022 – Dec 2024
- **Product Name** — 10 products (Laptop, Smartphone, Tablet, …)
- **Category** — Electronics, Office, Accessories
- **Region** — North, East, South, West
- **Quantity**, **Sales**, **Profit**

---

## FastAPI Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Single customer prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "frequency_orders": 5,
    "monetary_sum": 1200.0,
    "monetary_mean": 240.0,
    "active_days": 12,
    "recency_days": 15,
    "customer_tenure_days": 90,
    "avg_order_gap_days": 18.0
  }'
```

---

## Docker

```bash
docker-compose up --build
# API at http://localhost:8000
```

---

## Full Pipeline

```bash
# Run full MLOps pipeline (make_dataset → build_features → train_models → evaluate)
python scripts/run_pipeline.py

# Or step by step:
python -m src.data.make_dataset
python -m src.features.build_features
python -m src.models.train
python -m src.models.evaluate
```

---

## Tests

```bash
pytest tests/ -v --cov=src/
# 1 test: test_features.py (build_customer_features)
```

---

## Project Structure

```
Ecommerce-retention-ml-pipeline/
├── src/
│   ├── data/         make_dataset.py   — CSV loader with auto column inference
│   ├── features/     build_features.py — RFM feature engineering
│   ├── models/       train.py  evaluate.py
│   ├── inference/    predict.py — score_customers()
│   ├── api/          main.py (FastAPI)
│   └── utils/        io.py
├── tests/            test_features.py
├── configs/          base.yaml
├── models/           demo_bundle.pkl
├── scripts/          run_pipeline.py
├── streamlit_app.py
├── train_demo.py
├── requirements.txt
├── requirements-ci.txt
└── .github/workflows/ci.yml
```

---

## License

MIT — see [LICENSE](LICENSE).