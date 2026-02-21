# Ecommerce Retention & Expected Spend Project

An end-to-end, production-style data science project that predicts:
1. **Probability of repeat purchase** in the next 30 days
2. **Expected customer spend** over the same horizon

This repository is designed for portfolio visibility and job interviews, with reproducible pipelines, model evaluation, deployment hooks, and CI.

## Owner

- Name: Muhammad Farooq
- Email: mfarooqshafee333@gmail.com
- GitHub: https://github.com/Muhammad-Farooq-13

## 1) Problem Statement

Marketing teams often spend retention budget on broad segments. This project scores customers by likely repeat purchase and expected spend so campaigns can prioritize high-value targets.

### Objectives
- Build reproducible customer-level features from transaction data
- Train a propensity model and expected-spend estimator
- Evaluate both ML and business metrics (lift and expected value)
- Deploy scoring behind a lightweight API

### Success Criteria
- Strong ranking quality (ROC-AUC / PR-AUC)
- Positive lift in top decile vs baseline
- Reproducible pipeline with one command

## 2) Project Structure

```text
.
├── configs/
│   └── base.yaml
├── src/
│   ├── data/make_dataset.py
│   ├── features/build_features.py
│   ├── models/train.py
│   ├── models/evaluate.py
│   ├── inference/predict.py
│   └── api/main.py
├── scripts/run_pipeline.py
├── tests/test_features.py
├── notebooks/
│   └── 01_eda_customer_behavior.ipynb
├── artifacts/                 # generated outputs (ignored by git)
├── Dockerfile
├── Makefile
└── pyproject.toml
```

### Architecture Diagram

```mermaid
flowchart TD
  A[ecommerce_sales_data.csv] --> B[src/data/make_dataset.py]
  B --> C[artifacts/data/transactions.parquet]
  C --> D[src/features/build_features.py]
  D --> E[artifacts/data/customer_features.parquet]
  E --> F[src/models/train.py]
  F --> G[artifacts/models/model_bundle.joblib]
  F --> H[artifacts/models/metrics.json]
  E --> I[src/models/evaluate.py]
  I --> J[artifacts/models/business_metrics.json]
  G --> K[src/api/main.py]
  K --> L[/predict endpoint]
```

## 3) Quickstart

### Setup
```bash
python -m pip install -U pip
python -m pip install -e .[dev]
```

### Run Full Pipeline
```bash
python scripts/run_pipeline.py
```

Generated outputs:
- `artifacts/data/transactions.parquet`
- `artifacts/data/customer_features.parquet`
- `artifacts/models/model_bundle.joblib`
- `artifacts/models/metrics.json`
- `artifacts/models/business_metrics.json`

### EDA Notebook
- `notebooks/01_eda_customer_behavior.ipynb` contains portfolio-ready analysis visuals and EDA-to-model decisions.

## 4) Data Pipeline

- Input source configured in `configs/base.yaml` (`ecommerce_sales_data.csv`)
- Schema inference for common ecommerce column names
- Canonical transaction schema: `customer_id`, `order_id`, `order_date`, `amount`
- Temporal feature and label generation with configurable history and horizon windows

## 5) Modeling Approach

- Classification model: `RandomForestClassifier` for repeat purchase propensity
- Regression model: `RandomForestRegressor` (trained on positive-repeat customers) for conditional spend
- Expected spend computed as:

$$
\text{ExpectedSpend} = P(\text{Repeat}) \times E(\text{Spend} \mid \text{Repeat})
$$

## 6) Evaluation

Technical metrics:
- ROC-AUC (cross-validated)
- PR-AUC (cross-validated)
- RMSE for expected spend on validation split

Business metrics:
- Lift@10%
- Lift@20%
- Average expected spend among top-ranked customers

## 7) API Deployment

### Local API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Endpoints
- `GET /health`
- `POST /predict` with payload:

```json
{
  "rows": [
    {
      "frequency_orders": 7,
      "monetary_sum": 1200,
      "monetary_mean": 171.4,
      "active_days": 6,
      "recency_days": 12,
      "customer_tenure_days": 190,
      "avg_order_gap_days": 22
    }
  ]
}
```

### Docker
```bash
docker build -t ecommerce-retention .
docker run -p 8000:8000 ecommerce-retention
```

## 8) GitHub & Collaboration Standards

- Use conventional commits (`feat:`, `fix:`, `docs:`)
- Open PRs with objective, approach, validation evidence
- CI runs lint + tests on each push/PR
- Keep notebooks for analysis only; production code stays in `src/`

### Ready-to-Upload Checklist

- [x] Reproducible project structure with config-driven pipeline
- [x] API + Docker deployment files
- [x] CI workflow for lint and tests
- [x] Contributing guide and project documentation
- [x] Owner and contact details

### GitHub Upload Commands

```bash
git add .
git commit -m "feat: initial end-to-end ecommerce retention project"
git branch -M main
git remote add origin https://github.com/Muhammad-Farooq-13/<repo-name>.git
git push -u origin main
```

## 9) Job-Market Positioning Tips

For interviews and applications, emphasize:
- End-to-end ownership (data to deployment)
- Reproducibility and MLOps mindset (config-driven pipelines, CI, containerization)
- Business alignment (lift, expected value, campaign prioritization)
- Trade-off decisions and model limitations

### Role-Targeted Positioning

#### If applying for Data Scientist roles
- Lead with experimentation logic, feature engineering rationale, and model evaluation depth.
- Emphasize your choice of metrics by business objective (ranking quality + lift, not just accuracy).
- Highlight statistical thinking: leakage prevention, validation strategy, and assumptions.

#### If applying for ML Engineer roles
- Lead with production readiness: modular `src/`, API serving, containerization, and CI checks.
- Emphasize reproducibility and operational thinking (config-driven runs, artifact outputs, deployment path).
- Highlight maintainability decisions: test coverage, clear interfaces, and extensible pipeline stages.

#### If applying for Product Data Scientist roles
- Lead with business framing: budget allocation, customer prioritization, and decision support.
- Emphasize expected value interpretation and how scores map to campaign actions.
- Highlight communication: architecture diagram, EDA-to-decision narrative, and trade-off clarity.

## 10) Interview Case Study (1-Page Narrative)

### Business Problem
A marketing team needs to allocate limited retention budget more efficiently. Instead of broad campaigns, they want customer-level prioritization based on likely repeat purchase and expected short-term value.

### My Approach
1. Defined a prediction horizon (30 days) and translated business goals into measurable targets (ranking quality + lift).
2. Built a reproducible data pipeline with schema normalization and temporal feature/label generation.
3. Engineered customer behavior features (frequency, recency, spend intensity, tenure, order cadence).
4. Trained a two-stage model:
  - classification for repeat-purchase probability,
  - regression for conditional spend among repeat buyers.
5. Evaluated both technical and business outcomes (ROC-AUC, PR-AUC, RMSE, Lift@K).
6. Exposed the model through a FastAPI endpoint and added CI to enforce quality on changes.

### Demonstrated Impact (Portfolio Framing)
- Delivered end-to-end ownership from raw data to deployable service.
- Produced ranked customer outputs that support budget-aware targeting decisions.
- Built a reusable project template with config-driven workflows, tests, and containerization.

### Trade-Offs & Engineering Decisions
- **Model choice:** Random forests favored for strong tabular baseline performance and lower complexity.
- **Validation choice:** practical CV baseline in code; recommend strict time-based validation as next step for production.
- **Data limitation handling:** implemented fallback entity keying when explicit customer IDs are absent.
- **Interpretability vs speed:** prioritized explainable feature construction and maintainable modular code.

### What I Would Improve Next
- Add probability calibration and threshold optimization against campaign budget constraints.
- Add drift monitoring and scheduled retraining policy.
- Add experiment tracking for model lineage and faster iteration in team settings.

### 30-Second Interview Pitch Variants
- **Data Scientist:** “I built an end-to-end retention model that converts transaction history into customer-level repeat-purchase probability and expected spend, validated with both ML and business metrics to support campaign targeting.”
- **ML Engineer:** “I productionized a tabular ML workflow with reproducible data/feature pipelines, model artifact management, CI checks, and a FastAPI inference service that is container-ready.”
- **Product Data Scientist:** “I translated a retention budget problem into a prioritization system, producing interpretable ranked recommendations tied to expected value and actionable campaign decisions.”

## 11) Next Enhancements

- Time-based cross-validation and backtesting by cohort
- Probability calibration (Platt/Isotonic)
- Drift monitoring and scheduled retraining
- Experiment tracking (MLflow / Weights & Biases)
