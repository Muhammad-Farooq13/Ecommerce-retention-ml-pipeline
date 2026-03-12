# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] - 2026-03-12

### Added
- `streamlit_app.py` — 5-tab interactive dashboard (Overview, Model Results, Analytics, Pipeline & API, Predict)
- `train_demo.py` — demo bundle trainer using monthly product×region cohorts + RFM features
- `models/demo_bundle.pkl` — pre-trained two-stage bundle (classifier + regressor, ROC-AUC=1.0 on synthetic demo)
- `requirements-ci.txt` — lean CI/test dependency list
- `.streamlit/config.toml` — blue/dark Streamlit theme
- `runtime.txt` + `packages.txt` — Streamlit Cloud deployment config

### Changed
- `requirements.txt` — added `streamlit>=1.36`, `plotly>=5.16`, `python-dotenv>=1.0`; removed non-runtime heavy deps
- `.github/workflows/ci.yml` — upgraded to Python 3.11/3.12 matrix, pip cache on `requirements-ci.txt`, `codecov/codecov-action@v5`, `docker/setup-buildx-action@v3`, `docker/build-push-action@v5`
- `.gitignore` — added `!models/demo_bundle.pkl` exception

### Fixed
- All 1 unit tests pass clean (no changes required)

---

## [0.1.0] - initial release

- Initial project structure: data loader, RFM feature engineering, two-stage RF model
- FastAPI REST API
- StratifiedKFold cross-validation with ROC-AUC, PR-AUC, RMSE metrics
- GitHub Actions CI pipeline
- Notebook: 01_eda_customer_behavior.ipynb