# Creator Lifecycle Intelligence Platform

Three-module survival analysis on podcast creator data. Predicts which creators
survive, measures the causal effect of monetization on retention, and classifies
dormant creators vs. permanently churned using competing risks.

Work in progress — this README will be filled in with real results as each
module is built.

## Modules

| Module | Method | Key Metric | Result |
|--------|--------|-----------|--------|
| 1. Cold Start | Cox PH + XGBoost | C-index / AUC | TBD |
| 2. Monetization | DiD + RD | ATT / C-index | TBD |
| 3. Competing Risks | Fine-Gray | C-index | TBD |

## Tech Stack

Survival analysis: lifelines, scikit-survival
Causal inference: statsmodels, econml
ML: XGBoost, LightGBM, SHAP, Optuna
Tracking: MLflow
Validation: Great Expectations
Dashboard: Streamlit
Data: [ageitgey/all-podcasts-dataset](https://github.com/ageitgey/all-podcasts-dataset)

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python src/data/loader.py
python src/data/features.py
python src/data/validate.py
streamlit run dashboard/app.py
```
