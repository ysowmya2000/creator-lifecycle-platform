# Creator Lifecycle Intelligence Platform — Claude Code Build Instructions

## What This Is

You are building the Creator Lifecycle Intelligence Platform end to end: a three-module
survival analysis project on podcast creator data. This file is your single source of truth.
Read it fully before writing any code. Follow the phases in order. Do not skip steps.

---

## Repo Structure to Create

```
creator-lifecycle-platform/
├── CLAUDE.md                         ← this file (copy here too)
├── README.md                         ← generate at the end
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                          ← downloaded dataset goes here
│   ├── processed/                    ← SQLite db and parquet files
│   └── README.md                     ← data provenance note
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_module1_cold_start.ipynb
│   ├── 03_module2_monetization.ipynb
│   └── 04_module3_competing_risks.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 ← download + parse podcast dataset
│   │   ├── features.py               ← all feature engineering
│   │   └── validate.py               ← Great Expectations suite
│   ├── models/
│   │   ├── __init__.py
│   │   ├── module1_survival.py       ← Cox PH + XGBoost early warning
│   │   ├── module2_causal.py         ← DiD + RD + LTV estimator
│   │   └── module3_competing.py      ← Fine-Gray competing risks
│   └── utils/
│       ├── __init__.py
│       └── mlflow_utils.py           ← experiment logging helpers
├── dashboard/
│   ├── app.py                        ← main Streamlit entry point
│   ├── pages/
│   │   ├── 1_survival_explorer.py
│   │   ├── 2_early_warning.py
│   │   ├── 3_monetization_impact.py
│   │   └── 4_dormancy_intelligence.py
│   └── components/
│       ├── charts.py                 ← reusable plotly chart functions
│       └── sidebar.py                ← shared sidebar filters
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_pipeline.py
├── mlruns/                           ← MLflow artifacts (git-tracked)
└── outputs/
    ├── figures/                      ← saved charts for README
    └── models/                       ← serialized model artifacts
```

---

## Phase 0 — Repository Setup

1. Create the full directory structure above with empty `__init__.py` files and placeholder READMEs
2. Create `.gitignore`:
   ```
   data/raw/
   __pycache__/
   *.pyc
   .env
   .DS_Store
   *.egg-info/
   dist/
   .ipynb_checkpoints/
   ```
3. Create `requirements.txt` with exact versions:
   ```
   pandas==2.2.2
   numpy==1.26.4
   scipy==1.13.1
   lifelines==0.29.0
   scikit-survival==0.23.0
   statsmodels==0.14.2
   econml==0.15.1
   xgboost==2.0.3
   lightgbm==4.3.0
   scikit-learn==1.5.0
   shap==0.45.1
   optuna==3.6.1
   mlflow==2.13.0
   great-expectations==0.18.19
   spacy==3.7.4
   feedparser==6.0.11
   plotly==5.22.0
   streamlit==1.35.0
   pytest==8.2.2
   matplotlib==3.9.0
   seaborn==0.13.2
   pyarrow==16.1.0
   ```
4. Initialize git repo, make initial commit: `feat: initial repo structure`

---

## Phase 1 — Data Pipeline (`src/data/loader.py`)

### Dataset
Primary: `ageitgey/all-podcasts-dataset` on GitHub
- URL: https://github.com/ageitgey/all-podcasts-dataset
- Download the `podcasts.db` SQLite file directly (it is a single file, ~1-2GB)
- If that specific file is unavailable, fall back to constructing from the RSS feed CSVs in the same repo
- Store in `data/raw/podcasts.db`

### What to extract from the dataset
Query the SQLite database to extract per-podcast:
- `podcast_id`: unique identifier
- `title`: podcast title
- `language`: language code
- `country`: country of origin
- `category`: primary category (e.g. "True Crime", "Business", "Comedy")
- `episode_count`: total number of episodes
- `first_episode_date`: date of first episode published
- `last_episode_date`: date of most recent episode published
- `is_active`: whether RSS feed is currently active (boolean)
- `episode_dates`: sorted list of all episode publish dates (for cadence calculation)
- `episode_descriptions`: list of episode descriptions (for monetization NLP)

Write all extracted records to:
1. `data/processed/creators.parquet` — one row per podcast/creator
2. `data/processed/creators.db` — SQLite for dashboard queries

### Filtering rules
- Keep only podcasts with `first_episode_date` not null
- Keep only podcasts with at least 2 episodes (single-episode shows are not creators)
- Keep only English-language podcasts for consistency
- Drop podcasts with no episode date history

### Logging
Print progress every 10,000 records. Log final counts: total records, records after filtering, null rates per column.

---

## Phase 2 — Feature Engineering (`src/data/features.py`)

Compute the following features for each creator. All features computed from `episode_dates` list.

### Cadence features
- `posting_freq_mean`: mean days between consecutive episodes
- `posting_freq_std`: standard deviation of inter-episode gaps
- `posting_freq_median`: median inter-episode gap
- `longest_gap_days`: maximum gap between any two consecutive episodes
- `current_gap_days`: days from last episode to dataset snapshot date (use max date in dataset as snapshot)
- `first_30d_episode_count`: number of episodes in first 30 days
- `first_14d_episode_count`: number of episodes in first 14 days

### Consistency features
- `consistency_score_90d`: fraction of weeks in first 90 days with at least one episode
- `consistency_score_30d`: fraction of weeks in first 30 days with at least one episode
- `posted_week1`: binary — did creator post in first 7 days?
- `posted_week2`: binary — did creator post in days 8–14?
- `cadence_variance_ratio`: posting_freq_std / posting_freq_mean (coefficient of variation)

### Growth / velocity features
- `eps_per_month_total`: total episodes / total active months
- `eps_per_month_first3`: episodes in first 3 months / 3
- `eps_per_month_last3`: episodes in last 3 months / 3
- `velocity_change`: eps_per_month_last3 - eps_per_month_first3 (positive = accelerating)

### Lifecycle features
- `total_active_days`: days from first to last episode
- `total_active_months`: total_active_days / 30.44
- `days_since_last_episode`: current_gap_days (alias for clarity)
- `dormancy_30d`: binary — current gap > 30 days
- `dormancy_60d`: binary — current gap > 60 days
- `dormancy_90d`: binary — current gap > 90 days

### Monetization proxy (NLP)
Use spaCy's `en_core_web_sm` model and keyword matching on `episode_descriptions`:
- Keywords indicating monetization: "patreon", "sponsor", "ad-free", "premium", "subscribe", "membership", "supporters", "ad supported", "brought to you by"
- `has_monetization_signal`: binary — any episode description contains monetization keywords
- `monetization_episode_index`: which episode (index) first contained a monetization keyword (None if never)
- `monetization_timing_months`: months from first episode to first monetization signal

### Survival labels
- `survived_90d`: binary — creator posted at least once after day 90 from first episode
- `survived_180d`: binary — creator posted at least once after day 180 from first episode
- `time_to_first_long_gap`: days until first gap > 30 days (survival time for Module 1)
- `event_occurred`: binary — did a 30-day gap occur? (event indicator for Module 1)

### Competing risks labels (for Module 3)
- `event_type`: 0 = censored (still active), 1 = dormant (gap > 60d but eventually returned), 2 = permanently departed (gap > 60d, never returned, RSS inactive)
- Logic: if `is_active = True` → 0; if `is_active = False` and creator has episodes after a 60d gap → 1; if `is_active = False` and no return → 2

Save feature matrix to `data/processed/features.parquet`.

---

## Phase 3 — Data Validation (`src/data/validate.py`)

Use Great Expectations to validate the feature matrix. Create a validation suite with these expectations:

```python
# No nulls in key columns
expect_column_values_to_not_be_null("podcast_id")
expect_column_values_to_not_be_null("first_episode_date")
expect_column_values_to_not_be_null("posting_freq_mean")

# Range checks
expect_column_values_to_be_between("consistency_score_90d", 0, 1)
expect_column_values_to_be_between("consistency_score_30d", 0, 1)
expect_column_values_to_be_between("cadence_variance_ratio", 0, 100)

# Distribution checks
expect_column_mean_to_be_between("posting_freq_mean", 1, 60)
expect_column_median_to_be_between("first_30d_episode_count", 0, 30)

# Binary column checks
expect_column_values_to_be_in_set("event_occurred", [0, 1])
expect_column_values_to_be_in_set("survived_90d", [0, 1])
expect_column_values_to_be_in_set("event_type", [0, 1, 2])
```

Run validation and save the HTML report to `outputs/data_validation_report.html`.
Raise an assertion error if critical expectations fail (nulls in key columns, out-of-range values).

---

## Phase 4 — Module 1: Cold Start Survival (`src/models/module1_survival.py`)

### 4a. Kaplan-Meier Analysis
```python
from lifelines import KaplanMeierFitter

# Fit overall KM curve
kmf = KaplanMeierFitter()
kmf.fit(durations=df['time_to_first_long_gap'], event_observed=df['event_occurred'])

# Fit by category (top 8 categories by count)
# Fit by cohort year (year of first episode)
# Fit by country (top 5 countries)

# Log-rank tests between category groups
from lifelines.statistics import multivariate_logrank_test
```

Save KM plots to `outputs/figures/km_curves_overall.png`, `km_curves_by_category.png`, `km_curves_by_cohort.png`.

### 4b. Cox Proportional Hazards Model
```python
from lifelines import CoxPHFitter

features_cox = [
    'posting_freq_mean', 'posting_freq_std', 'consistency_score_30d',
    'first_14d_episode_count', 'first_30d_episode_count',
    'posted_week1', 'posted_week2', 'cadence_variance_ratio',
    'eps_per_month_first3', 'has_monetization_signal'
]

cph = CoxPHFitter(penalizer=0.1)
cph.fit(df[features_cox + ['time_to_first_long_gap', 'event_occurred']],
        duration_col='time_to_first_long_gap',
        event_col='event_occurred')

# Print summary with hazard ratios and confidence intervals
cph.print_summary()

# Schoenfeld residuals test (proportional hazards assumption check)
cph.check_assumptions(df, show_plots=True)
```

Target: C-index >= 0.72. Log result to MLflow.

### 4c. XGBoost Early Warning Classifier
```python
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold

# Features: only day-14 observable signals
features_14d = [
    'first_14d_episode_count', 'posting_freq_mean', 'posting_freq_std',
    'posted_week1', 'posted_week2', 'consistency_score_30d'
]
target = 'survived_180d'

# Train/val/test split: 70/15/15, stratified
# Optuna hyperparameter tuning: 50 trials, AUC objective
# Final evaluation on held-out test set
# SHAP values for feature importance
```

Target: AUC-ROC >= 0.80 on test set. Log to MLflow.

### 4d. Prediction Window Learning Curve
Train the XGBoost classifier with features observable at day 7, 14, 21, 30 separately.
Plot AUC vs. prediction window. Save to `outputs/figures/prediction_window_curve.png`.

### 4e. Ablation 1 — Prediction Window Sweep
Already covered by 4d. Save results table to `outputs/ablation1_window_sweep.csv`.

Save all Module 1 artifacts: models to `outputs/models/module1_cox.pkl`, `outputs/models/module1_xgb.pkl`.

---

## Phase 5 — Module 2: Monetization Causal Analysis (`src/models/module2_causal.py`)

### 5a. Monetization Proxy Extraction
Already done in feature engineering. Use `has_monetization_signal` and `monetization_episode_index`.

### 5b. Difference-in-Differences
```python
import statsmodels.formula.api as smf

# Treatment: creators who eventually monetize
# Control: matched creators who never monetize (matched on category, cohort year, early episode count)
# Pre-period: first 3 months. Post-period: months 4-12
# Outcome: eps_per_month (posting frequency as retention proxy)

# DiD formula: outcome ~ treated * post + treated + post + controls
model_did = smf.ols(
    'eps_per_month ~ treated * post + C(category) + C(cohort_year)',
    data=did_df
).fit(cov_type='HC3')  # heteroskedasticity-robust standard errors
```

Test parallel trends assumption: run DiD on pre-period only, check treated coefficient is not significant.
Save DiD results table to `outputs/module2_did_results.csv`.

### 5c. Regression Discontinuity
```python
# Threshold: 20 episodes (proxy for monetization eligibility cutoff)
# Running variable: episode_count at 6 months
# Outcome: is_active at 12 months (binary retention)

# Bandwidth selection via cross-validation
from scipy.optimize import minimize_scalar

# Local linear regression on both sides of threshold
# Plot with confidence intervals
```

McCrary density test: check for manipulation at the threshold.
Save RD plot to `outputs/figures/rd_plot.png`.

### 5d. LTV Estimator
```python
# LTV = survival_probability_12m * eps_per_month * estimated_revenue_per_episode
# estimated_revenue_per_episode: use $18 CPM * average episode downloads
# Segment by category and cohort

# Output: LTV distribution by creator tier (by episode count quartile)
```

Save LTV distribution plot to `outputs/figures/ltv_distribution.png`.

### 5e. Ablation 4 — Monetization Proxy Quality
Compare NLP keyword extraction vs. simple string matching (no spaCy).
Measure agreement on 500 randomly sampled episodes (manually labeled subset).
Save comparison to `outputs/ablation4_monetization_proxy.csv`.

---

## Phase 6 — Module 3: Competing Risks (`src/models/module3_competing.py`)

### 6a. Event Labeling
Use `event_type` column from feature engineering:
- 0 = censored (still active)
- 1 = dormant (went silent >60d but returned)
- 2 = permanently departed (went silent >60d, never returned)

### 6b. Fine-Gray Competing Risks Model
```python
from sksurv.linear_model import CoxPHSurvivalAnalysis
# Note: use lifelines' competing risks or scikit-survival's Fine-Gray implementation

from lifelines import AalenJohansenFitter

# Aalen-Johansen estimator for cumulative incidence functions
ajf_dormant = AalenJohansenFitter(cause_of_interest=1)
ajf_dormant.fit(durations, event_observed=events, event_col='event_type')

ajf_departed = AalenJohansenFitter(cause_of_interest=2)
ajf_departed.fit(durations, event_observed=events, event_col='event_type')
```

Plot cumulative incidence functions for both events, stratified by monetization status.
Save to `outputs/figures/cif_curves.png`.

### 6c. Time-Varying Covariates
Model `monetization_status` as a time-varying covariate (changes when creator first monetizes):
```python
# Expand dataset to long format: one row per creator per time interval
# Each row: (id, start_time, stop_time, event, monetization_at_interval)
from lifelines import CoxTimeVaryingFitter

ctv = CoxTimeVaryingFitter()
ctv.fit(long_format_df, id_col='podcast_id', start_col='start', stop_col='stop',
        event_col='event', weights_col=None)
```

### 6d. Dormancy Duration Model
For creators with `event_type=1` (dormant, eventually returned):
```python
import lightgbm as lgb

# Features: all baseline features + time_to_dormancy
# Target: duration_of_dormancy_days (how long the break lasted)
# Model: LightGBM regressor with Optuna tuning
# Metric: MAE in days
```

### 6e. Platform Action Matrix
```python
# For each creator: compute (dormancy_prob, departure_prob) from competing risks model
# Action rules:
# departure_prob > 0.7 → "write off"
# dormancy_prob > 0.6 AND departure_prob < 0.3 → "aggressive re-engagement"  
# dormancy_prob > 0.4 AND departure_prob 0.3–0.6 → "gentle nudge"
# both < 0.3 → "monitor"

# Output: action distribution across creator base
```

### 6f. Ablations 2 and 3
**Ablation 2 — Competing risks vs. standard Cox:**
Fit a standard Cox model treating dormancy as censoring. Compare C-index to Fine-Gray.
Save to `outputs/ablation2_competing_vs_standard.csv`.

**Ablation 3 — Time-varying vs. fixed covariates:**
Compare CoxTimeVaryingFitter vs. standard CoxPHFitter with baseline monetization.
Report C-index difference. Save to `outputs/ablation3_time_varying.csv`.

---

## Phase 7 — Tests (`tests/`)

### `tests/test_features.py`
```python
def test_consistency_score_bounds():
    # All consistency scores must be between 0 and 1

def test_event_type_valid():
    # event_type must be 0, 1, or 2 only

def test_no_negative_gaps():
    # posting_freq_mean must be >= 0

def test_survival_label_consistency():
    # if survived_180d = 1, creator must have total_active_days >= 180
```

### `tests/test_models.py`
```python
def test_cox_cindex_above_threshold():
    # C-index on test set >= 0.65 (conservative lower bound)

def test_xgb_auc_above_threshold():
    # AUC-ROC >= 0.75 (conservative lower bound)

def test_cif_curves_sum_to_one():
    # At any time t, P(dormant) + P(departed) + P(censored) = 1
```

### `tests/test_pipeline.py`
```python
def test_pipeline_runs_on_sample():
    # Run full pipeline on 1000-row sample, assert no errors

def test_output_files_exist():
    # Assert all expected output files were created
```

Run all tests with `pytest tests/ -v`. All must pass before Phase 8.

---

## Phase 8 — MLflow Experiment Tracking (`src/utils/mlflow_utils.py`)

Log every model run with:
```python
import mlflow

with mlflow.start_run(run_name="module1_cox_v1"):
    mlflow.log_param("penalizer", 0.1)
    mlflow.log_param("features", features_cox)
    mlflow.log_metric("c_index", cph.concordance_index_)
    mlflow.log_artifact("outputs/figures/km_curves_overall.png")
    mlflow.sklearn.log_model(cph, "cox_model")
```

Create one experiment per module: `module1_cold_start`, `module2_monetization`, `module3_competing_risks`.

Run `mlflow ui` locally to verify the dashboard works. Commit `mlruns/` to the repo so reviewers can see experiment history.

---

## Phase 9 — Streamlit Dashboard (`dashboard/app.py`)

### Entry point (`dashboard/app.py`)
```python
import streamlit as st

st.set_page_config(
    page_title="Creator Lifecycle Intelligence Platform",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Page 1: Survival Explorer (`pages/1_survival_explorer.py`)
- Plotly KM survival curves with confidence intervals
- Multiselect filter: category (top 8)
- Slider: cohort year range
- Radio: stratify by category / cohort / country
- Data table: survival rates at 30, 90, 180 days by segment

### Page 2: Early Warning Simulator (`pages/2_early_warning.py`)
- Number inputs: episodes in first 14 days, posting frequency, consistency score
- Toggle: posted in week 1, posted in week 2
- Output: predicted 6-month survival probability (gauge chart)
- SHAP waterfall chart for this specific prediction
- Comparison bar: "this creator vs. median creator in same category"

### Page 3: Monetization Impact (`pages/3_monetization_impact.py`)
- DiD visualization: before/after monetization, treatment vs. control line chart
- RD scatter plot with discontinuity line and CI shading
- LTV calculator: dropdowns for category and creator tier → expected LTV output
- Key findings callout box (hardcoded from model results)

### Page 4: Dormancy Intelligence (`pages/4_dormancy_intelligence.py`)
- Competing risks CIF curves (plotly, interactive)
- Filters: monetization status, category
- Creator profile simulator: sliders for dormancy prob and departure prob → action recommendation
- Platform action matrix: plotly heatmap
- Table: action distribution across creator base

### Shared sidebar (`components/sidebar.py`)
- Project title and description
- Links: GitHub repo, data source attribution
- Module selector (auto-navigates)

---

## Phase 10 — Notebooks (`notebooks/`)

Create four clean Jupyter notebooks — one per phase. Each notebook:
- Has markdown cells explaining the analysis in plain English before every code block
- Imports from `src/` (not inline code)
- Shows all key plots inline
- Has a "Key Findings" markdown cell at the end summarizing what was learned

Notebooks are for storytelling, not computation. Keep code cells short; put logic in `src/`.

---

## Phase 11 — README.md

Generate a README with:

```markdown
# Creator Lifecycle Intelligence Platform

Three-module survival analysis on 550,000+ podcast creators.
Predicts which creators survive, measures the causal effect of monetization,
and classifies dormant creators vs. permanently churned using competing risks.

## Key Findings
[Fill in actual numbers after running]
- 6-month survival rate: X% overall; Y% for creators who posted in both week 1 and 2
- Monetization causal effect on retention: ATT = X (95% CI: [Y, Z])
- Of creators silent for 45 days: X% eventually return; drops to Y% after 90 days

## Modules
| Module | Method | Key Metric | Result |
|--------|--------|-----------|--------|
| 1. Cold Start | Cox PH + XGBoost | C-index / AUC | X / Y |
| 2. Monetization | DiD + RD | ATT / C-index | X / Y |
| 3. Competing Risks | Fine-Gray | C-index | X |

## Live Demo
[HuggingFace Spaces link — add after deployment]

## Tech Stack
Survival analysis: lifelines, scikit-survival
Causal inference: statsmodels, econml
ML: XGBoost, LightGBM, SHAP, Optuna
Tracking: MLflow
Validation: Great Expectations
Dashboard: Streamlit
Data: ageitgey/all-podcasts-dataset (GitHub)

## Setup
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python src/data/loader.py
python src/data/features.py
python src/data/validate.py
streamlit run dashboard/app.py

## Ablation Results
[Fill in after running all ablations]
```

---

## Phase 12 — Git Commit Structure

Make one commit per phase with clean messages (no em dashes, no AI phrasing):

```
feat: initial repo structure and requirements
feat: data pipeline - podcast loader and SQLite storage
feat: feature engineering - 25 creator behavioral features
feat: data validation - Great Expectations suite
feat: module 1 - KM curves and Cox PH survival model
feat: module 1 - XGBoost early warning classifier and ablation
feat: module 2 - DiD and regression discontinuity causal analysis
feat: module 2 - LTV estimator
feat: module 3 - Fine-Gray competing risks model
feat: module 3 - time varying covariates and dormancy duration model
feat: ablations 2 and 3 - competing risks comparison
feat: tests - all unit tests passing
feat: mlflow experiment tracking integrated
feat: streamlit dashboard - all four pages
feat: notebooks - analysis walkthroughs
docs: README with results and setup instructions
```

Push to GitHub. Make repo public. Add HuggingFace Spaces deployment link to README once live.

---

## HuggingFace Spaces Deployment

Create a Space at huggingface.co/spaces with:
- SDK: Streamlit
- `app.py` pointing to `dashboard/app.py`
- `requirements.txt` in root (same file)
- Space title: "Creator Lifecycle Intelligence Platform"
- Space description: "Survival analysis on 550K podcast creators — cold start prediction, monetization causal inference, and competing risks dormancy classification"

---

## Key Constraints

- Zero monetary spend. No paid APIs, no cloud credits.
- All data from `ageitgey/all-podcasts-dataset` only (free GitHub download).
- `en_core_web_sm` spaCy model only (free, small).
- Groq free tier if any LLM calls are needed (none planned in current scope).
- All models must run on CPU (Colab free tier compatible).
- No hardcoded file paths — use `pathlib.Path` and relative paths throughout.
- No `print()` for logging — use Python `logging` module with INFO level.
- Every function must have a docstring.
- Type hints on all function signatures.

---

## Evaluation Targets (Do Not Ship Without Hitting These)

| Module | Metric | Minimum Target |
|--------|--------|----------------|
| Module 1 Cox | C-index | >= 0.68 |
| Module 1 XGBoost | AUC-ROC | >= 0.78 |
| Module 2 DiD | Parallel trends p-value | > 0.05 |
| Module 3 Fine-Gray | C-index | >= 0.65 |
| Tests | Pass rate | 100% |
| Dashboard | Loads without error | Required |

If a target is not hit, tune before moving on. Log all attempts in MLflow.

---

## Start Here

Run this sequence:
```bash
git init creator-lifecycle-platform
cd creator-lifecycle-platform
cp /path/to/CLAUDE.md .
# Then tell Claude Code: "Read CLAUDE.md and start Phase 0"
```
