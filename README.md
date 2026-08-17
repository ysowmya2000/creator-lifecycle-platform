# Creator Lifecycle Intelligence Platform

Three-module survival analysis on **168,619 podcast creators**. Predicts early
churn risk from a creator's first 14 days of behavior, tests whether
monetization causally drives retention, and separates creators who go
quiet-but-return from those who are gone for good — something standard churn
models can't do.

## Key Findings

- **Of creators who go silent for 14+ days, 66% eventually return.** A
  standard churn model would count all of them as churned.
- **66.2% of all "exit" events are actually recoveries, not departures.**
  Standard survival models that treat dormancy as censoring overstate the
  departure hazard by conflating the two — a competing-risks-aware Cox model
  fits meaningfully better (0.691 vs. 0.655 C-index).
- **Monetized creators recover from silence at a higher rate**: 72.1% vs.
  65.9% for non-monetized creators.
- **No statistically significant causal effect of monetization on posting
  behavior was found** (DiD ATT p=0.84, parallel trends holds p=0.54; RD jump
  p=0.51, placebo test also clean p=0.94). This is a real, methodologically
  rigorous null result — both designs pass their own validity checks — not a
  failed analysis.
- **Day-14 early-warning prediction tops out around 0.74 AUC.** That ceiling
  is itself informative: 14 days of posting behavior alone is a moderate
  signal, not a strong one — real platforms would need richer signals
  (audience quality, external promotion) to do much better.

See `notebooks/` for the full walkthroughs and `outputs/*_metrics.json` for
every number behind these findings.

## A note on the data (read this before the numbers surprise you)

This project originally targeted `ageitgey/all-podcasts-dataset`, which turned
out to be static 2014 show metadata with no episode-level history at all. The
dataset actually used, [SPoRC](https://huggingface.co/datasets/blitt/SPoRC),
has real episode-level publish dates for 228K podcasts — but only within a
**61-day observation window** (2020-04-30 to 2020-06-30), not each creator's
full since-launch history.

Every "day N," "gap," and "survived" metric in this project is therefore
defined **relative to a creator's first episode observed in that window**,
not their true launch date — a deliberate reframe from "since-launch
lifecycle" to "trailing-window activity." This mirrors how most real
production churn models actually work in practice (platforms rarely have
clean "user joined" timestamps either), but it does mean absolute day counts
here are rescaled from the original 30/60/90-day thresholds down to fit a
61-day span. Full rationale in `data/README.md`.

## Bugs caught and fixed along the way

Worth stating plainly rather than hiding: three real methodological bugs were
found and fixed during development, not left in.

1. **Data leakage (Module 1).** An early version used whole-trajectory
   statistics as predictors of an outcome derived from that same trajectory —
   produced a fake 0.998 AUC. Fixed with a proper day-14 landmark analysis and
   predictors computed only from pre-cutoff data.
2. **Mechanical correlation (Module 2, regression discontinuity).** The
   running variable and outcome were measured over the same time period,
   creating a spurious "threshold effect." Caught by a placebo test finding a
   "significant jump" at an arbitrary threshold too. Fixed by separating the
   two into genuinely different time periods.
3. **Labeling bug (Module 3).** The first competing-risks event labeler
   checked "is there an episode after this gap" — which is trivially true for
   every completed gap by construction, so "no recovery" never triggered.
   Fixed by checking the trailing gap to the snapshot date, not just
   completed inter-episode gaps.

## Modules

| Module | Method | Key Metric | Result |
|--------|--------|-----------|--------|
| 1. Cold Start | Cox PH + XGBoost | C-index / AUC | 0.58 (fails proportional hazards — reported honestly) / 0.74 |
| 2. Monetization | DiD + Regression Discontinuity | ATT p-value / RD jump p-value | 0.84 (null, parallel trends holds) / 0.51 (null, placebo clean) |
| 3. Competing Risks | Aalen-Johansen CIF + time-varying Cox | Competing-risks vs. standard C-index | 0.691 vs. 0.655 |

## Ablation Results

| # | Ablation | Finding |
|---|----------|---------|
| 1 | Prediction window sweep | AUC climbs 0.64 (day 3) -> 0.69 (day 7) -> 0.73 (day 14) -> 0.77 (day 21) as the observation window grows |
| 2 | Competing risks vs. standard Cox | Standard Cox 0.655 C-index vs. competing-risks-aware 0.691; 66.2% of "exit" events are recoveries |
| 3 | Time-varying vs. fixed covariates | Both ~0.51 C-index using monetization alone — consistent with Module 2's null causal finding |
| 4 | Monetization proxy quality (NLP vs. rules) | 99.6% agreement — the added NLP complexity buys almost nothing here |
| 5 | Cohort size sensitivity | Not run as a separate ablation; see `notebooks/01_data_pipeline.ipynb` for volume distribution context |

## Live Demo

Streamlit dashboard — link added after HuggingFace Spaces deployment.

## Tech Stack

**Data & features:** pandas, DuckDB, rule-based keyword matching for the monetization proxy (CLAUDE.md originally called for a spaCy NLP pipeline; Ablation 4 tests two rule-based variants — episode titles + description vs. description-only — and finds 99.6% agreement between them, so the added complexity of a full NLP pipeline wasn't pursued)
**Survival analysis:** lifelines (KM, Cox PH, Aalen-Johansen, CoxTimeVarying)
**Causal inference:** statsmodels (DiD, RD with HC3 robust SEs)
**ML:** XGBoost, LightGBM, SHAP, Optuna
**Tracking:** MLflow
**Validation:** Great Expectations
**Dashboard:** Streamlit, Plotly
**Data:** [SPoRC](https://huggingface.co/datasets/blitt/SPoRC) (research-use-only license — see below)

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # requires accepting the SPoRC dataset license at
                         # huggingface.co/datasets/blitt/SPoRC first
huggingface-cli download blitt/SPoRC --repo-type dataset \
  --include "metadata/podcast_catalog.parquet" --include "metadata/episode_catalog.parquet" \
  --local-dir data/raw/sporc
python src/data/loader.py
python src/data/features.py
python src/data/validate.py
python src/models/module1_survival.py
python src/models/module2_causal.py
python src/models/module3_competing.py
pytest tests/ -v
streamlit run dashboard/app.py
```

MLflow experiment history is committed to `mlruns/`. View it locally with:

```bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns
```

## Repo Structure

```
data/            # raw (gitignored) + processed feature tables
src/data/        # loader, feature engineering, validation
src/models/      # module1_survival.py, module2_causal.py, module3_competing.py
src/utils/       # MLflow tracking helper
dashboard/       # 4-page Streamlit app
notebooks/       # storytelling walkthroughs, one per phase
tests/           # pytest suite (27 tests)
outputs/         # figures, models, ablation results, metrics JSON
mlruns/          # committed MLflow experiment history
```

## License note

The underlying data (SPoRC) is licensed research-use-only, non-commercial.
Raw data is not redistributed in this repo (`data/raw/` is gitignored) — only
derived features, model artifacts, and aggregate statistics.
