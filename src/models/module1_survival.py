"""Module 1 — Cold Start Survival.

Kaplan-Meier curves, a Cox Proportional Hazards model, and an XGBoost
early-warning classifier that scores a creator's likelihood of remaining
active from day-14 behavioral signals.

Two substitutions from CLAUDE.md, both driven by what SPoRC's metadata
actually contains:
- No country field is available (only language), so the "by country" KM
  stratification is replaced with "by first-seen week" (a cohort proxy,
  since the observation window is only ~61 days and doesn't span years).
- "6-month survival" becomes survived_30d (the longest horizon that fits
  inside the window) — see data/README.md for the windowing rationale.
"""

import json
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.mlflow_utils import tracked_run
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES_PATH = Path("data/processed/features.parquet")
FIGURES_DIR = Path("outputs/figures")
MODELS_DIR = Path("outputs/models")

XGB_TARGET = "survived_30d"
TOP_N_CATEGORIES = 8


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    df["first_seen_week"] = ((pd.to_datetime(df["first_seen_date"]) - pd.to_datetime(df["first_seen_date"]).min()).dt.days // 7)
    return df


def add_early_window_features(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Cadence/consistency/monetization features computed using ONLY
    episodes observed within the first n_days of a creator's window —
    unlike features.parquet's whole-trajectory posting_freq_mean/std
    (which include the very gap that defines event_occurred, and are
    therefore unusable as predictors: using them caused near-perfect but
    circular AUC/C-index in an earlier version of this script)."""
    suffix = f"_{n_days}d"
    prefix = "early_"

    def _stats(row):
        dates = np.sort(pd.to_datetime(row["episode_dates"]))
        first_seen = pd.Timestamp(row["first_seen_date"])
        cutoff = first_seen + pd.Timedelta(days=n_days)
        early_dates = dates[dates <= cutoff]
        gaps = np.diff(early_dates) / np.timedelta64(1, "D") if len(early_dates) > 1 else np.array([])

        n_weeks = max(1, int(np.ceil(n_days / 7)))
        weeks_with_ep = {min((d - first_seen).days // 7, n_weeks - 1) for d in early_dates}
        consistency = len(weeks_with_ep) / n_weeks

        mean_gap = gaps.mean() if len(gaps) else np.nan
        std_gap = gaps.std() if len(gaps) > 1 else 0.0
        return pd.Series({
            f"{prefix}episode_count{suffix}": len(early_dates),
            f"{prefix}posting_freq_mean{suffix}": mean_gap,
            f"{prefix}posting_freq_std{suffix}": std_gap,
            f"{prefix}cadence_variance_ratio{suffix}": (std_gap / mean_gap) if mean_gap else 0.0,
            f"{prefix}consistency{suffix}": consistency,
            f"{prefix}multi_post{suffix}": int(len(early_dates) > 1),
        })

    stats = df.apply(_stats, axis=1)
    df = pd.concat([df, stats], axis=1)
    # monetization signal restricted to what was observable by day n_days
    df[f"{prefix}has_monetization{suffix}"] = (
        df["has_monetization_signal"].fillna(0).astype(int)
        & (df["monetization_timing_days"].fillna(999) <= n_days).astype(int)
    )
    # acceleration: ratio of posting rate in the second half of the window
    # to the first half — captures whether a creator is speeding up or
    # slowing down, not just their average pace.
    half = max(n_days // 2, 1)
    first_half_count = df.apply(
        lambda r: int((pd.to_datetime(r["episode_dates"]) <= pd.Timestamp(r["first_seen_date"]) + pd.Timedelta(days=half)).sum()),
        axis=1,
    )
    df[f"{prefix}accel{suffix}"] = (df[f"{prefix}episode_count{suffix}"] - first_half_count) - first_half_count
    return df


TOP_CATEGORIES_FOR_MODELING = None  # set in main(), used by both Cox and XGBoost


def add_category_feature(df: pd.DataFrame, top_categories: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["category_grouped"] = df["category"].where(df["category"].isin(top_categories), "other")
    return df


# --- 4a. Kaplan-Meier -------------------------------------------------

def run_km_analysis(df: pd.DataFrame) -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["time_to_first_long_gap"], event_observed=df["event_occurred"])
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf.plot_survival_function(ax=ax)
    ax.set_title("Overall Survival Curve (time to first long gap)")
    ax.set_xlabel("Days since first observed episode")
    ax.set_ylabel("Survival probability")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "km_curves_overall.png", dpi=120)
    plt.close(fig)
    results["overall_median_survival_days"] = kmf.median_survival_time_

    top_categories = df["category"].value_counts().head(TOP_N_CATEGORIES).index.tolist()
    fig, ax = plt.subplots(figsize=(9, 6))
    for cat in top_categories:
        subset = df[df["category"] == cat]
        kmf_cat = KaplanMeierFitter()
        kmf_cat.fit(subset["time_to_first_long_gap"], subset["event_occurred"], label=cat)
        kmf_cat.plot_survival_function(ax=ax, ci_show=False)
    ax.set_title(f"Survival Curves by Category (top {TOP_N_CATEGORIES})")
    ax.set_xlabel("Days since first observed episode")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "km_curves_by_category.png", dpi=120)
    plt.close(fig)

    cat_subset = df[df["category"].isin(top_categories)]
    logrank_cat = multivariate_logrank_test(
        cat_subset["time_to_first_long_gap"], cat_subset["category"], cat_subset["event_occurred"]
    )
    results["logrank_category_p_value"] = float(logrank_cat.p_value)

    week_bins = sorted(df["first_seen_week"].unique())[:8]
    fig, ax = plt.subplots(figsize=(9, 6))
    for wk in week_bins:
        subset = df[df["first_seen_week"] == wk]
        if len(subset) < 30:
            continue
        kmf_wk = KaplanMeierFitter()
        kmf_wk.fit(subset["time_to_first_long_gap"], subset["event_occurred"], label=f"week {wk}")
        kmf_wk.plot_survival_function(ax=ax, ci_show=False)
    ax.set_title("Survival Curves by First-Seen Week (cohort proxy)")
    ax.set_xlabel("Days since first observed episode")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "km_curves_by_cohort.png", dpi=120)
    plt.close(fig)

    logger.info("KM analysis done: median survival=%.1fd, logrank p=%.4g",
                results["overall_median_survival_days"], results["logrank_category_p_value"])
    return results


# --- 4b. Cox Proportional Hazards -------------------------------------

COX_WINDOW_DAYS = 14
COX_FEATURES = [
    "early_posting_freq_mean_14d", "early_posting_freq_std_14d",
    "early_consistency_14d", "early_cadence_variance_ratio_14d",
    "early_episode_count_14d", "early_has_monetization_14d", "early_accel_14d",
]


def run_cox_ph(df: pd.DataFrame) -> tuple[CoxPHFitter, dict]:
    """df must already have add_early_window_features(df, 14) and
    add_category_feature applied. Stratifies on category_grouped rather
    than including it as a linear covariate: the KM analysis found category
    is a highly significant driver of survival (logrank p<1e-290), and
    several continuous covariates fail the proportional-hazards test, so a
    per-category baseline hazard is a more honest fit than forcing one
    global hazard shape across very different content categories."""
    cols = COX_FEATURES + ["category_grouped", "time_to_first_long_gap", "event_occurred"]
    cox_df = df[cols].dropna()

    # Landmark analysis at day 14: covariates must be measured strictly
    # before the risk period they predict, so keep only creators still
    # event-free at the day-14 landmark, and re-zero duration to time-since-
    # landmark. Without this, covariates computed from data through day 14
    # would be fit against events that already happened inside that same
    # window — the circularity that produced a near-perfect but meaningless
    # C-index in an earlier version of this script.
    cox_df = cox_df[cox_df["time_to_first_long_gap"] >= COX_WINDOW_DAYS]
    cox_df = cox_df.assign(duration_since_landmark=cox_df["time_to_first_long_gap"] - COX_WINDOW_DAYS)
    cols = [c for c in cox_df.columns if c not in ("time_to_first_long_gap",)]
    cox_df = cox_df[cols]

    train_df, test_df = train_test_split(cox_df, test_size=0.2, random_state=42)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_df, duration_col="duration_since_landmark", event_col="event_occurred", strata=["category_grouped"])

    train_cindex = cph.concordance_index_
    test_cindex = cph.score(test_df, scoring_method="concordance_index")

    logger.info("Cox PH: train C-index=%.4f, test C-index=%.4f", train_cindex, test_cindex)

    summary_path = FIGURES_DIR / "cox_ph_summary.txt"
    with open(summary_path, "w") as f:
        f.write(cph.summary.to_string())

    try:
        fig = plt.figure(figsize=(8, 6))
        cph.check_assumptions(train_df, show_plots=False)
        assumptions_ok = True
    except Exception as e:
        logger.warning("Proportional hazards assumption check raised: %s", e)
        assumptions_ok = False
    finally:
        plt.close("all")

    metrics = {
        "train_c_index": float(train_cindex),
        "test_c_index": float(test_cindex),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "proportional_hazards_check_ran": assumptions_ok,
    }
    return cph, metrics


# --- 4c. XGBoost early warning classifier ------------------------------

DAY14_FEATURES = [
    "early_episode_count_14d", "early_posting_freq_mean_14d",
    "early_posting_freq_std_14d", "early_multi_post_14d",
    "early_consistency_14d", "early_accel_14d", "category_grouped",
]


def run_xgb_classifier(df: pd.DataFrame) -> tuple[xgb.XGBClassifier, dict, pd.DataFrame]:
    """df must already have add_early_window_features(df, 14) and
    add_category_feature applied. category_grouped is included as a native
    XGBoost categorical feature — the KM analysis found category is a
    strong, immediately-known (day-0) survival signal that the day-14
    model was otherwise missing."""
    cols = DAY14_FEATURES + [XGB_TARGET]
    xgb_df = df[cols].dropna().copy()
    xgb_df["category_grouped"] = xgb_df["category_grouped"].astype("category")
    X, y = xgb_df[DAY14_FEATURES], xgb_df[XGB_TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "eval_metric": "auc",
        }
        model = xgb.XGBClassifier(**params, random_state=42, enable_categorical=True)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best_params = study.best_params
    final_model = xgb.XGBClassifier(**best_params, eval_metric="auc", random_state=42, enable_categorical=True)
    final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    test_preds = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)

    logger.info("XGBoost: best val AUC=%.4f, test AUC=%.4f", study.best_value, test_auc)

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test.iloc[:2000])

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test.iloc[:2000], show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "xgb_shap_summary.png", dpi=120)
    plt.close(fig)

    metrics = {
        "val_auc": float(study.best_value),
        "test_auc": float(test_auc),
        "best_params": best_params,
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
    }
    return final_model, metrics, X_test


# --- 4d/4e. Prediction window sweep (also Ablation 1) -------------------

WINDOW_SWEEP_DAYS = (3, 7, 14, 21)


def run_window_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """df must already have add_early_window_features(df, n) applied for
    every n in WINDOW_SWEEP_DAYS."""
    rows = []
    for window_day in WINDOW_SWEEP_DAYS:
        suffix = f"_{window_day}d"
        feats = [
            f"early_episode_count{suffix}", f"early_posting_freq_mean{suffix}",
            f"early_posting_freq_std{suffix}", f"early_multi_post{suffix}",
            f"early_consistency{suffix}",
        ]
        cols = feats + [XGB_TARGET]
        wdf = df[cols].dropna()
        X, y = wdf[feats], wdf[XGB_TARGET]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        model = xgb.XGBClassifier(max_depth=4, n_estimators=150, learning_rate=0.1, eval_metric="auc", random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        rows.append({"window_day": window_day, "n_features": len(feats), "auc": auc, "n_test": len(X_test)})
        logger.info("Window day %d: AUC=%.4f (%d features)", window_day, auc, len(feats))

    result_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(result_df["window_day"], result_df["auc"], marker="o")
    ax.set_xlabel("Prediction window (days since first observed episode)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Early Warning AUC vs. Prediction Window")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prediction_window_curve.png", dpi=120)
    plt.close(fig)

    result_df.to_csv(Path("outputs/ablation1_window_sweep.csv"), index=False)
    return result_df


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    logger.info("Loaded %d creators for Module 1", len(df))

    for n in WINDOW_SWEEP_DAYS:
        df = add_early_window_features(df, n)
    top_categories = df["category"].value_counts().head(TOP_N_CATEGORIES).index.tolist()
    df = add_category_feature(df, top_categories)
    logger.info("Computed early-window (pre-cutoff-only) features for days %s", WINDOW_SWEEP_DAYS)

    with tracked_run("module1", "km_analysis"):
        km_results = run_km_analysis(df)
        mlflow.log_metrics({k: v for k, v in km_results.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(FIGURES_DIR / "km_curves_overall.png"))
        mlflow.log_artifact(str(FIGURES_DIR / "km_curves_by_category.png"))
        mlflow.log_artifact(str(FIGURES_DIR / "km_curves_by_cohort.png"))

    with tracked_run("module1", "cox_ph_v1"):
        cph, cox_metrics = run_cox_ph(df)
        mlflow.log_param("penalizer", 0.1)
        mlflow.log_param("features", COX_FEATURES)
        mlflow.log_metrics({k: v for k, v in cox_metrics.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(FIGURES_DIR / "cox_ph_summary.txt"))
        with open(MODELS_DIR / "module1_cox.pkl", "wb") as f:
            pickle.dump(cph, f)

    with tracked_run("module1", "xgb_early_warning_v1"):
        xgb_model, xgb_metrics, X_test = run_xgb_classifier(df)
        mlflow.log_params(xgb_metrics["best_params"])
        mlflow.log_metric("val_auc", xgb_metrics["val_auc"])
        mlflow.log_metric("test_auc", xgb_metrics["test_auc"])
        mlflow.log_artifact(str(FIGURES_DIR / "xgb_shap_summary.png"))
        mlflow.xgboost.log_model(xgb_model, name="xgb_early_warning")
        with open(MODELS_DIR / "module1_xgb.pkl", "wb") as f:
            pickle.dump(xgb_model, f)

    with tracked_run("module1", "window_sweep_ablation"):
        sweep_df = run_window_sweep(df)
        for _, row in sweep_df.iterrows():
            mlflow.log_metric(f"auc_day{int(row['window_day'])}", row["auc"])
        mlflow.log_artifact(str(FIGURES_DIR / "prediction_window_curve.png"))
        mlflow.log_artifact("outputs/ablation1_window_sweep.csv")

    all_metrics = {
        "km": km_results,
        "cox_ph": cox_metrics,
        "xgb": {k: v for k, v in xgb_metrics.items() if k != "best_params"} | {"best_params": xgb_metrics["best_params"]},
        "window_sweep": sweep_df.to_dict(orient="records"),
    }
    with open(Path("outputs/module1_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    logger.info("Module 1 complete. Cox test C-index=%.4f, XGB test AUC=%.4f",
                cox_metrics["test_c_index"], xgb_metrics["test_auc"])


if __name__ == "__main__":
    main()
