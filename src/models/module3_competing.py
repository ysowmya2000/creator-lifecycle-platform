"""Module 3 — Dormancy vs. Death: Competing Risks Survival Model.

Distinguishes creators who go quiet and come back (dormant) from those who
go quiet and don't (departed, as observed within the window) — something a
standard survival model that treats all silence as censoring cannot do.

Event labeling here is more precise than features.parquet's event_type
column: that column folds "never went quiet" and "went quiet but already
recovered" into the same "0 = censored" bucket, which conflates two
different populations for a competing-risks question ("of those who go
quiet, do they come back?"). This module instead computes, locally:
  - time_to_dormancy_onset: time from first episode to the first gap > 14
    days (same LONG_GAP_THRESHOLD_DAYS as Module 1), or right-censored at
    window end if no such gap occurs
  - competing_event: 0 = censored (no long gap observed in the window),
    1 = dormant (posted again after the gap started, before window end),
    2 = no recovery observed (silent since the gap started through window
    end — a proxy for departure, not confirmed: the window ending doesn't
    mean they never post again in July)
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
import lightgbm as lgb
from lifelines import AalenJohansenFitter, CoxPHFitter, CoxTimeVaryingFitter
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.module1_survival import add_early_window_features, add_category_feature
from src.data.features import LONG_GAP_THRESHOLD_DAYS
from src.utils.mlflow_utils import tracked_run
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES_PATH = Path("data/processed/features.parquet")
FIGURES_DIR = Path("outputs/figures")
MODELS_DIR = Path("outputs/models")
TOP_N_CATEGORIES = 8


def load_data() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


# --- 6a. Event labeling ---------------------------------------------------

def compute_competing_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = pd.to_datetime(df["last_seen_date"]).max()

    def _labels(row):
        dates = np.sort(pd.to_datetime(row["episode_dates"]))
        first_seen = pd.Timestamp(row["first_seen_date"])
        # append a virtual final checkpoint at snapshot_date so the
        # trailing (incomplete, "still silent as of window end") interval
        # is itself checked for being a long gap — otherwise every
        # detected gap is definitionally followed by an episode (that's
        # what makes it a completed gap in the array), which was making
        # every long gap register as "recovered."
        points = np.append(dates, np.datetime64(snapshot_date))
        gaps = np.diff(points) / np.timedelta64(1, "D")
        long_idx = np.where(gaps > LONG_GAP_THRESHOLD_DAYS)[0]

        if len(long_idx) == 0:
            duration = (snapshot_date - first_seen) / np.timedelta64(1, "D")
            return pd.Series({"time_to_dormancy_onset": duration, "competing_event": 0})

        onset_idx = long_idx[0]
        onset_date = points[onset_idx]
        duration = (onset_date - first_seen) / np.timedelta64(1, "D")
        is_trailing_gap = onset_idx == len(points) - 2  # this gap runs to the snapshot checkpoint, not a real episode
        return pd.Series({"time_to_dormancy_onset": max(duration, 0.0), "competing_event": 2 if is_trailing_gap else 1})

    labels = df.apply(_labels, axis=1)
    return pd.concat([df, labels], axis=1)


# --- 6b. Cumulative incidence functions -----------------------------------

def run_cif_analysis(df: pd.DataFrame) -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    ajf_dormant = AalenJohansenFitter()
    ajf_dormant.fit(df["time_to_dormancy_onset"], df["competing_event"], event_of_interest=1)
    ajf_departed = AalenJohansenFitter()
    ajf_departed.fit(df["time_to_dormancy_onset"], df["competing_event"], event_of_interest=2)

    fig, ax = plt.subplots(figsize=(9, 6))
    ajf_dormant.plot(ax=ax, label="Dormant (recovered)")
    ajf_departed.plot(ax=ax, label="No recovery observed")
    ax.set_xlabel("Days since first observed episode")
    ax.set_ylabel("Cumulative incidence")
    ax.set_title("Cumulative Incidence: Dormancy vs. No-Recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cif_curves.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, (label, subset) in zip(axes, [
        ("Monetized", df[df["has_monetization_signal"] == 1]),
        ("Not monetized", df[df["has_monetization_signal"] == 0]),
    ]):
        for event, ev_label in [(1, "Dormant"), (2, "No recovery")]:
            ajf = AalenJohansenFitter()
            ajf.fit(subset["time_to_dormancy_onset"], subset["competing_event"], event_of_interest=event)
            ajf.plot(ax=ax, label=ev_label)
        ax.set_title(label)
        ax.set_xlabel("Days")
    axes[0].set_ylabel("Cumulative incidence")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cif_curves_by_monetization.png", dpi=120)
    plt.close(fig)

    # Gray's-test-style check: compare CIFs for monetized vs. not, via the
    # recovery rate among creators who ever went quiet (a simpler, more
    # transparent stand-in for the formal Gray's test, which lifelines
    # doesn't implement)
    ever_quiet = df[df["competing_event"].isin([1, 2])]
    recovery_by_monetization = ever_quiet.groupby("has_monetization_signal")["competing_event"].apply(
        lambda s: (s == 1).mean()
    )
    results["recovery_rate_monetized"] = float(recovery_by_monetization.get(1, np.nan))
    results["recovery_rate_not_monetized"] = float(recovery_by_monetization.get(0, np.nan))
    results["n_ever_quiet"] = int(len(ever_quiet))
    results["n_recovered"] = int((df["competing_event"] == 1).sum())
    results["n_no_recovery"] = int((df["competing_event"] == 2).sum())
    results["n_censored"] = int((df["competing_event"] == 0).sum())

    logger.info("CIF: %d recovered, %d no-recovery, %d censored. Recovery rate monetized=%.3f vs not=%.3f",
                results["n_recovered"], results["n_no_recovery"], results["n_censored"],
                results["recovery_rate_monetized"], results["recovery_rate_not_monetized"])
    return results


# --- 6c. Time-varying covariates -------------------------------------------

def build_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """One row per creator per interval between consecutive episodes, with
    monetization_status updated at the interval where it's first observed
    — a time-varying covariate rather than fixed at baseline."""
    rows = []
    for _, row in df.iterrows():
        dates = np.sort(pd.to_datetime(row["episode_dates"]))
        first_seen = pd.Timestamp(row["first_seen_date"])
        mon_day = row["monetization_timing_days"] if pd.notna(row["monetization_timing_days"]) else np.inf

        if len(dates) < 2:
            continue
        day_offsets = [(d - first_seen) / np.timedelta64(1, "D") for d in dates]
        for i in range(len(day_offsets) - 1):
            start, stop = day_offsets[i], day_offsets[i + 1]
            event = 1 if (i == len(day_offsets) - 2 and row["competing_event"] == 2 and stop - start > LONG_GAP_THRESHOLD_DAYS) else 0
            monetized_at_interval = int(start >= mon_day)
            rows.append({
                "podcast_id": row["podcast_id"], "start": start, "stop": stop,
                "event": event, "monetized": monetized_at_interval,
            })
    return pd.DataFrame(rows)


def run_time_varying_cox(df: pd.DataFrame) -> dict:
    from lifelines.utils import concordance_index

    sample = df.sample(min(30000, len(df)), random_state=42)
    train_ids, test_ids = train_test_split(sample["podcast_id"], test_size=0.2, random_state=42)

    long_df = build_long_format(sample)
    long_df = long_df[long_df["stop"] > long_df["start"]]
    train_long = long_df[long_df["podcast_id"].isin(train_ids)]
    test_long = long_df[long_df["podcast_id"].isin(test_ids)]

    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(train_long, id_col="podcast_id", start_col="start", stop_col="stop", event_col="event")

    # Time-varying C-index: use each held-out creator's LAST interval's
    # partial hazard as their summary risk score (lifelines has no built-in
    # concordance for time-varying models), scored against their true
    # (duration, departure-event) outcome.
    test_last_rows = test_long.sort_values("stop").groupby("podcast_id").tail(1).copy()
    test_last_rows["risk_score"] = ctv.predict_partial_hazard(test_last_rows)
    outcomes = df.set_index("podcast_id").loc[test_last_rows["podcast_id"]].reset_index()
    scoring_df = pd.DataFrame({
        "duration": outcomes["time_to_dormancy_onset"].values,
        "risk_score": test_last_rows["risk_score"].values,
        "event": (outcomes["competing_event"] == 2).astype(int).values,
    }).dropna()
    tv_cindex = concordance_index(
        scoring_df["duration"], -scoring_df["risk_score"], event_observed=scoring_df["event"],
    )

    # Ablation 3 comparison: same train/test split, but monetization fixed
    # at baseline (whatever it was at start of window) instead of time-varying
    def _fixed_cox_cindex(id_subset):
        sub = df[df["podcast_id"].isin(id_subset)].copy()
        sub["monetized_baseline"] = (sub["monetization_timing_days"].fillna(np.inf) <= 0).astype(int)
        sub["event_binary"] = (sub["competing_event"] == 2).astype(int)
        cols = ["monetized_baseline", "time_to_dormancy_onset", "event_binary"]
        return sub[cols].dropna()

    fixed_train = _fixed_cox_cindex(train_ids)
    fixed_test = _fixed_cox_cindex(test_ids)
    cph_fixed = CoxPHFitter(penalizer=0.1)
    cph_fixed.fit(fixed_train, duration_col="time_to_dormancy_onset", event_col="event_binary")
    fixed_cindex = cph_fixed.score(fixed_test, scoring_method="concordance_index")

    with open(FIGURES_DIR / "module3_ctv_summary.txt", "w") as f:
        f.write(ctv.summary.to_string())

    results = {
        "time_varying_log_likelihood": float(ctv.log_likelihood_),
        "time_varying_concordance": float(tv_cindex),
        "fixed_baseline_concordance": float(fixed_cindex),
        "n_intervals": len(long_df), "n_creators_train": len(train_ids), "n_creators_test": len(test_ids),
    }
    logger.info("Time-varying Cox: held-out C-index=%.4f vs fixed-baseline held-out C-index=%.4f",
                results["time_varying_concordance"], results["fixed_baseline_concordance"])
    pd.DataFrame([results]).to_csv(Path("outputs/ablation3_time_varying.csv"), index=False)
    return results


# --- 6d. Dormancy duration model -------------------------------------------

def compute_dormancy_durations(df: pd.DataFrame) -> pd.DataFrame:
    recovered = df[df["competing_event"] == 1].copy()

    def _duration(row):
        dates = np.sort(pd.to_datetime(row["episode_dates"]))
        gaps = np.diff(dates) / np.timedelta64(1, "D")
        long_idx = np.where(gaps > LONG_GAP_THRESHOLD_DAYS)[0]
        return float(gaps[long_idx[0]]) if len(long_idx) else np.nan

    recovered["dormancy_duration_days"] = recovered.apply(_duration, axis=1)
    return recovered.dropna(subset=["dormancy_duration_days"])


def run_dormancy_duration_model(df: pd.DataFrame) -> dict:
    recovered = compute_dormancy_durations(df)
    recovered = add_early_window_features(recovered, 14)
    top_categories = df["category"].value_counts().head(TOP_N_CATEGORIES).index.tolist()
    recovered = add_category_feature(recovered, top_categories)
    recovered["category_grouped"] = recovered["category_grouped"].astype("category")

    features = ["early_episode_count_14d", "early_posting_freq_mean_14d", "early_consistency_14d", "category_grouped"]
    cols = features + ["dormancy_duration_days"]
    lgb_df = recovered[cols].dropna()
    X, y = lgb_df[features], lgb_df["dormancy_duration_days"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
        model = lgb.LGBMRegressor(**params, random_state=42, verbosity=-1)
        model.fit(X_train, y_train, categorical_feature=["category_grouped"])
        preds = model.predict(X_test)
        return mean_absolute_error(y_test, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    best_model = lgb.LGBMRegressor(**study.best_params, random_state=42, verbosity=-1)
    best_model.fit(X_train, y_train, categorical_feature=["category_grouped"])
    test_mae = mean_absolute_error(y_test, best_model.predict(X_test))

    logger.info("Dormancy duration model: test MAE=%.2f days (n=%d)", test_mae, len(lgb_df))
    with open(MODELS_DIR / "module3_dormancy_duration.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return {"test_mae_days": float(test_mae), "n_samples": len(lgb_df), "best_params": study.best_params}


# --- 6e. Platform action matrix ---------------------------------------------

def run_platform_action_matrix(df: pd.DataFrame) -> dict:
    df = add_early_window_features(df, 14)
    top_categories = df["category"].value_counts().head(TOP_N_CATEGORIES).index.tolist()
    df = add_category_feature(df, top_categories)
    df["category_grouped"] = df["category_grouped"].astype("category")

    features = ["early_episode_count_14d", "early_posting_freq_mean_14d", "early_consistency_14d",
                "early_has_monetization_14d", "category_grouped"]

    import xgboost as xgb
    from sklearn.metrics import roc_auc_score as auc_score

    results = {}
    probs = {}
    for target_name, target_event in [("dormancy_prob", 1), ("departure_prob", 2)]:
        df[f"is_{target_name}"] = (df["competing_event"] == target_event).astype(int)
        cols = features + [f"is_{target_name}"]
        model_df = df[cols].dropna().copy()
        X, y = model_df[features], model_df[f"is_{target_name}"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        model = xgb.XGBClassifier(max_depth=5, n_estimators=200, learning_rate=0.08, eval_metric="auc",
                                   random_state=42, enable_categorical=True)
        model.fit(X_train, y_train)
        test_auc = auc_score(y_test, model.predict_proba(X_test)[:, 1])
        results[f"{target_name}_auc"] = float(test_auc)
        logger.info("%s classifier: test AUC=%.4f", target_name, test_auc)

        full_preds = model.predict_proba(df[features])[:, 1]
        probs[target_name] = full_preds

    df["dormancy_prob"] = probs["dormancy_prob"]
    df["departure_prob"] = probs["departure_prob"]

    def _action(row):
        if row["departure_prob"] > 0.7:
            return "write off"
        if row["dormancy_prob"] > 0.6 and row["departure_prob"] < 0.3:
            return "aggressive re-engagement"
        if row["dormancy_prob"] > 0.4 and 0.3 <= row["departure_prob"] <= 0.6:
            return "gentle nudge"
        return "monitor"

    df["recommended_action"] = df.apply(_action, axis=1)
    action_dist = df["recommended_action"].value_counts(normalize=True).to_dict()
    results["action_distribution"] = {k: float(v) for k, v in action_dist.items()}

    fig, ax = plt.subplots(figsize=(8, 6))
    heat = pd.crosstab(pd.cut(df["dormancy_prob"], 5), pd.cut(df["departure_prob"], 5), normalize=True)
    im = ax.imshow(heat.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([f"{i.left:.1f}-{i.right:.1f}" for i in heat.columns], rotation=45)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels([f"{i.left:.1f}-{i.right:.1f}" for i in heat.index])
    ax.set_xlabel("Departure probability")
    ax.set_ylabel("Dormancy probability")
    ax.set_title("Platform Action Matrix (creator density)")
    fig.colorbar(im, ax=ax, label="fraction of creators")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "action_matrix_heatmap.png", dpi=120)
    plt.close(fig)

    logger.info("Action distribution: %s", results["action_distribution"])
    return results


# --- 6f. Ablation 2: competing risks vs. standard Cox ------------------------

def run_ablation2(df: pd.DataFrame) -> dict:
    df = add_early_window_features(df, 14)
    cox_features = ["early_posting_freq_mean_14d", "early_consistency_14d", "early_episode_count_14d"]

    # Standard Cox: treats ALL exits from observation (dormant OR departed)
    # as the event, ignoring which competing outcome occurred
    std_df = df[cox_features + ["time_to_dormancy_onset", "competing_event"]].dropna().copy()
    std_df["event_any"] = (std_df["competing_event"] != 0).astype(int)
    std_df = std_df[std_df["time_to_dormancy_onset"] > 0]
    std_train, std_test = train_test_split(std_df, test_size=0.2, random_state=42)
    cph_std = CoxPHFitter(penalizer=0.1)
    cph_std.fit(std_train[cox_features + ["time_to_dormancy_onset", "event_any"]],
                duration_col="time_to_dormancy_onset", event_col="event_any")
    std_cindex = cph_std.score(std_test[cox_features + ["time_to_dormancy_onset", "event_any"]],
                                scoring_method="concordance_index")

    # Competing-risks-aware: only the "departure" (no-recovery) outcome
    # counts as the event; "dormant-recovered" is treated as censored for
    # this specific cause-specific model, which is the correct way to
    # isolate the departure hazard in a competing risks setting
    cr_df = df[cox_features + ["time_to_dormancy_onset", "competing_event"]].dropna().copy()
    cr_df["event_departure_only"] = (cr_df["competing_event"] == 2).astype(int)
    cr_df = cr_df[cr_df["time_to_dormancy_onset"] > 0]
    cr_train, cr_test = train_test_split(cr_df, test_size=0.2, random_state=42)
    cph_cr = CoxPHFitter(penalizer=0.1)
    cph_cr.fit(cr_train[cox_features + ["time_to_dormancy_onset", "event_departure_only"]],
               duration_col="time_to_dormancy_onset", event_col="event_departure_only")
    cr_cindex = cph_cr.score(cr_test[cox_features + ["time_to_dormancy_onset", "event_departure_only"]],
                              scoring_method="concordance_index")

    n_dormant = (df["competing_event"] == 1).sum()
    n_total_events = (df["competing_event"] != 0).sum()
    bias_estimate = n_dormant / n_total_events if n_total_events else np.nan

    results = {
        "standard_cox_cindex": float(std_cindex),
        "competing_risks_cindex": float(cr_cindex),
        "fraction_of_events_that_are_recoveries": float(bias_estimate),
        "interpretation": (
            "Standard Cox treats dormant-then-recovered creators as a 'churn' event "
            f"identical to true no-recovery cases. {bias_estimate:.1%} of all 'exit' events "
            "are actually recoveries, so a standard model overstates the departure hazard "
            "by conflating the two."
        ),
    }
    logger.info("Ablation 2: standard Cox C-index=%.4f, competing-risks Cox C-index=%.4f, %.1f%% of events are recoveries",
                std_cindex, cr_cindex, bias_estimate * 100)
    pd.DataFrame([results]).to_csv(Path("outputs/ablation2_competing_vs_standard.csv"), index=False)
    return results


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    logger.info("Loaded %d creators for Module 3", len(df))

    df = compute_competing_risk_labels(df)
    logger.info("Competing risk labels: %s", df["competing_event"].value_counts().to_dict())

    with tracked_run("module3", "cif_analysis"):
        cif_results = run_cif_analysis(df)
        mlflow.log_metrics({k: v for k, v in cif_results.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(FIGURES_DIR / "cif_curves.png"))
        mlflow.log_artifact(str(FIGURES_DIR / "cif_curves_by_monetization.png"))

    with tracked_run("module3", "time_varying_cox"):
        tv_results = run_time_varying_cox(df)
        mlflow.log_metrics({k: v for k, v in tv_results.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(FIGURES_DIR / "module3_ctv_summary.txt"))
        mlflow.log_artifact("outputs/ablation3_time_varying.csv")

    with tracked_run("module3", "dormancy_duration"):
        duration_results = run_dormancy_duration_model(df)
        mlflow.log_metric("test_mae_days", duration_results["test_mae_days"])
        mlflow.log_params(duration_results["best_params"])

    with tracked_run("module3", "action_matrix"):
        action_results = run_platform_action_matrix(df)
        mlflow.log_metric("dormancy_prob_auc", action_results["dormancy_prob_auc"])
        mlflow.log_metric("departure_prob_auc", action_results["departure_prob_auc"])
        mlflow.log_artifact(str(FIGURES_DIR / "action_matrix_heatmap.png"))

    with tracked_run("module3", "ablation2_competing_vs_standard"):
        ablation2_results = run_ablation2(df)
        mlflow.log_metrics({k: v for k, v in ablation2_results.items() if isinstance(v, (int, float))})
        mlflow.log_artifact("outputs/ablation2_competing_vs_standard.csv")

    all_results = {
        "cif": cif_results, "time_varying": tv_results, "dormancy_duration": duration_results,
        "action_matrix": action_results, "ablation2": ablation2_results,
    }
    with open(Path("outputs/module3_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("Module 3 complete.")


if __name__ == "__main__":
    main()
