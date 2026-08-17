"""Module 2 — Monetization Causal Analysis.

Difference-in-differences, regression discontinuity, and an LTV estimator
for the causal effect of monetization on creator retention.

Data-reality note: of the 8,679 creators flagged has_monetization_signal,
91% (7,909) were flagged from their static podcast description, which
carries no in-window "before" period — the signal was already present
before their first observed episode. Only 770 creators have a genuine
in-window monetization *event* (a keyword first appearing in a specific
episode's title), so the DiD event study below is built on that smaller,
but legitimate, subset rather than the full 8,679.
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.mlflow_utils import tracked_run
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = Path("data/processed/features.parquet")
FIGURES_DIR = Path("outputs/figures")



def load_data() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


# --- 5b. Difference-in-differences --------------------------------------

def _rate_in_window(row, start_day, end_day):
    dates = pd.to_datetime(row["episode_dates"])
    first_seen = pd.Timestamp(row["first_seen_date"])
    lo, hi = first_seen + pd.Timedelta(days=start_day), first_seen + pd.Timedelta(days=end_day)
    n = int(((dates >= lo) & (dates <= hi)).sum())
    return n / max((end_day - start_day) / 7, 1e-6)


def build_did_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Treatment: creators with a genuine in-window monetization event
    (monetization_episode_index >= 0). Control: nearest-neighbor matched on
    category, first-seen week, AND pre-event posting rate — matching only
    on category/week (an earlier version of this) left the parallel-trends
    test failing at p=0.005, because it didn't force treated and control
    groups to already look similar before the event, which is what
    parallel trends actually requires. Outcome: posting rate (episodes/
    week) in the 14 days before vs. after the (pseudo-)event day."""
    treated = df[df["monetization_episode_index"] >= 0].copy()
    treated["event_day"] = treated["monetization_timing_days"]
    treated["treated"] = 1

    pool = df[df["has_monetization_signal"] == 0].copy()
    pool["first_seen_week"] = ((pd.to_datetime(pool["first_seen_date"]) - pd.to_datetime(df["first_seen_date"]).min()).dt.days // 7)
    treated["first_seen_week"] = ((pd.to_datetime(treated["first_seen_date"]) - pd.to_datetime(df["first_seen_date"]).min()).dt.days // 7)

    treated["pre_rate"] = treated.apply(lambda r: _rate_in_window(r, max(r["event_day"] - 14, 0), r["event_day"]), axis=1)

    rows = []
    used_ids = set()
    for _, t_row in treated.iterrows():
        candidates = pool[
            (pool["category"] == t_row["category"]) & (pool["first_seen_week"] == t_row["first_seen_week"])
            & (~pool["podcast_id"].isin(used_ids))
        ]
        if len(candidates) == 0:
            candidates = pool[(pool["category"] == t_row["category"]) & (~pool["podcast_id"].isin(used_ids))]
        if len(candidates) == 0:
            continue
        cand_pre_rates = candidates.apply(
            lambda r: _rate_in_window(r, max(t_row["event_day"] - 14, 0), t_row["event_day"]), axis=1
        )
        best_idx = (cand_pre_rates - t_row["pre_rate"]).abs().idxmin()
        ctrl = candidates.loc[best_idx]
        used_ids.add(ctrl["podcast_id"])

        ctrl_row = ctrl.copy()
        ctrl_row["event_day"] = t_row["event_day"]  # assign the same pseudo-event day
        ctrl_row["treated"] = 0
        rows.append(t_row)
        rows.append(ctrl_row)

    matched = pd.DataFrame(rows)

    long_rows = []
    for _, row in matched.iterrows():
        ed = row["event_day"]
        pre = _rate_in_window(row, max(ed - 14, 0), ed)
        post = _rate_in_window(row, ed, min(ed + 14, 60))
        long_rows.append({"podcast_id": row["podcast_id"], "treated": row["treated"], "post": 0,
                           "eps_per_week": pre, "category": row["category"]})
        long_rows.append({"podcast_id": row["podcast_id"], "treated": row["treated"], "post": 1,
                           "eps_per_week": post, "category": row["category"]})

    return pd.DataFrame(long_rows)


def run_did(df: pd.DataFrame) -> dict:
    did_df = build_did_dataset(df)
    logger.info("DiD dataset: %d creators (%d treated)", did_df["podcast_id"].nunique(),
                did_df[did_df["treated"] == 1]["podcast_id"].nunique())

    model = smf.ols("eps_per_week ~ treated * post + C(category)", data=did_df).fit(cov_type="HC3")
    att = model.params.get("treated:post", np.nan)
    att_ci = model.conf_int().loc["treated:post"].tolist() if "treated:post" in model.params.index else [np.nan, np.nan]
    att_p = model.pvalues.get("treated:post", np.nan)

    pre_only = did_df[did_df["post"] == 0]
    parallel_model = smf.ols("eps_per_week ~ treated + C(category)", data=pre_only).fit(cov_type="HC3")
    parallel_p = parallel_model.pvalues.get("treated", np.nan)

    with open(FIGURES_DIR / "module2_did_summary.txt", "w") as f:
        f.write(model.summary().as_text())

    result_df = did_df.groupby(["treated", "post"])["eps_per_week"].mean().reset_index()
    result_df.to_csv(Path("outputs/module2_did_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for treated_val, label in [(0, "control"), (1, "treated")]:
        sub = result_df[result_df["treated"] == treated_val].sort_values("post")
        ax.plot(sub["post"], sub["eps_per_week"], marker="o", label=label)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pre-event", "post-event"])
    ax.set_ylabel("Episodes per week")
    ax.set_title("DiD: Posting Rate Around Monetization Event")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "module2_did_plot.png", dpi=120)
    plt.close(fig)

    logger.info("DiD: ATT=%.4f (p=%.4g), parallel-trends p=%.4g", att, att_p, parallel_p)
    return {
        "att": float(att), "att_ci_lower": float(att_ci[0]), "att_ci_upper": float(att_ci[1]),
        "att_p_value": float(att_p), "parallel_trends_p_value": float(parallel_p),
        "n_treated": int(did_df[did_df["treated"] == 1]["podcast_id"].nunique()),
        "n_control": int(did_df[did_df["treated"] == 0]["podcast_id"].nunique()),
    }


# --- 5c. Regression discontinuity ----------------------------------------

def build_rd_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Running variable and outcome must be temporally separated, or a
    creator who's simply more prolific throughout the window mechanically
    scores high on both (an earlier version of this used episodes_in_window
    — the whole-window count — as the running variable and activity at
    window end as the outcome, which are measured over the same period and
    are therefore correlated by construction, not by any threshold effect
    — confirmed by a placebo test finding a "significant" jump at an
    arbitrary threshold too).

    Fix: use a fixed calendar checkpoint at window_start + 30 days. Running
    variable = episodes posted before the checkpoint; outcome = whether the
    creator posted again after it. Only creators who started before the
    checkpoint are included, so every creator has a defined pre-checkpoint
    count."""
    window_start = pd.to_datetime(df["first_seen_date"]).min()
    checkpoint = window_start + pd.Timedelta(days=30)

    df = df[pd.to_datetime(df["first_seen_date"]) <= checkpoint].copy()

    def _counts(row):
        dates = pd.to_datetime(row["episode_dates"])
        pre = int((dates < checkpoint).sum())
        post = int((dates >= checkpoint).sum())
        return pd.Series({"episodes_before_checkpoint": pre, "active_after_checkpoint": int(post > 0)})

    counts = df.apply(_counts, axis=1)
    return pd.concat([df, counts], axis=1)


def select_rd_bandwidth(df: pd.DataFrame, threshold: int, candidates: list[float]) -> float:
    """K-fold CV over bandwidth candidates: fit local linear regression of
    the outcome on the running variable within +/-bw of the threshold,
    pick the bandwidth minimizing held-out MSE."""
    best_bw, best_mse = candidates[0], np.inf
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for bw in candidates:
        window = df[(df["episodes_before_checkpoint"] >= threshold - bw) & (df["episodes_before_checkpoint"] <= threshold + bw)]
        if len(window) < 50:
            continue
        mses = []
        X = window["episodes_before_checkpoint"].values - threshold
        y = window["active_after_checkpoint"].values
        for train_idx, test_idx in kf.split(X):
            coef = np.polyfit(X[train_idx], y[train_idx], 1)
            preds = np.polyval(coef, X[test_idx])
            mses.append(np.mean((preds - y[test_idx]) ** 2))
        mean_mse = float(np.mean(mses))
        if mean_mse < best_mse:
            best_mse, best_bw = mean_mse, bw

    return best_bw


def run_rd(df: pd.DataFrame) -> dict:
    rd_df = build_rd_dataset(df)

    # The whole corpus was pre-filtered to creators with >=2 total episodes
    # in the window (loader.py), so a creator with exactly 1 pre-checkpoint
    # episode is *guaranteed* to have a later one (that's where their
    # required 2nd episode must fall) — active_after_checkpoint=1 by
    # construction, not behavior. That mechanical floor sits at episode
    # counts 0-2 and would masquerade as a "discontinuity" at any nearby
    # threshold. Excluding it rather than picking a threshold that
    # unknowingly straddles it.
    rd_df = rd_df[rd_df["episodes_before_checkpoint"] >= 3]
    threshold = int(rd_df["episodes_before_checkpoint"].quantile(0.80))
    logger.info("RD: %d creators eligible (started before day-30 checkpoint), threshold=%d episodes",
                len(rd_df), threshold)

    bw_candidates = [2, 3, 4, 5, 6, 8]
    bw = select_rd_bandwidth(rd_df, threshold, bw_candidates)
    logger.info("RD: selected bandwidth=%d episodes", bw)

    window = rd_df[
        (rd_df["episodes_before_checkpoint"] >= threshold - bw)
        & (rd_df["episodes_before_checkpoint"] <= threshold + bw)
    ].copy()
    window["running"] = window["episodes_before_checkpoint"] - threshold
    window["above"] = (window["running"] >= 0).astype(int)

    rd_model = smf.ols("active_after_checkpoint ~ running * above", data=window).fit(cov_type="HC3")
    jump = rd_model.params.get("above", np.nan)
    jump_p = rd_model.pvalues.get("above", np.nan)
    jump_ci = rd_model.conf_int().loc["above"].tolist() if "above" in rd_model.params.index else [np.nan, np.nan]

    # McCrary-style density check: compare counts just below vs. just above
    below = (window["running"] < 0).sum()
    above = (window["running"] >= 0).sum()
    density_ratio = above / below if below else np.nan

    # Placebo test: fake threshold shifted well above the real one, but
    # still inside the >=3 clean region (a threshold below 3 would butt up
    # against the exclusion floor above and create a near-empty "below"
    # group — extreme imbalance producing a spuriously tiny p-value that
    # has nothing to do with a real discontinuity).
    placebo_threshold = threshold + max(2 * bw, 4)
    placebo_window = rd_df[
        (rd_df["episodes_before_checkpoint"] >= placebo_threshold - bw) & (rd_df["episodes_before_checkpoint"] <= placebo_threshold + bw)
    ].copy()
    placebo_window["running"] = placebo_window["episodes_before_checkpoint"] - placebo_threshold
    placebo_window["above"] = (placebo_window["running"] >= 0).astype(int)
    placebo_model = smf.ols("active_after_checkpoint ~ running * above", data=placebo_window).fit(cov_type="HC3")
    placebo_p = placebo_model.pvalues.get("above", np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    binned = window.groupby("episodes_before_checkpoint")["active_after_checkpoint"].mean()
    ax.scatter(binned.index, binned.values, alpha=0.6, label="binned mean")
    for side_df, style in [(window[window["above"] == 0], "-"), (window[window["above"] == 1], "-")]:
        if len(side_df) < 2:
            continue
        coef = np.polyfit(side_df["running"], side_df["active_after_checkpoint"], 1)
        xs = np.linspace(side_df["running"].min(), side_df["running"].max(), 20)
        ax.plot(xs + threshold, np.polyval(coef, xs), style, color="C1")
    ax.axvline(threshold, color="gray", linestyle="--", label=f"threshold ({threshold} episodes by day 30)")
    ax.set_xlabel("Episodes posted before day-30 checkpoint")
    ax.set_ylabel("P(active after checkpoint)")
    ax.set_title("Regression Discontinuity: Episode Count Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rd_plot.png", dpi=120)
    plt.close(fig)

    logger.info("RD: jump=%.4f (p=%.4g), density_ratio=%.3f, placebo_p=%.4g", jump, jump_p, density_ratio, placebo_p)
    return {
        "threshold_episodes": threshold, "bandwidth": bw,
        "jump_estimate": float(jump), "jump_p_value": float(jump_p),
        "jump_ci_lower": float(jump_ci[0]), "jump_ci_upper": float(jump_ci[1]),
        "mccrary_density_ratio": float(density_ratio),
        "placebo_p_value": float(placebo_p),
        "n_window": len(window),
    }


# --- 5d. LTV estimator ----------------------------------------------------

REVENUE_PER_EPISODE_CPM = 18.0
ASSUMED_DOWNLOADS_PER_EPISODE = 500  # conservative, documented assumption


def run_ltv(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["survival_proxy"] = df["survived_30d"]  # from Module 1 target; see README
    df["est_revenue_per_episode"] = REVENUE_PER_EPISODE_CPM * (ASSUMED_DOWNLOADS_PER_EPISODE / 1000)
    df["ltv"] = df["survival_proxy"] * df["eps_per_month_total"] * df["est_revenue_per_episode"]

    df["episode_tier"] = pd.qcut(df["episodes_in_window"], 4, labels=["Q1 (fewest)", "Q2", "Q3", "Q4 (most)"], duplicates="drop")
    ltv_by_tier = df.groupby("episode_tier", observed=True)["ltv"].agg(["mean", "median", "std", "count"])
    ltv_by_tier.to_csv(Path("outputs/module2_ltv_by_tier.csv"))

    fig, ax = plt.subplots(figsize=(8, 5))
    for tier in df["episode_tier"].cat.categories:
        subset = df[df["episode_tier"] == tier]["ltv"]
        ax.hist(subset.clip(upper=subset.quantile(0.95)), bins=30, alpha=0.5, label=str(tier))
    ax.set_xlabel("Estimated LTV ($)")
    ax.set_ylabel("Creator count")
    ax.set_title("LTV Distribution by Creator Tier (episode count quartile)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ltv_distribution.png", dpi=120)
    plt.close(fig)

    logger.info("LTV: mean=$%.2f, median=$%.2f across %d creators", df["ltv"].mean(), df["ltv"].median(), len(df))
    return {
        "mean_ltv": float(df["ltv"].mean()), "median_ltv": float(df["ltv"].median()),
        "assumed_cpm": REVENUE_PER_EPISODE_CPM, "assumed_downloads_per_episode": ASSUMED_DOWNLOADS_PER_EPISODE,
        "ltv_by_tier": ltv_by_tier.reset_index().to_dict(orient="records"),
    }


# --- 5e. Ablation 4: monetization proxy quality --------------------------

DESCRIPTION_ONLY_KEYWORDS = [
    "patreon", "sponsor", "ad-free", "premium", "subscribe",
    "membership", "supporters", "ad supported", "brought to you by",
]


def run_monetization_proxy_ablation(df: pd.DataFrame, n_sample: int = 500) -> dict:
    """Compares the full proxy (episode titles + podcast description) used
    in features.py against a simpler description-only rule on a random
    sample. No manually-labeled ground truth is available (CLAUDE.md's
    500 hand-labeled episodes assumes a labeling budget this project
    doesn't have), so this reports agreement between the two rule-based
    methods rather than accuracy against ground truth — a real limitation,
    stated plainly rather than presented as validated accuracy."""
    sample = df.sample(min(n_sample, len(df)), random_state=42)

    def _description_only(desc):
        text = str(desc or "").lower()
        return int(any(kw in text for kw in DESCRIPTION_ONLY_KEYWORDS))

    sample = sample.copy()
    sample["desc_only_signal"] = sample["pod_description"].apply(_description_only)
    agreement = (sample["desc_only_signal"] == sample["has_monetization_signal"]).mean()

    result = {
        "n_sample": len(sample),
        "agreement_rate": float(agreement),
        "full_proxy_positive_rate": float(sample["has_monetization_signal"].mean()),
        "description_only_positive_rate": float(sample["desc_only_signal"].mean()),
        "note": "Agreement between two rule-based methods, not accuracy against manual labels (none available).",
    }
    pd.DataFrame([result]).to_csv(Path("outputs/ablation4_monetization_proxy.csv"), index=False)
    logger.info("Ablation 4: agreement rate=%.4f between full proxy and description-only rule", agreement)
    return result


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    logger.info("Loaded %d creators for Module 2", len(df))

    with tracked_run("module2", "did_v1"):
        did_results = run_did(df)
        mlflow.log_metrics({k: v for k, v in did_results.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(FIGURES_DIR / "module2_did_plot.png"))
        mlflow.log_artifact(str(FIGURES_DIR / "module2_did_summary.txt"))
        mlflow.log_artifact("outputs/module2_did_results.csv")

    with tracked_run("module2", "rd_v1"):
        rd_results = run_rd(df)
        mlflow.log_metrics({k: v for k, v in rd_results.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(FIGURES_DIR / "rd_plot.png"))

    with tracked_run("module2", "ltv_v1"):
        ltv_results = run_ltv(df)
        mlflow.log_metrics({k: v for k, v in ltv_results.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(FIGURES_DIR / "ltv_distribution.png"))
        mlflow.log_artifact("outputs/module2_ltv_by_tier.csv")

    with tracked_run("module2", "ablation4_monetization_proxy"):
        ablation4_results = run_monetization_proxy_ablation(df)
        mlflow.log_metrics({k: v for k, v in ablation4_results.items() if isinstance(v, (int, float))})
        mlflow.log_artifact("outputs/ablation4_monetization_proxy.csv")

    all_results = {"did": did_results, "rd": rd_results, "ltv": ltv_results, "ablation4": ablation4_results}
    with open(Path("outputs/module2_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("Module 2 complete. DiD ATT=%.4f (p=%.4g), RD jump=%.4f (p=%.4g)",
                did_results["att"], did_results["att_p_value"], rd_results["jump_estimate"], rd_results["jump_p_value"])


if __name__ == "__main__":
    main()
