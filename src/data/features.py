"""Feature engineering on the creators table.

IMPORTANT — window-relative definitions:
SPoRC's episode_catalog only covers episodes published in a ~61-day
observation window (2020-04-30 to 2020-06-30). We do not know each
creator's true first-ever episode date, so every "day N" / "gap" /
"survived" feature below is defined relative to the creator's first
episode *observed in this window*, not their true launch date. Gap and
survival thresholds are rescaled down from CLAUDE.md's originals
(30/60/90 days, day 7/14/21/30) to fit inside the 61-day span
(7/14/21 day gaps, day 3/7/14 signals). See data/README.md.

There are also no per-episode descriptions in this dataset (only episode
titles and a podcast-level description), so the monetization proxy scans
episode titles + the podcast description rather than episode descriptions.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

MONETIZATION_KEYWORDS = [
    "patreon", "sponsor", "ad-free", "premium", "subscribe",
    "membership", "supporters", "ad supported", "brought to you by",
]

# Gap/survival thresholds in days, rescaled from CLAUDE.md's 30/60/90 to fit
# a 61-day observation window. LONG_GAP_THRESHOLD_DAYS is set empirically:
# across all ~896K individual inter-episode gaps in the corpus, the median
# gap is 5.25 days and the 90th percentile is 14.02 days, with the
# distribution's knee around there (p80=8.16d, p85=12.12d, p90=14.02d,
# p95=20.92d). 14 days is therefore the point where a gap stops looking
# like normal posting-cadence variance and starts looking like a real
# behavioral change, so it's used as the "long gap" / dormancy floor rather
# than a naive proportional rescale of the spec's 30-day threshold (which,
# at 7 days, was tighter than most creators' typical cadence and made
# nearly every biweekly show register a spurious "event").
LONG_GAP_THRESHOLD_DAYS = 14
DORMANCY_THRESHOLDS_DAYS = (14, 21, 30)
SURVIVAL_CHECKPOINTS_DAYS = (14, 30)


def _episode_gaps_days(dates: np.ndarray) -> np.ndarray:
    """Return sorted inter-episode gaps in days for a single creator."""
    dates = np.sort(pd.to_datetime(dates))
    if len(dates) < 2:
        return np.array([])
    deltas = np.diff(dates) / np.timedelta64(1, "D")
    return deltas


def compute_cadence_features(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    gaps = df["episode_dates"].apply(_episode_gaps_days)

    df["posting_freq_mean"] = gaps.apply(lambda g: g.mean() if len(g) else np.nan)
    df["posting_freq_std"] = gaps.apply(lambda g: g.std() if len(g) > 1 else 0.0)
    df["posting_freq_median"] = gaps.apply(lambda g: np.median(g) if len(g) else np.nan)
    df["longest_gap_days"] = gaps.apply(lambda g: g.max() if len(g) else 0.0)
    df["current_gap_days"] = (snapshot_date - pd.to_datetime(df["last_seen_date"])) / np.timedelta64(1, "D")

    def _episodes_within(dates, first_seen, n_days):
        dates = pd.to_datetime(dates)
        cutoff = first_seen + pd.Timedelta(days=n_days)
        return int((dates <= cutoff).sum())

    df["first_14d_episode_count"] = df.apply(
        lambda r: _episodes_within(r["episode_dates"], pd.Timestamp(r["first_seen_date"]), 14), axis=1
    )
    df["first_7d_episode_count"] = df.apply(
        lambda r: _episodes_within(r["episode_dates"], pd.Timestamp(r["first_seen_date"]), 7), axis=1
    )
    return df


def compute_consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    def _weekly_consistency(dates, first_seen, n_days):
        dates = pd.to_datetime(dates)
        first_seen = pd.Timestamp(first_seen)
        window_dates = dates[dates <= first_seen + pd.Timedelta(days=n_days)]
        n_weeks = max(1, int(np.ceil(n_days / 7)))
        weeks_with_episode = {min((d - first_seen).days // 7, n_weeks - 1) for d in window_dates}
        return len(weeks_with_episode) / n_weeks

    df["consistency_score_30d"] = df.apply(
        lambda r: _weekly_consistency(r["episode_dates"], r["first_seen_date"], 30), axis=1
    )
    df["consistency_score_60d"] = df.apply(
        lambda r: _weekly_consistency(r["episode_dates"], r["first_seen_date"], 60), axis=1
    )

    def _posted_in_range(dates, first_seen, start_day, end_day):
        dates = pd.to_datetime(dates)
        first_seen = pd.Timestamp(first_seen)
        lo, hi = first_seen + pd.Timedelta(days=start_day), first_seen + pd.Timedelta(days=end_day)
        return int(((dates >= lo) & (dates <= hi)).any())

    # "day 0" is defined as this creator's first observed episode, so posting
    # in week 1 is tautologically always true. posted_week1 is redefined here
    # as posting *again* within the first week (i.e. a second episode before
    # day 7) to keep it informative rather than constant.
    df["posted_week1"] = (df["first_7d_episode_count"] > 1).astype(int)
    df["posted_week2"] = df.apply(lambda r: _posted_in_range(r["episode_dates"], r["first_seen_date"], 8, 14), axis=1)
    df["cadence_variance_ratio"] = (df["posting_freq_std"] / df["posting_freq_mean"].replace(0, np.nan)).fillna(0)
    return df


def compute_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    df["total_active_days"] = (
        pd.to_datetime(df["last_seen_date"]) - pd.to_datetime(df["first_seen_date"])
    ) / np.timedelta64(1, "D")
    df["total_active_months"] = (df["total_active_days"] / 30.44).clip(lower=1 / 30.44)
    df["eps_per_month_total"] = df["episodes_in_window"] / df["total_active_months"]

    def _half_window_rates(dates, first_seen, last_seen):
        dates = pd.to_datetime(dates)
        first_seen, last_seen = pd.Timestamp(first_seen), pd.Timestamp(last_seen)
        midpoint = first_seen + (last_seen - first_seen) / 2
        first_half = (dates <= midpoint).sum()
        second_half = (dates > midpoint).sum()
        half_span_months = max((last_seen - first_seen).days / 2, 1) / 30.44
        return first_half / half_span_months, second_half / half_span_months

    rates = df.apply(
        lambda r: _half_window_rates(r["episode_dates"], r["first_seen_date"], r["last_seen_date"]), axis=1
    )
    df["eps_per_month_first_half"] = rates.apply(lambda t: t[0])
    df["eps_per_month_second_half"] = rates.apply(lambda t: t[1])
    df["velocity_change"] = df["eps_per_month_second_half"] - df["eps_per_month_first_half"]
    return df


def compute_lifecycle_features(df: pd.DataFrame) -> pd.DataFrame:
    df["days_since_last_episode"] = df["current_gap_days"]
    for threshold in DORMANCY_THRESHOLDS_DAYS:
        df[f"dormancy_{threshold}d"] = (df["current_gap_days"] > threshold).astype(int)
    return df


def compute_monetization_features(df: pd.DataFrame) -> pd.DataFrame:
    keyword_pattern = "|".join(MONETIZATION_KEYWORDS)

    def _first_monetization_hit(row):
        texts = list(row["episode_titles"])
        dates = pd.to_datetime(row["episode_dates"])
        order = np.argsort(dates)
        for idx in order:
            text = str(texts[idx]).lower()
            if any(kw in text for kw in MONETIZATION_KEYWORDS):
                return idx, dates[idx]
        pod_desc = str(row.get("pod_description") or "").lower()
        if any(kw in pod_desc for kw in MONETIZATION_KEYWORDS):
            return -1, pd.Timestamp(row["first_seen_date"])
        return None, None

    hits = df.apply(_first_monetization_hit, axis=1)
    df["monetization_episode_index"] = hits.apply(lambda h: h[0])
    df["has_monetization_signal"] = df["monetization_episode_index"].notna().astype(int)

    monetization_dates = hits.apply(lambda h: h[1])
    df["monetization_timing_days"] = (
        (monetization_dates - pd.to_datetime(df["first_seen_date"])) / np.timedelta64(1, "D")
    )
    return df


def compute_survival_labels(df: pd.DataFrame) -> pd.DataFrame:
    for checkpoint in SURVIVAL_CHECKPOINTS_DAYS:
        def _survived(row, checkpoint=checkpoint):
            dates = pd.to_datetime(row["episode_dates"])
            first_seen = pd.Timestamp(row["first_seen_date"])
            return int((dates > first_seen + pd.Timedelta(days=checkpoint)).any())

        df[f"survived_{checkpoint}d"] = df.apply(_survived, axis=1)

    def _time_to_first_long_gap(row):
        gaps = _episode_gaps_days(row["episode_dates"])
        if len(gaps) == 0:
            return np.nan, 0
        long_gap_idx = np.where(gaps > LONG_GAP_THRESHOLD_DAYS)[0]
        dates = np.sort(pd.to_datetime(row["episode_dates"]))
        if len(long_gap_idx) > 0:
            first_idx = long_gap_idx[0]
            duration = (dates[first_idx] - dates[0]) / np.timedelta64(1, "D")
            return float(duration), 1
        duration = (dates[-1] - dates[0]) / np.timedelta64(1, "D")
        return float(duration), 0

    results = df.apply(_time_to_first_long_gap, axis=1)
    df["time_to_first_long_gap"] = results.apply(lambda t: t[0])
    df["event_occurred"] = results.apply(lambda t: t[1])
    return df


def compute_competing_risks_labels(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """event_type: 0 = censored (active at window end), 1 = dormant (quiet
    then returned before window end), 2 = quiet at window end with no
    return observed. Because the observation window ends 2020-06-30, "2"
    is a proxy for permanent departure, not confirmed churn: a creator
    could return in July and we would not see it."""

    def _event_type(row):
        dates = np.sort(pd.to_datetime(row["episode_dates"]))
        gaps = np.diff(dates) / np.timedelta64(1, "D")
        long_gaps = gaps > LONG_GAP_THRESHOLD_DAYS
        current_gap = (snapshot_date - dates[-1]) / np.timedelta64(1, "D")

        if current_gap <= LONG_GAP_THRESHOLD_DAYS:
            return 0  # still active at window end
        if long_gaps.any():
            return 1  # went quiet earlier in the window but returned at least once
        return 2  # quiet since first long gap, no return observed before window end

    df["event_type"] = df.apply(_event_type, axis=1)
    return df


def build_features(df: pd.DataFrame, pod_descriptions: pd.Series | None = None) -> pd.DataFrame:
    snapshot_date = pd.to_datetime(df["last_seen_date"]).max()
    logger.info("Snapshot date (window end): %s", snapshot_date)

    df = compute_cadence_features(df, snapshot_date)
    df = compute_consistency_features(df)
    df = compute_growth_features(df)
    df = compute_lifecycle_features(df)
    df = compute_monetization_features(df)
    df = compute_survival_labels(df)
    df = compute_competing_risks_labels(df, snapshot_date)
    return df


def main() -> None:
    creators_path = PROCESSED_DIR / "creators.parquet"
    df = pd.read_parquet(creators_path)
    logger.info("Loaded %d creators", len(df))

    df = build_features(df)

    out_path = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows, %d columns)", out_path, *df.shape)


if __name__ == "__main__":
    main()
