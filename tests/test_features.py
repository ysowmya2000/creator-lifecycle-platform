"""Unit tests for the feature engineering pipeline."""

from pathlib import Path

import pandas as pd
import pytest

FEATURES_PATH = Path("data/processed/features.parquet")


@pytest.fixture(scope="module")
def features_df():
    if not FEATURES_PATH.exists():
        pytest.skip("features.parquet not built yet; run src/data/features.py first")
    return pd.read_parquet(FEATURES_PATH)


def test_consistency_score_bounds(features_df):
    for col in ["consistency_score_30d", "consistency_score_60d"]:
        assert features_df[col].between(0, 1).all(), f"{col} has values outside [0, 1]"


def test_event_type_valid(features_df):
    assert set(features_df["event_type"].unique()).issubset({0, 1, 2})


def test_no_negative_gaps(features_df):
    assert (features_df["posting_freq_mean"].dropna() >= 0).all()
    assert (features_df["longest_gap_days"].dropna() >= 0).all()


def test_survival_label_consistency(features_df):
    # if survived_30d=1, the creator must have an episode after day 30
    # relative to their first observed episode
    survived = features_df[features_df["survived_30d"] == 1]

    def _has_episode_after_30d(row):
        dates = pd.to_datetime(row["episode_dates"])
        first_seen = pd.Timestamp(row["first_seen_date"])
        return (dates > first_seen + pd.Timedelta(days=30)).any()

    assert survived.sample(min(500, len(survived)), random_state=42).apply(_has_episode_after_30d, axis=1).all()


def test_dormancy_flags_are_binary(features_df):
    for col in ["dormancy_14d", "dormancy_21d", "dormancy_30d"]:
        assert set(features_df[col].unique()).issubset({0, 1})


def test_no_duplicate_creators(features_df):
    assert features_df["podcast_id"].is_unique
