"""End-to-end smoke tests: the pipeline runs without errors, and the
expected output files exist after a full run."""

from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.features import build_features

CREATORS_PATH = Path("data/processed/creators.parquet")


def test_pipeline_runs_on_sample():
    if not CREATORS_PATH.exists():
        pytest.skip("creators.parquet not built yet; run src/data/loader.py first")
    df = pd.read_parquet(CREATORS_PATH).sample(min(1000, len(pd.read_parquet(CREATORS_PATH))), random_state=42)
    result = build_features(df)
    assert len(result) == len(df)
    assert result["event_type"].isin([0, 1, 2]).all()


@pytest.mark.parametrize("path", [
    "data/processed/creators.parquet",
    "data/processed/features.parquet",
    "outputs/data_validation_report.html",
    "outputs/models/module1_cox.pkl",
    "outputs/models/module1_xgb.pkl",
    "outputs/models/module3_dormancy_duration.pkl",
    "outputs/module1_metrics.json",
    "outputs/module2_metrics.json",
    "outputs/module3_metrics.json",
    "outputs/ablation1_window_sweep.csv",
    "outputs/ablation2_competing_vs_standard.csv",
    "outputs/ablation3_time_varying.csv",
    "outputs/ablation4_monetization_proxy.csv",
])
def test_output_files_exist(path):
    assert Path(path).exists(), f"expected output file missing: {path}"
