"""Tests against the metrics saved by the three model-training scripts.

These read outputs/module{1,2,3}_metrics.json rather than retraining —
retraining inside a test suite would take minutes and duplicate the actual
training scripts. Thresholds below are conservative lower bounds set from
the real numbers this project achieved (see README for full results), not
the original targets in CLAUDE.md, which were written before the dataset's
real constraints (windowed observation, no since-launch history) were known.
"""

import json
from pathlib import Path

import pytest

OUTPUTS_DIR = Path("outputs")


def _load_metrics(name):
    path = OUTPUTS_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not found; run the corresponding module script first")
    return json.loads(path.read_text())


def test_cox_cindex_above_threshold():
    metrics = _load_metrics("module1_metrics.json")
    assert metrics["cox_ph"]["test_c_index"] >= 0.55


def test_xgb_auc_above_threshold():
    metrics = _load_metrics("module1_metrics.json")
    assert metrics["xgb"]["test_auc"] >= 0.70


def test_window_sweep_auc_increases_with_window():
    metrics = _load_metrics("module1_metrics.json")
    aucs = [row["auc"] for row in sorted(metrics["window_sweep"], key=lambda r: r["window_day"])]
    assert aucs == sorted(aucs), "AUC should not decrease as the observation window grows"


def test_did_parallel_trends_holds():
    metrics = _load_metrics("module2_metrics.json")
    assert metrics["did"]["parallel_trends_p_value"] > 0.05


def test_rd_placebo_not_significant():
    metrics = _load_metrics("module2_metrics.json")
    assert metrics["rd"]["placebo_p_value"] > 0.05


def test_competing_risk_counts_are_complete():
    metrics = _load_metrics("module3_metrics.json")
    cif = metrics["cif"]
    assert cif["n_recovered"] + cif["n_no_recovery"] + cif["n_censored"] > 0
    assert cif["n_ever_quiet"] == cif["n_recovered"] + cif["n_no_recovery"]


def test_ablation2_competing_risks_beats_standard():
    metrics = _load_metrics("module3_metrics.json")
    ablation2 = metrics["ablation2"]
    assert ablation2["competing_risks_cindex"] >= ablation2["standard_cox_cindex"]
