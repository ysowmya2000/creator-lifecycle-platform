"""Creator Lifecycle Intelligence Platform — Streamlit entry point."""

import json
from pathlib import Path

import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Creator Lifecycle Intelligence Platform",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

st.title("🎙 Creator Lifecycle Intelligence Platform")
st.markdown(
    "Three-module survival analysis on **168,619 podcast creators**. Predicts early "
    "churn risk, tests whether monetization causally drives retention, and separates "
    "creators who go quiet-but-return from those who are gone for good — something "
    "standard churn models can't do."
)

with open("outputs/module1_metrics.json") as f:
    m1 = json.load(f)
with open("outputs/module2_metrics.json") as f:
    m2 = json.load(f)
with open("outputs/module3_metrics.json") as f:
    m3 = json.load(f)

st.markdown("## Key Findings")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Of creators who go quiet, recover", f"{m3['cif']['n_recovered'] / m3['cif']['n_ever_quiet']:.0%}")
    st.caption("A standard churn model would count all of them as churned — the core finding of Module 3.")
with col2:
    st.metric("'Exit' events that are actually recoveries", f"{m3['ablation2']['fraction_of_events_that_are_recoveries']:.1%}")
    st.caption("Standard survival models overstate the departure hazard by conflating recovery with churn.")
with col3:
    st.metric("Monetized creators' recovery rate", f"{m3['cif']['recovery_rate_monetized']:.0%}",
               delta=f"+{(m3['cif']['recovery_rate_monetized'] - m3['cif']['recovery_rate_not_monetized']) * 100:.0f}pp vs. non-monetized")
    st.caption("Monetized creators who go quiet are meaningfully more likely to come back.")

st.markdown("### Module summary")
st.markdown(
    f"""
| Module | Method | Key result |
|---|---|---|
| 1. Cold Start | Cox PH + XGBoost | Cox C-index {m1['cox_ph']['test_c_index']:.2f} (fails proportional-hazards — reported honestly); XGBoost AUC {m1['xgb']['test_auc']:.2f} |
| 2. Monetization | DiD + Regression Discontinuity | No significant causal effect on posting behavior detected (parallel trends holds, p={m2['did']['parallel_trends_p_value']:.2f}) — a real null result, not a failure |
| 3. Competing Risks | Aalen-Johansen CIF + time-varying Cox | Competing-risks model beats standard Cox ({m3['ablation2']['competing_risks_cindex']:.2f} vs {m3['ablation2']['standard_cox_cindex']:.2f} C-index) |
"""
)

st.info(
    "**Data note:** the free dataset available (SPoRC) only captures a ~61-day observation "
    "window rather than each creator's full since-launch history. Every metric here is defined "
    "relative to that window, not a creator's true lifetime — see `data/README.md` for the full "
    "reframing rationale.",
    icon="ℹ️",
)

st.markdown("Use the sidebar to explore each module in detail.")
