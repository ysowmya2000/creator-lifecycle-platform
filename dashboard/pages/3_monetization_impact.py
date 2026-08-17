"""Page 3 — Monetization Impact Analyzer (Module 2)."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.components.sidebar import render_sidebar

st.set_page_config(page_title="Monetization Impact", page_icon="💰", layout="wide")
render_sidebar()
st.title("💰 Monetization Impact Analyzer")

with open("outputs/module2_metrics.json") as f:
    m2 = json.load(f)

st.warning(
    f"**Key finding: no statistically significant causal effect detected.** "
    f"DiD ATT p-value = {m2['did']['att_p_value']:.2f} (parallel trends holds, "
    f"p={m2['did']['parallel_trends_p_value']:.2f}); RD jump p-value = {m2['rd']['jump_p_value']:.2f} "
    f"(placebo test also not significant, p={m2['rd']['placebo_p_value']:.2f}). "
    "This is a real, methodologically clean null result, not a failed analysis — see README for the full write-up.",
    icon="🔍",
)

st.markdown("### Difference-in-Differences: Posting Rate Around Monetization Event")
did_df = pd.read_csv("outputs/module2_did_results.csv")
fig = go.Figure()
for treated_val, label, color in [(0, "Control", "#4C72B0"), (1, "Treated (monetized)", "#DD8452")]:
    sub = did_df[did_df["treated"] == treated_val].sort_values("post")
    fig.add_trace(go.Scatter(x=["pre-event", "post-event"], y=sub["eps_per_week"], mode="lines+markers",
                              name=label, line=dict(color=color)))
fig.update_layout(yaxis_title="Episodes per week", template="plotly_white", height=400)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"n={m2['did']['n_treated']} treated (genuine in-window monetization event), matched to {m2['did']['n_control']} controls.")

st.markdown("### Regression Discontinuity: Episode Count Threshold")


@st.cache_data
def get_rd_data():
    # Precomputed by src/models/module2_causal.py — this involves a
    # row-wise scan of every creator's episode list, which takes 15-20s
    # and is far too slow to redo on every dashboard cold start.
    return pd.read_parquet("data/processed/rd_dataset.parquet")


rd_df = get_rd_data()
threshold = m2["rd"]["threshold_episodes"]
bw = m2["rd"]["bandwidth"]
window = rd_df[
    (rd_df["episodes_before_checkpoint"] >= threshold - bw) & (rd_df["episodes_before_checkpoint"] <= threshold + bw)
]
binned = window.groupby("episodes_before_checkpoint")["active_after_checkpoint"].mean().reset_index()
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=binned["episodes_before_checkpoint"], y=binned["active_after_checkpoint"],
                           mode="markers", marker=dict(size=10), name="binned mean"))
fig2.add_vline(x=threshold, line_dash="dash", line_color="gray", annotation_text=f"threshold ({threshold} episodes)")
fig2.update_layout(xaxis_title="Episodes posted before day-30 checkpoint",
                    yaxis_title="P(active after checkpoint)", template="plotly_white", height=400)
st.plotly_chart(fig2, use_container_width=True)
st.caption(f"Jump estimate: {m2['rd']['jump_estimate']:.4f} (p={m2['rd']['jump_p_value']:.2f}, not significant)")

st.markdown("### LTV Calculator")
ltv_by_tier = pd.read_csv("outputs/module2_ltv_by_tier.csv")
col1, col2 = st.columns(2)
with col1:
    selected_tier = st.selectbox("Creator tier (by episode count quartile)", ltv_by_tier["episode_tier"])
tier_row = ltv_by_tier[ltv_by_tier["episode_tier"] == selected_tier].iloc[0]
with col2:
    st.metric(f"Expected LTV — {selected_tier}", f"${tier_row['mean']:.2f}", help=f"Median: ${tier_row['median']:.2f}, n={int(tier_row['count'])}")

st.bar_chart(ltv_by_tier.set_index("episode_tier")["mean"])
st.caption(
    f"LTV = survival probability x episodes/month x assumed revenue/episode "
    f"(${m2['ltv']['assumed_cpm']:.0f} CPM x {m2['ltv']['assumed_downloads_per_episode']} assumed downloads/episode). "
    "Assumptions are illustrative, not fitted to real ad-revenue data."
)
