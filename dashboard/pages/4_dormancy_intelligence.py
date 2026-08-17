"""Page 4 — Dormancy Intelligence Center (Module 3)."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.components.sidebar import render_sidebar
from dashboard.components.charts import cif_figure, action_matrix_heatmap

st.set_page_config(page_title="Dormancy Intelligence", page_icon="🌙", layout="wide")
render_sidebar()
st.title("🌙 Dormancy Intelligence Center")

with open("outputs/module3_metrics.json") as f:
    m3 = json.load(f)


@st.cache_data
def load_labeled_data():
    # Precomputed by src/models/module3_competing.py — labeling is a
    # row-wise scan of every creator's episode list (~15s), too slow to
    # redo on every dashboard cold start.
    df = pd.read_parquet("data/processed/features.parquet")
    labels = pd.read_parquet("data/processed/competing_risk_labels.parquet")
    return df.merge(labels, on="podcast_id")


df = load_labeled_data()

st.markdown(
    f"Of **{m3['cif']['n_ever_quiet']:,}** creators who went silent for more than 14 days, "
    f"**{m3['cif']['n_recovered'] / m3['cif']['n_ever_quiet']:.0%} eventually returned**. "
    "A standard churn model can't tell the difference between them and creators who are truly gone."
)

col1, col2 = st.columns(2)
with col1:
    monetization_filter = st.selectbox("Monetization status", ["All", "Monetized only", "Not monetized"])
with col2:
    top_categories = df["category"].value_counts().head(8).index.tolist()
    category_filter = st.multiselect("Category", top_categories, default=top_categories[:5])

filtered = df.copy()
if monetization_filter == "Monetized only":
    filtered = filtered[filtered["has_monetization_signal"] == 1]
elif monetization_filter == "Not monetized":
    filtered = filtered[filtered["has_monetization_signal"] == 0]
if category_filter:
    filtered = filtered[filtered["category"].isin(category_filter)]

st.plotly_chart(cif_figure(filtered, "time_to_dormancy_onset", "competing_event"), use_container_width=True)

st.markdown("### Creator Profile Simulator")
col1, col2, col3 = st.columns(3)
with col1:
    sim_dormancy = st.slider("Dormancy probability", 0.0, 1.0, 0.5)
with col2:
    sim_departure = st.slider("Departure probability", 0.0, 1.0, 0.3)


def recommend_action(dormancy_prob: float, departure_prob: float) -> str:
    if departure_prob > 0.7:
        return "write off"
    if dormancy_prob > 0.6 and departure_prob < 0.3:
        return "aggressive re-engagement"
    if dormancy_prob > 0.4 and 0.3 <= departure_prob <= 0.6:
        return "gentle nudge"
    return "monitor"


with col3:
    action = recommend_action(sim_dormancy, sim_departure)
    color = {"write off": "🔴", "aggressive re-engagement": "🟢", "gentle nudge": "🟡", "monitor": "⚪"}[action]
    st.metric("Recommended action", f"{color} {action}")

st.markdown("### Platform Action Matrix")
action_dist = pd.Series(m3["action_matrix"]["action_distribution"]).sort_values(ascending=False)
col1, col2 = st.columns([2, 1])
with col1:
    st.bar_chart(action_dist)
with col2:
    st.dataframe((action_dist * 100).round(1).astype(str) + "%", use_container_width=True)

st.caption(
    f"Dormancy classifier AUC={m3['action_matrix']['dormancy_prob_auc']:.2f}, "
    f"departure classifier AUC={m3['action_matrix']['departure_prob_auc']:.2f}. "
    "Both trained on day-14 behavioral features only, matching Module 1's early-warning design."
)
