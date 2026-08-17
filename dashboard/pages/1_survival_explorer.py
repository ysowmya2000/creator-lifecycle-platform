"""Page 1 — Creator Survival Explorer."""

from pathlib import Path

import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.components.sidebar import render_sidebar
from dashboard.components.charts import km_curve_figure

st.set_page_config(page_title="Survival Explorer", page_icon="📈", layout="wide")
render_sidebar()
st.title("📈 Creator Survival Explorer")


@st.cache_data
def load_features():
    df = pd.read_parquet("data/processed/features.parquet")
    df["first_seen_week"] = ((pd.to_datetime(df["first_seen_date"]) - pd.to_datetime(df["first_seen_date"]).min()).dt.days // 7)
    return df


df = load_features()
top_categories = df["category"].value_counts().head(8).index.tolist()

st.markdown(
    "Kaplan-Meier survival curves for **time to first long gap** (a gap greater than "
    f"{14} days between episodes) — the moment a creator's posting rhythm breaks, not "
    "necessarily permanent churn (see Module 4 for dormancy vs. departure)."
)

col1, col2 = st.columns([2, 1])
with col1:
    selected_categories = st.multiselect("Filter by category", top_categories, default=top_categories[:5])
with col2:
    stratify_by = st.radio("Stratify by", ["category", "first_seen_week", "none"], index=0)

filtered = df[df["category"].isin(selected_categories)] if selected_categories else df

if stratify_by == "none":
    fig = km_curve_figure(filtered, "time_to_first_long_gap", "event_occurred", None, "Overall Survival Curve")
elif stratify_by == "category":
    fig = km_curve_figure(filtered, "time_to_first_long_gap", "event_occurred", "category", "Survival by Category")
else:
    week_bins = sorted(filtered["first_seen_week"].unique())[:8]
    fig = km_curve_figure(filtered[filtered["first_seen_week"].isin(week_bins)],
                           "time_to_first_long_gap", "event_occurred", "first_seen_week", "Survival by First-Seen Week")

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Survival rates at day 14 / day 30, by category")
table = filtered.groupby("category", observed=True).agg(
    n_creators=("podcast_id", "count"),
    survived_14d=("survived_14d", "mean"),
    survived_30d=("survived_30d", "mean"),
).sort_values("n_creators", ascending=False).head(10)
table["survived_14d"] = (table["survived_14d"] * 100).round(1).astype(str) + "%"
table["survived_30d"] = (table["survived_30d"] * 100).round(1).astype(str) + "%"
st.dataframe(table, use_container_width=True)
