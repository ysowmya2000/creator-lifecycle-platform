"""Page 2 — Early Warning Simulator (Module 1)."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.components.sidebar import render_sidebar
from dashboard.components.charts import gauge_figure

st.set_page_config(page_title="Early Warning Simulator", page_icon="⚠️", layout="wide")
render_sidebar()
st.title("⚠️ Early Warning Simulator")
st.markdown(
    "Simulate a creator's first 14 days of posting behavior and get a predicted "
    "probability they're still active at day 30, plus a SHAP explanation of the prediction."
)


@st.cache_resource
def load_model():
    with open("outputs/models/module1_xgb.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_features():
    return pd.read_parquet("data/processed/features.parquet")


model = load_model()
df = load_features()
top_categories = df["category"].value_counts().head(8).index.tolist()

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### Creator profile")
    episode_count = st.number_input("Episodes posted in first 14 days", 1, 20, 3)
    posting_freq_mean = st.slider("Average days between episodes", 0.5, 14.0, 4.0)
    posting_freq_std = st.slider("Std. dev. of days between episodes", 0.0, 10.0, 1.5)
    multi_post = st.toggle("Posted more than once in first 7 days?", value=True)
    consistency = st.slider("Weeks with >=1 episode (of first 2 weeks)", 0.0, 1.0, 0.75)
    accel = st.slider("Acceleration (2nd week vs 1st week episode count diff)", -5, 5, 0)
    category = st.selectbox("Category", top_categories)

input_row = pd.DataFrame([{
    "early_episode_count_14d": episode_count,
    "early_posting_freq_mean_14d": posting_freq_mean,
    "early_posting_freq_std_14d": posting_freq_std,
    "early_multi_post_14d": int(multi_post),
    "early_consistency_14d": consistency,
    "early_accel_14d": accel,
    "category_grouped": category,
}])
input_row["category_grouped"] = input_row["category_grouped"].astype(
    pd.CategoricalDtype(categories=top_categories + ["other"])
)

pred_proba = model.predict_proba(input_row)[0, 1]

with col2:
    st.plotly_chart(gauge_figure(pred_proba, "Predicted probability of still being active at day 30"),
                     use_container_width=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_row)

    st.markdown("#### Why this prediction? (SHAP feature contributions)")
    contrib_df = pd.DataFrame({
        "feature": input_row.columns,
        "shap_value": shap_values[0] if shap_values.ndim == 2 else shap_values,
    }).sort_values("shap_value")
    st.bar_chart(contrib_df.set_index("feature")["shap_value"])

    st.markdown("#### How does this compare to the median creator in this category?")
    median_creator = df[df["category"] == category][
        ["early_episode_count_14d", "early_posting_freq_mean_14d", "early_consistency_14d"]
    ].median() if "early_episode_count_14d" in df.columns else None
    comparison = pd.DataFrame({
        "this creator": [episode_count, posting_freq_mean, consistency],
        "category median": [
            df[df["category"] == category]["first_14d_episode_count"].median(),
            df[df["category"] == category]["posting_freq_mean"].median(),
            df[df["category"] == category]["consistency_score_30d"].median(),
        ],
    }, index=["episodes (14d)", "avg gap (days)", "consistency"])
    st.dataframe(comparison, use_container_width=True)
