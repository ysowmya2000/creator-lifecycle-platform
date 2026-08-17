"""Reusable Plotly chart builders shared across dashboard pages."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lifelines import AalenJohansenFitter, KaplanMeierFitter


def km_curve_figure(df: pd.DataFrame, duration_col: str, event_col: str, group_col: str | None = None,
                     title: str = "Survival Curve") -> go.Figure:
    fig = go.Figure()
    groups = df[group_col].unique() if group_col else [None]

    for g in groups:
        subset = df[df[group_col] == g] if group_col else df
        if len(subset) < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(subset[duration_col], subset[event_col], label=str(g) if g is not None else "overall")
        sf = kmf.survival_function_
        fig.add_trace(go.Scatter(x=sf.index, y=sf.iloc[:, 0], mode="lines", name=str(g) if g is not None else "overall"))

    fig.update_layout(title=title, xaxis_title="Days since first observed episode",
                       yaxis_title="Survival probability", template="plotly_white", height=450)
    return fig


def gauge_figure(value: float, title: str = "Predicted probability") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value * 100,
        title={"text": title},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4C72B0"},
               "steps": [{"range": [0, 40], "color": "#f8d7da"},
                         {"range": [40, 70], "color": "#fff3cd"},
                         {"range": [70, 100], "color": "#d4edda"}]},
        number={"suffix": "%"},
    ))
    fig.update_layout(height=300)
    return fig


def cif_figure(df: pd.DataFrame, duration_col: str, event_col: str, title: str = "Cumulative Incidence") -> go.Figure:
    fig = go.Figure()
    for event, label in [(1, "Dormant (recovered)"), (2, "No recovery observed")]:
        ajf = AalenJohansenFitter()
        ajf.fit(df[duration_col], df[event_col], event_of_interest=event)
        cif = ajf.cumulative_density_
        fig.add_trace(go.Scatter(x=cif.index, y=cif.iloc[:, 0], mode="lines", name=label, fill="tozeroy"))
    fig.update_layout(title=title, xaxis_title="Days since first observed episode",
                       yaxis_title="Cumulative incidence", template="plotly_white", height=450)
    return fig


def action_matrix_heatmap(df: pd.DataFrame, dormancy_col: str = "dormancy_prob",
                           departure_col: str = "departure_prob") -> go.Figure:
    heat = pd.crosstab(pd.cut(df[dormancy_col], 5), pd.cut(df[departure_col], 5), normalize=True)
    fig = go.Figure(go.Heatmap(
        z=heat.values,
        x=[f"{i.left:.1f}-{i.right:.1f}" for i in heat.columns],
        y=[f"{i.left:.1f}-{i.right:.1f}" for i in heat.index],
        colorscale="YlOrRd",
    ))
    fig.update_layout(title="Platform Action Matrix (creator density)",
                       xaxis_title="Departure probability", yaxis_title="Dormancy probability", height=450)
    return fig
