"""Archived: bar-chart comparison of U.S. states vs. countries by 2023 GDP.

Superseded by the world-map visualization (`world_top30_economies`) in
Dashboard/build_story_dashboard.py. Kept here for reference.
"""

import pandas as pd
import plotly.graph_objects as go


def state_vs_country_gdp(latest: pd.DataFrame, country_gdp_2023_billions, colors, plot_layout) -> go.Figure:
    states = (
        latest[["state", "gdp"]]
        .dropna()
        .assign(label=lambda d: d["state"], kind="State", gdp_b=lambda d: d["gdp"] / 1e9)
        [["label", "gdp_b", "kind"]]
    )
    countries = pd.DataFrame(
        [{"label": name, "gdp_b": float(val), "kind": "Country"} for name, val in country_gdp_2023_billions]
    )
    combined = pd.concat([states, countries], ignore_index=True)
    combined = combined.sort_values("gdp_b", ascending=False).reset_index(drop=True)
    order = combined["label"].tolist()

    state_rows = combined[combined["kind"] == "State"]
    country_rows = combined[combined["kind"] == "Country"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=country_rows["label"], y=country_rows["gdp_b"], name="Country",
        marker=dict(color=colors["gold"], line=dict(width=0)), opacity=0.35,
        hovertemplate="<b>%{x}</b><br>2023 GDP: $%{y:,.0f} B<extra>Country</extra>",
    ))
    fig.add_trace(go.Bar(
        x=state_rows["label"], y=state_rows["gdp_b"], name="U.S. state",
        marker=dict(color=colors["teal"], line=dict(width=0)), opacity=0.95,
        hovertemplate="<b>%{x}</b><br>2023 GDP: $%{y:,.0f} B<extra>U.S. state</extra>",
    ))
    fig.update_xaxes(categoryorder="array", categoryarray=order, tickfont=dict(size=9), tickangle=-60)
    fig.update_yaxes(title="2023 GDP (USD, billions, nominal)", type="log", tickformat="$~s")
    fig.update_layout(
        barmode="overlay", bargap=0.15,
        legend=dict(orientation="h", y=1.04, x=0, yanchor="bottom"),
        margin=dict(l=64, r=24, t=24, b=120),
    )
    return plot_layout(fig, height=520)
