"""
Builds a more narrative dashboard for the GDP / human-capital story.

Inputs:
  - Clean Data/merged_data.csv
  - Raw Data/acs_state_2023_profile.json
  - Raw Data/acs_county_2023_profile.json
  - Raw Data/CAGDP1.zip
  - Raw Data/CAINC1.zip

Source downloads used:
  curl -L -o "Raw Data/acs_state_2023_profile.json" "https://api.census.gov/data/2023/acs/acs5/profile?get=NAME,DP02_0068PE,DP03_0062E,DP03_0128PE,DP05_0001E&for=state:*"
  curl -L -o "Raw Data/acs_county_2023_profile.json" "https://api.census.gov/data/2023/acs/acs5/profile?get=NAME,DP02_0068PE,DP03_0062E,DP03_0128PE,DP05_0001E&for=county:*"
  curl -L -o "Raw Data/acs_county_2023_extended_profile.json" "https://api.census.gov/data/2023/acs/acs5/profile?get=NAME,DP02_0066PE,DP02_0067PE,DP02_0068PE,DP02_0084PE,DP02_0094PE,DP02_0154PE,DP03_0009PE,DP03_0024PE,DP03_0025E,DP03_0027PE,DP03_0041PE,DP03_0062E,DP03_0128PE,DP04_0089E,DP05_0018E,DP05_0001E&for=county:*"
  curl -L -o "Raw Data/CAGDP1.zip" "https://apps.bea.gov/regional/zip/CAGDP1.zip"
  curl -L -o "Raw Data/CAGDP2.zip" "https://apps.bea.gov/regional/zip/CAGDP2.zip"
  curl -L -o "Raw Data/CAINC1.zip" "https://apps.bea.gov/regional/zip/CAINC1.zip"
  curl -L -o "Raw Data/geojson-counties-fips.json" "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
  curl -L -o "Raw Data/2023-rural-urban-continuum-codes.csv" "https://www.ers.usda.gov/media/5768/2023-rural-urban-continuum-codes.csv?v=97310"

Outputs:
  - Clean Data/state_compounding_metrics.csv
  - Clean Data/county_gdp_concentration.csv
  - Clean Data/county_income_breakouts_1969_2024.csv
  - Clean Data/county_gdp_breakouts_2001_2024.csv
  - Clean Data/state_story_2023.csv
  - Clean Data/county_story_2023.csv
  - Dashboard/story_dashboard.html

Run from the project root:
  python Dashboard/build_story_dashboard.py
"""

from __future__ import annotations

import base64
import json
import textwrap
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "Clean Data"
RAW = ROOT / "Raw Data"
OUT = ROOT / "Dashboard"

STATE_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}
ABBR_STATE = {abbr: state for state, abbr in STATE_ABBR.items()}
STATE_FIPS = {
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    "10": "Delaware",
    "11": "District of Columbia",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
}

COLORS = {
    "ink": "#171717",
    "muted": "#686868",
    "grid": "#dedede",
    "paper": "#fbfbfb",
    "red": "#b43b45",
    "teal": "#227c80",
    "gold": "#c19a30",
    "green": "#44724a",
    "blue": "#3d6f9f",
}
PLOTLY_JS_VERSION = get_plotlyjs_version()
BODY_FONT = '"Avenir Next", Avenir, "Helvetica Neue", Helvetica, Arial, sans-serif'
DISPLAY_FONT = '"New York", Georgia, "Times New Roman", serif'


def read_census_profile(path: Path, geography: str) -> pd.DataFrame:
    with path.open() as f:
        rows = json.load(f)
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.rename(
        columns={
            "NAME": "name",
            "DP02_0068PE": "bachelors_or_higher_pct",
            "DP03_0062E": "median_household_income",
            "DP03_0128PE": "poverty_pct",
            "DP05_0001E": "acs_population",
        }
    )
    for col in [
        "bachelors_or_higher_pct",
        "median_household_income",
        "poverty_pct",
        "acs_population",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan

    if geography == "county":
        df["county_fips"] = df["state"] + df["county"]
        df["state_name"] = df["state"].map(STATE_FIPS)
        df = df[df["state_name"].notna()].copy()
    elif geography == "state":
        df["state_name"] = df["name"]
    else:
        raise ValueError(f"Unknown geography: {geography}")

    return df


def read_county_gdp(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        with zf.open("CAGDP1__ALL_AREAS_2001_2024.csv") as f:
            bea = pd.read_csv(f, dtype={"GeoFIPS": str}, encoding="latin1")

    bea["GeoFIPS"] = bea["GeoFIPS"].str.replace('"', "", regex=False).str.strip()
    bea = bea[(bea["LineCode"] == 3) & (bea["GeoFIPS"].str.len() == 5)].copy()
    bea = bea[~bea["GeoFIPS"].str.endswith("000")].copy()
    bea["county_gdp_current_dollars"] = (
        pd.to_numeric(bea["2023"].replace("(NA)", np.nan), errors="coerce") * 1_000
    )
    return bea[["GeoFIPS", "GeoName", "county_gdp_current_dollars"]].rename(
        columns={"GeoFIPS": "county_fips", "GeoName": "bea_county_name"}
    )


def read_county_gdp_panel(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        with zf.open("CAGDP1__ALL_AREAS_2001_2024.csv") as f:
            bea = pd.read_csv(f, dtype={"GeoFIPS": str}, encoding="latin1")

    bea["GeoFIPS"] = bea["GeoFIPS"].str.replace('"', "", regex=False).str.strip()
    bea = bea[(bea["LineCode"] == 3) & (bea["GeoFIPS"].str.len() == 5)].copy()
    bea = bea[~bea["GeoFIPS"].str.endswith("000")].copy()
    years = [str(year) for year in range(2001, 2025)]
    panel = bea[["GeoFIPS", "GeoName", *years]].melt(
        id_vars=["GeoFIPS", "GeoName"],
        var_name="year",
        value_name="county_gdp_current_thousands",
    )
    panel["year"] = panel["year"].astype(int)
    panel["county_gdp_current_dollars"] = (
        pd.to_numeric(
            panel["county_gdp_current_thousands"].replace("(NA)", np.nan),
            errors="coerce",
        )
        * 1_000
    )
    panel = panel.dropna(subset=["county_gdp_current_dollars"]).copy()
    panel = panel.rename(columns={"GeoFIPS": "county_fips", "GeoName": "bea_county_name"})
    return panel[["county_fips", "bea_county_name", "year", "county_gdp_current_dollars"]]


def read_county_income_panel(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        with zf.open("CAINC1__ALL_AREAS_1969_2024.csv") as f:
            inc = pd.read_csv(f, dtype={"GeoFIPS": str}, encoding="latin1", low_memory=False)

    inc["GeoFIPS"] = inc["GeoFIPS"].str.replace('"', "", regex=False).str.strip()
    years = [str(year) for year in range(1969, 2025)]

    county = inc[(inc["GeoFIPS"].str.len() == 5) & (~inc["GeoFIPS"].str.endswith("000"))].copy()
    county = county[county["LineCode"].isin([2, 3])].copy()
    panel = county[["GeoFIPS", "GeoName", "LineCode", "Description", *years]].melt(
        id_vars=["GeoFIPS", "GeoName", "LineCode", "Description"],
        var_name="year",
        value_name="value",
    )
    panel["year"] = panel["year"].astype(int)
    panel["value"] = pd.to_numeric(panel["value"].replace("(NA)", np.nan), errors="coerce")
    panel = panel.pivot_table(
        index=["GeoFIPS", "GeoName", "year"],
        columns="LineCode",
        values="value",
        aggfunc="first",
    ).reset_index()
    panel = panel.rename(
        columns={
            "GeoFIPS": "county_fips",
            "GeoName": "bea_county_name",
            2: "population",
            3: "per_capita_personal_income",
        }
    )
    panel["state_fips"] = panel["county_fips"].str[:2]
    panel["state_name"] = panel["state_fips"].map(STATE_FIPS)
    panel = panel[panel["state_name"].notna()].copy()

    us = inc[(inc["GeoFIPS"] == "00000") & (inc["LineCode"] == 3)][years].iloc[0]
    us = pd.to_numeric(us.replace("(NA)", np.nan), errors="coerce")
    us = pd.DataFrame({"year": us.index.astype(int), "us_per_capita_personal_income": us.values})
    panel = panel.merge(us, on="year", how="left")
    panel["income_index_us_100"] = (
        panel["per_capita_personal_income"] / panel["us_per_capita_personal_income"] * 100
    )
    return panel


def state_compounding_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged[merged["state"] != "District of Columbia"].copy()
    rows = []
    for year, group in df.groupby("year"):
        sorted_group = group.sort_values("gdp_per_capita")
        bottom10 = sorted_group.head(10)["gdp_per_capita"].mean()
        top10 = sorted_group.tail(10)["gdp_per_capita"].mean()
        p10 = group["gdp_per_capita"].quantile(0.10)
        p90 = group["gdp_per_capita"].quantile(0.90)
        rows.append(
            {
                "year": int(year),
                "top10_avg_gdp_per_capita": top10,
                "bottom10_avg_gdp_per_capita": bottom10,
                "top10_bottom10_gap": top10 - bottom10,
                "top10_bottom10_ratio": top10 / bottom10,
                "p90_gdp_per_capita": p90,
                "p10_gdp_per_capita": p10,
                "p90_p10_gap": p90 - p10,
                "p90_p10_ratio": p90 / p10,
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def county_concentration_metrics(county_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in county_panel.groupby("year"):
        values = group["county_gdp_current_dollars"].dropna().sort_values(ascending=False)
        county_count = len(values)
        top_1pct_n = max(1, int(np.ceil(county_count * 0.01)))
        total = values.sum()
        rows.append(
            {
                "year": int(year),
                "county_count": county_count,
                "top_1pct_share": values.head(top_1pct_n).sum() / total,
                "top_25_counties_share": values.head(25).sum() / total,
                "top_100_counties_share": values.head(100).sum() / total,
                "top_county_share": values.head(1).sum() / total,
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def county_income_breakouts(
    income_panel: pd.DataFrame, start_year: int = 1969, end_year: int = 2024
) -> pd.DataFrame:
    start = income_panel[income_panel["year"] == start_year][
        [
            "county_fips",
            "bea_county_name",
            "state_name",
            "population",
            "per_capita_personal_income",
            "income_index_us_100",
        ]
    ].rename(
        columns={
            "population": "population_start",
            "per_capita_personal_income": "pcpi_start",
            "income_index_us_100": "income_index_start",
        }
    )
    end = income_panel[income_panel["year"] == end_year][
        [
            "county_fips",
            "population",
            "per_capita_personal_income",
            "income_index_us_100",
        ]
    ].rename(
        columns={
            "population": "population_end",
            "per_capita_personal_income": "pcpi_end",
            "income_index_us_100": "income_index_end",
        }
    )
    out = start.merge(end, on="county_fips", how="inner")
    out = out.dropna(subset=["income_index_start", "income_index_end", "population_end"]).copy()
    out["income_index_change"] = out["income_index_end"] - out["income_index_start"]
    out["income_index_ratio"] = out["income_index_end"] / out["income_index_start"]
    out["population_change_pct"] = (
        out["population_end"] / out["population_start"].replace(0, np.nan) - 1
    ) * 100
    out["rank_start"] = out["income_index_start"].rank(ascending=False, method="min")
    out["rank_end"] = out["income_index_end"].rank(ascending=False, method="min")
    out["rank_gain"] = out["rank_start"] - out["rank_end"]
    out["period"] = f"{start_year}-{end_year}"
    return out.sort_values("income_index_change", ascending=False)


def county_gdp_breakouts(
    gdp_panel: pd.DataFrame,
    income_panel: pd.DataFrame,
    start_year: int = 2001,
    end_year: int = 2024,
) -> pd.DataFrame:
    panel = county_gdp_index_panel(gdp_panel, income_panel)

    start = panel[panel["year"] == start_year][
        [
            "county_fips",
            "bea_county_name",
            "state_name",
            "population",
            "county_gdp_per_capita",
            "gdp_index_us_100",
        ]
    ].rename(
        columns={
            "population": "population_start",
            "county_gdp_per_capita": "gdp_per_capita_start",
            "gdp_index_us_100": "gdp_index_start",
        }
    )
    end = panel[panel["year"] == end_year][
        ["county_fips", "population", "county_gdp_per_capita", "gdp_index_us_100"]
    ].rename(
        columns={
            "population": "population_end",
            "county_gdp_per_capita": "gdp_per_capita_end",
            "gdp_index_us_100": "gdp_index_end",
        }
    )
    out = start.merge(end, on="county_fips", how="inner")
    out = out.dropna(subset=["gdp_index_start", "gdp_index_end", "population_end"]).copy()
    out["gdp_index_change"] = out["gdp_index_end"] - out["gdp_index_start"]
    out["gdp_index_ratio"] = out["gdp_index_end"] / out["gdp_index_start"]
    out["rank_start"] = out["gdp_index_start"].rank(ascending=False, method="min")
    out["rank_end"] = out["gdp_index_end"].rank(ascending=False, method="min")
    out["rank_gain"] = out["rank_start"] - out["rank_end"]
    out["period"] = f"{start_year}-{end_year}"
    return out.sort_values("gdp_index_change", ascending=False)


def county_quality_of_life_index(county: pd.DataFrame) -> pd.DataFrame:
    """
    Build a county-level quality-of-life index from available county indicators.
    Score is percentile-scaled to [0, 100] so it is easy to interpret.
    """
    base = county[
        [
            "county_fips",
            "bea_county_name",
            "state_name",
            "acs_population",
            "median_household_income",
            "poverty_pct",
            "bachelors_or_higher_pct",
            "county_gdp_per_capita",
        ]
    ].copy()

    mobility_path = CLEAN / "county_mobility_ml_dataset.csv"
    if mobility_path.exists():
        mobility = pd.read_csv(
            mobility_path,
            dtype={"county_fips": str},
            usecols=[
                "county_fips",
                "broadband_pct",
                "unemployment_pct",
                "mean_commute_minutes",
                "worked_from_home_pct",
            ],
        )
        base = base.merge(mobility, on="county_fips", how="left")

    metric_specs = [
        ("median_household_income", 1.0, 1),
        ("bachelors_or_higher_pct", 1.0, 1),
        ("poverty_pct", 1.0, -1),
        ("county_gdp_per_capita", 0.8, 1),
        ("broadband_pct", 0.8, 1),
        ("unemployment_pct", 0.8, -1),
        ("mean_commute_minutes", 0.6, -1),
        ("worked_from_home_pct", 0.5, 1),
    ]

    component_cols: list[str] = []
    weight_cols: list[str] = []
    for col, weight, direction in metric_specs:
        if col not in base.columns:
            continue
        series = pd.to_numeric(base[col], errors="coerce")
        if series.notna().mean() < 0.4:
            continue

        lower = series.quantile(0.05)
        upper = series.quantile(0.95)
        clipped = series.clip(lower=lower, upper=upper)
        std = clipped.std(ddof=0)
        if std == 0 or np.isnan(std):
            continue

        z = (clipped - clipped.mean()) / std
        z = z * direction
        component_col = f"qol_component_{col}"
        weight_col = f"qol_weight_{col}"
        base[component_col] = z
        base[weight_col] = np.where(z.notna(), weight, 0.0)
        component_cols.append(component_col)
        weight_cols.append(weight_col)

    if not component_cols:
        out = base[["county_fips", "bea_county_name", "state_name", "acs_population"]].copy()
        out["qol_raw_score"] = np.nan
        out["qol_score_0_100"] = np.nan
        out["qol_tier"] = "Insufficient data"
        out["qol_components_used"] = 0
        return out

    weight_sum = base[weight_cols].sum(axis=1)
    weighted_sum = np.zeros(len(base))
    for component_col, weight_col in zip(component_cols, weight_cols):
        weighted_sum += base[component_col].fillna(0.0) * base[weight_col]
    base["qol_raw_score"] = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)
    base["qol_score_0_100"] = base["qol_raw_score"].rank(pct=True) * 100
    base["qol_components_used"] = (base[weight_cols] > 0).sum(axis=1)
    base["qol_tier"] = pd.cut(
        base["qol_score_0_100"],
        bins=[-np.inf, 20, 40, 60, 80, np.inf],
        labels=["Very low", "Low", "Mid", "High", "Very high"],
    ).astype(str)
    base.loc[base["qol_score_0_100"].isna(), "qol_tier"] = "Insufficient data"

    return base[
        [
            "county_fips",
            "bea_county_name",
            "state_name",
            "acs_population",
            "qol_raw_score",
            "qol_score_0_100",
            "qol_tier",
            "qol_components_used",
        ]
    ].copy()


def county_gdp_index_panel(gdp_panel: pd.DataFrame, income_panel: pd.DataFrame) -> pd.DataFrame:
    population = income_panel[["county_fips", "year", "population", "state_name"]].copy()
    panel = gdp_panel.merge(population, on=["county_fips", "year"], how="left")
    panel = panel.dropna(subset=["county_gdp_current_dollars", "population"]).copy()
    panel["county_gdp_per_capita"] = panel["county_gdp_current_dollars"] / panel["population"]
    national = (
        panel.groupby("year", as_index=False)
        .agg(total_gdp=("county_gdp_current_dollars", "sum"), total_pop=("population", "sum"))
        .assign(us_county_gdp_per_capita=lambda d: d["total_gdp"] / d["total_pop"])
    )
    panel = panel.merge(national[["year", "us_county_gdp_per_capita"]], on="year", how="left")
    panel["gdp_index_us_100"] = (
        panel["county_gdp_per_capita"] / panel["us_county_gdp_per_capita"] * 100
    )
    return panel


def enrich_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    merged = pd.read_csv(CLEAN / "merged_data.csv")
    merged["state_abbr"] = merged["state"].map(STATE_ABBR)
    merged["rd_per_capita"] = merged["research_spending"] * 1000 / merged["population"]
    merged["f500_per_million"] = merged["f500_count"] / (merged["population"] / 1_000_000)
    state_compounding = state_compounding_metrics(merged)

    latest_year = int(merged["year"].max())
    state_acs = read_census_profile(RAW / "acs_state_2023_profile.json", "state")
    latest = merged[merged["year"] == latest_year].merge(
        state_acs[
            [
                "state_name",
                "bachelors_or_higher_pct",
                "median_household_income",
                "poverty_pct",
                "acs_population",
            ]
        ],
        left_on="state",
        right_on="state_name",
        how="left",
    )
    latest = latest[latest["state"] != "District of Columbia"].copy()

    county = read_census_profile(RAW / "acs_county_2023_profile.json", "county")
    county = county.merge(read_county_gdp(RAW / "CAGDP1.zip"), on="county_fips", how="left")
    county = county[county["county_gdp_current_dollars"].notna()].copy()
    county["county_gdp_per_capita"] = county["county_gdp_current_dollars"] / county["acs_population"]
    county = county[county["acs_population"] > 0].copy()

    county = county.merge(
        latest[["state", "gdp_per_capita", "bachelors_or_higher_pct"]].rename(
            columns={
                "state": "state_name",
                "gdp_per_capita": "state_gdp_per_capita",
                "bachelors_or_higher_pct": "state_bachelors_or_higher_pct",
            }
        ),
        on="state_name",
        how="left",
    )
    county_gdp_panel = read_county_gdp_panel(RAW / "CAGDP1.zip")
    county_income_panel = read_county_income_panel(RAW / "CAINC1.zip")
    county_gdp_index = county_gdp_index_panel(county_gdp_panel, county_income_panel)
    county_concentration = county_concentration_metrics(county_gdp_panel)
    income_breakouts = county_income_breakouts(county_income_panel)
    gdp_breakouts = county_gdp_breakouts(county_gdp_panel, county_income_panel)
    county_qol = county_quality_of_life_index(county)

    CLEAN.mkdir(exist_ok=True)
    state_compounding.to_csv(CLEAN / "state_compounding_metrics.csv", index=False)
    county_concentration.to_csv(CLEAN / "county_gdp_concentration.csv", index=False)
    income_breakouts.to_csv(CLEAN / "county_income_breakouts_1969_2024.csv", index=False)
    gdp_breakouts.to_csv(CLEAN / "county_gdp_breakouts_2001_2024.csv", index=False)
    latest.to_csv(CLEAN / "state_story_2023.csv", index=False)
    county.to_csv(CLEAN / "county_story_2023.csv", index=False)
    county_qol.to_csv(CLEAN / "county_quality_of_life_index.csv", index=False)
    return (
        merged,
        latest,
        county,
        state_compounding,
        county_concentration,
        income_breakouts,
        gdp_breakouts,
        county_income_panel,
        county_gdp_index,
        county_qol,
    )


def plot_layout(fig: go.Figure, height: int = 560) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY_FONT, color=COLORS["ink"], size=13),
        title_font=dict(family=DISPLAY_FONT, size=24, color=COLORS["ink"]),
        margin=dict(l=64, r=34, t=54, b=64),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=COLORS["grid"], font_size=13),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=False,
        ticks="outside",
        linecolor=COLORS["ink"],
        mirror=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=False,
        ticks="outside",
        linecolor=COLORS["ink"],
        mirror=False,
    )
    return fig


def add_ols_line(fig: go.Figure, df: pd.DataFrame, x: str, y: str, name: str) -> None:
    fit = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    coeff = np.polyfit(fit[x], fit[y], 1)
    xs = np.linspace(fit[x].min(), fit[x].max(), 80)
    ys = coeff[0] * xs + coeff[1]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=COLORS["ink"], width=2),
            name=name,
            hoverinfo="skip",
        )
    )


def compounding_opening(
    state_metrics: pd.DataFrame, county_metrics: pd.DataFrame
) -> go.Figure:
    state = state_metrics.copy()
    county = county_metrics.copy()
    fig = go.Figure()

    state_gap_start = state["top10_bottom10_gap"].iloc[0]
    state_gap_end = state["top10_bottom10_gap"].iloc[-1]
    p90_gap_start = state["p90_p10_gap"].iloc[0]
    p90_gap_end = state["p90_p10_gap"].iloc[-1]
    state_annotations = [
        dict(
            x=state["year"].iloc[-1],
            y=state["top10_avg_gdp_per_capita"].iloc[-1],
            xanchor="right",
            yanchor="bottom",
            text=(
                f"Top-bottom state gap: ${state_gap_start/1000:.0f}k"
                f" to ${state_gap_end/1000:.0f}k"
            ),
            showarrow=False,
            font=dict(color=COLORS["ink"], size=13),
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor=COLORS["grid"],
            borderpad=5,
        ),
        dict(
            x=state["year"].iloc[-1],
            y=state["p90_p10_gap"].iloc[-1],
            xref="x",
            yref="y2",
            xanchor="right",
            yanchor="top",
            text=f"P90-P10: ${p90_gap_start/1000:.0f}k to ${p90_gap_end/1000:.0f}k",
            showarrow=False,
            font=dict(color=COLORS["ink"], size=13),
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor=COLORS["grid"],
            borderpad=5,
        ),
    ]

    fig.add_trace(
        go.Scatter(
            x=state["year"],
            y=state["top10_avg_gdp_per_capita"],
            mode="lines+markers",
            line=dict(color=COLORS["teal"], width=4),
            marker=dict(size=7),
            name="Top 10 states",
            hovertemplate="Top 10 average<br>%{x}: $%{y:,.0f}<extra></extra>",
            visible=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=state["year"],
            y=state["bottom10_avg_gdp_per_capita"],
            mode="lines+markers",
            line=dict(color=COLORS["red"], width=4),
            marker=dict(size=7),
            name="Bottom 10 states",
            fill="tonexty",
            fillcolor="rgba(180, 59, 69, 0.10)",
            hovertemplate="Bottom 10 average<br>%{x}: $%{y:,.0f}<extra></extra>",
            visible=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=state["year"],
            y=state["p90_p10_gap"],
            mode="lines",
            line=dict(color=COLORS["gold"], width=3, dash="dot"),
            name="P90-P10 gap",
            yaxis="y2",
            hovertemplate="P90-P10 gap<br>%{x}: $%{y:,.0f}<extra></extra>",
            visible=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=county["year"],
            y=county["top_1pct_share"] * 100,
            mode="lines+markers",
            line=dict(color=COLORS["teal"], width=4),
            marker=dict(size=7),
            name="Top 1% counties",
            hovertemplate="Top 1% county share<br>%{x}: %{y:.1f}%<extra></extra>",
            visible=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=county["year"],
            y=county["top_25_counties_share"] * 100,
            mode="lines+markers",
            line=dict(color=COLORS["gold"], width=3),
            marker=dict(size=6),
            name="Top 25 counties",
            hovertemplate="Top 25 county share<br>%{x}: %{y:.1f}%<extra></extra>",
            visible=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=county["year"],
            y=county["top_100_counties_share"] * 100,
            mode="lines+markers",
            line=dict(color=COLORS["green"], width=3),
            marker=dict(size=6),
            name="Top 100 counties",
            hovertemplate="Top 100 county share<br>%{x}: %{y:.1f}%<extra></extra>",
            visible=False,
        )
    )

    fig.update_layout(
        title="Before asking why states differ, ask whether the gap compounds",
        legend=dict(orientation="h", y=-0.22, x=0),
        yaxis2=dict(
            title="P90-P10 dollar gap",
            overlaying="y",
            side="right",
            tickprefix="$",
            tickformat=",.0f",
            showgrid=False,
            zeroline=False,
        ),
        annotations=state_annotations,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0,
                y=1.16,
                xanchor="left",
                buttons=[
                    dict(
                        label="State per-person gap",
                        method="update",
                        args=[
                            {"visible": [True, True, True, False, False, False]},
                            {
                                "yaxis": {
                                    "title": "Average GDP per person",
                                    "tickprefix": "$",
                                    "tickformat": ",.0f",
                                    "range": [30000, 95000],
                                },
                                "yaxis2": {
                                    "title": "P90-P10 dollar gap",
                                    "overlaying": "y",
                                    "side": "right",
                                    "tickprefix": "$",
                                    "tickformat": ",.0f",
                                    "showgrid": False,
                                    "zeroline": False,
                                    "visible": True,
                                },
                                "annotations": state_annotations,
                            },
                        ],
                    ),
                    dict(
                        label="County GDP concentration",
                        method="update",
                        args=[
                            {"visible": [False, False, False, True, True, True]},
                            {
                                "yaxis": {
                                    "title": "Share of all county GDP",
                                    "ticksuffix": "%",
                                    "tickformat": ".0f",
                                    "range": [20, 58],
                                },
                                "yaxis2": {"visible": False},
                                "annotations": [],
                            },
                        ],
                    ),
                ],
            )
        ],
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(
        title="Average GDP per person",
        tickprefix="$",
        tickformat=",.0f",
        range=[30000, 95000],
    )
    return plot_layout(fig, height=610)


def county_breakout_scatter(
    income_breakouts: pd.DataFrame, gdp_breakouts: pd.DataFrame
) -> go.Figure:
    fig = go.Figure()

    income = income_breakouts[income_breakouts["population_end"] >= 50_000].copy()
    gdp = gdp_breakouts[gdp_breakouts["population_end"] >= 50_000].copy()
    income_labels = set(income.head(12)["county_fips"]) | set(income.tail(12)["county_fips"])
    gdp_labels = set(gdp.head(12)["county_fips"]) | set(gdp.tail(12)["county_fips"])

    fig.add_trace(
        go.Scattergl(
            x=income["income_index_start"],
            y=income["income_index_end"],
            mode="markers+text",
            text=[
                row.bea_county_name if row.county_fips in income_labels else ""
                for row in income.itertuples()
            ],
            textposition="top center",
            marker=dict(
                size=np.clip(np.sqrt(income["population_end"]) / 55, 5, 34),
                color=income["income_index_change"],
                colorscale=[[0, COLORS["red"]], [0.5, "#eeeeee"], [1, COLORS["teal"]]],
                cmin=-80,
                cmax=80,
                colorbar=dict(title="Index change"),
                opacity=0.72,
                line=dict(width=0),
            ),
            customdata=np.stack(
                [
                    income["bea_county_name"],
                    income["state_name"],
                    income["population_end"],
                    income["income_index_change"],
                    income["rank_gain"],
                    income["pcpi_start"],
                    income["pcpi_end"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "1969 income index: %{x:.1f}<br>"
                "2024 income index: %{y:.1f}<br>"
                "Index change: %{customdata[3]:+.1f}<br>"
                "Rank gain: %{customdata[4]:+.0f}<br>"
                "2024 population: %{customdata[2]:,.0f}<br>"
                "PCPI: $%{customdata[5]:,.0f} to $%{customdata[6]:,.0f}<extra></extra>"
            ),
            name="Income mobility",
            visible=True,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=gdp["gdp_index_start"],
            y=gdp["gdp_index_end"],
            mode="markers+text",
            text=[row.bea_county_name if row.county_fips in gdp_labels else "" for row in gdp.itertuples()],
            textposition="top center",
            marker=dict(
                size=np.clip(np.sqrt(gdp["population_end"]) / 55, 5, 34),
                color=gdp["gdp_index_change"],
                colorscale=[[0, COLORS["red"]], [0.5, "#eeeeee"], [1, COLORS["teal"]]],
                cmin=-100,
                cmax=100,
                colorbar=dict(title="Index change"),
                opacity=0.72,
                line=dict(width=0),
            ),
            customdata=np.stack(
                [
                    gdp["bea_county_name"],
                    gdp["state_name"],
                    gdp["population_end"],
                    gdp["gdp_index_change"],
                    gdp["rank_gain"],
                    gdp["gdp_per_capita_start"],
                    gdp["gdp_per_capita_end"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "2001 GDP index: %{x:.1f}<br>"
                "2024 GDP index: %{y:.1f}<br>"
                "Index change: %{customdata[3]:+.1f}<br>"
                "Rank gain: %{customdata[4]:+.0f}<br>"
                "2024 population: %{customdata[2]:,.0f}<br>"
                "GDP/person: $%{customdata[5]:,.0f} to $%{customdata[6]:,.0f}<extra></extra>"
            ),
            name="GDP mobility",
            visible=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 300],
            y=[0, 300],
            mode="lines",
            line=dict(color=COLORS["ink"], width=2, dash="dot"),
            hoverinfo="skip",
            name="No change",
            visible=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 500],
            y=[0, 500],
            mode="lines",
            line=dict(color=COLORS["ink"], width=2, dash="dot"),
            hoverinfo="skip",
            name="No change",
            visible=False,
        )
    )

    fig.update_layout(
        title="Breakouts are counties that moved relative to the national average",
        legend=dict(orientation="h", y=-0.22, x=0),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0,
                y=1.16,
                xanchor="left",
                buttons=[
                    dict(
                        label="Personal income, 1969 to 2024",
                        method="update",
                        args=[
                            {"visible": [True, False, True, False]},
                            {
                                "xaxis": {
                                    "title": "1969 per-capita personal income index, U.S. = 100",
                                    "range": [0, 300],
                                },
                                "yaxis": {
                                    "title": "2024 per-capita personal income index, U.S. = 100",
                                    "range": [0, 300],
                                },
                            },
                        ],
                    ),
                    dict(
                        label="County GDP, 2001 to 2024",
                        method="update",
                        args=[
                            {"visible": [False, True, False, True]},
                            {
                                "xaxis": {
                                    "title": "2001 county GDP per person index, U.S. = 100",
                                    "range": [0, 500],
                                },
                                "yaxis": {
                                    "title": "2024 county GDP per person index, U.S. = 100",
                                    "range": [0, 500],
                                },
                            },
                        ],
                    ),
                ],
            )
        ],
    )
    fig.update_xaxes(title="1969 per-capita personal income index, U.S. = 100", range=[0, 300])
    fig.update_yaxes(title="2024 per-capita personal income index, U.S. = 100", range=[0, 300])
    return plot_layout(fig, height=640)


def county_mobility_animation(income_panel: pd.DataFrame) -> go.Figure:
    latest_pop = income_panel[income_panel["year"] == 2024][["county_fips", "population"]].rename(
        columns={"population": "population_2024"}
    )
    baseline = income_panel[income_panel["year"] == 1969][
        ["county_fips", "bea_county_name", "state_name", "income_index_us_100"]
    ].rename(columns={"income_index_us_100": "baseline_index"})
    panel = income_panel.merge(latest_pop, on="county_fips", how="left").merge(
        baseline, on=["county_fips", "bea_county_name", "state_name"], how="inner"
    )
    panel = panel[
        (panel["population_2024"] >= 75_000)
        & panel["baseline_index"].notna()
        & panel["income_index_us_100"].notna()
    ].copy()
    panel["index_change_since_1969"] = panel["income_index_us_100"] - panel["baseline_index"]

    final = panel[panel["year"] == 2024].copy()
    focus_counties = set(final.nlargest(8, "index_change_since_1969")["county_fips"])
    focus_counties |= set(final.nsmallest(8, "index_change_since_1969")["county_fips"])

    def frame_data(year: int) -> pd.DataFrame:
        df = panel[panel["year"] == year].sort_values("county_fips").copy()
        df["label"] = np.where(
            df["county_fips"].isin(focus_counties) & (year == 2024),
            df["bea_county_name"],
            "",
        )
        return df

    initial = frame_data(1969)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=initial["baseline_index"],
            y=initial["income_index_us_100"],
            mode="markers+text",
            text=initial["label"],
            textposition="top center",
            marker=dict(
                size=np.clip(np.sqrt(initial["population_2024"]) / 62, 5, 30),
                color=initial["index_change_since_1969"],
                colorscale=[[0, COLORS["red"]], [0.5, "#eeeeee"], [1, COLORS["teal"]]],
                cmin=-80,
                cmax=80,
                colorbar=dict(title="Change vs. 1969"),
                opacity=0.74,
                line=dict(width=0),
            ),
            customdata=np.stack(
                [
                    initial["bea_county_name"],
                    initial["state_name"],
                    initial["population_2024"],
                    initial["index_change_since_1969"],
                    initial["per_capita_personal_income"],
                    initial["year"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "1969 index: %{x:.1f}<br>"
                "%{customdata[5]:.0f} index: %{y:.1f}<br>"
                "Change since 1969: %{customdata[3]:+.1f}<br>"
                "Per-capita personal income: $%{customdata[4]:,.0f}<br>"
                "2024 population: %{customdata[2]:,.0f}<extra></extra>"
            ),
            name="Counties",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 320],
            y=[0, 320],
            mode="lines",
            line=dict(color=COLORS["ink"], width=2, dash="dot"),
            hoverinfo="skip",
            name="No relative change",
        )
    )

    frames = []
    for year in sorted(panel["year"].unique()):
        df = frame_data(int(year))
        frames.append(
            go.Frame(
                name=str(year),
                data=[
                    go.Scattergl(
                        x=df["baseline_index"],
                        y=df["income_index_us_100"],
                        text=df["label"],
                        marker=dict(
                            size=np.clip(np.sqrt(df["population_2024"]) / 62, 5, 30),
                            color=df["index_change_since_1969"],
                        ),
                        customdata=np.stack(
                            [
                                df["bea_county_name"],
                                df["state_name"],
                                df["population_2024"],
                                df["index_change_since_1969"],
                                df["per_capita_personal_income"],
                                df["year"],
                            ],
                            axis=-1,
                        ),
                    ),
                    go.Scatter(x=[0, 320], y=[0, 320]),
                ],
            )
        )
    fig.frames = frames

    steps = [
        dict(
            method="animate",
            label=str(year),
            args=[
                [str(year)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 220, "redraw": True},
                    "transition": {"duration": 120},
                },
            ],
        )
        for year in sorted(panel["year"].unique())
    ]

    fig.update_layout(
        title="",
        showlegend=False,
        sliders=[],
        updatemenus=[],
        annotations=[
            dict(
                x=218,
                y=296,
                text="Above the line: gained ground relative to the U.S.",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.82)",
                bordercolor=COLORS["grid"],
                borderpad=5,
                font=dict(size=13, color=COLORS["ink"]),
            ),
            dict(
                x=222,
                y=58,
                text="Below the line: fell behind its own 1969 position",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.82)",
                bordercolor=COLORS["grid"],
                borderpad=5,
                font=dict(size=13, color=COLORS["ink"]),
            ),
        ],
        margin=dict(l=64, r=34, t=36, b=68),
    )
    fig.update_xaxes(title="1969 per-capita personal income index, U.S. = 100", range=[0, 320])
    fig.update_yaxes(title="Current-year per-capita personal income index, U.S. = 100", range=[0, 320])
    fig = plot_layout(fig, height=510)
    fig.update_layout(margin=dict(l=64, r=34, t=36, b=68), sliders=[], updatemenus=[])
    return fig


def county_mobility_snapshot(income_panel: pd.DataFrame, year: int = 2024) -> go.Figure:
    latest_pop = income_panel[income_panel["year"] == 2024][["county_fips", "population"]].rename(
        columns={"population": "population_2024"}
    )
    baseline = income_panel[income_panel["year"] == 1969][
        ["county_fips", "bea_county_name", "state_name", "income_index_us_100"]
    ].rename(columns={"income_index_us_100": "baseline_index"})
    panel = income_panel.merge(latest_pop, on="county_fips", how="left").merge(
        baseline, on=["county_fips", "bea_county_name", "state_name"], how="inner"
    )
    panel = panel[
        (panel["population_2024"] >= 75_000)
        & (panel["year"] == year)
        & panel["baseline_index"].notna()
        & panel["income_index_us_100"].notna()
    ].copy()
    panel["index_change_since_1969"] = panel["income_index_us_100"] - panel["baseline_index"]

    final = income_panel.merge(latest_pop, on="county_fips", how="left").merge(
        baseline, on=["county_fips", "bea_county_name", "state_name"], how="inner"
    )
    final = final[(final["population_2024"] >= 75_000) & (final["year"] == 2024)].copy()
    final["index_change_since_1969"] = final["income_index_us_100"] - final["baseline_index"]
    focus_counties = set(final.nlargest(8, "index_change_since_1969")["county_fips"])
    focus_counties |= set(final.nsmallest(8, "index_change_since_1969")["county_fips"])
    panel["label"] = np.where(
        panel["county_fips"].isin(focus_counties) & (year == 2024),
        panel["bea_county_name"],
        "",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=panel["baseline_index"],
            y=panel["income_index_us_100"],
            mode="markers+text",
            text=panel["label"],
            textposition="top center",
            marker=dict(
                size=np.clip(np.sqrt(panel["population_2024"]) / 62, 5, 30),
                color=panel["index_change_since_1969"],
                colorscale=[[0, COLORS["red"]], [0.5, "#eeeeee"], [1, COLORS["teal"]]],
                cmin=-80,
                cmax=80,
                colorbar=dict(title="Change vs. 1969"),
                opacity=0.74,
                line=dict(width=0),
            ),
            customdata=np.stack(
                [
                    panel["bea_county_name"],
                    panel["state_name"],
                    panel["population_2024"],
                    panel["index_change_since_1969"],
                    panel["per_capita_personal_income"],
                    panel["year"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "1969 index: %{x:.1f}<br>"
                "%{customdata[5]:.0f} index: %{y:.1f}<br>"
                "Change since 1969: %{customdata[3]:+.1f}<br>"
                "Per-capita personal income: $%{customdata[4]:,.0f}<br>"
                "2024 population: %{customdata[2]:,.0f}<extra></extra>"
            ),
            name="Counties",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 320],
            y=[0, 320],
            mode="lines",
            line=dict(color=COLORS["ink"], width=2, dash="dot"),
            hoverinfo="skip",
            name="No relative change",
        )
    )
    fig.update_layout(
        title=f"County income mobility, {year}",
        legend=dict(orientation="h", y=-0.16, x=0),
        annotations=[
            dict(
                x=218,
                y=296,
                text="Above the line: gained ground relative to the U.S.",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.82)",
                bordercolor=COLORS["grid"],
                borderpad=5,
                font=dict(size=13, color=COLORS["ink"]),
            ),
        ],
    )
    fig.update_xaxes(title="1969 per-capita personal income index, U.S. = 100", range=[0, 320])
    fig.update_yaxes(title=f"{year} per-capita personal income index, U.S. = 100", range=[0, 320])
    fig = plot_layout(fig, height=690)
    fig.update_layout(margin=dict(l=64, r=34, t=86, b=86))
    return fig


def county_breakout_atlas(mode: str = "mobility", show_buttons: bool = True) -> go.Figure:
    dataset_path = CLEAN / "county_mobility_ml_dataset.csv"
    residual_path = CLEAN / "county_mobility_residuals.csv"
    geojson_path = RAW / "geojson-counties-fips.json"
    if not dataset_path.exists() or not residual_path.exists() or not geojson_path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Run Dashboard/analyze_county_mobility.py and download county GeoJSON to generate the atlas.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=520)

    df = pd.read_csv(dataset_path, dtype={"county_fips": str})
    residuals = pd.read_csv(residual_path, dtype={"county_fips": str})
    with geojson_path.open() as f:
        full_geojson = json.load(f)
    df = df[df["population_2024"] >= 50_000].copy()
    if "county_fips" not in residuals.columns:
        residuals = residuals.merge(
            df[["county_fips", "bea_county_name"]],
            on="bea_county_name",
            how="left",
            suffixes=("", "_from_dataset"),
        )

    raw_selected = pd.concat(
        [
            df.nlargest(18, "mobility_5yr_avg").assign(atlas_group="Top breakout"),
            df.nsmallest(18, "mobility_5yr_avg").assign(atlas_group="Largest loss"),
        ],
        ignore_index=True,
    )
    residual_selected = pd.concat(
        [
            residuals.nlargest(18, "residual_mobility").assign(atlas_group="Model overperformer"),
            residuals.nsmallest(18, "residual_mobility").assign(atlas_group="Model underperformer"),
        ],
        ignore_index=True,
    )
    residual_selected = residual_selected.merge(
        df[
            [
                "county_fips",
                "income_index_1969_1973",
                "income_index_2020_2024",
                "pcpi_2024",
            ]
        ],
        on="county_fips",
        how="left",
        suffixes=("", "_dataset"),
    )

    selected_fips = set(raw_selected["county_fips"].dropna()) | set(residual_selected["county_fips"].dropna())
    filtered_geojson = {
        "type": "FeatureCollection",
        "features": [feature for feature in full_geojson["features"] if feature["id"] in selected_fips],
    }

    def note(row: pd.Series) -> str:
        parts = []
        if row.get("bachelors_or_higher_pct", np.nan) >= 50:
            parts.append("high-education labor market")
        if row.get("county_gdp_per_capita", np.nan) >= 150_000:
            parts.append("very high local production")
        if row.get("population_growth_pct", np.nan) >= 200:
            parts.append("rapid population growth")
        if row.get("poverty_pct", np.nan) >= 18:
            parts.append("high poverty headwind")
        if row.get("mobility_5yr_avg", np.nan) >= 60:
            parts.append("large long-run income gain")
        if row.get("mobility_5yr_avg", np.nan) <= -35:
            parts.append("large long-run income loss")
        if row.get("residual_mobility", np.nan) >= 35:
            parts.append("beat model expectations")
        if row.get("residual_mobility", np.nan) <= -30:
            parts.append("fell short of model expectations")
        if isinstance(row.get("engine_label", np.nan), str):
            parts.append(row["engine_label"].lower())
        return "; ".join(parts[:3]) if parts else "notable outlier"

    def hover_text(frame: pd.DataFrame, value_col: str) -> pd.Series:
        return frame.apply(
            lambda row: (
                f"<b>{row['bea_county_name']}</b><br>"
                f"{row['state_name']}<br>"
                f"{row['atlas_group']}<br>"
                f"Map value: {row[value_col]:+.1f}<br>"
                f"Income index: {row['income_index_1969_1973']:.1f} to {row['income_index_2020_2024']:.1f}<br>"
                f"2024 population: {row['population_2024']:,.0f}<br>"
                f"Bachelor's share: {row.get('bachelors_or_higher_pct', np.nan):.1f}%<br>"
                f"GDP per person: ${row.get('county_gdp_per_capita', np.nan):,.0f}<br>"
                f"Economic engine: {row.get('engine_label', 'not classified')}<br>"
                f"Note: {note(row)}"
            ),
            axis=1,
        )

    raw_selected["hover_label"] = hover_text(raw_selected, "mobility_5yr_avg")
    residual_selected["hover_label"] = hover_text(residual_selected, "residual_mobility")

    fig = go.Figure()
    fig.add_trace(
        go.Choroplethmap(
                geojson=filtered_geojson,
                locations=raw_selected["county_fips"],
                z=raw_selected["mobility_5yr_avg"],
                featureidkey="id",
                colorscale=[
                    [0, COLORS["red"]],
                    [0.48, "#f2f2f2"],
                    [0.52, "#f2f2f2"],
                    [1, COLORS["teal"]],
                ],
                zmid=0,
                zmin=-90,
                zmax=110,
                marker_line_width=0.8,
                marker_line_color="#ffffff",
                colorbar=dict(title="Breakout<br>score"),
                text=raw_selected["hover_label"],
                customdata=raw_selected["county_fips"],
                hovertemplate="%{text}<extra></extra>",
                name="Breakout Score",
                visible=(mode == "mobility"),
        )
    )
    fig.add_trace(
        go.Choroplethmap(
                geojson=filtered_geojson,
                locations=residual_selected["county_fips"],
                z=residual_selected["residual_mobility"],
                featureidkey="id",
                colorscale=[
                    [0, COLORS["red"]],
                    [0.48, "#f2f2f2"],
                    [0.52, "#f2f2f2"],
                    [1, COLORS["teal"]],
                ],
                zmid=0,
                zmin=-65,
                zmax=65,
                marker_line_width=0.8,
                marker_line_color="#ffffff",
                colorbar=dict(title="Residual<br>index points"),
                text=residual_selected["hover_label"],
                customdata=residual_selected["county_fips"],
                hovertemplate="%{text}<extra></extra>",
                name="Model residual",
                visible=(mode == "residual"),
        )
    )
    if mode == "residual":
        title = "Breakout atlas: counties that beat or missed model expectations"
        annotation = "Residual: actual Breakout Score minus Random Forest predicted score. This finds surprising outliers after observed county traits."
    else:
        title = "Breakout atlas: largest smoothed gains and losses"
        annotation = "Breakout Score: change in income position relative to the U.S., 1969-1973 average to 2020-2024 average."

    updatemenus = []
    if show_buttons:
        updatemenus = [
            dict(
                type="buttons",
                direction="right",
                x=0,
                y=1.08,
                xanchor="left",
                buttons=[
                    dict(
                        label="Largest gains/losses",
                        method="update",
                        args=[
                            {"visible": [True, False]},
                            {
                                "title": "Breakout atlas: largest smoothed gains and losses",
                                "annotations[0].text": "Breakout Score: change in income position relative to the U.S., 1969-1973 average to 2020-2024 average.",
                            },
                        ],
                    ),
                    dict(
                        label="Model residuals",
                        method="update",
                        args=[
                            {"visible": [False, True]},
                            {
                                "title": "Breakout atlas: counties that beat or missed model expectations",
                                "annotations[0].text": "Residual: actual Breakout Score minus Random Forest predicted score. This finds surprising outliers after observed county traits.",
                            },
                        ],
                    ),
                ],
            )
        ]
    fig.update_layout(
        title=title if show_buttons else "",
        updatemenus=updatemenus,
        map=dict(
            style="open-street-map",
            center=dict(lat=37.8, lon=-96.4),
            zoom=3.15,
            bounds=dict(west=-126, east=-65, south=23, north=51),
        ),
        paper_bgcolor="#fbfbfb",
        plot_bgcolor="#fbfbfb",
        margin=dict(l=8, r=8, t=70 if show_buttons else 10, b=8),
        annotations=[
            dict(
                x=0.02,
                y=-0.02,
                xref="paper",
                yref="paper",
                text=annotation if show_buttons else "",
                showarrow=False,
                bgcolor="rgba(244,241,232,0.86)",
                bordercolor="rgba(23,23,23,0.18)",
                borderpad=5,
                font=dict(size=12, color=COLORS["ink"]),
                align="left",
            )
        ],
    )
    return plot_layout(fig, height=700)


ATLAS_STORY_OVERRIDES = {
    "06085": {
        "headline": "Silicon Valley's county engine",
        "image": "https://commons.wikimedia.org/wiki/Special:Redirect/file/San_Jose_from_above.jpg",
        "note": "Santa Clara is the cleanest version of the thesis: high-skill labor, headquarters, venture-backed firms, and research capacity all concentrate in one labor market.",
    },
    "06075": {
        "headline": "Command functions in a dense city",
        "image": "https://commons.wikimedia.org/wiki/Special:Redirect/file/San_Francisco_skyline_from_Marin_Headlands.jpg",
        "note": "San Francisco rises through finance, technology, professional services, and the urban edge of the Bay Area knowledge economy.",
    },
    "36061": {
        "headline": "The command-center county",
        "image": "https://commons.wikimedia.org/wiki/Special:Redirect/file/New_York_City_(New_York,_USA),_Manhattan,_Skyline_--_2012_--_6677.jpg",
        "note": "Manhattan is less a normal county than a national command center: finance, law, media, corporate headquarters, and global business services.",
    },
    "48329": {
        "headline": "A resource boom exception",
        "image": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Pump_Jack_in_Texas.jpg",
        "note": "Midland is a reminder that not all breakouts are knowledge economies. Resource extraction can create spectacular gains, but the mechanism is different.",
    },
    "05007": {
        "headline": "Retail headquarters as local infrastructure",
        "image": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Crystal_Bridges_Museum_of_American_Art.jpg",
        "note": "Benton County reflects Northwest Arkansas's corporate ecosystem around Walmart, suppliers, logistics, and a fast-growing professional class.",
    },
    "32023": {
        "headline": "A high-start county that fell back",
        "image": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Nye_County_Courthouse,_Tonopah,_Nevada.jpg",
        "note": "Nye shows why the map needs losses too: some counties began unusually high relative to the U.S. and then lost position over time.",
    },
    "02020": {
        "headline": "Alaska's relative-income reset",
        "image": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Anchorage_skyline_and_Chugach_Mountains.jpg",
        "note": "Anchorage remains important, but its relative income position fell as the national economy and resource cycle changed around it.",
    },
    "12087": {
        "headline": "Amenity wealth at the edge",
        "image": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Key_West_from_Smathers_Beach.jpg",
        "note": "Monroe County is an amenity and wealth story: high resident income in a constrained, highly desirable place.",
    },
    "08037": {
        "headline": "Mountain amenities and high incomes",
        "image": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Vail_Colorado.jpg",
        "note": "Eagle County illustrates the resort/amenity path: high incomes can concentrate in places built around scarce lifestyle assets.",
    },
}


def atlas_story_records(limit_each: int = 18) -> list[dict[str, object]]:
    dataset_path = CLEAN / "county_mobility_ml_dataset.csv"
    if not dataset_path.exists():
        return []

    df = pd.read_csv(dataset_path, dtype={"county_fips": str})
    df = df[df["population_2024"] >= 50_000].copy()
    selected = pd.concat(
        [
            df.nlargest(limit_each, "mobility_5yr_avg").assign(atlas_group="Top breakout"),
            df.nsmallest(limit_each, "mobility_5yr_avg").assign(atlas_group="Largest loss"),
        ],
        ignore_index=True,
    )
    selected = selected.drop_duplicates("county_fips").sort_values(["state_name", "bea_county_name"])

    records = []
    for _, row in selected.iterrows():
        fips = row["county_fips"]
        override = ATLAS_STORY_OVERRIDES.get(fips, {})
        direction = "gained" if row["mobility_5yr_avg"] >= 0 else "lost"
        engine = row.get("engine_label", "not classified")
        default_note = (
            f"{row['bea_county_name']} {direction} {abs(row['mobility_5yr_avg']):.1f} points of national income position. "
            f"Its current engine label is {str(engine).lower()}."
        )
        records.append(
            {
                "fips": fips,
                "name": row["bea_county_name"],
                "state": row["state_name"],
                "group": row["atlas_group"],
                "headline": override.get("headline", row["bea_county_name"]),
                "image": override.get("image", "https://commons.wikimedia.org/wiki/Special:Redirect/file/United_States_relief_location_map.jpg"),
                "note": override.get("note", default_note),
                "breakout": round(float(row["mobility_5yr_avg"]), 1),
                "start": round(float(row["income_index_1969_1973"]), 1),
                "end": round(float(row["income_index_2020_2024"]), 1),
                "population": int(row["population_2024"]),
                "engine": engine if isinstance(engine, str) else "not classified",
            }
        )
    return records


def metro_nonmetro_lens() -> go.Figure:
    breakouts_path = CLEAN / "county_income_breakouts_1969_2024.csv"
    rucc_path = RAW / "2023-rural-urban-continuum-codes.csv"
    fallback_path = CLEAN / "county_mobility_ml_dataset.csv"
    if not breakouts_path.exists() and not fallback_path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Run Dashboard/analyze_county_mobility.py after downloading USDA Rural-Urban Continuum Codes.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=520)

    order = [
        "Metro, 1M+",
        "Metro, 250k-1M",
        "Metro, <250k",
        "Nonmetro urban, adjacent",
        "Nonmetro urban, remote",
        "Nonmetro town, adjacent",
        "Nonmetro town, remote",
        "Nonmetro rural, adjacent",
        "Nonmetro rural, remote",
    ]

    if breakouts_path.exists() and rucc_path.exists():
        counties = pd.read_csv(breakouts_path, dtype={"county_fips": str})
        counties = counties.rename(
            columns={
                "income_index_change": "mobility_5yr_avg",
                "population_end": "population_2024",
            }
        )
        rucc_raw = pd.read_csv(rucc_path, dtype={"FIPS": str}, encoding="latin1")
        rucc_wide = (
            rucc_raw.pivot_table(
                index=["FIPS", "State", "County_Name"],
                columns="Attribute",
                values="Value",
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(None, axis=1)
            .rename(columns={"FIPS": "county_fips", "RUCC_2023": "rucc_2023", "Description": "rucc_description"})
        )
        rucc_wide["county_fips"] = rucc_wide["county_fips"].str.zfill(5)
        rucc_wide["rucc_2023"] = pd.to_numeric(rucc_wide["rucc_2023"], errors="coerce")
        group_map = {
            1: "Metro, 1M+",
            2: "Metro, 250k-1M",
            3: "Metro, <250k",
            4: "Nonmetro urban, adjacent",
            5: "Nonmetro urban, remote",
            6: "Nonmetro town, adjacent",
            7: "Nonmetro town, remote",
            8: "Nonmetro rural, adjacent",
            9: "Nonmetro rural, remote",
        }
        rucc_wide["rucc_group"] = rucc_wide["rucc_2023"].map(group_map)
        counties = counties.merge(
            rucc_wide[["county_fips", "rucc_group"]],
            on="county_fips",
            how="left",
        )
    else:
        counties = pd.read_csv(fallback_path)

    counties = counties.dropna(subset=["rucc_group", "mobility_5yr_avg"]).copy()
    counties["rucc_group"] = pd.Categorical(counties["rucc_group"], categories=order, ordered=True)
    counties = counties.sort_values("rucc_group")
    if counties.empty:
        fig = go.Figure()
        fig.add_annotation(text="No RUCC-county mobility rows available.", showarrow=False, x=0.5, y=0.5)
        return plot_layout(fig, height=520)

    counts = counties.groupby("rucc_group", observed=False)["county_fips"].count()

    # Display-only trimming: only drop ultra-rare leverage points (1-2 outliers per group)
    # so the density shape is readable without flattening genuine heavy tails.
    filtered_frames = []
    removed_outliers = 0
    for group in order:
        subset = counties[counties["rucc_group"] == group].copy()
        if subset.empty:
            continue
        values = subset["mobility_5yr_avg"].astype(float)
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr <= 0:
            filtered_frames.append(subset)
            continue
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        outlier_mask = (values < lower) | (values > upper)
        outlier_count = int(outlier_mask.sum())
        if 1 <= outlier_count <= 2:
            subset = subset.loc[~outlier_mask].copy()
            removed_outliers += outlier_count
        filtered_frames.append(subset)

    counties = pd.concat(filtered_frames, ignore_index=True)
    if counties.empty:
        fig = go.Figure()
        fig.add_annotation(text="No rows left after outlier filtering.", showarrow=False, x=0.5, y=0.5)
        return plot_layout(fig, height=520)

    def to_rgba(hex_color: str, alpha: float) -> str:
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            return f"rgba(34,124,128,{alpha})"
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    group_values: dict[str, np.ndarray] = {}
    for group in order:
        vals = counties.loc[counties["rucc_group"] == group, "mobility_5yr_avg"].dropna().to_numpy(dtype=float)
        if len(vals):
            group_values[group] = vals

    if not group_values:
        fig = go.Figure()
        fig.add_annotation(text="No valid mobility values available.", showarrow=False, x=0.5, y=0.5)
        return plot_layout(fig, height=520)

    all_values = np.concatenate(list(group_values.values()))
    x_lo = float(np.quantile(all_values, 0.01))
    x_hi = float(np.quantile(all_values, 0.99))
    if x_hi <= x_lo:
        x_lo = float(all_values.min())
        x_hi = float(all_values.max())
    pad = max(2.5, 0.08 * (x_hi - x_lo))
    grid = np.linspace(x_lo - pad, x_hi + pad, 320)

    ridge_height = 0.82
    fig = go.Figure()

    y_positions = {group: float(len(order) - idx - 1) for idx, group in enumerate(order)}
    for group in order:
        if group not in group_values:
            continue
        vals = group_values[group]
        y_base = y_positions[group]
        median_val = float(np.median(vals))
        color = COLORS["teal"] if median_val >= 0 else COLORS["red"]

        bandwidth = max(1.6, min(4.0, float(np.std(vals, ddof=0) * 0.35)))
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(vals.reshape(-1, 1))
        density = np.exp(kde.score_samples(grid.reshape(-1, 1)))
        max_density = float(density.max())
        if max_density <= 0:
            continue
        density_scaled = density / max_density * ridge_height

        x_poly = np.concatenate([grid, grid[::-1]])
        y_poly = np.concatenate([y_base + density_scaled, np.full_like(grid, y_base)])
        fig.add_trace(
            go.Scatter(
                x=x_poly,
                y=y_poly,
                mode="lines",
                line=dict(color=color, width=1.2),
                fill="toself",
                fillcolor=to_rgba(color, 0.34),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=y_base + density_scaled,
                mode="lines",
                line=dict(color=color, width=2.0),
                hovertemplate=(
                    f"<b>{group} (n={len(vals):,})</b><br>"
                    "Mobility: %{x:+.1f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    for group in order:
        if group not in group_values:
            continue
        y_base = y_positions[group]
        fig.add_shape(
            type="line",
            x0=float(grid.min()),
            x1=float(grid.max()),
            y0=y_base,
            y1=y_base,
            line=dict(color="rgba(23,23,23,0.18)", width=1),
            layer="below",
        )

    covered = int(counties["county_fips"].nunique())
    max_group = int(max(len(v) for v in group_values.values()))

    tickvals = [y_positions[g] for g in order if g in group_values]
    ticktext = [f"{g} (n={len(group_values[g]):,})" for g in order if g in group_values]

    fig.update_layout(
        title="County breakout density by metro/nonmetro setting",
        margin=dict(l=240, r=34, t=84, b=108),
    )
    fig.update_xaxes(title="County breakout mobility (income-index change)")
    fig.update_yaxes(
        title="",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        range=[-0.35, len(order) - 0.1],
        showgrid=False,
        zeroline=False,
    )
    note = (
        "Ridgeline density view: each profile is a smoothed county distribution and points upward from its baseline. "
        f"Coverage: {covered:,} counties; largest group has {max_group:,} counties."
    )
    if removed_outliers > 0:
        note += f" Trimmed {removed_outliers} extreme county outlier(s) where only 1-2 points stretched a group's scale."
    fig.add_annotation(
        x=0.02,
        y=-0.22,
        xref="paper",
        yref="paper",
        text=note,
        showarrow=False,
        font=dict(size=12, color=COLORS["muted"]),
        align="left",
    )
    return plot_layout(fig, height=760)


def industry_composition_lens() -> go.Figure:
    summary_path = CLEAN / "county_industry_mobility_summary.csv"
    if not summary_path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Run Dashboard/analyze_industry_composition.py and Dashboard/analyze_county_mobility.py to generate industry diagnostics.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=520)

    summary = pd.read_csv(summary_path)
    order = summary.sort_values("pop_weighted_mobility", ascending=True)["engine_label"].tolist()
    color_map = {
        "Knowledge / command": COLORS["teal"],
        "Resource extraction": COLORS["gold"],
        "Manufacturing": COLORS["blue"],
        "Trade / logistics": COLORS["green"],
        "Education / health": "#7a7a7a",
        "Leisure / amenity": "#9b6a2f",
        "Government": COLORS["red"],
    }

    fig = go.Figure()
    summary["engine_label"] = pd.Categorical(summary["engine_label"], categories=order, ordered=True)
    summary = summary.sort_values("engine_label")
    fig.add_trace(
        go.Bar(
            x=summary["pop_weighted_mobility"],
            y=summary["engine_label"],
            orientation="h",
            marker=dict(
                color=[color_map.get(label, COLORS["muted"]) for label in summary["engine_label"].astype(str)],
                line=dict(color="#ffffff", width=0.8),
            ),
            text=[f"{v:.1f}%" for v in summary["avg_engine_share"]],
            textposition="outside",
            customdata=np.stack(
                [
                    summary["counties"],
                    summary["population_2024"],
                    summary["median_mobility"],
                    summary["avg_current_income_index"],
                    summary["avg_bachelors"],
                    summary["avg_engine_share"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Population-weighted mobility: %{x:+.1f}<br>"
                "Median mobility: %{customdata[2]:+.1f}<br>"
                "Current income index: %{customdata[3]:.1f}<br>"
                "Average bachelor's share: %{customdata[4]:.1f}%<br>"
                "Average engine share: %{customdata[5]:.1f}%<br>"
                "Counties: %{customdata[0]:,.0f}<br>"
                "Population: %{customdata[1]:,.0f}<extra></extra>"
            ),
            name="Engine summary",
            showlegend=False,
        )
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=0,
        y1=1,
        xref="x1",
        yref="paper",
        line=dict(color=COLORS["ink"], width=1, dash="dot"),
    )
    fig.update_layout(
        title="The breakout pattern has a clear industry signature",
        margin=dict(l=196, r=34, t=84, b=96),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    fig.update_xaxes(title="Population-weighted mobility (income-index change)")
    fig.update_yaxes(title="")
    fig.add_annotation(
        x=0.02,
        y=-0.2,
        xref="paper",
        yref="paper",
        text="Bar labels show the average GDP share of that county engine. Source: BEA CAGDP2 current-dollar county GDP by industry.",
        showarrow=False,
        font=dict(size=12, color=COLORS["muted"]),
        align="left",
    )
    return plot_layout(fig, height=700)


def county_quality_of_life_lens() -> go.Figure:
    path = CLEAN / "county_quality_of_life_index.csv"
    if not path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Run Dashboard/build_story_dashboard.py to generate county quality-of-life metrics.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=520)

    df = pd.read_csv(path, dtype={"county_fips": str})
    df = df.dropna(subset=["qol_score_0_100"]).copy()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Quality-of-life data is not available for this build.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=520)

    # Prefer larger counties for readability, then fill if needed.
    display_pool = df[df["acs_population"] >= 50_000].copy()
    if len(display_pool) < 20:
        display_pool = df.copy()
    top = display_pool.nlargest(10, "qol_score_0_100").copy()
    bottom = display_pool.nsmallest(10, "qol_score_0_100").copy()
    display = pd.concat([bottom, top], ignore_index=True)
    display["label"] = display["bea_county_name"] + " (" + display["state_name"] + ")"
    display = display.sort_values("qol_score_0_100")

    fig = go.Figure(
        go.Bar(
            x=display["qol_score_0_100"],
            y=display["label"],
            orientation="h",
            marker=dict(
                color=display["qol_score_0_100"],
                colorscale=[[0.0, COLORS["red"]], [0.5, COLORS["gold"]], [1.0, COLORS["teal"]]],
                cmin=0,
                cmax=100,
                line=dict(color="#ffffff", width=0.7),
            ),
            text=[f"{v:.1f}" for v in display["qol_score_0_100"]],
            textposition="outside",
            customdata=np.stack(
                [
                    display["acs_population"],
                    display["qol_tier"],
                    display["qol_components_used"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Quality-of-life score: %{x:.1f} / 100<br>"
                "Tier: %{customdata[1]}<br>"
                "Population: %{customdata[0]:,.0f}<br>"
                "Components used: %{customdata[2]:.0f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        title="County quality-of-life index (composite, 2023)",
        margin=dict(l=260, r=34, t=84, b=98),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    fig.update_xaxes(title="Quality-of-life score (county percentile, 0-100)", range=[0, 100])
    fig.update_yaxes(title="")
    fig.add_annotation(
        x=0.02,
        y=-0.2,
        xref="paper",
        yref="paper",
        text=(
            "Composite index uses income, poverty, education, and county GDP per person; "
            "where available it also includes broadband, unemployment, commute time, and remote-work share."
        ),
        showarrow=False,
        font=dict(size=12, color=COLORS["muted"]),
        align="left",
    )
    return plot_layout(fig, height=640)


def qol_breakout_correlation_lens() -> go.Figure:
    qol_path = CLEAN / "county_quality_of_life_index.csv"
    mobility_path = CLEAN / "county_mobility_ml_dataset.csv"
    if not qol_path.exists() or not mobility_path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Run Dashboard/build_story_dashboard.py and Dashboard/analyze_county_mobility.py to generate QoL-breakout correlation inputs.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=520)

    qol = pd.read_csv(qol_path, dtype={"county_fips": str})
    mobility = pd.read_csv(
        mobility_path,
        dtype={"county_fips": str},
        usecols=[
            "county_fips",
            "bea_county_name",
            "state_name",
            "mobility_5yr_avg",
            "population_2024",
            "income_index_1969_1973",
            "income_index_2020_2024",
        ],
    )
    df = mobility.merge(
        qol[["county_fips", "qol_score_0_100", "qol_tier", "qol_components_used"]],
        on="county_fips",
        how="inner",
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["mobility_5yr_avg", "qol_score_0_100", "population_2024"]
    )
    if len(df) < 30:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough joined county records to estimate QoL-breakout correlation.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=520)

    pearson_r = df["qol_score_0_100"].corr(df["mobility_5yr_avg"])
    spearman_rho = df["qol_score_0_100"].corr(df["mobility_5yr_avg"], method="spearman")

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df["qol_score_0_100"],
            y=df["mobility_5yr_avg"],
            mode="markers",
            marker=dict(
                size=np.clip(np.sqrt(df["population_2024"]) / 82, 4, 21),
                color=df["qol_score_0_100"],
                colorscale=[[0.0, COLORS["red"]], [0.5, COLORS["gold"]], [1.0, COLORS["teal"]]],
                cmin=0,
                cmax=100,
                opacity=0.7,
                line=dict(width=0),
                colorbar=dict(title="QoL score"),
            ),
            customdata=np.stack(
                [
                    df["bea_county_name"],
                    df["state_name"],
                    df["population_2024"],
                    df["qol_tier"],
                    df["qol_components_used"],
                    df["income_index_1969_1973"],
                    df["income_index_2020_2024"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "QoL score: %{x:.1f} / 100<br>"
                "Breakout mobility: %{y:+.1f}<br>"
                "Tier: %{customdata[3]}<br>"
                "Components used: %{customdata[4]:.0f}<br>"
                "Income index: %{customdata[5]:.1f} to %{customdata[6]:.1f}<br>"
                "Population: %{customdata[2]:,.0f}<extra></extra>"
            ),
            name="Counties",
        )
    )
    add_ols_line(fig, df, "qol_score_0_100", "mobility_5yr_avg", "OLS fit")
    fig.add_hline(y=0, line_color=COLORS["ink"], line_width=1, line_dash="dot")
    fig.update_layout(
        title="How strongly is county quality of life correlated with breakout?",
        showlegend=False,
    )
    fig.update_xaxes(title="County quality-of-life score (0-100 percentile)")
    fig.update_yaxes(title="Breakout mobility (income-index change)")
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        text=f"Pearson r = {pearson_r:+.2f}<br>Spearman rho = {spearman_rho:+.2f}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.86)",
        bordercolor=COLORS["grid"],
        borderpad=6,
        font=dict(size=12, color=COLORS["ink"]),
        align="left",
    )
    fig.add_annotation(
        x=0.02,
        y=-0.18,
        xref="paper",
        yref="paper",
        text="Each point is a county in the mobility model sample. Marker size tracks population; line is an OLS fit.",
        showarrow=False,
        font=dict(size=12, color=COLORS["muted"]),
        align="left",
    )
    return plot_layout(fig, height=620)


def state_human_capital_scatter(latest: pd.DataFrame) -> go.Figure:
    df = latest.sort_values("gdp_per_capita").copy()
    labels = set(df.head(5)["state"]) | set(df.tail(5)["state"]) | {"Texas", "California", "New York"}
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["bachelors_or_higher_pct"],
            y=df["gdp_per_capita"],
            mode="markers+text",
            text=[STATE_ABBR[s] if s in labels else "" for s in df["state"]],
            textposition="top center",
            marker=dict(
                size=np.clip(df["median_household_income"] / 3400, 12, 32),
                color=df["rd_per_capita"],
                colorscale=[[0, COLORS["red"]], [0.5, COLORS["gold"]], [1, COLORS["teal"]]],
                colorbar=dict(title="R&D per person", tickprefix="$"),
                line=dict(color="#ffffff", width=1.2),
                opacity=0.9,
            ),
            customdata=np.stack(
                [
                    df["state"],
                    df["median_household_income"],
                    df["poverty_pct"],
                    df["rd_per_capita"],
                    df["f500_per_million"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Bachelor's or higher: %{x:.1f}%<br>"
                "GDP per person: $%{y:,.0f}<br>"
                "Median household income: $%{customdata[1]:,.0f}<br>"
                "Poverty: %{customdata[2]:.1f}%<br>"
                "R&D per person: $%{customdata[3]:,.0f}<br>"
                "F500 per million residents: %{customdata[4]:.2f}<extra></extra>"
            ),
            name="States",
        )
    )
    add_ols_line(fig, df, "bachelors_or_higher_pct", "gdp_per_capita", "OLS fit")
    fig.update_layout(
        title="The cleanest predictor is knowledge density, not raw size",
        showlegend=False,
    )
    fig.update_xaxes(title="Adults with a bachelor's degree or higher, 2023 ACS")
    fig.update_yaxes(title="Real GDP per person, 2023", tickprefix="$", tickformat=",.0f")
    return plot_layout(fig)


def correlation_bars(latest: pd.DataFrame) -> go.Figure:
    cols = {
        "Bachelor's share": "bachelors_or_higher_pct",
        "Median income": "median_household_income",
        "Poverty": "poverty_pct",
        "R&D per person": "rd_per_capita",
        "F500 per million": "f500_per_million",
        "Population": "population",
        "Total GDP": "gdp",
    }
    values = []
    for label, col in cols.items():
        values.append((label, latest["gdp_per_capita"].corr(latest[col])))
    corr = pd.DataFrame(values, columns=["factor", "correlation"]).sort_values("correlation")
    fig = go.Figure(
        go.Bar(
            x=corr["correlation"],
            y=corr["factor"],
            orientation="h",
            marker_color=[COLORS["red"] if v < 0 else COLORS["teal"] for v in corr["correlation"]],
            text=[f"{v:+.2f}" for v in corr["correlation"]],
            textposition="outside",
            hovertemplate="%{y}<br>Correlation with GDP per person: %{x:.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=COLORS["ink"], line_width=1)
    fig.update_layout(
        title="A four-minute model: prosperity rises with skill and income, falls with poverty",
        showlegend=False,
    )
    fig.update_xaxes(title="Pearson correlation with 2023 state GDP per person", range=[-0.75, 0.9])
    fig.update_yaxes(title="")
    return plot_layout(fig, height=480)


def education_income_diagnostics() -> go.Figure:
    partial_path = CLEAN / "education_income_partial_residuals.csv"
    diagnostics_path = CLEAN / "education_income_diagnostics.csv"
    if not partial_path.exists() or not diagnostics_path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Run Dashboard/analyze_education_income.py to generate education diagnostics.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=420)

    df = pd.read_csv(partial_path)
    diagnostics = pd.read_csv(diagnostics_path)
    current = diagnostics[diagnostics["outcome"] == "income_index_2020_2024"].iloc[0]
    mobility = diagnostics[diagnostics["outcome"] == "mobility_5yr_avg"].iloc[0]
    labels = set(df.nlargest(7, "income_residual_after_controls")["bea_county_name"])
    labels |= set(df.nsmallest(7, "income_residual_after_controls")["bea_county_name"])

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.52, 0.48],
        horizontal_spacing=0.14,
        subplot_titles=(
            "Raw county gradient",
            "After starting income, population, and state controls",
        ),
    )
    marker = dict(
        size=np.clip(np.sqrt(df["population_2024"]) / 76, 4, 24),
        color=df["mobility_5yr_avg"],
        colorscale=[[0, COLORS["red"]], [0.5, "#eeeeee"], [1, COLORS["teal"]]],
        cmin=-60,
        cmax=60,
        colorbar=dict(title="Mobility"),
        opacity=0.68,
        line=dict(width=0),
    )
    fig.add_trace(
        go.Scattergl(
            x=df["bachelors_or_higher_pct"],
            y=df["income_index_2020_2024"],
            mode="markers",
            marker=marker,
            customdata=np.stack(
                [
                    df["bea_county_name"],
                    df["state_name"],
                    df["population_2024"],
                    df["mobility_5yr_avg"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Bachelor's share: %{x:.1f}%<br>"
                "Current income index: %{y:.1f}<br>"
                "Mobility: %{customdata[3]:+.1f}<br>"
                "2024 population: %{customdata[2]:,.0f}<extra></extra>"
            ),
            name="Counties",
        ),
        row=1,
        col=1,
    )
    raw_fit = np.polyfit(df["bachelors_or_higher_pct"], df["income_index_2020_2024"], 1)
    xs = np.linspace(df["bachelors_or_higher_pct"].min(), df["bachelors_or_higher_pct"].max(), 80)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=raw_fit[0] * xs + raw_fit[1],
            mode="lines",
            line=dict(color=COLORS["ink"], width=2),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df["education_residual_after_controls"],
            y=df["income_residual_after_controls"],
            mode="markers+text",
            text=[name if name in labels else "" for name in df["bea_county_name"]],
            textposition="top center",
            marker=marker,
            customdata=np.stack(
                [
                    df["bea_county_name"],
                    df["state_name"],
                    df["population_2024"],
                    df["bachelors_or_higher_pct"],
                    df["income_index_2020_2024"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Education residual: %{x:+.1f}<br>"
                "Income residual: %{y:+.1f}<br>"
                "Bachelor's share: %{customdata[3]:.1f}%<br>"
                "Income index: %{customdata[4]:.1f}<br>"
                "2024 population: %{customdata[2]:,.0f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    resid_fit = np.polyfit(
        df["education_residual_after_controls"],
        df["income_residual_after_controls"],
        1,
    )
    xs2 = np.linspace(
        df["education_residual_after_controls"].min(),
        df["education_residual_after_controls"].max(),
        80,
    )
    fig.add_trace(
        go.Scatter(
            x=xs2,
            y=resid_fit[0] * xs2 + resid_fit[1],
            mode="lines",
            line=dict(color=COLORS["ink"], width=2),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line_color=COLORS["grid"], line_width=1, row=1, col=2)
    fig.add_vline(x=0, line_color=COLORS["grid"], line_width=1, row=1, col=2)
    fig.update_layout(
        title=(
            "Education is not just correlated with income; it adds signal after controls"
        ),
        annotations=[
            *list(fig.layout.annotations),
            dict(
                x=0.03,
                y=1.02,
                xref="paper",
                yref="paper",
                text=f"Raw r = {current['raw_corr']:.2f}; education-only R2 = {current['education_only_r2']:.2f}",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.86)",
                bordercolor=COLORS["grid"],
                borderpad=5,
                font=dict(size=12, color=COLORS["ink"]),
            ),
            dict(
                x=0.73,
                y=1.02,
                xref="paper",
                yref="paper",
                text=(
                    f"Education adds +{current['incremental_education_r2']:.2f} R2 for current income; "
                    f"+{mobility['incremental_education_r2']:.2f} for mobility"
                ),
                showarrow=False,
                bgcolor="rgba(255,255,255,0.86)",
                bordercolor=COLORS["grid"],
                borderpad=5,
                font=dict(size=12, color=COLORS["ink"]),
            ),
        ],
    )
    fig.update_xaxes(title="Adults with bachelor's degree or higher", row=1, col=1)
    fig.update_yaxes(title="Current income index, U.S. = 100", row=1, col=1)
    fig.update_xaxes(title="Education residual after controls", row=1, col=2)
    fig.update_yaxes(title="Income residual after controls", row=1, col=2)
    fig = plot_layout(fig, height=650)
    fig.update_layout(margin=dict(l=72, r=34, t=108, b=76))
    return fig


def model_diagnostics() -> go.Figure:
    ml_path = CLEAN / "county_mobility_ml_dataset.csv"
    if not ml_path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Run Dashboard/analyze_county_mobility.py to generate county model inputs.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=420)

    df = pd.read_csv(ml_path)
    feature_cols = [
        "income_index_1969_1973",
        "bachelors_or_higher_pct",
        "unemployment_pct",
        "median_household_income",
        "poverty_pct",
        "county_gdp_per_capita",
        "broadband_pct",
        "mean_commute_minutes",
        "management_science_arts_occupation_pct",
        "worked_from_home_pct",
    ]
    model_df = df[["mobility_5yr_avg", *feature_cols]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    model_df["breakout_up"] = (model_df["mobility_5yr_avg"] > 0).astype(int)
    if len(model_df) < 120:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough complete rows to estimate a logistic model.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=420)

    X = model_df[feature_cols].copy()
    y = model_df["breakout_up"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegressionCV(
        Cs=np.logspace(-2, 2, 16),
        cv=5,
        penalty="l1",
        solver="saga",
        scoring="roc_auc",
        max_iter=6000,
        n_jobs=-1,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    base_rate = y.mean()

    def pretty_feature(name: str) -> str:
        return (
            name.replace("_pct", " share")
            .replace("county_gdp_per_capita", "county GDP per capita")
            .replace("_", " ")
            .title()
        )

    coef = pd.Series(model.coef_.ravel(), index=feature_cols)
    kept = coef[np.abs(coef) > 1e-5].copy()
    if kept.empty:
        kept = coef.abs().sort_values(ascending=False).head(6)
        kept = coef.loc[kept.index]
    kept = kept.sort_values()
    coef_df = pd.DataFrame(
        {
            "feature": kept.index,
            "feature_label": [pretty_feature(c) for c in kept.index],
            "coef": kept.values,
            "odds_ratio": np.exp(kept.values),
        }
    )

    calib = pd.DataFrame({"pred_prob": probs, "actual": y_test.values})
    calib["bin"] = pd.qcut(calib["pred_prob"], q=8, duplicates="drop")
    calib = (
        calib.groupby("bin", observed=True, as_index=False)
        .agg(
            mean_pred_prob=("pred_prob", "mean"),
            actual_rate=("actual", "mean"),
            count=("actual", "size"),
        )
        .sort_values("mean_pred_prob")
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.12,
        subplot_titles=(
            "Logistic calibration: predicted vs observed breakout rate",
            "Kept predictors (L1 logistic odds ratios)",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=calib["mean_pred_prob"] * 100,
            y=calib["actual_rate"] * 100,
            marker=dict(color=COLORS["teal"], line=dict(color="#ffffff", width=0.8)),
            customdata=np.stack([calib["count"]], axis=-1),
            hovertemplate=(
                "Predicted breakout probability: %{x:.1f}%<br>"
                "Observed breakout rate: %{y:.1f}%<br>"
                "Counties in bin: %{customdata[0]:.0f}<extra></extra>"
            ),
            name="Observed",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=calib["mean_pred_prob"] * 100,
            y=calib["mean_pred_prob"] * 100,
            mode="lines+markers",
            line=dict(color=COLORS["ink"], width=2, dash="dot"),
            marker=dict(size=6),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=coef_df["odds_ratio"],
            y=coef_df["feature_label"],
            orientation="h",
            marker=dict(
                color=[COLORS["teal"] if c > 0 else COLORS["red"] for c in coef_df["coef"]],
                line=dict(color="#ffffff", width=0.8),
            ),
            customdata=np.stack([coef_df["coef"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Odds ratio (1 SD increase): %{x:.2f}<br>"
                "Coefficient: %{customdata[0]:+.2f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_vline(x=1, line_color=COLORS["ink"], line_width=1, line_dash="dot", row=1, col=2)

    fig.update_layout(
        title="Simple breakout model: multivariable logistic regression (L1-pruned)",
        showlegend=False,
    )
    fig.update_xaxes(title="Predicted breakout probability (%)", row=1, col=1)
    fig.update_yaxes(title="Observed breakout rate (%)", row=1, col=1)
    fig.update_xaxes(title="Odds ratio for moving up", row=1, col=2)
    fig.update_yaxes(title="", automargin=True, row=1, col=2)
    fig = plot_layout(fig, height=620)
    fig.update_layout(margin=dict(l=90, r=34, t=96, b=80))
    fig.add_annotation(
        x=0.01,
        y=1.08,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        text=(
            f"n={len(model_df):,} counties | baseline breakout rate={base_rate*100:.1f}% | "
            f"AUC={auc:.2f} | accuracy={acc:.2f} | kept predictors={len(coef_df)}"
        ),
        showarrow=False,
        font=dict(size=12, color=COLORS["muted"]),
        align="left",
    )
    return fig


def kmeans_cluster_lens() -> go.Figure:
    path = CLEAN / "county_mobility_cluster_summary.csv"
    if not path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Run Dashboard/analyze_county_mobility.py to generate K-means cluster summaries.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=420)

    df = pd.read_csv(path).sort_values("avg_mobility")
    fig = go.Figure(
        go.Bar(
            x=df["avg_mobility"],
            y=df["cluster_label"],
            orientation="h",
            marker=dict(
                color=df["avg_mobility"],
                colorscale=[[0.0, COLORS["red"]], [0.5, COLORS["gold"]], [1.0, COLORS["teal"]]],
                line=dict(color="#ffffff", width=0.8),
            ),
            text=[f"{int(c)} counties" for c in df["counties"]],
            textposition="outside",
            customdata=np.stack(
                [
                    df["avg_bachelors"],
                    df["avg_poverty"],
                    df["avg_population_growth"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Average breakout mobility: %{x:+.1f}<br>"
                "Avg bachelor's share: %{customdata[0]:.1f}%<br>"
                "Avg poverty: %{customdata[1]:.1f}%<br>"
                "Avg population growth: %{customdata[2]:+.1f}%<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color=COLORS["ink"], width=1, dash="dot"),
    )
    fig.update_layout(
        title="K-means county clusters: which county types are moving up?",
        margin=dict(l=210, r=34, t=86, b=86),
    )
    fig.update_xaxes(title="Average breakout mobility (income-index change)")
    fig.update_yaxes(title="")
    fig.add_annotation(
        x=0.02,
        y=-0.19,
        xref="paper",
        yref="paper",
        text="Cluster labels summarize county archetypes in the filtered modeling sample.",
        showarrow=False,
        font=dict(size=12, color=COLORS["muted"]),
        align="left",
    )
    return plot_layout(fig, height=520)


def kmeans_cluster_map() -> go.Figure:
    cluster_path = CLEAN / "county_mobility_clusters.csv"
    geojson_path = RAW / "geojson-counties-fips.json"
    if not cluster_path.exists() or not geojson_path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Run Dashboard/analyze_county_mobility.py and ensure county GeoJSON is available.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=520)

    df = pd.read_csv(cluster_path, dtype={"county_fips": str})
    df = df.dropna(subset=["cluster", "cluster_label"]).copy()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No county clusters available to map.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=520)

    df["county_fips"] = df["county_fips"].str.zfill(5)
    order = (
        df.groupby(["cluster", "cluster_label"], as_index=False)["mobility_5yr_avg"]
        .mean()
        .sort_values("mobility_5yr_avg", ascending=False)
    )
    labels = order["cluster_label"].tolist()
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    df["cluster_idx"] = df["cluster_label"].map(label_to_index)

    with geojson_path.open() as f:
        full_geojson = json.load(f)
    selected_fips = set(df["county_fips"])
    filtered_geojson = {
        "type": "FeatureCollection",
        "features": [feature for feature in full_geojson["features"] if feature["id"] in selected_fips],
    }

    palette = [COLORS["teal"], COLORS["blue"], COLORS["gold"], "#8a6b5c", COLORS["red"]]
    n = max(len(labels), 1)
    colorscale = []
    for i in range(n):
        c = palette[i % len(palette)]
        v = i / (n - 1) if n > 1 else 0
        colorscale.append([v, c])

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=filtered_geojson,
            locations=df["county_fips"],
            z=df["cluster_idx"],
            zmin=0,
            zmax=max(n - 1, 0),
            colorscale=colorscale,
            marker_line_width=0.2,
            marker_line_color="rgba(255,255,255,0.45)",
            customdata=np.stack(
                [
                    df["bea_county_name"],
                    df["state_name"],
                    df["cluster_label"],
                    df["mobility_5yr_avg"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}, %{customdata[1]}</b><br>"
                "Cluster: %{customdata[2]}<br>"
                "Breakout mobility: %{customdata[3]:+.1f}<extra></extra>"
            ),
            colorbar=dict(
                title="Cluster type",
                tickvals=list(range(n)),
                ticktext=labels,
                len=0.72,
                thickness=16,
                x=1.02,
            ),
        )
    )
    fig.update_layout(
        title="K-means county clusters on the map",
        margin=dict(l=18, r=90, t=88, b=48),
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=37.8, lon=-96.4),
            zoom=2.7,
        ),
    )
    coverage_base = pd.read_csv(CLEAN / "county_income_breakouts_1969_2024.csv", dtype={"county_fips": str})
    coverage_pct = len(df) / max(len(coverage_base), 1) * 100
    fig.add_annotation(
        x=0.01,
        y=-0.12,
        xref="paper",
        yref="paper",
        text=f"Map covers {len(df):,} counties ({coverage_pct:.1f}% of counties in the BEA income panel).",
        showarrow=False,
        font=dict(size=12, color=COLORS["muted"]),
        align="left",
    )
    return plot_layout(fig, height=700)


def kmeans_k_diagnostic() -> go.Figure:
    cluster_path = CLEAN / "county_mobility_clusters.csv"
    if not cluster_path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Run Dashboard/analyze_county_mobility.py to generate county clusters first.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=460)

    df = pd.read_csv(cluster_path)
    features = [
        "income_index_1969_1973",
        "mobility_5yr_avg",
        "population_growth_pct",
        "bachelors_or_higher_pct",
        "median_household_income",
        "county_gdp_per_capita",
        "poverty_pct",
    ]
    model_df = df.dropna(subset=features).copy()
    if len(model_df) < 50:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough county rows to evaluate candidate K values.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=460)

    for col in features:
        lo, hi = model_df[col].quantile([0.01, 0.99])
        model_df[col] = model_df[col].clip(lo, hi)
    x = StandardScaler().fit_transform(model_df[features])
    k_values = list(range(2, 11))
    inertia = []
    silhouette = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = km.fit_predict(x)
        inertia.append(km.inertia_)
        silhouette.append(silhouette_score(x, labels))

    silhouette_arr = np.array(silhouette)
    best_k = k_values[int(np.argmax(silhouette_arr))]
    current_k = 3

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=inertia,
            mode="lines+markers",
            line=dict(color=COLORS["blue"], width=3),
            marker=dict(size=8),
            name="Inertia (lower is better)",
            hovertemplate="K=%{x}<br>Inertia=%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=silhouette,
            mode="lines+markers",
            line=dict(color=COLORS["teal"], width=3),
            marker=dict(size=8),
            name="Silhouette (higher is better)",
            hovertemplate="K=%{x}<br>Silhouette=%{y:.3f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_vline(x=current_k, line_dash="dot", line_color=COLORS["ink"], line_width=1)
    fig.add_annotation(
        x=current_k,
        y=max(inertia),
        xref="x",
        yref="y",
        text=f"Current K={current_k}",
        showarrow=True,
        arrowhead=2,
        ax=30,
        ay=-30,
        bgcolor="rgba(255,255,255,0.85)",
    )
    fig.update_layout(
        title="Is K=3 reasonable? Elbow + silhouette check",
        margin=dict(l=64, r=68, t=88, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(title="Number of clusters (K)", tickmode="linear", dtick=1)
    fig.update_yaxes(title="Inertia", secondary_y=False)
    fig.update_yaxes(title="Silhouette score", secondary_y=True, range=[0, max(0.4, silhouette_arr.max() + 0.04)])
    fig.add_annotation(
        x=0.01,
        y=-0.23,
        xref="paper",
        yref="paper",
        text=f"Best silhouette in tested range is K={best_k}. Use this as guidance, not proof.",
        showarrow=False,
        font=dict(size=12, color=COLORS["muted"]),
        align="left",
    )
    return plot_layout(fig, height=470)


def cluster_model_check() -> go.Figure:
    cluster_path = CLEAN / "county_mobility_clusters.csv"
    if not cluster_path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Run Dashboard/analyze_county_mobility.py to generate county clusters first.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=360)

    df = pd.read_csv(cluster_path)
    features = [
        "income_index_1969_1973",
        "mobility_5yr_avg",
        "population_growth_pct",
        "bachelors_or_higher_pct",
        "median_household_income",
        "county_gdp_per_capita",
        "poverty_pct",
    ]
    model_df = df.dropna(subset=features).copy()
    if len(model_df) < 50:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough county rows to compare clustering models.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=360)

    for col in features:
        lo, hi = model_df[col].quantile([0.01, 0.99])
        model_df[col] = model_df[col].clip(lo, hi)
    x = StandardScaler().fit_transform(model_df[features])
    k = 3

    rows = []
    km = KMeans(n_clusters=k, random_state=42, n_init=30)
    km_labels = km.fit_predict(x)
    km_sizes = pd.Series(km_labels).value_counts(normalize=True)
    rows.append(
        {
            "model": "KMeans",
            "silhouette": silhouette_score(x, km_labels),
            "bic": np.nan,
            "aic": np.nan,
            "min_cluster_share": km_sizes.min(),
        }
    )

    for cov in ["full", "diag", "tied", "spherical"]:
        gmm = GaussianMixture(n_components=k, covariance_type=cov, random_state=42, n_init=5)
        gmm.fit(x)
        labels = gmm.predict(x)
        sizes = pd.Series(labels).value_counts(normalize=True)
        rows.append(
            {
                "model": f"GMM ({cov})",
                "silhouette": silhouette_score(x, labels),
                "bic": gmm.bic(x),
                "aic": gmm.aic(x),
                "min_cluster_share": sizes.min(),
            }
        )

    result = pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)

    fig = go.Figure(
        go.Bar(
            x=result["silhouette"],
            y=result["model"],
            orientation="h",
            marker=dict(
                color=result["silhouette"],
                colorscale=[[0.0, COLORS["red"]], [0.5, COLORS["gold"]], [1.0, COLORS["teal"]]],
                line=dict(color="#ffffff", width=0.8),
            ),
            customdata=np.stack(
                [
                    result["min_cluster_share"] * 100,
                    result["bic"].fillna(np.nan),
                    result["aic"].fillna(np.nan),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Silhouette: %{x:.3f}<br>"
                "Smallest cluster: %{customdata[0]:.1f}%<br>"
                "BIC: %{customdata[1]:,.0f}<br>"
                "AIC: %{customdata[2]:,.0f}<extra></extra>"
            ),
            text=[f"{v:.3f}" for v in result["silhouette"]],
            textposition="outside",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="K=3 model check: KMeans vs GMM variants",
        margin=dict(l=170, r=24, t=78, b=66),
    )
    fig.update_xaxes(title="Silhouette (higher is better)")
    fig.update_yaxes(title="")
    best_row = result.iloc[0]
    fig.add_annotation(
        x=0.01,
        y=-0.16,
        xref="paper",
        yref="paper",
        text=(
            f"Highest silhouette here: {best_row['model']} ({best_row['silhouette']:.3f}). "
            "Also check smallest-cluster share to avoid collapsed/over-imbalanced partitions."
        ),
        showarrow=False,
        font=dict(size=12, color=COLORS["muted"]),
        align="left",
    )
    return plot_layout(fig, height=360)


def gmm_tied_cluster_map() -> go.Figure:
    base_path = CLEAN / "county_income_breakouts_1969_2024.csv"
    story_path = CLEAN / "county_story_2023.csv"
    geojson_path = RAW / "geojson-counties-fips.json"
    if not base_path.exists() or not story_path.exists() or not geojson_path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Missing files for GMM visualization. Rebuild clean data and county GeoJSON.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=700)

    base = pd.read_csv(base_path, dtype={"county_fips": str}).rename(
        columns={
            "income_index_start": "income_index_1969_1973",
            "income_index_change": "mobility_5yr_avg",
            "population_change_pct": "population_growth_pct",
        }
    )
    story = pd.read_csv(story_path, dtype={"county_fips": str})
    df = base.merge(
        story[
            [
                "county_fips",
                "bea_county_name",
                "state_name",
                "bachelors_or_higher_pct",
                "median_household_income",
                "poverty_pct",
                "county_gdp_per_capita",
            ]
        ],
        on=["county_fips", "bea_county_name", "state_name"],
        how="left",
    )
    features = [
        "income_index_1969_1973",
        "mobility_5yr_avg",
        "population_growth_pct",
        "bachelors_or_higher_pct",
        "median_household_income",
        "county_gdp_per_capita",
        "poverty_pct",
    ]
    df = df.dropna(subset=features).copy()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No rows available for GMM map.", showarrow=False, x=0.5, y=0.5)
        return plot_layout(fig, height=700)
    for col in features:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)

    x = StandardScaler().fit_transform(df[features])
    gmm = GaussianMixture(n_components=3, covariance_type="tied", random_state=42, n_init=5)
    df["cluster"] = gmm.fit_predict(x)
    label_order = (
        df.groupby("cluster", as_index=False)["mobility_5yr_avg"].mean().sort_values("mobility_5yr_avg")
    )
    labels = ["lower breakout", "middle breakout", "higher breakout"]
    mapping = {row.cluster: labels[i] for i, row in enumerate(label_order.itertuples(index=False))}
    df["cluster_label"] = df["cluster"].map(mapping)
    idx_map = {label: i for i, label in enumerate(labels)}
    df["cluster_idx"] = df["cluster_label"].map(idx_map)

    with geojson_path.open() as f:
        full_geojson = json.load(f)
    selected = set(df["county_fips"])
    filtered_geojson = {
        "type": "FeatureCollection",
        "features": [feat for feat in full_geojson["features"] if feat["id"] in selected],
    }

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=filtered_geojson,
            locations=df["county_fips"],
            z=df["cluster_idx"],
            zmin=0,
            zmax=2,
            colorscale=[[0.0, COLORS["red"]], [0.5, COLORS["gold"]], [1.0, COLORS["teal"]]],
            marker_line_width=0.2,
            marker_line_color="rgba(255,255,255,0.45)",
            customdata=np.stack(
                [df["bea_county_name"], df["state_name"], df["cluster_label"], df["mobility_5yr_avg"]],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}, %{customdata[1]}</b><br>"
                "GMM tied cluster: %{customdata[2]}<br>"
                "Breakout mobility: %{customdata[3]:+.1f}<extra></extra>"
            ),
            colorbar=dict(title="GMM tied", tickvals=[0, 1, 2], ticktext=labels, len=0.72, thickness=16, x=1.02),
        )
    )
    fig.update_layout(
        title="GMM (tied covariance) county clusters on the map",
        margin=dict(l=18, r=90, t=88, b=48),
        mapbox=dict(style="carto-positron", center=dict(lat=37.8, lon=-96.4), zoom=2.7),
    )
    fig.add_annotation(
        x=0.01,
        y=-0.12,
        xref="paper",
        yref="paper",
        text=f"GMM tied, K=3, coverage: {len(df):,} counties.",
        showarrow=False,
        font=dict(size=12, color=COLORS["muted"]),
        align="left",
    )
    return plot_layout(fig, height=700)


def county_scatter(county: pd.DataFrame) -> go.Figure:
    df = county.copy()
    df = df[df["county_gdp_per_capita"].between(5_000, 350_000)].copy()
    df = df.dropna(
        subset=[
            "bachelors_or_higher_pct",
            "median_household_income",
            "poverty_pct",
            "county_gdp_per_capita",
        ]
    ).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df["bachelors_or_higher_pct"],
            y=df["median_household_income"],
            mode="markers",
            marker=dict(
                size=np.clip(np.sqrt(df["acs_population"]) / 44, 4, 34),
                color=df["county_gdp_per_capita"],
                colorscale=[[0, COLORS["red"]], [0.55, COLORS["gold"]], [1, COLORS["teal"]]],
                cmin=df["county_gdp_per_capita"].quantile(0.05),
                cmax=df["county_gdp_per_capita"].quantile(0.95),
                colorbar=dict(title="County GDP per person", tickprefix="$"),
                opacity=0.66,
                line=dict(width=0),
            ),
            customdata=np.stack(
                [
                    df["name"],
                    df["state_name"],
                    df["acs_population"],
                    df["poverty_pct"],
                    df["county_gdp_per_capita"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Bachelor's or higher: %{x:.1f}%<br>"
                "Median household income: $%{y:,.0f}<br>"
                "Population: %{customdata[2]:,.0f}<br>"
                "Poverty: %{customdata[3]:.1f}%<br>"
                "County GDP per person: $%{customdata[4]:,.0f}<extra></extra>"
            ),
            name="Counties",
        )
    )
    add_ols_line(fig, df, "bachelors_or_higher_pct", "median_household_income", "OLS fit")
    fig.update_layout(
        title="Granular view: state averages are built from unequal county engines",
        showlegend=False,
    )
    fig.update_xaxes(title="Adults with a bachelor's degree or higher")
    fig.update_yaxes(title="Median household income", tickprefix="$", tickformat=",.0f")
    return plot_layout(fig, height=610)


def top_bottom_lollipop(latest: pd.DataFrame) -> go.Figure:
    top = latest.nlargest(8, "gdp_per_capita").assign(group="Top 8")
    bottom = latest.nsmallest(8, "gdp_per_capita").assign(group="Bottom 8")
    df = pd.concat([bottom, top]).sort_values("gdp_per_capita")
    colors = [COLORS["red"] if g == "Bottom 8" else COLORS["teal"] for g in df["group"]]

    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_shape(
            type="line",
            x0=0,
            x1=row["gdp_per_capita"],
            y0=row["state"],
            y1=row["state"],
            line=dict(color="#d6d6d6", width=2),
        )
    fig.add_trace(
        go.Scatter(
            x=df["gdp_per_capita"],
            y=df["state"],
            mode="markers",
            marker=dict(size=15, color=colors, line=dict(color="#ffffff", width=1)),
            customdata=np.stack(
                [df["bachelors_or_higher_pct"], df["median_household_income"], df["poverty_pct"]],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "GDP per person: $%{x:,.0f}<br>"
                "Bachelor's or higher: %{customdata[0]:.1f}%<br>"
                "Median household income: $%{customdata[1]:,.0f}<br>"
                "Poverty: %{customdata[2]:.1f}%<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(title="The tails tell the story most clearly")
    fig.update_xaxes(title="2023 real GDP per person", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title="", categoryorder="array", categoryarray=df["state"].tolist())
    return plot_layout(fig, height=700)


def time_paths(merged: pd.DataFrame, latest: pd.DataFrame) -> go.Figure:
    winners = latest.nlargest(5, "gdp_per_capita")["state"].tolist()
    laggards = latest.nsmallest(5, "gdp_per_capita")["state"].tolist()
    focus = winners + laggards
    df = merged[merged["state"].isin(focus)].copy()
    baseline = df[df["year"] == df["year"].min()][["state", "gdp_per_capita"]].rename(
        columns={"gdp_per_capita": "baseline_gdp_per_capita"}
    )
    df = df.merge(baseline, on="state")
    df["index_2006"] = df["gdp_per_capita"] / df["baseline_gdp_per_capita"] * 100

    fig = go.Figure()
    for state in focus:
        sub = df[df["state"] == state]
        is_winner = state in winners
        fig.add_trace(
            go.Scatter(
                x=sub["year"],
                y=sub["index_2006"],
                mode="lines",
                line=dict(
                    color=COLORS["teal"] if is_winner else COLORS["red"],
                    width=3 if is_winner else 2,
                ),
                opacity=0.92 if is_winner else 0.72,
                name=STATE_ABBR[state],
                hovertemplate=f"<b>{state}</b><br>%{{x}} index: %{{y:.1f}}<extra></extra>",
            )
        )
    fig.add_hline(y=100, line_color=COLORS["ink"], line_width=1, line_dash="dot")
    fig.update_layout(
        title="The hierarchy is persistent; the question is why the engines compound",
        legend=dict(orientation="h", y=-0.2, x=0),
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title="GDP per person index, 2006 = 100")
    return plot_layout(fig, height=520)


def metro_concentration(county: pd.DataFrame) -> go.Figure:
    df = county.copy()
    state_summary = (
        df.sort_values(["state_name", "county_gdp_current_dollars"], ascending=[True, False])
        .groupby("state_name")
        .head(1)
        .copy()
    )
    state_totals = df.groupby("state_name", as_index=False).agg(
        state_county_gdp=("county_gdp_current_dollars", "sum"),
        state_county_population=("acs_population", "sum"),
    )
    state_summary = state_summary.merge(state_totals, on="state_name")
    state_summary["top_county_gdp_share"] = (
        state_summary["county_gdp_current_dollars"] / state_summary["state_county_gdp"]
    )
    state_summary = state_summary.sort_values("top_county_gdp_share").tail(14)

    fig = go.Figure(
        go.Bar(
            x=state_summary["top_county_gdp_share"] * 100,
            y=state_summary["state_name"],
            orientation="h",
            marker_color=COLORS["gold"],
            customdata=np.stack(
                [
                    state_summary["name"],
                    state_summary["county_gdp_current_dollars"],
                    state_summary["county_gdp_per_capita"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Share of state county GDP: %{x:.1f}%<br>"
                "County GDP: $%{customdata[1]:,.0f}<br>"
                "County GDP per person: $%{customdata[2]:,.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Many state economies depend on one dominant county engine",
        showlegend=False,
    )
    fig.update_xaxes(title="Largest county share of state county GDP")
    fig.update_yaxes(title="")
    return plot_layout(fig, height=520)


def state_engine_breakdown(county: pd.DataFrame, latest: pd.DataFrame) -> go.Figure:
    df = county[county["state_name"].notna()].copy()
    df = df.dropna(subset=["county_gdp_current_dollars", "acs_population"]).copy()
    df = df[df["acs_population"] > 0].copy()
    us_gdp_per_person = df["county_gdp_current_dollars"].sum() / df["acs_population"].sum()
    state_totals = df.groupby("state_name", as_index=False).agg(
        state_county_gdp=("county_gdp_current_dollars", "sum"),
        state_county_population=("acs_population", "sum"),
    )
    state_totals["state_county_gdp_per_person"] = (
        state_totals["state_county_gdp"] / state_totals["state_county_population"]
    )
    state_totals["state_premium_per_person"] = (
        state_totals["state_county_gdp_per_person"] - us_gdp_per_person
    )
    df = df.merge(state_totals, on="state_name", how="left")
    df["county_gdp_per_person"] = df["county_gdp_current_dollars"] / df["acs_population"]
    df["premium_contribution_per_state_resident"] = (
        (df["county_gdp_current_dollars"] - df["acs_population"] * us_gdp_per_person)
        / df["state_county_population"]
    )
    positive = df[df["premium_contribution_per_state_resident"] > 0].copy()
    pos_summary = positive.groupby("state_name", as_index=False).agg(
        positive_premium=("premium_contribution_per_state_resident", "sum"),
        top_county_positive_contribution=("premium_contribution_per_state_resident", "max"),
    )
    state_totals = state_totals.merge(pos_summary, on="state_name", how="left")
    state_totals[["positive_premium", "top_county_positive_contribution"]] = state_totals[
        ["positive_premium", "top_county_positive_contribution"]
    ].fillna(0)
    state_totals["largest_engine_share_of_positive_premium"] = np.where(
        state_totals["positive_premium"] > 0,
        state_totals["top_county_positive_contribution"] / state_totals["positive_premium"] * 100,
        np.nan,
    )
    focus_states = state_totals.nlargest(14, "state_premium_per_person").copy()
    focus_states = focus_states.sort_values("state_premium_per_person")

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.44, 0.56],
        vertical_spacing=0.16,
        subplot_titles=(
            "Top states: GDP per person premium over the U.S. county average",
            "Selected state: county contributions to the state premium",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=focus_states["state_premium_per_person"],
            y=focus_states["state_name"],
            orientation="h",
            marker=dict(
                color=focus_states["largest_engine_share_of_positive_premium"],
                colorscale=[[0, COLORS["teal"]], [0.5, COLORS["gold"]], [1, COLORS["red"]]],
                cmin=0,
                cmax=70,
                colorbar=dict(title="Largest<br>engine share"),
                line=dict(color="#ffffff", width=0.8),
            ),
            customdata=np.stack(
                [
                    focus_states["state_county_gdp_per_person"],
                    focus_states["state_county_population"],
                    focus_states["largest_engine_share_of_positive_premium"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Premium over U.S. county average: $%{x:,.0f}<br>"
                "State county GDP per person: $%{customdata[0]:,.0f}<br>"
                "Population: %{customdata[1]:,.0f}<br>"
                "Largest county share of positive premium: %{customdata[2]:.1f}%<extra></extra>"
            ),
            name="State premium",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    state_order = ["California", "New York", "Massachusetts", "Washington", "Texas", "Florida"]
    state_order += [
        state
        for state in state_totals.sort_values("state_premium_per_person", ascending=False)["state_name"].tolist()
        if state not in state_order
    ]
    state_order = [state for state in state_order if state in set(df["state_name"])]
    bar_trace_start = 1
    for i, state in enumerate(state_order):
        sub = df[df["state_name"] == state].copy()
        top_pos = sub.nlargest(9, "premium_contribution_per_state_resident").copy()
        bottom_neg = sub.nsmallest(4, "premium_contribution_per_state_resident").copy()
        selected = pd.concat([top_pos, bottom_neg]).drop_duplicates("county_fips")
        other_contribution = (
            sub["premium_contribution_per_state_resident"].sum()
            - selected["premium_contribution_per_state_resident"].sum()
        )
        other = pd.DataFrame(
            {
                "name": ["All other counties"],
                "premium_contribution_per_state_resident": [other_contribution],
                "county_gdp_per_person": [np.nan],
                "county_gdp_current_dollars": [np.nan],
                "acs_population": [np.nan],
            }
        )
        plot_df = pd.concat(
            [
                selected[
                    [
                        "name",
                        "premium_contribution_per_state_resident",
                        "county_gdp_per_person",
                        "county_gdp_current_dollars",
                        "acs_population",
                    ]
                ],
                other,
            ]
        )
        plot_df = plot_df.sort_values("premium_contribution_per_state_resident")
        fig.add_trace(
            go.Bar(
                x=plot_df["premium_contribution_per_state_resident"],
                y=[
                    name.replace(" County", "").replace(", California", "").replace(", Texas", "")
                    for name in plot_df["name"]
                ],
                orientation="h",
                marker_color=[
                    "#d9d9d9"
                    if name == "All other counties"
                    else COLORS["teal"]
                    if value >= 0
                    else COLORS["red"]
                    for name, value in zip(plot_df["name"], plot_df["premium_contribution_per_state_resident"])
                ],
                customdata=np.stack(
                    [
                        plot_df["county_gdp_per_person"],
                        plot_df["county_gdp_current_dollars"],
                        plot_df["acs_population"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Contribution to state premium: $%{x:,.0f} per state resident<br>"
                    "County GDP per person: $%{customdata[0]:,.0f}<br>"
                    "County GDP: $%{customdata[1]:,.0f}<br>"
                    "County population: %{customdata[2]:,.0f}<extra></extra>"
                ),
                name=state,
                visible=(state == "California"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    buttons = []
    for i, state in enumerate(state_order):
        visible = [True] + [False] * len(state_order)
        visible[bar_trace_start + i] = True
        buttons.append(
            dict(
                label=state,
                method="update",
                args=[
                    {"visible": visible},
                    {"annotations[1].text": f"{state}: who lifts the state above average?"},
                ],
            )
        )

    fig.update_layout(
        title="Which counties actually make a state exceptional?",
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                x=1,
                y=1.08,
                xanchor="right",
                yanchor="top",
            )
        ],
        showlegend=False,
        annotations=[
            *list(fig.layout.annotations),
            dict(
                x=7400,
                y="California",
                xref="x",
                yref="y",
                text="Color asks concentration: does one county create most of the positive premium?",
                showarrow=True,
                arrowhead=2,
                ax=70,
                ay=-36,
                bgcolor="rgba(255,255,255,0.86)",
                bordercolor=COLORS["grid"],
                borderpad=5,
                font=dict(size=12, color=COLORS["ink"]),
            ),
        ],
    )
    fig.add_vline(x=0, line_color=COLORS["ink"], line_width=1, line_dash="dot", row=2, col=1)
    fig.update_xaxes(title="GDP per person premium over U.S. county average", tickprefix="$", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title="", automargin=True, row=1, col=1)
    fig.update_xaxes(title="Contribution to selected state's premium, dollars per state resident", tickprefix="$", tickformat=",.0f", row=2, col=1)
    fig.update_yaxes(title="", automargin=True, row=2, col=1)
    fig = plot_layout(fig, height=625)
    fig.update_layout(margin=dict(l=150, r=34, t=86, b=64))
    return fig


def state_mobility_distributions() -> go.Figure:
    dataset_path = CLEAN / "county_mobility_ml_dataset.csv"
    summary_path = CLEAN / "county_mobility_state_summary.csv"
    if not dataset_path.exists() or not summary_path.exists():
        fig = go.Figure()
        fig.add_annotation(
            text="Run Dashboard/analyze_county_mobility.py to generate state mobility summaries.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        return plot_layout(fig, height=420)

    counties = pd.read_csv(dataset_path)
    summary = pd.read_csv(summary_path)
    summary = summary[summary["state_name"] != "District of Columbia"].copy()
    counties = counties[counties["state_name"].isin(summary["state_name"])].copy()
    order = summary.sort_values("pop_weighted_mobility")["state_name"].tolist()
    summary["state_name"] = pd.Categorical(summary["state_name"], categories=order, ordered=True)
    counties["state_name"] = pd.Categorical(counties["state_name"], categories=order, ordered=True)
    summary = summary.sort_values("state_name")

    fig = go.Figure()
    for _, row in summary.iterrows():
        fig.add_shape(
            type="line",
            x0=row["p25_mobility"],
            x1=row["p75_mobility"],
            y0=row["state_name"],
            y1=row["state_name"],
            line=dict(color="#bdbdbd", width=7),
            xref="x",
            yref="y",
            layer="below",
        )
    fig.add_trace(
        go.Scatter(
            x=counties["mobility_5yr_avg"],
            y=counties["state_name"],
            mode="markers",
            marker=dict(
                size=np.clip(np.sqrt(counties["population_2024"]) / 92, 4, 22),
                color="rgba(23, 23, 23, 0.32)",
                line=dict(width=0),
            ),
            customdata=np.stack(
                [
                    counties["bea_county_name"],
                    counties["population_2024"],
                    counties["income_index_1969_1973"],
                    counties["income_index_2020_2024"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{y}<br>"
                "Mobility: %{x:+.1f}<br>"
                "Index: %{customdata[2]:.1f} to %{customdata[3]:.1f}<br>"
                "2024 population: %{customdata[1]:,.0f}<extra></extra>"
            ),
            name="Counties",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary["pop_weighted_mobility"],
            y=summary["state_name"],
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=11,
                color=COLORS["teal"],
                line=dict(color="#ffffff", width=1),
            ),
            customdata=np.stack(
                [
                    summary["counties"],
                    summary["median_mobility"],
                    summary["p25_mobility"],
                    summary["p75_mobility"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Population-weighted mobility: %{x:+.1f}<br>"
                "Median county: %{customdata[1]:+.1f}<br>"
                "Middle 50%: %{customdata[2]:+.1f} to %{customdata[3]:+.1f}<br>"
                "Counties modeled: %{customdata[0]:.0f}<extra></extra>"
            ),
            name="State average",
        )
    )
    fig.add_vline(x=0, line_color=COLORS["ink"], line_width=1, line_dash="dot")
    fig.update_layout(
        title="States differ, but counties inside the same state often diverge",
        legend=dict(orientation="h", y=-0.08, x=0),
        annotations=[
            dict(
                x=28,
                y="Massachusetts",
                text="State context is visible",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.86)",
                bordercolor=COLORS["grid"],
                borderpad=5,
                font=dict(size=12, color=COLORS["ink"]),
            ),
            dict(
                x=-28,
                y="Nevada",
                text="But within-state county spread remains large",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.86)",
                bordercolor=COLORS["grid"],
                borderpad=5,
                font=dict(size=12, color=COLORS["ink"]),
            ),
        ],
    )
    fig.update_xaxes(title="County mobility: income-index change, 1969-1973 to 2020-2024")
    fig.update_yaxes(title="", categoryorder="array", categoryarray=order)
    fig = plot_layout(fig, height=980)
    fig.update_layout(margin=dict(l=148, r=34, t=86, b=86))
    return fig


def fig_html(fig: go.Figure, slug: str) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        div_id=slug,
        auto_play=False,
        config={
            "displayModeBar": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


def make_html(
    merged: pd.DataFrame,
    latest: pd.DataFrame,
    county: pd.DataFrame,
    state_compounding: pd.DataFrame,
    county_concentration: pd.DataFrame,
    income_breakouts: pd.DataFrame,
    gdp_breakouts: pd.DataFrame,
    county_income_panel: pd.DataFrame,
    county_gdp_index: pd.DataFrame,
) -> str:
    top_states = latest.nlargest(5, "gdp_per_capita")
    bottom_states = latest.nsmallest(5, "gdp_per_capita")
    corr_degree = latest["gdp_per_capita"].corr(latest["bachelors_or_higher_pct"])
    corr_income = latest["gdp_per_capita"].corr(latest["median_household_income"])
    corr_poverty = latest["gdp_per_capita"].corr(latest["poverty_pct"])
    gap_start = state_compounding["top10_bottom10_gap"].iloc[0]
    gap_end = state_compounding["top10_bottom10_gap"].iloc[-1]
    county_share_start = county_concentration["top_1pct_share"].iloc[0] * 100
    county_share_end = county_concentration["top_1pct_share"].iloc[-1] * 100

    charts = [
        (
            "1",
            "Counties Move",
            "The national state hierarchy is stable, but counties inside it can radically reorder.",
            county_mobility_animation(county_income_panel),
        ),
        (
            "2",
            "States Differ",
            "State context matters, but each state still contains counties moving in very different directions.",
            state_mobility_distributions(),
        ),
        (
            "3",
            "Metro Lens",
            "This summary strips away point-cloud noise: large metros host most upside, but several nonmetro settings still produce breakouts.",
            metro_nonmetro_lens(),
        ),
        (
            "4",
            "Industry Signature",
            "Breakout gains cluster in command-style county economies; engine share labels show how dominant each engine is in its counties.",
            industry_composition_lens(),
        ),
        (
            "5",
            "What Factors Matter?",
            "A simple multivariable logistic model shows which county factors increase or reduce the odds of moving up.",
            model_diagnostics(),
        ),
        (
            "6",
            "Breakout Atlas",
            "After the pattern is established, the atlas lets you explore who broke out and who fell behind.",
            county_breakout_atlas(show_buttons=False),
        ),
    ]
    support_charts = [
        (
            "A",
            "Education Gradient",
            "Education remains a strong income signal even after controlling for where counties started, how large they were, and which state they are in.",
            education_income_diagnostics(),
        ),
        (
            "B",
            "State Engines",
            "The best states are portfolios of county engines, not just one famous superstar county.",
            state_engine_breakdown(county, latest),
        ),
        (
            "C",
            "County Quality Of Life",
            "A county-level quality-of-life composite makes it easier to benchmark local outcomes with one comparable score.",
            county_quality_of_life_lens(),
        ),
        (
            "D",
            "QoL vs Breakout Correlation",
            "This links the composite quality-of-life score directly to breakout mobility to show how tightly they move together.",
            qol_breakout_correlation_lens(),
        ),
        (
            "E",
            "County Clusters (K-Means)",
            "K-means groups counties into simple archetypes so you can see which county types are consistently moving up.",
            kmeans_cluster_lens(),
        ),
        (
            "F",
            "Clusters On The Map",
            "Same K-means archetypes, now mapped so we can see where each county type is concentrated.",
            kmeans_cluster_map(),
        ),
        (
            "G",
            "Picking K",
            "Quick elbow + silhouette diagnostics to sanity-check whether K=3 is a defensible cluster count.",
            kmeans_k_diagnostic(),
        ),
        (
            "H",
            "KMeans vs GMM",
            "Side-by-side metrics for K=3 so you can see whether Gaussian mixtures outperform standard K-means.",
            cluster_model_check(),
        ),
        (
            "I",
            "GMM Tied Map",
            "A direct map of GMM (tied covariance, K=3) clusters so you can compare geographic structure vs K-means.",
            gmm_tied_cluster_map(),
        ),
    ]
    atlas_records = atlas_story_records()
    atlas_json = json.dumps(atlas_records, ensure_ascii=False).replace("</", "<\\/")

    def chart_section_html(num: str, title: str, note: str, fig: go.Figure) -> str:
        is_atlas = title == "Breakout Atlas"
        section_class = "viz-block atlas-block" if is_atlas else "viz-block"
        kicker = f"Visual {num} · Atlas Plate" if is_atlas else f"Visual {num}"
        controls = ""
        plot_html = f'<div class="plot-wrap">{fig_html(fig, f"story-{num}")}</div>'
        if num == "1":
            tick_years = [1969, 1977, 1985, 1993, 2001, 2009, 2017, 2024]
            ticks = "".join(f"<span>{year}</span>" for year in tick_years)
            controls = f"""
          <div class="custom-timeline" data-plot="story-1">
            <div class="timeline-row">
              <div>
                <div class="timeline-label">Animation year</div>
                <div class="timeline-year"><span data-year-label>1969</span></div>
              </div>
              <div class="timeline-actions">
                <button type="button" data-action="play">Play</button>
                <button type="button" data-action="pause">Pause</button>
              </div>
            </div>
            <input aria-label="Select year" data-year-slider type="range" min="1969" max="2024" value="1969" step="1">
            <div class="timeline-ticks">{ticks}</div>
            <div class="timeline-caption">Drag through income history, or play the animation from the 1969 baseline.</div>
          </div>
            """
        if is_atlas:
            options = "\n".join(
                f'<option value="{record["fips"]}">{record["state"]} · {record["name"]}</option>'
                for record in atlas_records
            )
            plot_html = f"""
          <div class="atlas-stage" data-atlas-plot="story-{num}">
            <div class="plot-wrap">{fig_html(fig, f"story-{num}")}</div>
            <button type="button" class="atlas-card-toggle" data-atlas-toggle aria-expanded="false">
              Show breakout county profile
            </button>
            <aside class="atlas-card is-collapsed" data-atlas-card>
              <div class="atlas-card-image" data-atlas-image></div>
              <div class="atlas-card-body">
                <div class="atlas-card-kicker" data-atlas-kicker>Breakout County</div>
                <h3 data-atlas-title>Choose a county</h3>
                <p data-atlas-note>Use the dropdown or click a highlighted county on the map.</p>
                <div class="atlas-card-stats">
                  <div><span>Score</span><strong data-atlas-score>--</strong></div>
                  <div><span>Index</span><strong data-atlas-index>--</strong></div>
                  <div><span>Engine</span><strong data-atlas-engine>--</strong></div>
                </div>
                <label class="atlas-select-label">
                  Select county
                  <select data-atlas-select>{options}</select>
                </label>
              </div>
            </aside>
          </div>
          <script type="application/json" id="atlas-card-data">{atlas_json}</script>
            """
        return f"""
        <section class="{section_class}" id="chart-{num}">
          <div class="section-kicker">{kicker}</div>
          <div class="section-head">
            <h2>{title}</h2>
            <p>{note}</p>
          </div>
          {plot_html}
          {controls}
        </section>
        """

    chart_sections = "\n".join(
        chart_section_html(num, title, note, fig)
        for num, title, note, fig in charts
    )
    support_sections = "\n".join(
        f"""
      <section class="viz-block support-block" id="support-block-{num}">
          <div class="section-kicker">Supporting Evidence {num}</div>
          <div class="section-head">
            <h2>{title}</h2>
            <p>{note}</p>
          </div>
          <div class="plot-wrap">{fig_html(fig, f"support-{num}")}</div>
        </section>
        """
        for num, title, note, fig in support_charts
    )

    top_list = ", ".join(f"{STATE_ABBR[s]} ${v:,.0f}" for s, v in zip(top_states["state"], top_states["gdp_per_capita"]))
    bottom_list = ", ".join(f"{STATE_ABBR[s]} ${v:,.0f}" for s, v in zip(bottom_states["state"], bottom_states["gdp_per_capita"]))
    bg_path = OUT / "assets" / "us_space_view.jpg"
    if not bg_path.exists():
        bg_path = OUT / "assets" / "economic_map_background.png"
    bg_uri = ""
    if bg_path.exists():
        bg_uri = "data:image/png;base64," + base64.b64encode(bg_path.read_bytes()).decode("ascii")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Why Some State Economies Compound</title>
  <script src="https://cdn.plot.ly/plotly-{PLOTLY_JS_VERSION}.min.js"></script>
  <style>
    :root {{
      --ink: {COLORS["ink"]};
      --muted: {COLORS["muted"]};
      --paper: {COLORS["paper"]};
      --paper-alt: #f5f3ed;
      --panel: #ffffff;
      --line: {COLORS["grid"]};
      --red: {COLORS["red"]};
      --teal: {COLORS["teal"]};
      --gold: {COLORS["gold"]};
      --green: {COLORS["green"]};
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 22px;
      --shadow-soft: 0 10px 26px rgba(23,23,23,0.06);
      --shadow-strong: 0 22px 54px rgba(23,23,23,0.14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      background-image:
        radial-gradient(circle at 7% -8%, rgba(34,124,128,0.14), transparent 26%),
        radial-gradient(circle at 92% 6%, rgba(193,154,48,0.13), transparent 24%),
        linear-gradient(180deg, #fcfcfa 0%, var(--paper) 48%, #f6f5f1 100%);
      font-family: {BODY_FONT} !important;
      line-height: 1.45;
      letter-spacing: 0;
    }}
    .page {{
      width: min(1260px, calc(100% - 136px));
      margin: 0 auto;
      padding-bottom: 24px;
    }}
    header {{
      position: relative;
      overflow: hidden;
      min-height: 84vh;
      display: grid;
      align-content: center;
      border-bottom: 1px solid rgba(23,23,23,0.12);
      padding: 42px 0 36px;
    }}
    header::before {{
      content: "";
      position: absolute;
      inset: 34px -210px -90px 28%;
      background-image: var(--hero-bg);
      background-position: right 12% top 16%;
      background-repeat: no-repeat;
      background-size: min(82vw, 1180px) auto;
      opacity: 0.56;
      transform-origin: 84% 12%;
      transform: perspective(1180px) rotateX(22deg) rotateZ(-9deg) scale(1.08);
      filter: saturate(0.9) contrast(1.05) brightness(0.98);
      mask-image: radial-gradient(circle at 48% 44%, rgba(0,0,0,0.92), rgba(0,0,0,0.42) 64%, transparent 88%);
      pointer-events: none;
    }}
    header::after {{
      content: "";
      position: absolute;
      inset: 18% -8% 0 42%;
      background:
        radial-gradient(circle at 26% 30%, rgba(193,154,48,0.24), transparent 31%),
        radial-gradient(circle at 72% 42%, rgba(34,124,128,0.22), transparent 38%),
        radial-gradient(circle at 44% 78%, rgba(180,59,69,0.13), transparent 34%);
      pointer-events: none;
    }}
    header > * {{
      position: relative;
      z-index: 1;
    }}
    .masthead {{
      align-self: start;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 24px;
      border-bottom: 1px solid rgba(23,23,23,0.14);
      padding-bottom: 16px;
      margin-bottom: 54px;
      font-size: 13px;
      backdrop-filter: blur(3px);
    }}
    .masthead .mark {{
      font-family: {DISPLAY_FONT} !important;
      font-size: 18px;
    }}
    .masthead .links {{
      display: flex;
      gap: 18px;
      color: var(--muted);
      text-transform: uppercase;
      font-size: 11px;
      font-weight: 800;
    }}
    .eyebrow {{
      color: var(--teal);
      text-transform: uppercase;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0;
      margin-bottom: 22px;
    }}
    h1 {{
      max-width: 940px;
      margin: 0;
      font-family: {DISPLAY_FONT} !important;
      font-weight: 400;
      font-size: clamp(54px, 7vw, 104px);
      line-height: 0.92;
      font-weight: 500;
      letter-spacing: 0;
    }}
    .thesis {{
      max-width: 820px;
      margin: 30px 0 0;
      font-size: clamp(19px, 2.15vw, 27px);
      color: #303030;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 1px;
      background: rgba(23,23,23,0.11);
      margin-top: 52px;
      border: 1px solid rgba(23,23,23,0.14);
      border-radius: var(--radius-md);
      overflow: hidden;
      box-shadow: var(--shadow-soft);
      max-width: 1120px;
    }}
    .metric {{
      background:
        linear-gradient(155deg, rgba(255,255,255,0.96), rgba(255,255,255,0.9));
      padding: 22px;
      min-height: 138px;
      transition: transform 180ms ease, box-shadow 180ms ease;
    }}
    .metric:hover {{
      transform: translateY(-2px);
      box-shadow: inset 0 0 0 1px rgba(34,124,128,0.18);
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 800;
    }}
    .metric .label::before {{
      content: "";
      display: block;
      width: 36px;
      height: 3px;
      border-radius: 999px;
      margin-bottom: 10px;
      background: var(--line);
    }}
    .metric:nth-child(1) .label::before {{ background: var(--red); }}
    .metric:nth-child(2) .label::before {{ background: var(--gold); }}
    .metric:nth-child(3) .label::before {{ background: var(--teal); }}
    .metric:nth-child(4) .label::before {{ background: var(--green); }}
    .metric:nth-child(5) .label::before {{ background: var(--ink); }}
    .metric .value {{
      margin-top: 16px;
      font-family: {DISPLAY_FONT} !important;
      font-size: 34px;
      line-height: 1;
    }}
    .metric .sub {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 14px;
    }}
    .metric-definition {{
      border-left: 3px solid var(--teal);
      padding: 12px 16px;
      margin-top: 24px;
      max-width: 880px;
      color: #2f2f2f;
      background: rgba(255,255,255,0.78);
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
      font-size: 16px;
    }}
    .storyline {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 38px;
      padding: 58px 0 40px;
      border-bottom: 1px solid rgba(23,23,23,0.12);
    }}
    .storyline h2, .viz-block h2, .support-intro h2 {{
      margin: 0;
      font-family: {DISPLAY_FONT} !important;
      font-size: clamp(30px, 4vw, 54px);
      line-height: 1;
      font-weight: 500;
    }}
    .storyline p, .section-head p {{
      margin: 16px 0 0;
      color: #353535;
      font-size: 18px;
      max-width: 760px;
    }}
    .script {{
      border-left: 3px solid var(--teal);
      padding: 6px 0 6px 22px;
      color: #2f2f2f;
      font-size: 16px;
    }}
    .script strong {{
      color: var(--ink);
    }}
    .viz-block {{
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding: 44px 0 46px;
      border-bottom: 1px solid rgba(23,23,23,0.12);
      scroll-margin-top: 18px;
    }}
    .support-intro {{
      padding: 58px 0 20px;
      border-bottom: 1px solid rgba(23,23,23,0.12);
    }}
    .support-intro p {{
      color: #353535;
      font-size: 18px;
      max-width: 820px;
    }}
    .section-kicker {{
      color: var(--red);
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 800;
      letter-spacing: 0.04em;
      margin-bottom: 8px;
    }}
    .section-head {{
      display: grid;
      grid-template-columns: 0.9fr 1.1fr;
      gap: 34px;
      align-items: start;
      margin-bottom: 14px;
    }}
    .section-head h2 {{
      font-size: clamp(28px, 3.25vw, 46px);
    }}
    .section-head p {{
      font-size: 16px;
      line-height: 1.4;
    }}
    .plot-wrap {{
      position: relative;
      background: var(--panel);
      border: 1px solid rgba(23,23,23,0.14);
      border-radius: var(--radius-md);
      padding: 12px;
      box-shadow: var(--shadow-soft);
      overflow: hidden;
    }}
    .plot-wrap::before {{
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.55), transparent 28%);
    }}
    #chart-1 {{
      align-items: center;
    }}
    #chart-1 .section-kicker,
    #chart-1 .section-head,
    #chart-1 .plot-wrap,
    #chart-1 .custom-timeline {{
      width: min(100%, 1120px);
    }}
    #chart-1 .section-head {{
      grid-template-columns: 0.82fr 1.18fr;
    }}
    #chart-1 .plot-wrap {{
      padding: 14px;
    }}
    .atlas-block {{
      background:
        linear-gradient(90deg, rgba(34,124,128,0.08), rgba(255,255,255,0.9));
      border-left: 3px solid var(--teal);
      padding-left: 24px;
      padding-right: 0;
      min-height: 100vh;
    }}
    .atlas-block .section-kicker {{
      color: var(--teal);
    }}
    .atlas-block .plot-wrap {{
      background: #fbfbfb;
      border-color: rgba(23,23,23,0.16);
      box-shadow: 0 16px 38px rgba(23,23,23,0.08);
    }}
    .atlas-stage {{
      position: relative;
    }}
    .atlas-stage .plot-wrap {{
      min-height: 560px;
    }}
    .atlas-card-toggle {{
      position: absolute;
      left: 22px;
      top: 22px;
      z-index: 4;
      appearance: none;
      border: 1px solid rgba(23,23,23,0.2);
      border-radius: var(--radius-sm);
      background: rgba(255,255,255,0.9);
      color: var(--ink);
      font: inherit;
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      padding: 9px 12px;
      cursor: pointer;
      box-shadow: 0 7px 22px rgba(23,23,23,0.12);
      transition: border-color 180ms ease, color 180ms ease, background-color 180ms ease;
    }}
    .atlas-card-toggle:hover {{
      border-color: var(--teal);
      color: var(--teal);
      background: #ffffff;
    }}
    .atlas-card {{
      position: absolute;
      left: 22px;
      top: 64px;
      width: min(360px, calc(100% - 44px));
      overflow: hidden;
      background: rgba(255,255,255,0.94);
      border: 1px solid rgba(23,23,23,0.16);
      border-radius: var(--radius-md);
      box-shadow: 0 16px 38px rgba(23,23,23,0.18);
      z-index: 3;
      backdrop-filter: blur(4px);
      transition: transform 260ms ease, opacity 220ms ease;
      transform: translateX(0);
      opacity: 1;
    }}
    .atlas-card.is-collapsed {{
      transform: translateX(calc(-100% - 18px));
      opacity: 0;
      pointer-events: none;
    }}
    .atlas-card-image {{
      min-height: 142px;
      background:
        linear-gradient(180deg, rgba(23,23,23,0.10), rgba(23,23,23,0.52)),
        linear-gradient(135deg, rgba(34,124,128,0.34), rgba(193,154,48,0.20));
      background-position: center;
      background-size: cover;
    }}
    .atlas-card-body {{
      padding: 18px 18px 16px;
    }}
    .atlas-card-kicker {{
      color: var(--teal);
      text-transform: uppercase;
      font-size: 11px;
      font-weight: 800;
      margin-bottom: 8px;
    }}
    .atlas-card h3 {{
      font-family: {DISPLAY_FONT} !important;
      font-size: 28px;
      line-height: 1.02;
      margin: 0 0 10px;
    }}
    .atlas-card p {{
      color: #353535;
      font-size: 14px;
      line-height: 1.4;
      margin: 0 0 14px;
    }}
    .atlas-card-stats {{
      display: grid;
      grid-template-columns: 0.8fr 1fr 1.2fr;
      gap: 1px;
      background: var(--line);
      border: 1px solid var(--line);
      margin-bottom: 14px;
    }}
    .atlas-card-stats div {{
      background: #ffffff;
      padding: 9px 10px;
      min-width: 0;
    }}
    .atlas-card-stats span {{
      display: block;
      color: var(--muted);
      text-transform: uppercase;
      font-size: 10px;
      font-weight: 800;
      margin-bottom: 4px;
    }}
    .atlas-card-stats strong {{
      display: block;
      font-size: 13px;
      line-height: 1.2;
      overflow-wrap: anywhere;
    }}
    .atlas-select-label {{
      display: block;
      color: var(--muted);
      text-transform: uppercase;
      font-size: 10px;
      font-weight: 800;
    }}
    .atlas-select-label select {{
      width: 100%;
      margin-top: 6px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: #ffffff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
      text-transform: none;
      padding: 8px 9px;
    }}
    .plot-wrap .js-plotly-plot {{
      width: 100%;
    }}
    .custom-timeline {{
      border: 1px solid rgba(23,23,23,0.14);
      border-top: 0;
      background:
        linear-gradient(90deg, rgba(34,124,128,0.055), rgba(255,255,255,0.96) 38%),
        #ffffff;
      padding: 16px 20px 16px;
      border-radius: 0 0 var(--radius-md) var(--radius-md);
    }}
    .timeline-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 18px;
      margin-bottom: 14px;
    }}
    .timeline-label {{
      color: var(--muted);
      font-size: 11px;
      line-height: 1;
      text-transform: uppercase;
      font-weight: 800;
      margin-bottom: 6px;
    }}
    .timeline-year {{
      font-family: {DISPLAY_FONT} !important;
      font-size: 38px;
      line-height: 0.9;
      color: var(--ink);
    }}
    .timeline-year span {{
      font-variant-numeric: tabular-nums;
    }}
    .timeline-actions {{
      display: flex;
      gap: 8px;
    }}
    .timeline-actions button {{
      appearance: none;
      border: 1px solid rgba(23,23,23,0.18);
      border-radius: var(--radius-sm);
      background: rgba(255,255,255,0.74);
      color: var(--ink);
      font: inherit;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      padding: 8px 13px;
      cursor: pointer;
    }}
    .timeline-actions button:hover {{
      border-color: var(--teal);
      color: var(--teal);
      background: #ffffff;
    }}
    .custom-timeline input[type="range"] {{
      --progress: 0%;
      appearance: none;
      width: 100%;
      height: 22px;
      margin: 0;
      background: transparent;
      cursor: pointer;
    }}
    .custom-timeline input[type="range"]::-webkit-slider-runnable-track {{
      height: 3px;
      border-radius: 999px;
      background:
        linear-gradient(90deg, var(--teal) 0 var(--progress), rgba(23,23,23,0.16) var(--progress) 100%);
    }}
    .custom-timeline input[type="range"]::-moz-range-track {{
      height: 3px;
      border-radius: 999px;
      background: rgba(23,23,23,0.16);
    }}
    .custom-timeline input[type="range"]::-moz-range-progress {{
      height: 3px;
      border-radius: 999px;
      background: var(--teal);
    }}
    .custom-timeline input[type="range"]::-webkit-slider-thumb {{
      appearance: none;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      border: 2px solid #ffffff;
      background: var(--teal);
      box-shadow: 0 2px 9px rgba(23,23,23,0.22);
      margin-top: -7.5px;
    }}
    .custom-timeline input[type="range"]::-moz-range-thumb {{
      width: 16px;
      height: 16px;
      border-radius: 50%;
      border: 2px solid #ffffff;
      background: var(--teal);
      box-shadow: 0 2px 9px rgba(23,23,23,0.22);
    }}
    .timeline-ticks {{
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
      font-variant-numeric: tabular-nums;
    }}
    .timeline-caption {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      margin-top: 10px;
    }}
    footer {{
      padding: 48px 0 62px;
      color: var(--muted);
      font-size: 14px;
      border-top: 1px solid rgba(23,23,23,0.08);
    }}
    body.presenter .page {{
      width: min(1360px, calc(100% - 128px));
    }}
    body.presenter .viz-block {{
      min-height: 100svh;
      padding-top: 24px;
      padding-bottom: 24px;
    }}
    body.presenter #chart-1 .section-kicker,
    body.presenter #chart-1 .section-head,
    body.presenter #chart-1 .plot-wrap,
    body.presenter #chart-1 .custom-timeline {{
      width: min(100%, 1260px);
    }}
    body.presenter .section-head {{
      margin-bottom: 10px;
    }}
    body.presenter .custom-timeline {{
      padding-top: 12px;
      padding-bottom: 12px;
    }}
    a {{ color: var(--teal); }}
    .reveal-ready .viz-block,
    .reveal-ready .support-intro {{
      opacity: 0;
      transform: translateY(26px);
      transition: opacity 620ms ease, transform 620ms cubic-bezier(0.2, 0.65, 0.15, 1);
      will-change: opacity, transform;
    }}
    .reveal-ready .viz-block.is-visible,
    .reveal-ready .support-intro.is-visible {{
      opacity: 1;
      transform: translateY(0);
    }}
    @media (prefers-reduced-motion: reduce) {{
      .reveal-ready .viz-block,
      .reveal-ready .support-intro {{
        opacity: 1;
        transform: none;
        transition: none;
      }}
    }}
    @media (max-width: 800px) {{
      .page {{ width: min(100% - 28px, 1240px); }}
      header {{ min-height: 78vh; padding-top: 34px; }}
      .metrics, .storyline, .section-head, .masthead {{
        grid-template-columns: 1fr;
      }}
      .masthead {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .masthead .links {{
        flex-wrap: wrap;
      }}
      .metric {{ min-height: auto; }}
      .atlas-block {{
        padding-left: 14px;
      }}
      .atlas-card {{
        position: relative;
        left: auto;
        top: auto;
        width: 100%;
        margin-top: 12px;
      }}
      .atlas-card-toggle {{
        position: relative;
        left: auto;
        top: auto;
        margin-top: 12px;
      }}
      .timeline-row {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .timeline-ticks span:nth-child(even) {{
        display: none;
      }}
      .plot-wrap {{
        padding: 2px;
        border-left: 0;
        border-right: 0;
        border-radius: var(--radius-sm);
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header style="--hero-bg: url('{bg_uri}')">
      <div class="masthead">
        <div class="mark">County Engines</div>
        <div class="links">
          <span>States</span>
          <span>Metro</span>
          <span>Industry</span>
          <span>Factors</span>
          <span>Atlas</span>
        </div>
      </div>
      <div class="eyebrow">State GDP, human capital, and county engines</div>
      <h1>Why some county economies break out.</h1>
      <p class="thesis">Some county economies do not rise because their state magically compounds. They rise when high-skill labor, command functions, and specialized industries concentrate in particular labor markets.</p>
      <div class="metric-definition">Breakout Score = change in county per-capita personal-income position from the 1969-1973 average to the 2020-2024 average, with the U.S. average equal to 100 each year.</div>
      <div class="metrics">
        <div class="metric">
          <div class="label">Outcome</div>
          <div class="value">Income</div>
          <div class="sub">BEA CAINC1 relative income-position change</div>
        </div>
        <div class="metric">
          <div class="label">Production</div>
          <div class="value">GDP</div>
          <div class="sub">BEA CAGDP1/CAGDP2 county output and industry</div>
        </div>
        <div class="metric">
          <div class="label">Human capital</div>
          <div class="value">ACS</div>
          <div class="sub">Bachelor's share and county traits</div>
        </div>
        <div class="metric">
          <div class="label">Metro class</div>
          <div class="value">RUCC</div>
          <div class="sub">USDA ERS rural-urban continuum codes</div>
        </div>
        <div class="metric">
          <div class="label">Model residuals</div>
          <div class="value">Logit</div>
          <div class="sub">L1 logistic breakout model + K-means county archetypes</div>
        </div>
      </div>
    </header>

    <section class="storyline">
      <div>
        <h2>The six-visual story</h2>
        <p>Counties move. States still differ. Metro scale and industry structure shape the odds, and the model helps rank which factors matter most before the atlas reveals specific winners and laggards.</p>
      </div>
      <div class="script">
        <p><strong>Opening claim:</strong> county engines, not state magic.</p>
        <p><strong>State spread:</strong> states differ, but counties inside a state still diverge.</p>
        <p><strong>Mechanism:</strong> metro scale and industry composition shape breakout potential.</p>
        <p><strong>Model lens:</strong> factor importance quantifies which traits carry signal.</p>
      </div>
    </section>

    {chart_sections}

    <section class="support-intro">
      <div class="section-kicker">Appendix</div>
      <h2>What supports the claim?</h2>
      <p>These diagnostics are useful for questions after the six-visual story. They deepen the core narrative without adding clutter to the main sequence.</p>
    </section>

    {support_sections}

    <footer>
      Sources: BEA CAINC1 county personal income, BEA CAGDP1 county GDP, BEA CAGDP2 county GDP by industry, Census ACS 2023 profile variables, USDA ERS 2023 Rural-Urban Continuum Codes, and county GeoJSON from Plotly datasets. County GDP per person combines workplace GDP with resident population, so treat it as an economic-intensity proxy rather than a household welfare measure.
    </footer>
  </main>
  <script>
    (function () {{
      const revealTargets = document.querySelectorAll(".viz-block, .support-intro");
      if (!revealTargets.length) return;
      if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {{
        revealTargets.forEach((el) => el.classList.add("is-visible"));
        return;
      }}
      document.body.classList.add("reveal-ready");
      const observer = new IntersectionObserver(
        (entries) => {{
          entries.forEach((entry) => {{
            if (!entry.isIntersecting) return;
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }});
        }},
        {{ rootMargin: "0px 0px -12% 0px", threshold: 0.18 }}
      );
      revealTargets.forEach((target) => observer.observe(target));
    }})();

    (function () {{
      const control = document.querySelector(".custom-timeline[data-plot='story-1']");
      const plot = document.getElementById("story-1");
      if (new URLSearchParams(window.location.search).get("present") === "1") {{
        document.body.classList.add("presenter");
      }}
      if (!control || !plot || !window.Plotly) return;

      const slider = control.querySelector("[data-year-slider]");
      const label = control.querySelector("[data-year-label]");
      const play = control.querySelector("[data-action='play']");
      const pause = control.querySelector("[data-action='pause']");
      let timer = null;

      function updateProgress() {{
        const min = Number(slider.min);
        const max = Number(slider.max);
        const value = Number(slider.value);
        const progress = ((value - min) / (max - min)) * 100;
        slider.style.setProperty("--progress", progress + "%");
      }}

      function animateTo(year) {{
        label.textContent = year;
        slider.value = String(year);
        updateProgress();
        Plotly.animate(
          plot,
          [String(year)],
          {{
            mode: "immediate",
            frame: {{ duration: 180, redraw: true }},
            transition: {{ duration: 80 }}
          }}
        );
      }}

      function stop() {{
        if (timer) {{
          window.clearInterval(timer);
          timer = null;
        }}
      }}

      slider.addEventListener("input", function () {{
        stop();
        animateTo(slider.value);
      }});

      play.addEventListener("click", function () {{
        stop();
        timer = window.setInterval(function () {{
          const next = Number(slider.value) >= Number(slider.max)
            ? Number(slider.min)
            : Number(slider.value) + 1;
          animateTo(next);
        }}, 280);
      }});

      pause.addEventListener("click", stop);
      updateProgress();
    }})();

    (function () {{
      const atlasStage = document.querySelector("[data-atlas-plot]");
      const atlasPlotId = atlasStage ? atlasStage.getAttribute("data-atlas-plot") : null;
      const plot = atlasPlotId ? document.getElementById(atlasPlotId) : null;
      const dataEl = document.getElementById("atlas-card-data");
      const card = document.querySelector("[data-atlas-card]");
      const toggle = document.querySelector("[data-atlas-toggle]");
      if (!plot || !dataEl || !card || !window.Plotly) return;

      const records = JSON.parse(dataEl.textContent || "[]");
      const byFips = new Map(records.map((record) => [String(record.fips), record]));
      const select = card.querySelector("[data-atlas-select]");
      const image = card.querySelector("[data-atlas-image]");
      const kicker = card.querySelector("[data-atlas-kicker]");
      const title = card.querySelector("[data-atlas-title]");
      const note = card.querySelector("[data-atlas-note]");
      const score = card.querySelector("[data-atlas-score]");
      const index = card.querySelector("[data-atlas-index]");
      const engine = card.querySelector("[data-atlas-engine]");

      function setCardOpen(isOpen) {{
        card.classList.toggle("is-collapsed", !isOpen);
        if (!toggle) return;
        toggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
        toggle.textContent = isOpen
          ? "Hide breakout county profile"
          : "Show breakout county profile";
      }}

      function fmt(value) {{
        return value > 0 ? "+" + value.toFixed(1) : value.toFixed(1);
      }}

      function render(fips, shouldOpen) {{
        const record = byFips.get(String(fips)) || records[0];
        if (!record) return;
        if (select) select.value = record.fips;
        image.style.backgroundImage =
          "linear-gradient(180deg, rgba(23,23,23,0.04), rgba(23,23,23,0.58)), url('" + record.image + "')";
        kicker.textContent = record.group + " · " + record.state;
        title.textContent = record.headline;
        note.textContent = record.note;
        score.textContent = fmt(Number(record.breakout));
        index.textContent = record.start + " -> " + record.end;
        engine.textContent = record.engine;
        if (shouldOpen) setCardOpen(true);
      }}

      if (select) {{
        select.addEventListener("change", function () {{
          render(select.value, true);
        }});
      }}
      if (toggle) {{
        toggle.addEventListener("click", function () {{
          const isCollapsed = card.classList.contains("is-collapsed");
          setCardOpen(isCollapsed);
        }});
      }}

      plot.on("plotly_click", function (event) {{
        const point = event && event.points && event.points[0];
        const fips = point && (point.customdata || point.location);
        if (fips) render(fips, true);
      }});

      setCardOpen(false);
      render(byFips.has("06085") ? "06085" : records[0] && records[0].fips, false);
    }})();
  </script>
</body>
</html>
"""


def main() -> None:
    (
        merged,
        latest,
        county,
        state_compounding,
        county_concentration,
        income_breakouts,
        gdp_breakouts,
        county_income_panel,
        county_gdp_index,
        county_qol,
    ) = enrich_data()
    html = make_html(
        merged,
        latest,
        county,
        state_compounding,
        county_concentration,
        income_breakouts,
        gdp_breakouts,
        county_income_panel,
        county_gdp_index,
    )
    OUT.mkdir(exist_ok=True)
    output = OUT / "story_dashboard.html"
    output.write_text(textwrap.dedent(html).strip(), encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote Clean Data/state_compounding_metrics.csv ({len(state_compounding):,} rows)")
    print(f"Wrote Clean Data/county_gdp_concentration.csv ({len(county_concentration):,} rows)")
    print(f"Wrote Clean Data/county_income_breakouts_1969_2024.csv ({len(income_breakouts):,} rows)")
    print(f"Wrote Clean Data/county_gdp_breakouts_2001_2024.csv ({len(gdp_breakouts):,} rows)")
    print(f"Wrote Clean Data/state_story_2023.csv ({len(latest):,} rows)")
    print(f"Wrote Clean Data/county_story_2023.csv ({len(county):,} rows)")
    print(f"Wrote Clean Data/county_quality_of_life_index.csv ({len(county_qol):,} rows)")


if __name__ == "__main__":
    main()
