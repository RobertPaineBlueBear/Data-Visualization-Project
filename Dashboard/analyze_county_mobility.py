"""
County mobility diagnostics and lightweight ML.

This script asks three questions:
  1. Is 1969 a fragile starting year?
  2. Which counties rose/fell using a more stable 1969-1973 baseline?
  3. Which county traits help explain mobility?

Outputs:
  - Clean Data/county_mobility_ml_dataset.csv
  - Clean Data/county_mobility_feature_importance.csv
  - Clean Data/county_mobility_residuals.csv
  - Clean Data/county_mobility_metro_summary.csv
  - Project Docs/county_mobility_model_report.md

Run from the project root:
  python Dashboard/analyze_county_mobility.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Dashboard"))

from build_story_dashboard import read_county_income_panel  # noqa: E402


CLEAN = ROOT / "Clean Data"
RAW = ROOT / "Raw Data"
DOCS = ROOT / "Project Docs"


def period_average(panel: pd.DataFrame, years: range, value: str, name: str) -> pd.DataFrame:
    return (
        panel[panel["year"].isin(list(years))]
        .groupby("county_fips", as_index=False)
        .agg(**{name: (value, "mean")})
    )


def build_dataset() -> tuple[pd.DataFrame, dict[str, float]]:
    income = read_county_income_panel(RAW / "CAINC1.zip")
    county_2023 = read_extended_county_features()
    rucc = read_rural_urban_codes()
    industry = read_industry_features()

    base = income[income["year"] == 1969][
        ["county_fips", "bea_county_name", "state_name", "population", "income_index_us_100"]
    ].rename(
        columns={
            "population": "population_1969",
            "income_index_us_100": "income_index_1969",
        }
    )
    current = income[income["year"] == 2024][
        ["county_fips", "population", "income_index_us_100", "per_capita_personal_income"]
    ].rename(
        columns={
            "population": "population_2024",
            "income_index_us_100": "income_index_2024",
            "per_capita_personal_income": "pcpi_2024",
        }
    )
    base_avg = period_average(income, range(1969, 1974), "income_index_us_100", "income_index_1969_1973")
    current_avg = period_average(income, range(2020, 2025), "income_index_us_100", "income_index_2020_2024")
    base_pop_avg = period_average(income, range(1969, 1974), "population", "population_1969_1973")

    df = (
        base.merge(current, on="county_fips")
        .merge(base_avg, on="county_fips")
        .merge(current_avg, on="county_fips")
        .merge(base_pop_avg, on="county_fips")
        .merge(county_2023, on="county_fips", how="left")
        .merge(rucc, on="county_fips", how="left")
        .merge(industry, on="county_fips", how="left")
    )
    df = df.dropna(
        subset=[
            "income_index_1969",
            "income_index_2024",
            "income_index_1969_1973",
            "income_index_2020_2024",
            "population_1969_1973",
            "population_2024",
        ]
    ).copy()
    df = df[df["population_2024"] >= 50_000].copy()
    df["mobility_1969_to_2024"] = df["income_index_2024"] - df["income_index_1969"]
    df["mobility_5yr_avg"] = df["income_index_2020_2024"] - df["income_index_1969_1973"]
    df["log_population_1969_1973"] = np.log(df["population_1969_1973"])
    df["population_growth_pct"] = (
        df["population_2024"] / df["population_1969_1973"].replace(0, np.nan) - 1
    ) * 100
    df["baseline_rank"] = df["income_index_1969_1973"].rank(ascending=False, method="min")
    df["current_rank"] = df["income_index_2020_2024"].rank(ascending=False, method="min")
    df["rank_gain"] = df["baseline_rank"] - df["current_rank"]

    diagnostics = {
        "single_vs_5yr_change_corr": df["mobility_1969_to_2024"].corr(df["mobility_5yr_avg"]),
        "baseline_1969_vs_5yr_corr": df["income_index_1969"].corr(df["income_index_1969_1973"]),
        "counties_modeled": float(len(df)),
    }
    return df, diagnostics


def read_rural_urban_codes() -> pd.DataFrame:
    path = RAW / "2023-rural-urban-continuum-codes.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "county_fips",
                "rucc_2023",
                "rucc_description",
                "rucc_group",
                "metro_status",
                "rucc_population_2020",
            ]
        )

    raw = pd.read_csv(path, dtype={"FIPS": str}, encoding="latin1")
    wide = (
        raw.pivot_table(
            index=["FIPS", "State", "County_Name"],
            columns="Attribute",
            values="Value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide = wide.rename(
        columns={
            "FIPS": "county_fips",
            "RUCC_2023": "rucc_2023",
            "Description": "rucc_description",
            "Population_2020": "rucc_population_2020",
        }
    )
    wide["county_fips"] = wide["county_fips"].str.zfill(5)
    wide["rucc_2023"] = pd.to_numeric(wide["rucc_2023"], errors="coerce")
    wide["rucc_population_2020"] = pd.to_numeric(wide["rucc_population_2020"], errors="coerce")
    wide = wide.dropna(subset=["rucc_2023"]).copy()
    wide["rucc_2023"] = wide["rucc_2023"].astype(int)
    wide["metro_status"] = np.where(wide["rucc_2023"].le(3), "Metro", "Nonmetro")
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
    wide["rucc_group"] = wide["rucc_2023"].map(group_map)
    return wide[
        [
            "county_fips",
            "rucc_2023",
            "rucc_description",
            "rucc_group",
            "metro_status",
            "rucc_population_2020",
        ]
    ]


def read_industry_features() -> pd.DataFrame:
    path = CLEAN / "county_industry_composition.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "county_fips",
                "engine_label",
                "engine_share",
                "knowledge_command_share",
                "resource_share",
                "manufacturing_share",
                "trade_logistics_share",
                "education_health_share",
                "leisure_amenity_share",
                "government_share",
                "specialization_gap",
            ]
        )

    cols = [
        "county_fips",
        "engine_label",
        "engine_share",
        "knowledge_command_share",
        "resource_share",
        "manufacturing_share",
        "trade_logistics_share",
        "education_health_share",
        "leisure_amenity_share",
        "government_share",
        "specialization_gap",
    ]
    df = pd.read_csv(path, dtype={"county_fips": str}, usecols=cols)
    numeric_cols = [col for col in cols if col not in {"county_fips", "engine_label"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_extended_county_features() -> pd.DataFrame:
    path = RAW / "acs_county_2023_extended_profile.json"
    if path.exists():
        with path.open() as f:
            rows = json.load(f)
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df = df.rename(
            columns={
                "DP02_0066PE": "graduate_degree_pct",
                "DP02_0067PE": "high_school_or_higher_pct",
                "DP02_0068PE": "bachelors_or_higher_pct",
                "DP02_0084PE": "moved_from_different_county_pct",
                "DP02_0094PE": "foreign_born_pct",
                "DP02_0154PE": "broadband_pct",
                "DP03_0009PE": "unemployment_pct",
                "DP03_0024PE": "worked_from_home_pct",
                "DP03_0025E": "mean_commute_minutes",
                "DP03_0027PE": "management_science_arts_occupation_pct",
                "DP03_0041PE": "professional_scientific_industry_pct",
                "DP03_0062E": "median_household_income",
                "DP03_0128PE": "poverty_pct",
                "DP04_0089E": "median_home_value",
                "DP05_0018E": "median_age",
                "DP05_0001E": "acs_population",
            }
        )
        feature_cols = [
            "graduate_degree_pct",
            "high_school_or_higher_pct",
            "bachelors_or_higher_pct",
            "moved_from_different_county_pct",
            "foreign_born_pct",
            "broadband_pct",
            "unemployment_pct",
            "worked_from_home_pct",
            "mean_commute_minutes",
            "management_science_arts_occupation_pct",
            "professional_scientific_industry_pct",
            "median_household_income",
            "poverty_pct",
            "median_home_value",
            "median_age",
            "acs_population",
        ]
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan
        df["county_fips"] = df["state"] + df["county"]
        df = df[["county_fips", *feature_cols]].copy()
    else:
        df = pd.read_csv(CLEAN / "county_story_2023.csv", dtype={"county_fips": str})
        df = df[
            [
                "county_fips",
                "bachelors_or_higher_pct",
                "median_household_income",
                "poverty_pct",
            ]
        ].copy()

    gdp = pd.read_csv(CLEAN / "county_story_2023.csv", dtype={"county_fips": str})[
        ["county_fips", "county_gdp_per_capita"]
    ]
    return df.merge(gdp, on="county_fips", how="left")


def fit_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    target = "mobility_5yr_avg"
    numeric_baseline = ["income_index_1969_1973", "log_population_1969_1973"]
    numeric_full = numeric_baseline + [
        "population_growth_pct",
        "graduate_degree_pct",
        "high_school_or_higher_pct",
        "bachelors_or_higher_pct",
        "moved_from_different_county_pct",
        "foreign_born_pct",
        "broadband_pct",
        "unemployment_pct",
        "worked_from_home_pct",
        "mean_commute_minutes",
        "management_science_arts_occupation_pct",
        "professional_scientific_industry_pct",
        "median_household_income",
        "poverty_pct",
        "median_home_value",
        "median_age",
        "county_gdp_per_capita",
        "knowledge_command_share",
        "resource_share",
        "manufacturing_share",
        "trade_logistics_share",
        "education_health_share",
        "leisure_amenity_share",
        "government_share",
        "specialization_gap",
    ]
    state_categorical = ["state_name"]
    geography_categorical = ["rucc_group", "engine_label"]
    full_categorical = state_categorical + geography_categorical

    model_df = df.dropna(subset=[target, *numeric_full, *full_categorical]).copy()
    train, test = train_test_split(model_df, test_size=0.25, random_state=42)

    def preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ]
        )

    baseline_model = Pipeline(
        steps=[
            ("prep", preprocessor(numeric_baseline, state_categorical)),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )
    no_state_model = Pipeline(
        steps=[
            ("prep", preprocessor(numeric_full, geography_categorical)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=600,
                    min_samples_leaf=8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    full_model = Pipeline(
        steps=[
            ("prep", preprocessor(numeric_full, full_categorical)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=600,
                    min_samples_leaf=8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    weights_train = np.sqrt(train["population_2024"])
    weights_test = np.sqrt(test["population_2024"])
    baseline_model.fit(train[numeric_baseline + state_categorical], train[target], model__sample_weight=weights_train)
    no_state_model.fit(train[numeric_full + geography_categorical], train[target], model__sample_weight=weights_train)
    full_model.fit(train[numeric_full + full_categorical], train[target], model__sample_weight=weights_train)

    baseline_pred = baseline_model.predict(test[numeric_baseline + state_categorical])
    no_state_pred = no_state_model.predict(test[numeric_full + geography_categorical])
    full_pred = full_model.predict(test[numeric_full + full_categorical])
    metrics = {
        "baseline_r2": r2_score(test[target], baseline_pred, sample_weight=weights_test),
        "baseline_mae": mean_absolute_error(test[target], baseline_pred, sample_weight=weights_test),
        "no_state_r2": r2_score(test[target], no_state_pred, sample_weight=weights_test),
        "no_state_mae": mean_absolute_error(test[target], no_state_pred, sample_weight=weights_test),
        "full_r2": r2_score(test[target], full_pred, sample_weight=weights_test),
        "full_mae": mean_absolute_error(test[target], full_pred, sample_weight=weights_test),
        "test_counties": float(len(test)),
    }

    full_model.fit(model_df[numeric_full + full_categorical], model_df[target], model__sample_weight=np.sqrt(model_df["population_2024"]))
    perm = permutation_importance(
        full_model,
        model_df[numeric_full + full_categorical],
        model_df[target],
        n_repeats=20,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": numeric_full + full_categorical,
            "importance_mae_reduction": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mae_reduction", ascending=False)

    model_df["predicted_mobility"] = full_model.predict(model_df[numeric_full + full_categorical])
    model_df["residual_mobility"] = model_df[target] - model_df["predicted_mobility"]
    residual_cols = [
        "county_fips",
        "bea_county_name",
        "state_name",
        "population_2024",
        "income_index_1969_1973",
        "income_index_2020_2024",
        "mobility_5yr_avg",
        "predicted_mobility",
        "residual_mobility",
        "rank_gain",
        "graduate_degree_pct",
        "bachelors_or_higher_pct",
        "management_science_arts_occupation_pct",
        "worked_from_home_pct",
        "broadband_pct",
        "poverty_pct",
        "county_gdp_per_capita",
        "rucc_group",
        "metro_status",
        "engine_label",
        "engine_share",
        "knowledge_command_share",
        "resource_share",
        "manufacturing_share",
        "trade_logistics_share",
        "education_health_share",
        "leisure_amenity_share",
        "government_share",
        "specialization_gap",
    ]
    residuals = model_df[residual_cols].sort_values("residual_mobility", ascending=False)
    return importance, residuals, metrics


def metro_mobility_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_df = df.dropna(subset=["rucc_group", "mobility_5yr_avg", "population_2024"]).copy()
    top_100_fips = set(summary_df.nlargest(100, "mobility_5yr_avg")["county_fips"])
    summary_df["is_top_100_mobility"] = summary_df["county_fips"].isin(top_100_fips)
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
    rows = []
    for group_name, group in summary_df.groupby("rucc_group"):
        weights = group["population_2024"].clip(lower=1)
        rows.append(
            {
                "rucc_group": group_name,
                "rucc_2023": int(group["rucc_2023"].iloc[0]),
                "metro_status": group["metro_status"].iloc[0],
                "counties": len(group),
                "population_2024": weights.sum(),
                "pop_weighted_mobility": np.average(group["mobility_5yr_avg"], weights=weights),
                "median_mobility": group["mobility_5yr_avg"].median(),
                "p25_mobility": group["mobility_5yr_avg"].quantile(0.25),
                "p75_mobility": group["mobility_5yr_avg"].quantile(0.75),
                "top_100_mobility_count": int(group["is_top_100_mobility"].sum()),
                "avg_bachelors": np.average(
                    group["bachelors_or_higher_pct"].fillna(group["bachelors_or_higher_pct"].median()),
                    weights=weights,
                ),
                "avg_current_income_index": np.average(group["income_index_2020_2024"], weights=weights),
            }
        )
    out = pd.DataFrame(rows)
    out["rucc_group"] = pd.Categorical(out["rucc_group"], categories=order, ordered=True)
    return out.sort_values("rucc_group")


def cluster_counties(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # For clustering, use the broad county coverage dataset so the clusters are
    # not dominated by missingness from sparse specialty variables.
    broad = pd.read_csv(CLEAN / "county_income_breakouts_1969_2024.csv", dtype={"county_fips": str})
    story = pd.read_csv(CLEAN / "county_story_2023.csv", dtype={"county_fips": str})
    broad = broad.rename(
        columns={
            "income_index_start": "income_index_1969_1973",
            "income_index_end": "income_index_2020_2024",
            "income_index_change": "mobility_5yr_avg",
            "population_change_pct": "population_growth_pct",
        }
    )
    broad = broad.merge(
        story[
            [
                "county_fips",
                "bachelors_or_higher_pct",
                "median_household_income",
                "poverty_pct",
                "county_gdp_per_capita",
                "acs_population",
            ]
        ],
        on="county_fips",
        how="left",
    )
    broad["population_2024"] = broad["population_end"]

    cluster_features = [
        "income_index_1969_1973",
        "mobility_5yr_avg",
        "population_growth_pct",
        "bachelors_or_higher_pct",
        "median_household_income",
        "county_gdp_per_capita",
        "poverty_pct",
    ]
    cluster_df = broad.dropna(subset=cluster_features).copy()
    # Prevent a single extreme county from becoming its own cluster.
    for col in cluster_features:
        lo, hi = cluster_df[col].quantile([0.01, 0.99])
        cluster_df[col] = cluster_df[col].clip(lo, hi)
    x = StandardScaler().fit_transform(cluster_df[cluster_features])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=30)
    cluster_df["cluster"] = kmeans.fit_predict(x)
    cluster_df["pca_1"] = coords[:, 0]
    cluster_df["pca_2"] = coords[:, 1]

    summary = (
        cluster_df.groupby("cluster", as_index=False)
        .agg(
            counties=("county_fips", "count"),
            avg_mobility=("mobility_5yr_avg", "mean"),
            avg_start_index=("income_index_1969_1973", "mean"),
            avg_current_index=("income_index_2020_2024", "mean"),
            avg_bachelors=("bachelors_or_higher_pct", "mean"),
            avg_median_income=("median_household_income", "mean"),
            avg_gdp_per_capita=("county_gdp_per_capita", "mean"),
            avg_poverty=("poverty_pct", "mean"),
            avg_population_growth=("population_growth_pct", "mean"),
        )
        .sort_values("avg_mobility", ascending=False)
    )
    labels = {}
    ordered = summary["cluster"].tolist()
    fallback = [
        "high-skill breakouts",
        "affluent incumbents",
        "growth metros",
        "stable middle",
        "relative decliners",
    ]
    for label, cluster in zip(fallback, ordered):
        labels[cluster] = label
    cluster_df["cluster_label"] = cluster_df["cluster"].map(labels)
    summary["cluster_label"] = summary["cluster"].map(labels)
    summary["counties_coverage_pct"] = summary["counties"] / broad["county_fips"].nunique() * 100
    summary["pca_explained_variance_1"] = pca.explained_variance_ratio_[0]
    summary["pca_explained_variance_2"] = pca.explained_variance_ratio_[1]
    return cluster_df, summary


def state_mobility_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for state, group in df.groupby("state_name"):
        weights = group["population_2024"]
        rows.append(
            {
                "state_name": state,
                "counties": len(group),
                "population_2024": weights.sum(),
                "pop_weighted_mobility": np.average(group["mobility_5yr_avg"], weights=weights),
                "median_mobility": group["mobility_5yr_avg"].median(),
                "p25_mobility": group["mobility_5yr_avg"].quantile(0.25),
                "p75_mobility": group["mobility_5yr_avg"].quantile(0.75),
                "min_mobility": group["mobility_5yr_avg"].min(),
                "max_mobility": group["mobility_5yr_avg"].max(),
                "within_state_sd": group["mobility_5yr_avg"].std(),
            }
        )
    summary = pd.DataFrame(rows).sort_values("pop_weighted_mobility", ascending=False)

    y = df["mobility_5yr_avg"]
    grand_mean = y.mean()
    state_mean = df.groupby("state_name")["mobility_5yr_avg"].transform("mean")
    state_r2 = ((state_mean - grand_mean) ** 2).sum() / ((y - grand_mean) ** 2).sum()

    weights = df["population_2024"]
    weighted_grand = np.average(y, weights=weights)
    weighted_state_mean = (
        df.groupby("state_name")[["mobility_5yr_avg", "population_2024"]]
        .apply(lambda g: np.average(g["mobility_5yr_avg"], weights=g["population_2024"]))
    )
    mapped = df["state_name"].map(weighted_state_mean)
    weighted_state_r2 = (
        (weights * (mapped - weighted_grand) ** 2).sum()
        / (weights * (y - weighted_grand) ** 2).sum()
    )
    metrics = {
        "state_only_r2": float(state_r2),
        "weighted_state_only_r2": float(weighted_state_r2),
    }
    return summary, metrics


def write_report(
    df: pd.DataFrame,
    importance: pd.DataFrame,
    residuals: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    state_summary: pd.DataFrame,
    metro_summary: pd.DataFrame,
    diagnostics: dict[str, float],
    metrics: dict[str, float],
    state_metrics: dict[str, float],
) -> None:
    top_gain = df.nlargest(10, "mobility_5yr_avg")
    top_drop = df.nsmallest(10, "mobility_5yr_avg")
    over = residuals.head(10)
    under = residuals.tail(10).sort_values("residual_mobility")

    def table_md(table: pd.DataFrame, cols: list[str]) -> str:
        return table[cols].to_markdown(index=False, floatfmt=".1f")

    report = f"""# County Mobility Model Report

## Baseline Check

Using one year is slightly noisy, but 1969 is not breaking the story.

- Correlation between 1969-only mobility and five-year-smoothed mobility: `{diagnostics['single_vs_5yr_change_corr']:.3f}`
- Correlation between the 1969 income index and the 1969-1973 average index: `{diagnostics['baseline_1969_vs_5yr_corr']:.3f}`
- Counties modeled after filtering to 2024 population >= 50,000: `{diagnostics['counties_modeled']:.0f}`

Recommendation: use the animation for intuition, but use the 1969-1973 average baseline in analysis and narration.

## Model

Target: change in county per-capita personal-income index from the 1969-1973 average to the 2020-2024 average, where the U.S. average equals 100 each year.

Baseline model: starting income index, starting population, and state.

Full model: baseline model plus richer 2023 ACS features, population growth, county GDP per person, USDA rural-urban class, and BEA county industry composition.

- Baseline test R2: `{metrics['baseline_r2']:.3f}`
- Baseline weighted MAE: `{metrics['baseline_mae']:.1f}` index points
- No-state full model test R2: `{metrics['no_state_r2']:.3f}`
- No-state full model weighted MAE: `{metrics['no_state_mae']:.1f}` index points
- Full model test R2: `{metrics['full_r2']:.3f}`
- Full model weighted MAE: `{metrics['full_mae']:.1f}` index points
- State-only unweighted R2: `{state_metrics['state_only_r2']:.3f}`
- State-only population-weighted R2: `{state_metrics['weighted_state_only_r2']:.3f}`

State matters if the full model meaningfully beats the no-state model. This is descriptive ML, not causal proof. Many full-model predictors are measured near the end of the period, so they help describe what breakout counties became, not necessarily what caused the breakout.

## Metro And Nonmetro Lens

USDA Rural-Urban Continuum Codes add a useful geography test: are breakout counties mainly in large metros, smaller metros, or nonmetro places?

{table_md(metro_summary, ['rucc_group', 'counties', 'population_2024', 'pop_weighted_mobility', 'median_mobility', 'p25_mobility', 'p75_mobility', 'top_100_mobility_count', 'avg_bachelors', 'avg_current_income_index'])}

## Feature Importance

Permutation importance is measured as the increase in absolute error when a feature is shuffled.

{table_md(importance.head(10), ['feature', 'importance_mae_reduction', 'importance_std'])}

## County Clusters

The cluster view is exploratory. It uses KMeans on standardized county traits and a PCA projection for visualization. It is useful for naming county types, not for proving causal mechanisms.

{table_md(cluster_summary, ['cluster_label', 'counties', 'avg_mobility', 'avg_start_index', 'avg_current_index', 'avg_bachelors', 'avg_gdp_per_capita', 'avg_poverty'])}

## State Mobility Differences

States differ in their county mobility distributions, but the within-state spread is also large. A state-only model explains part of the variation, not all of it.

{table_md(state_summary.head(12), ['state_name', 'counties', 'pop_weighted_mobility', 'median_mobility', 'p25_mobility', 'p75_mobility'])}

## Biggest Smoothed Mobility Gains

{table_md(top_gain, ['bea_county_name', 'state_name', 'population_2024', 'income_index_1969_1973', 'income_index_2020_2024', 'mobility_5yr_avg', 'rank_gain'])}

## Biggest Smoothed Mobility Drops

{table_md(top_drop, ['bea_county_name', 'state_name', 'population_2024', 'income_index_1969_1973', 'income_index_2020_2024', 'mobility_5yr_avg', 'rank_gain'])}

## Overperformers After The Model

These counties gained more than the model expected.

{table_md(over, ['bea_county_name', 'state_name', 'population_2024', 'mobility_5yr_avg', 'predicted_mobility', 'residual_mobility'])}

## Underperformers After The Model

These counties gained less, or fell more, than the model expected.

{table_md(under, ['bea_county_name', 'state_name', 'population_2024', 'mobility_5yr_avg', 'predicted_mobility', 'residual_mobility'])}
"""
    DOCS.mkdir(exist_ok=True)
    (DOCS / "county_mobility_model_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    df, diagnostics = build_dataset()
    importance, residuals, metrics = fit_models(df)
    clustered, cluster_summary = cluster_counties(df)
    state_summary, state_metrics = state_mobility_summary(df)
    metro_summary = metro_mobility_summary(df)
    CLEAN.mkdir(exist_ok=True)
    df.to_csv(CLEAN / "county_mobility_ml_dataset.csv", index=False)
    importance.to_csv(CLEAN / "county_mobility_feature_importance.csv", index=False)
    residuals.to_csv(CLEAN / "county_mobility_residuals.csv", index=False)
    clustered.to_csv(CLEAN / "county_mobility_clusters.csv", index=False)
    cluster_summary.to_csv(CLEAN / "county_mobility_cluster_summary.csv", index=False)
    state_summary.to_csv(CLEAN / "county_mobility_state_summary.csv", index=False)
    metro_summary.to_csv(CLEAN / "county_mobility_metro_summary.csv", index=False)
    write_report(
        df,
        importance,
        residuals,
        cluster_summary,
        state_summary,
        metro_summary,
        diagnostics,
        metrics,
        state_metrics,
    )
    print(f"Wrote Clean Data/county_mobility_ml_dataset.csv ({len(df):,} rows)")
    print("Wrote Clean Data/county_mobility_feature_importance.csv")
    print("Wrote Clean Data/county_mobility_residuals.csv")
    print("Wrote Clean Data/county_mobility_clusters.csv")
    print("Wrote Clean Data/county_mobility_cluster_summary.csv")
    print("Wrote Clean Data/county_mobility_state_summary.csv")
    print("Wrote Clean Data/county_mobility_metro_summary.csv")
    print("Wrote Project Docs/county_mobility_model_report.md")
    print(f"1969-only vs smoothed mobility correlation: {diagnostics['single_vs_5yr_change_corr']:.3f}")
    print(f"Full model test R2: {metrics['full_r2']:.3f}")


if __name__ == "__main__":
    main()
