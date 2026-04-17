"""
Education, income, GDP, and model residual diagnostics.

Outputs:
  - Clean Data/education_income_diagnostics.csv
  - Clean Data/education_income_partial_residuals.csv
  - Clean Data/residual_profile_summary.csv
  - Project Docs/education_income_report.md

Run from the project root:
  python Dashboard/analyze_education_income.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "Clean Data"
DOCS = ROOT / "Project Docs"


def weighted_corr(x: pd.Series, y: pd.Series, w: pd.Series) -> float:
    mask = x.notna() & y.notna() & w.notna()
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    w = w[mask].astype(float)
    xbar = np.average(x, weights=w)
    ybar = np.average(y, weights=w)
    cov = np.average((x - xbar) * (y - ybar), weights=w)
    vx = np.average((x - xbar) ** 2, weights=w)
    vy = np.average((y - ybar) ** 2, weights=w)
    return float(cov / np.sqrt(vx * vy))


def weighted_r2(y: pd.Series, pred: np.ndarray, w: pd.Series) -> float:
    return float(r2_score(y, pred, sample_weight=w))


def fit_ridge(df: pd.DataFrame, x_cols: list[str], y_col: str, weights: pd.Series) -> Pipeline:
    cat = [col for col in x_cols if col == "state_name"]
    num = [col for col in x_cols if col != "state_name"]
    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("prep", prep),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )
    model.fit(df[x_cols], df[y_col], model__sample_weight=weights)
    return model


def residualize(df: pd.DataFrame, col: str, controls: list[str], weights: pd.Series) -> np.ndarray:
    model = fit_ridge(df, controls, col, weights)
    return df[col].to_numpy() - model.predict(df[controls])


def main() -> None:
    df = pd.read_csv(CLEAN / "county_mobility_ml_dataset.csv")
    residuals = pd.read_csv(CLEAN / "county_mobility_residuals.csv")
    df = df.dropna(
        subset=[
            "bachelors_or_higher_pct",
            "income_index_2020_2024",
            "mobility_5yr_avg",
            "income_index_1969_1973",
            "log_population_1969_1973",
            "state_name",
            "population_2024",
            "county_gdp_per_capita",
            "pcpi_2024",
        ]
    ).copy()
    weights = np.sqrt(df["population_2024"])

    controls = ["income_index_1969_1973", "log_population_1969_1973", "state_name"]
    full_controls = controls + ["bachelors_or_higher_pct"]
    outcomes = ["income_index_2020_2024", "mobility_5yr_avg"]

    rows = []
    for outcome in outcomes:
        edu_model = fit_ridge(df, ["bachelors_or_higher_pct"], outcome, weights)
        control_model = fit_ridge(df, controls, outcome, weights)
        full_model = fit_ridge(df, full_controls, outcome, weights)
        rows.append(
            {
                "outcome": outcome,
                "raw_corr": df["bachelors_or_higher_pct"].corr(df[outcome]),
                "weighted_corr": weighted_corr(df["bachelors_or_higher_pct"], df[outcome], df["population_2024"]),
                "education_only_r2": weighted_r2(df[outcome], edu_model.predict(df[["bachelors_or_higher_pct"]]), weights),
                "controls_only_r2": weighted_r2(df[outcome], control_model.predict(df[controls]), weights),
                "controls_plus_education_r2": weighted_r2(df[outcome], full_model.predict(df[full_controls]), weights),
            }
        )
    diagnostics = pd.DataFrame(rows)
    diagnostics["incremental_education_r2"] = (
        diagnostics["controls_plus_education_r2"] - diagnostics["controls_only_r2"]
    )

    y_resid = residualize(df, "income_index_2020_2024", controls, weights)
    x_resid = residualize(df, "bachelors_or_higher_pct", controls, weights)
    partial = df[
        [
            "county_fips",
            "bea_county_name",
            "state_name",
            "population_2024",
            "bachelors_or_higher_pct",
            "income_index_2020_2024",
            "mobility_5yr_avg",
            "county_gdp_per_capita",
            "pcpi_2024",
        ]
    ].copy()
    partial["education_residual_after_controls"] = x_resid
    partial["income_residual_after_controls"] = y_resid

    residual_features = [
        "bachelors_or_higher_pct",
        "graduate_degree_pct",
        "management_science_arts_occupation_pct",
        "worked_from_home_pct",
        "broadband_pct",
        "poverty_pct",
        "county_gdp_per_capita",
    ]
    over = residuals.nlargest(50, "residual_mobility")
    under = residuals.nsmallest(50, "residual_mobility")
    middle = residuals[
        residuals["residual_mobility"].between(
            residuals["residual_mobility"].quantile(0.45),
            residuals["residual_mobility"].quantile(0.55),
        )
    ]
    profiles = []
    for name, group in [("overperformers", over), ("underperformers", under), ("middle", middle)]:
        row = {"group": name, "counties": len(group), "avg_residual": group["residual_mobility"].mean()}
        for col in residual_features:
            row[col] = group[col].mean()
        profiles.append(row)
    profile = pd.DataFrame(profiles)

    gdp_pcpi_corr = df["county_gdp_per_capita"].corr(df["pcpi_2024"])
    gdp_pcpi_weighted_corr = weighted_corr(df["county_gdp_per_capita"], df["pcpi_2024"], df["population_2024"])

    CLEAN.mkdir(exist_ok=True)
    diagnostics.to_csv(CLEAN / "education_income_diagnostics.csv", index=False)
    partial.to_csv(CLEAN / "education_income_partial_residuals.csv", index=False)
    profile.to_csv(CLEAN / "residual_profile_summary.csv", index=False)

    report = f"""# Education, Income, And Residual Diagnostics

## Education And Income

Education is strongly related to current county income, and still contributes after controlling for starting income, starting population, and state.

{diagnostics.to_markdown(index=False, floatfmt=".3f")}

Interpretation:

- For current income level, bachelor's share alone explains a large part of the cross-county pattern.
- For long-run mobility, education still matters, but less mechanically; some counties moved for reasons not captured by education alone.
- The residualized relationship controls for starting income, starting population, and state, so it asks whether higher-education counties are richer than expected among otherwise similar starting positions.

## GDP Per Capita vs Personal Income

County GDP per capita and county per-capita personal income are correlated, but they are not the same thing.

- Unweighted correlation: `{gdp_pcpi_corr:.3f}`
- Population-weighted correlation: `{gdp_pcpi_weighted_corr:.3f}`

GDP per capita measures production located in the county divided by residents. Personal income measures income received by residents. They diverge when people commute across county lines, when local production is capital/resource intensive, when profits accrue to owners elsewhere, or when transfers, dividends, rents, and retirement income matter.

## Residual Profiles

These averages compare the 50 biggest model overperformers, 50 biggest underperformers, and a middle band of counties.

{profile.to_markdown(index=False, floatfmt=".2f")}
"""
    DOCS.mkdir(exist_ok=True)
    (DOCS / "education_income_report.md").write_text(report, encoding="utf-8")
    print("Wrote Clean Data/education_income_diagnostics.csv")
    print("Wrote Clean Data/education_income_partial_residuals.csv")
    print("Wrote Clean Data/residual_profile_summary.csv")
    print("Wrote Project Docs/education_income_report.md")
    print(f"GDP per capita vs PCPI correlation: {gdp_pcpi_corr:.3f}")


if __name__ == "__main__":
    main()
