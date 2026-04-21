"""
County industry composition from BEA CAGDP2.

This script builds a county-level industry profile using current-dollar GDP by
industry. Current dollars are used for shares because chained-dollar industry
components should not be treated as exactly additive.

Outputs:
  - Clean Data/county_industry_composition.csv
  - Clean Data/county_industry_mobility_summary.csv

Run from the project root:
  python Dashboard/analyze_industry_composition.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Dashboard"))

from build_story_dashboard import STATE_FIPS  # noqa: E402


CLEAN = ROOT / "Clean Data"
RAW = ROOT / "Raw Data"


MAJOR_INDUSTRIES = {
    3: "agriculture",
    6: "mining",
    11: "construction",
    12: "manufacturing",
    34: "wholesale",
    35: "retail",
    36: "transportation",
    45: "information",
    50: "finance_real_estate",
    59: "professional_business",
    68: "education_health",
    75: "leisure_hospitality",
    82: "other_services",
    83: "government",
}

ENGINE_GROUPS = {
    "knowledge_command": ["information", "finance_real_estate", "professional_business"],
    "resource": ["agriculture", "mining"],
    "manufacturing": ["manufacturing"],
    "trade_logistics": ["wholesale", "retail", "transportation"],
    "education_health": ["education_health"],
    "leisure_amenity": ["leisure_hospitality"],
    "government": ["government"],
}

ENGINE_LABELS = {
    "knowledge_command": "Knowledge / command",
    "resource": "Resource extraction",
    "manufacturing": "Manufacturing",
    "trade_logistics": "Trade / logistics",
    "education_health": "Education / health",
    "leisure_amenity": "Leisure / amenity",
    "government": "Government",
}


def read_cagdp2(path: Path = RAW / "CAGDP2.zip", year: int = 2024) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        with zf.open("CAGDP2__ALL_AREAS_2001_2024.csv") as f:
            raw = pd.read_csv(f, dtype={"GeoFIPS": str}, encoding="latin1", low_memory=False)

    raw["GeoFIPS"] = raw["GeoFIPS"].str.replace('"', "", regex=False).str.strip()
    raw["value"] = pd.to_numeric(raw[str(year)].replace("(NA)", np.nan), errors="coerce")
    raw = raw[raw["LineCode"].isin([1, *MAJOR_INDUSTRIES.keys()])].copy()
    county_raw = raw[(raw["GeoFIPS"].str.len() == 5) & (~raw["GeoFIPS"].str.endswith("000"))].copy()

    wide = county_raw.pivot_table(
        index=["GeoFIPS", "GeoName"],
        columns="LineCode",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide = wide.rename(columns={"GeoFIPS": "county_fips", "GeoName": "bea_county_name", 1: "total_gdp_thousands"})
    for code, name in MAJOR_INDUSTRIES.items():
        wide[name] = pd.to_numeric(wide.get(code), errors="coerce")

    wide["state_fips"] = wide["county_fips"].str[:2]
    wide["state_name"] = wide["state_fips"].map(STATE_FIPS)
    wide = wide[wide["state_name"].notna()].copy()
    wide = wide.dropna(subset=["total_gdp_thousands"])
    wide = wide[wide["total_gdp_thousands"] > 0].copy()

    for name in MAJOR_INDUSTRIES.values():
        wide[f"{name}_share"] = wide[name] / wide["total_gdp_thousands"] * 100

    for group_name, industries in ENGINE_GROUPS.items():
        wide[f"{group_name}_share"] = wide[[f"{industry}_share" for industry in industries]].sum(axis=1)

    group_share_cols = [f"{group}_share" for group in ENGINE_GROUPS]
    wide["engine_type"] = wide[group_share_cols].idxmax(axis=1).str.replace("_share", "", regex=False)
    wide["engine_label"] = wide["engine_type"].map(ENGINE_LABELS)
    wide["engine_share"] = wide[group_share_cols].max(axis=1)

    industry_share_cols = [f"{name}_share" for name in MAJOR_INDUSTRIES.values()]
    wide["dominant_industry"] = wide[industry_share_cols].idxmax(axis=1).str.replace("_share", "", regex=False)
    wide["dominant_industry_share"] = wide[industry_share_cols].max(axis=1)

    us = raw[(raw["GeoFIPS"] == "00000") & raw["LineCode"].isin(MAJOR_INDUSTRIES.keys())].copy()
    if us.empty:
        national_shares = wide[industry_share_cols].mean()
    else:
        total_us = pd.to_numeric(
            raw[(raw["GeoFIPS"] == "00000") & (raw["LineCode"] == 1)]["value"],
            errors="coerce",
        ).iloc[0]
        national_shares = {
            f"{MAJOR_INDUSTRIES[int(row.LineCode)]}_share": row.value / total_us * 100
            for row in us.itertuples()
        }

    specialization = []
    for _, row in wide.iterrows():
        diffs = {col: row[col] - national_shares.get(col, np.nan) for col in industry_share_cols}
        best_col = max(diffs, key=lambda col: -np.inf if pd.isna(diffs[col]) else diffs[col])
        specialization.append(
            {
                "county_fips": row["county_fips"],
                "specialized_industry": best_col.replace("_share", ""),
                "specialization_gap": diffs[best_col],
            }
        )
    wide = wide.merge(pd.DataFrame(specialization), on="county_fips", how="left")

    keep = [
        "county_fips",
        "bea_county_name",
        "state_name",
        "total_gdp_thousands",
        "engine_type",
        "engine_label",
        "engine_share",
        "dominant_industry",
        "dominant_industry_share",
        "specialized_industry",
        "specialization_gap",
        *industry_share_cols,
        *group_share_cols,
    ]
    return wide[keep].copy()


def mobility_summary(industry: pd.DataFrame) -> pd.DataFrame:
    mobility_path = CLEAN / "county_mobility_ml_dataset.csv"
    if not mobility_path.exists():
        return pd.DataFrame()

    mobility = pd.read_csv(mobility_path, dtype={"county_fips": str})
    merged = mobility.merge(industry, on="county_fips", how="inner", suffixes=("", "_industry"))
    rows = []
    for label, group in merged.groupby("engine_label"):
        weights = group["population_2024"].clip(lower=1)
        bachelors = group["bachelors_or_higher_pct"].fillna(group["bachelors_or_higher_pct"].median())
        if bachelors.isna().all():
            bachelors = pd.Series(np.zeros(len(group)), index=group.index)
        rows.append(
            {
                "engine_label": label,
                "counties": len(group),
                "population_2024": weights.sum(),
                "pop_weighted_mobility": np.average(group["mobility_5yr_avg"], weights=weights),
                "median_mobility": group["mobility_5yr_avg"].median(),
                "p25_mobility": group["mobility_5yr_avg"].quantile(0.25),
                "p75_mobility": group["mobility_5yr_avg"].quantile(0.75),
                "avg_current_income_index": np.average(group["income_index_2020_2024"], weights=weights),
                "avg_bachelors": np.average(bachelors, weights=weights),
                "avg_engine_share": np.average(group["engine_share"], weights=weights),
            }
        )
    return pd.DataFrame(rows).sort_values("pop_weighted_mobility", ascending=False)


def main() -> None:
    CLEAN.mkdir(exist_ok=True)
    industry = read_cagdp2()
    summary = mobility_summary(industry)
    industry.to_csv(CLEAN / "county_industry_composition.csv", index=False)
    summary.to_csv(CLEAN / "county_industry_mobility_summary.csv", index=False)
    print(f"Wrote Clean Data/county_industry_composition.csv ({len(industry):,} rows)")
    print(f"Wrote Clean Data/county_industry_mobility_summary.csv ({len(summary):,} rows)")
    # Pre-period (2001) snapshot for the mobility ML features — avoids leaking
    # end-of-period composition into the growth-rate target.
    industry_2001 = read_cagdp2(year=2001)
    industry_2001.to_csv(CLEAN / "county_industry_composition_2001.csv", index=False)
    print(f"Wrote Clean Data/county_industry_composition_2001.csv ({len(industry_2001):,} rows)")


if __name__ == "__main__":
    main()
