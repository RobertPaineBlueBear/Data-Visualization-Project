"""Archived: GDP-per-capita CAGR target for the county mobility Random Forest.

Superseded in April 2026 by an employment-CAGR target (BEA CAEMP25N, LineCode 10)
in Dashboard/build_story_dashboard.py::county_growth_prediction.

Why we moved away from GDP-per-capita CAGR:
  - GDP is measured at workplace; ACS population is resident — commuting makes the
    ratio noisy for small counties and for satellite counties of big metros.
  - A handful of oil/gas microcounties (e.g. Lea NM, McKenzie ND) dominate the tails
    even after winsorizing.
  - Employment growth is easier to narrate ("jobs added, not dollars per head") and
    maps more directly onto the compounding/agglomeration story the dashboard tells.

Kept here for reference — the construction below matches what shipped through
2026-04-19. Target window was 2001 → 2024 (23 years), log-ratio annualized,
winsorized at the 1st/99th percentile, min county population 5,000.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_gdp_cagr_target(gdp_breakouts: pd.DataFrame) -> pd.DataFrame:
    """Return a per-county frame with target column ``y`` = annualized log-CAGR of
    real GDP per capita from 2001 to 2024, winsorized at 1/99%."""
    raw = gdp_breakouts[[
        "county_fips", "bea_county_name", "state_name",
        "population_start", "gdp_per_capita_start", "gdp_index_start",
        "population_end", "gdp_per_capita_end",
    ]].copy()
    raw = raw[(raw["gdp_per_capita_start"] > 0) & (raw["gdp_per_capita_end"] > 0)]
    raw = raw[raw["population_start"] >= 5_000]
    years = 2024 - 2001
    raw["gdp_cagr_2001_2024"] = (
        np.log(raw["gdp_per_capita_end"]) - np.log(raw["gdp_per_capita_start"])
    ) / years
    lo_q, hi_q = raw["gdp_cagr_2001_2024"].quantile([0.01, 0.99])
    raw["gdp_cagr_2001_2024"] = raw["gdp_cagr_2001_2024"].clip(lo_q, hi_q)
    return raw.rename(columns={"gdp_cagr_2001_2024": "y"})
