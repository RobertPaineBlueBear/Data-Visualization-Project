# Data Viz Project — County Economic Mobility & Compounding

## Project Overview
Data visualization final project. The story: **state economic hierarchies are surprisingly stable, but counties move** — some break out of their starting position, others fall behind. The dashboard argues that GDP per capita (not total GDP) is the right prosperity metric, and that human capital clusters (education, income, R&D, headquarters density) explain most of the pattern.

Full narrative lives in `Project Docs/presentation_story.md`.

## The Deliverable

**`Dashboard/story_dashboard.html`** is the final visual. It is a standalone, self-contained HTML file built by `Dashboard/build_story_dashboard.py`. This is what gets presented — everything else is scaffolding.

Run the build from the project root:
```
python Dashboard/build_story_dashboard.py
```

The script reads from `Clean Data/` and `Raw Data/`, writes several summary CSVs back to `Clean Data/`, and emits the final `Dashboard/story_dashboard.html`.

## Directory Layout

```
Data Viz Project/
├── CLAUDE.md                    # this file
├── .gitignore
├── Dashboard/                   # all dashboard build scripts + output
│   ├── story_dashboard.html     # ← THE deliverable
│   ├── build_story_dashboard.py # builds story_dashboard.html
│   ├── analyze_county_mobility.py
│   ├── analyze_education_income.py
│   ├── analyze_industry_composition.py
│   ├── story_streamlit_app.py   # alt Streamlit view (not the final deliverable)
│   └── assets/                  # images used by the dashboard
├── Clean Data/                  # processed CSVs consumed by build scripts
├── Raw Data/                    # BEA, ACS, ERS, geojson, f500, etc.
├── Project Docs/
│   ├── presentation_story.md    # the narrative / storyboard
│   ├── county_mobility_model_report.md
│   ├── education_income_report.md
│   └── CS 329E DV Proposal-1.pdf / CS 329E DV Analysis.pdf
└── Archive/                     # superseded scratch notebooks and older dashboard iterations
```

## Data Sources

All raw inputs live in `Raw Data/`. Curl commands to reproduce downloads are documented at the top of `Dashboard/build_story_dashboard.py`.

| File | Source | Purpose |
|------|--------|---------|
| `CAINC1.zip` | BEA Regional | County per-capita personal income 1969–2024 (long-run mobility) |
| `CAGDP1.zip` | BEA Regional | County GDP 2001–2024 (production) |
| `CAGDP2.zip` / `CAGDP9.zip` | BEA Regional | County GDP by industry |
| `acs_state_2023_profile.json` | Census ACS 5-yr | State education / income / poverty / population |
| `acs_county_2023_profile.json` | Census ACS 5-yr | Same, county level |
| `acs_county_2023_extended_profile.json` | Census ACS 5-yr | Extended county variables (occupations, broadband, migration, etc.) |
| `geojson-counties-fips.json` | Plotly datasets | County polygons for choropleths |
| `2023-rural-urban-continuum-codes.csv` | USDA ERS | Rural/urban classification |
| `historical_state_population_by_year.csv` | Census | Population history |
| `f500_growth.xlsx` | Fortune | HQ density proxy |
| `ncses_*.xlsx` | NSF NCSES | R&D spending proxy |

## Key Clean Data Files

| File | Description |
|------|-------------|
| `county_income_breakouts_1969_2024.csv` | County per-capita income mobility (long-run headline) |
| `county_gdp_breakouts_2001_2024.csv` | County GDP mobility (production view) |
| `county_gdp_concentration.csv` | Top-1% county share of total county GDP over time |
| `county_quality_of_life_index.csv` | Composite QoL index by county |
| `state_compounding_metrics.csv` | Top-vs-bottom state GDP-per-capita gap over time |
| `state_story_2023.csv` / `county_story_2023.csv` | State/county 2023 cross-section for scatter views |
| `county_mobility_ml_dataset.csv` | Features + target for the Random Forest mobility model |
| `county_mobility_residuals.csv` | Actual − predicted mobility (the "surprise" view) |
| `county_mobility_clusters.csv` | Cluster assignments with features |
| `county_industry_composition.csv` | Industry mix by county |
| `merged_data.csv` | Legacy merged inputs still consumed by build scripts |

## Headline Findings (2023 data, excl. DC)

- Top-10 minus bottom-10 state GDP-per-capita gap: ~$26k in 2006 → ~$32k in 2023.
- Top-1% of counties: ~31.7% of county GDP in 2001 → ~33.3% in 2024.
- Bachelor's-or-higher vs. GDP per capita correlation: ~+0.66.
- Median household income vs. GDP per capita: ~+0.75.
- Poverty vs. GDP per capita: ~−0.57.
- County mobility model (test R²): ~0.05 with baseline features, ~0.57 with full county trait set. State label adds nothing once county traits are included.
- Biggest long-run per-capita income gainers (counties ≥50k pop): Marin CA, Monroe FL, San Mateo CA, Williamson TN, New York NY, Benton AR, Eagle CO, Santa Clara CA, Midland TX, San Francisco CA.
- Biggest relative droppers: Nye NV, Pulaski MO, Queens NY, Anchorage AK, Clayton GA.

## Technical Notes

- Mobility headline metric: change in county per-capita personal-income index from 1969–1973 avg to 2020–2024 avg, with U.S. avg = 100 each year. Smoothed endpoints — avoids letting one odd year drive the result.
- Mobility "surprise" metric: actual − Random Forest predicted. Good for finding unusual counties, harder to explain in a presentation.
- The model is **descriptive, not causal**: several explanatory features are measured near the end of the period.
- County GDP per person is a workplace-production intensity proxy — do not present it as household welfare (GDP is measured where work happens, population is where people live).
- The 1969 anchor is not fragile: 1969-only mobility correlates ~0.98 with the 1969–73 vs. 2020–24 smoothed version.
