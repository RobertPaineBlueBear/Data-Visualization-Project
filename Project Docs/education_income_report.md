# Education, Income, And Residual Diagnostics

## Education And Income

Education is strongly related to current county income, and still contributes after controlling for starting income, starting population, and state.

| outcome                |   raw_corr |   weighted_corr |   education_only_r2 |   controls_only_r2 |   controls_plus_education_r2 |   incremental_education_r2 |
|:-----------------------|-----------:|----------------:|--------------------:|-------------------:|-----------------------------:|---------------------------:|
| income_index_2020_2024 |      0.737 |           0.761 |               0.568 |              0.606 |                        0.738 |                      0.132 |
| mobility_5yr_avg       |      0.462 |           0.582 |               0.266 |              0.194 |                        0.466 |                      0.271 |

Interpretation:

- For current income level, bachelor's share alone explains a large part of the cross-county pattern.
- For long-run mobility, education still matters, but less mechanically; some counties moved for reasons not captured by education alone.
- The residualized relationship controls for starting income, starting population, and state, so it asks whether higher-education counties are richer than expected among otherwise similar starting positions.

## GDP Per Capita vs Personal Income

County GDP per capita and county per-capita personal income are correlated, but they are not the same thing.

- Unweighted correlation: `0.681`
- Population-weighted correlation: `0.798`

GDP per capita measures production located in the county divided by residents. Personal income measures income received by residents. They diverge when people commute across county lines, when local production is capital/resource intensive, when profits accrue to owners elsewhere, or when transfers, dividends, rents, and retirement income matter.

## Residual Profiles

These averages compare the 50 biggest model overperformers, 50 biggest underperformers, and a middle band of counties.

| group           |   counties |   avg_residual |   bachelors_or_higher_pct |   graduate_degree_pct |   management_science_arts_occupation_pct |   worked_from_home_pct |   broadband_pct |   poverty_pct |   county_gdp_per_capita |
|:----------------|-----------:|---------------:|--------------------------:|----------------------:|-----------------------------------------:|-----------------------:|----------------:|--------------:|------------------------:|
| overperformers  |         50 |          23.07 |                     40.81 |                 16.11 |                                    43.91 |                  14.69 |           91.07 |         10.06 |                99014.21 |
| underperformers |         50 |         -16.12 |                     40.89 |                 17.22 |                                    45.30 |                  15.29 |           91.58 |         11.00 |                87254.50 |
| middle          |         98 |          -0.59 |                     28.78 |                 10.90 |                                    37.38 |                  10.36 |           88.18 |         13.31 |                60026.38 |
