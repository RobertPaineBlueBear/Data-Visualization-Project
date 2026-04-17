# Presentation Story: Why Some State Economies Compound

## Recommended Hypothesis

The best-performing states do not win mainly because they are large. They win because skilled labor, high household income, research capacity, and corporate command functions cluster in the same places. The worst-performing states tend to lack that reinforcing cluster: lower education levels, lower incomes, higher poverty, and thinner innovation or headquarters density.

Use GDP per capita as the main outcome. Total GDP mostly tells a scale story: California, Texas, New York, and Florida are large because they have large populations. GDP per capita is the cleaner prosperity question for a four-minute presentation.

## Stronger Revised Angle

The state hierarchy is surprisingly stable. That makes county mobility more interesting than state mobility. The better question may be: **which local economies escaped their starting position, and what kinds of places fell behind?**

Use BEA county per-capita personal income for the long-run version because it runs from 1969 to 2024. County GDP is still useful, but only from 2001 to 2024.

The 1969 starting point is not fragile. The 1969-only mobility score correlates about `0.98` with a five-year-smoothed version that compares 1969-1973 to 2020-2024. For narration, use the animation from 1969 because it is intuitive; for claims, cite the smoothed analysis.

The strongest dashboard sequence is now:

1. **The state gap is stable**: establish the puzzle.
2. **Mobility is geographic**: map where counties gained and lost ground.
3. **Counties move**: animate county income mobility from 1969 to 2024.
4. **States are built from county engines**: show whether high-GDP states are broad-based or dependent on one county.
5. **Human capital explains the pattern**: education, income, poverty, R&D, and headquarters density.

## The Four-Minute Narrative

1. Open with compounding: the top-state vs. bottom-state GDP-per-person dollar gap is wider than it was in 2006, and county GDP is slightly more concentrated than it was in 2001.
2. Pivot to mobility: states are stable, but counties can break out or fall behind.
3. Ask whether state success is broad-based or dependent on a single county.
4. State the trap: total GDP confuses size with performance.
5. Reframe the outcome: GDP per capita asks what economic environment each resident is standing inside.
6. Show the pattern: GDP per capita rises with education and median income, and falls with poverty.
7. Add mechanism: R&D and Fortune 500 density are proxies for innovation capacity and command functions.
8. Add granularity: counties reveal that state averages are built from concentrated local engines.
9. Close with caveat: this is not proof of causality, but it is a compelling descriptive model.

## Best Five Visuals To Present

Use visuals 1 through 5 in `Dashboard/story_dashboard.html`.

1. **Compounding gap**: top-state vs. bottom-state GDP per capita over time, with a county concentration toggle.
2. **Breakout atlas**: curated map of top/bottom county outliers, with a toggle between raw mobility and model residuals.
3. **County breakouts**: animated county income mobility from 1969 to 2024.
4. **State context**: states differ, but within-state county spread remains large.
5. **State engines**: largest county share vs. state GDP per capita, with a dropdown for top county contributors.

Visuals 6 and 7 support the "why" section: education-income diagnostics and model residuals.

## Current Evidence Snapshot

Using the 2023 state data excluding DC:

- Top-10 minus bottom-10 state GDP-per-capita gap: about `$26k` in 2006 and `$32k` in 2023.
- State p90/p10 GDP-per-capita ratio: roughly flat, about `1.59` in 2006 and `1.60` in 2023.
- County GDP concentration: the top 1% of counties rose from about `31.7%` of county GDP in 2001 to about `33.3%` in 2024.
- Long-run county data: BEA county per-capita personal income runs from 1969 to 2024.
- Bachelor's-or-higher share vs. GDP per capita: correlation about `+0.66`.
- Median household income vs. GDP per capita: correlation about `+0.75`.
- Poverty rate vs. GDP per capita: correlation about `-0.57`.
- Top five GDP-per-capita states: New York, Massachusetts, Washington, California, Delaware.
- Bottom five GDP-per-capita states: Mississippi, West Virginia, Arkansas, Idaho, Alabama.

## Breakout County Clues

Among counties with at least 50,000 people in 2024, the biggest long-run per-capita personal-income gainers relative to the U.S. average include Marin CA, Monroe FL, San Mateo CA, Williamson TN, New York NY, Benton AR, Eagle CO, Santa Clara CA, Midland TX, and San Francisco CA.

The biggest relative droppers include Nye NV, Pulaski MO, Queens NY, Anchorage AK, Clayton GA, Norfolk VA, Riley KS, Hardin KY, Fairbanks North Star AK, and Sutter CA.

For county GDP from 2001 to 2024, the largest gainers include San Mateo CA, Midland TX, San Francisco CA, St. Charles LA, Lea NM, Eddy NM, New York NY, Lincoln SD, Santa Clara CA, and Bradford PA.

## Breakout Metrics

Use the smoothed mobility metric as the headline definition of "breakout": change in county per-capita personal-income index from the 1969-1973 average to the 2020-2024 average, with the U.S. average set to 100 each year. It is easy to explain and avoids letting one odd year define the result.

Use the model residual as the secondary definition of "surprise": actual mobility minus Random Forest predicted mobility. This is better for finding counties that did unusually well or poorly after accounting for observed county traits, but it is harder to explain quickly.

Other possible breakout definitions:

- Rank gain: how far a county moved up or down the national county ranking.
- Percent growth in real per-capita personal income: intuitive, but less good for comparing counties because national income also rose.
- Decade-specific mobility: useful if the story becomes about a specific era, such as Sun Belt growth or post-2000 tech concentration.
- GDP-per-person mobility: useful for production engines, but only available from 2001 and more sensitive to commuting and capital-intensive industries.

## ML Mobility Notes

The county mobility model predicts change in county per-capita personal-income index from the 1969-1973 average to the 2020-2024 average. The U.S. average equals 100 each year.

- Baseline-only model using starting income, starting population, and state: test R2 about `0.05`.
- Fuller descriptive model with current education, income, poverty, labor, housing, migration, broadband, population growth, and county GDP per person: test R2 about `0.57`.
- Adding the state label does not improve test R2 once county traits are included. The no-state model and with-state model both score about `0.57`.
- A state-only view still matters: state averages explain about `0.17` of unweighted county mobility variance and about `0.14` when population-weighted.
- Most important descriptive signals: starting income index, bachelor's-or-higher share, county GDP per person, median age, unemployment, and management/science/arts occupations.

This model should be presented as descriptive, not causal, because several explanatory features are measured near the end of the period.

## State Context

Counties are not independent of their states. Massachusetts, New Hampshire, South Dakota, Arkansas, Washington, Vermont, and Tennessee have positive population-weighted county mobility. Alaska, Nevada, Hawaii, Michigan, Delaware, and Maryland are negative in the current model sample.

The important nuance is that state context is visible but incomplete. Some states have large internal spread: a high-performing state can contain weak counties, and a lower-performing state can contain breakout counties.

## State Engine Example

California is not just San Francisco. In 2023 county GDP:

- Los Angeles County contributed about `25.0%` of California county GDP.
- Santa Clara County contributed about `10.6%`.
- Orange County contributed about `8.7%`.
- San Diego County contributed about `8.2%`.
- San Francisco County contributed about `6.8%`.

The stronger point is that California has several large county engines, including both very large population engines and extremely high-output Bay Area counties.

## Small Pivots

- **From “best states” to “best state systems”**: define performance as high GDP per capita plus high median income plus low poverty.
- **From GDP to opportunity**: ask which states convert education into income most efficiently.
- **From winners to outliers**: identify states that outperform or underperform their education level.
- **From Fortune 500 count to headquarters density**: use Fortune 500 per million residents instead of raw count.

## Radical Pivots

- **County engines, not states**: argue that states are administrative containers and the real economy is county or metro clusters.
- **The anti-size story**: show that population explains total GDP but not resident prosperity.
- **Compounding institutions**: frame R&D, universities, headquarters, and educated labor as a self-reinforcing institutional flywheel.
- **Two economies inside one state**: use county inequality to show that a prosperous state can still contain weak local economies.

## Data Additions

- Added Census ACS 2023 five-year state and county profile data:
  - `DP02_0068PE`: adults with bachelor's degree or higher.
  - `DP03_0062E`: median household income.
  - `DP03_0128PE`: poverty rate.
  - `DP05_0001E`: population.
- Added BEA `CAGDP1` county GDP data for county-level GDP per person.
- Added richer ACS county profile variables for the mobility model, including graduate degree share, unemployment, work from home, broadband, median age, median home value, foreign-born share, migration, management/science occupations, and professional/scientific industry share.

County GDP per person is an intensity proxy. GDP is measured by workplace production, while ACS population is resident population, so do not present it as household welfare.
