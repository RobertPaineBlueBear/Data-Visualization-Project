# County Mobility Model Report

## Baseline Check

Using one year is slightly noisy, but 1969 is not breaking the story.

- Correlation between 1969-only mobility and five-year-smoothed mobility: `0.980`
- Correlation between the 1969 income index and the 1969-1973 average index: `0.984`
- Counties modeled after filtering to 2024 population >= 50,000: `1000`

Recommendation: use the animation for intuition, but use the 1969-1973 average baseline in analysis and narration.

## Model

Target: change in county per-capita personal-income index from the 1969-1973 average to the 2020-2024 average, where the U.S. average equals 100 each year.

Baseline model: starting income index, starting population, and state.

Full model: baseline model plus richer 2023 ACS features, population growth, county GDP per person, USDA rural-urban class, and BEA county industry composition.

- Baseline test R2: `0.048`
- Baseline weighted MAE: `14.6` index points
- No-state full model test R2: `0.593`
- No-state full model weighted MAE: `8.4` index points
- Full model test R2: `0.593`
- Full model weighted MAE: `8.4` index points
- State-only unweighted R2: `0.171`
- State-only population-weighted R2: `0.136`

State matters if the full model meaningfully beats the no-state model. This is descriptive ML, not causal proof. Many full-model predictors are measured near the end of the period, so they help describe what breakout counties became, not necessarily what caused the breakout.

## Metro And Nonmetro Lens

USDA Rural-Urban Continuum Codes add a useful geography test: are breakout counties mainly in large metros, smaller metros, or nonmetro places?

| rucc_group               |   counties |   population_2024 |   pop_weighted_mobility |   median_mobility |   p25_mobility |   p75_mobility |   top_100_mobility_count |   avg_bachelors |   avg_current_income_index |
|:-------------------------|-----------:|------------------:|------------------------:|------------------:|---------------:|---------------:|-------------------------:|----------------:|---------------------------:|
| Metro, 1M+               |        320 |       187806658.0 |                     3.0 |               1.3 |           -9.5 |           12.2 |                       58 |            39.8 |                      111.5 |
| Metro, 250k-1M           |        239 |        64186192.0 |                    -3.6 |              -4.5 |          -11.8 |            4.6 |                       22 |            32.8 |                       89.3 |
| Metro, <250k             |        204 |        26336857.0 |                    -4.0 |              -4.1 |          -13.4 |            3.0 |                        8 |            30.1 |                       84.3 |
| Nonmetro urban, adjacent |        136 |        10711065.0 |                    -2.8 |              -4.1 |          -11.1 |            4.4 |                        7 |            24.3 |                       78.1 |
| Nonmetro urban, remote   |         36 |         2641845.0 |                    -2.8 |              -2.7 |           -8.4 |            5.9 |                        3 |            27.5 |                       81.7 |
| Nonmetro town, adjacent  |         37 |         2160630.0 |                    -1.2 |              -2.7 |           -8.9 |            6.1 |                        2 |            22.6 |                       77.2 |
| Nonmetro town, remote    |          6 |          360137.0 |                     3.2 |               5.0 |            0.2 |            7.5 |                        0 |            27.0 |                       80.9 |
| Nonmetro rural, adjacent |          4 |          237589.0 |                     5.6 |               8.2 |            3.4 |           10.9 |                        0 |            22.3 |                       78.3 |

## Feature Importance

Permutation importance is measured as the increase in absolute error when a feature is shuffled.

| feature                                |   importance_mae_reduction |   importance_std |
|:---------------------------------------|---------------------------:|-----------------:|
| income_index_1969_1973                 |                        6.4 |              0.1 |
| bachelors_or_higher_pct                |                        3.7 |              0.1 |
| management_science_arts_occupation_pct |                        0.7 |              0.0 |
| county_gdp_per_capita                  |                        0.6 |              0.0 |
| unemployment_pct                       |                        0.6 |              0.0 |
| government_share                       |                        0.6 |              0.0 |
| median_age                             |                        0.5 |              0.0 |
| median_home_value                      |                        0.5 |              0.0 |
| median_household_income                |                        0.4 |              0.0 |
| resource_share                         |                        0.2 |              0.0 |

## County Clusters

The cluster view is exploratory. It uses KMeans on standardized county traits and a PCA projection for visualization. It is useful for naming county types, not for proving causal mechanisms.

| cluster_label        |   counties |   avg_mobility |   avg_start_index |   avg_current_index |   avg_bachelors |   avg_gdp_per_capita |   avg_poverty |
|:---------------------|-----------:|---------------:|------------------:|--------------------:|----------------:|---------------------:|--------------:|
| high-skill breakouts |        452 |           13.1 |              95.6 |               113.3 |            39.5 |              82699.2 |           9.0 |
| affluent incumbents  |        984 |            7.1 |              59.9 |                67.1 |            17.0 |              44017.6 |          20.0 |
| growth metros        |       1619 |           -3.1 |              85.2 |                82.0 |            23.7 |              69869.2 |          12.2 |

## State Mobility Differences

States differ in their county mobility distributions, but the within-state spread is also large. A state-only model explains part of the variation, not all of it.

| state_name           |   counties |   pop_weighted_mobility |   median_mobility |   p25_mobility |   p75_mobility |
|:---------------------|-----------:|------------------------:|------------------:|---------------:|---------------:|
| Massachusetts        |         12 |                    22.0 |               9.9 |            3.9 |           28.5 |
| New Hampshire        |          8 |                    18.6 |              13.1 |           11.0 |           19.6 |
| District of Columbia |          1 |                    18.2 |              18.2 |           18.2 |           18.2 |
| South Dakota         |          3 |                    15.4 |              15.0 |           12.1 |           20.5 |
| Arkansas             |         13 |                    14.6 |               0.6 |           -3.2 |            3.7 |
| Washington           |         21 |                    12.9 |             -10.8 |          -15.0 |           -3.9 |
| Vermont              |          5 |                    11.6 |               8.2 |            4.4 |           16.5 |
| Tennessee            |         33 |                    11.6 |               5.9 |           -2.4 |           12.4 |
| Montana              |          6 |                     9.4 |               7.0 |           -9.9 |           12.4 |
| Louisiana            |         21 |                     9.2 |               7.5 |            5.0 |           15.4 |
| Colorado             |         15 |                     8.2 |               4.3 |           -8.7 |           24.7 |
| Maine                |          9 |                     7.5 |               0.6 |           -4.3 |           11.0 |

## Biggest Smoothed Mobility Gains

| bea_county_name   | state_name   |   population_2024 |   income_index_1969_1973 |   income_index_2020_2024 |   mobility_5yr_avg |   rank_gain |
|:------------------|:-------------|------------------:|-------------------------:|-------------------------:|-------------------:|------------:|
| San Mateo, CA     | California   |          742893.0 |                    143.5 |                    248.6 |              105.1 |         8.0 |
| Marin, CA         | California   |          256400.0 |                    153.0 |                    255.0 |              102.0 |         6.0 |
| New York, NY      | New York     |         1660664.0 |                    199.6 |                    295.7 |               96.1 |         0.0 |
| Santa Clara, CA   | California   |         1926325.0 |                    119.9 |                    214.7 |               94.9 |        42.0 |
| Williamson, TN    | Tennessee    |          269136.0 |                     90.6 |                    181.5 |               90.9 |       457.0 |
| Benton, AR        | Arkansas     |          321566.0 |                     71.3 |                    159.6 |               88.4 |       855.0 |
| Midland, TX       | Texas        |          183587.0 |                    113.8 |                    197.5 |               83.7 |        72.0 |
| San Francisco, CA | California   |          827526.0 |                    153.4 |                    235.5 |               82.1 |         3.0 |
| Monroe, FL        | Florida      |           80908.0 |                    101.0 |                    182.3 |               81.3 |       207.0 |
| Eagle, CO         | Colorado     |           54330.0 |                     91.4 |                    172.2 |               80.8 |       434.0 |

## Biggest Smoothed Mobility Drops

| bea_county_name                | state_name   |   population_2024 |   income_index_1969_1973 |   income_index_2020_2024 |   mobility_5yr_avg |   rank_gain |
|:-------------------------------|:-------------|------------------:|-------------------------:|-------------------------:|-------------------:|------------:|
| Nye, NV                        | Nevada       |           55990.0 |                    137.6 |                     65.3 |              -72.3 |      -938.0 |
| Pulaski, MO                    | Missouri     |           53964.0 |                    117.1 |                     69.7 |              -47.4 |      -819.0 |
| Anchorage Municipality, AK     | Alaska       |          289600.0 |                    156.5 |                    114.3 |              -42.2 |      -102.0 |
| Clayton, GA                    | Georgia      |          297703.0 |                     93.6 |                     52.0 |              -41.6 |      -616.0 |
| Matanuska-Susitna Borough, AK  | Alaska       |          117613.0 |                    128.5 |                     87.2 |              -41.3 |      -373.0 |
| Queens, NY                     | New York     |         2316841.0 |                    127.9 |                     87.1 |              -40.8 |      -375.0 |
| Prince George's, MD            | Maryland     |          966629.0 |                    122.6 |                     82.5 |              -40.1 |      -484.0 |
| Lyon, NV                       | Nevada       |           63718.0 |                    110.3 |                     72.3 |              -38.1 |      -699.0 |
| Sutter, CA                     | California   |           98545.0 |                    117.9 |                     80.3 |              -37.6 |      -527.0 |
| Norfolk (Independent City), VA | Virginia     |          231105.0 |                    107.9 |                     71.4 |              -36.5 |      -711.0 |

## Overperformers After The Model

These counties gained more than the model expected.

| bea_county_name   | state_name   |   population_2024 |   mobility_5yr_avg |   predicted_mobility |   residual_mobility |
|:------------------|:-------------|------------------:|-------------------:|---------------------:|--------------------:|
| Monroe, FL        | Florida      |           80908.0 |               81.3 |                 19.2 |                62.2 |
| Benton, AR        | Arkansas     |          321566.0 |               88.4 |                 28.6 |                59.7 |
| Midland, TX       | Texas        |          183587.0 |               83.7 |                 25.5 |                58.2 |
| Marin, CA         | California   |          256400.0 |              102.0 |                 44.8 |                57.2 |
| San Mateo, CA     | California   |          742893.0 |              105.1 |                 57.6 |                47.5 |
| Eagle, CO         | Colorado     |           54330.0 |               80.8 |                 35.0 |                45.8 |
| Williamson, TN    | Tennessee    |          269136.0 |               90.9 |                 47.4 |                43.5 |
| Walton, FL        | Florida      |           89666.0 |               64.5 |                 22.9 |                41.6 |
| Indian River, FL  | Florida      |          172139.0 |               51.9 |                 13.8 |                38.1 |
| New York, NY      | New York     |         1660664.0 |               96.1 |                 58.6 |                37.5 |

## Underperformers After The Model

These counties gained less, or fell more, than the model expected.

| bea_county_name            | state_name           |   population_2024 |   mobility_5yr_avg |   predicted_mobility |   residual_mobility |
|:---------------------------|:---------------------|------------------:|-------------------:|---------------------:|--------------------:|
| Arlington, VA              | Virginia             |          239807.0 |               -8.9 |                 47.8 |               -56.7 |
| Nye, NV                    | Nevada               |           55990.0 |              -72.3 |                -29.6 |               -42.8 |
| Montgomery, MD             | Maryland             |         1082273.0 |              -27.0 |                  7.1 |               -34.1 |
| Anchorage Municipality, AK | Alaska               |          289600.0 |              -42.2 |                -19.4 |               -22.8 |
| Riley, KS                  | Kansas               |           72557.0 |              -28.3 |                 -6.3 |               -22.1 |
| Durham, NC                 | North Carolina       |          343628.0 |                0.1 |                 22.0 |               -21.9 |
| District of Columbia, DC   | District of Columbia |          702250.0 |               18.2 |                 39.9 |               -21.8 |
| Madison, ID                | Idaho                |           55549.0 |              -19.8 |                  1.9 |               -21.7 |
| Pulaski, MO                | Missouri             |           53964.0 |              -47.4 |                -26.7 |               -20.7 |
| Douglas, CO                | Colorado             |          393995.0 |               15.3 |                 33.7 |               -18.4 |
