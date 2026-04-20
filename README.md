# Data Viz Project

County-level economic mobility and compounding analysis for a data visualization final project.

The main deliverable is [`Dashboard/story_dashboard.html`](./Dashboard/story_dashboard.html), a standalone narrative dashboard built from the project data and analysis scripts.

## Project Layout

```text
Data Viz Project/
|- Dashboard/      # dashboard builders, analysis scripts, assets, final HTML
|- Clean Data/     # processed CSVs consumed by the dashboard
|- Raw Data/       # downloaded source data files
|- Project Docs/   # narrative, reports, and course PDFs
|- Archive/        # older notebooks, experiments, and legacy charts
|- CLAUDE.md       # detailed project notes and dataset descriptions
```

## Quick Start

Create a Python environment, install the dependencies, and run the dashboard build from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python Dashboard/build_story_dashboard.py
```

This rebuilds the derived CSV outputs in `Clean Data/` and writes the final dashboard to `Dashboard/story_dashboard.html`.

To view the alternate Streamlit version:

```bash
streamlit run Dashboard/story_streamlit_app.py
```

## Core Files

- `Dashboard/build_story_dashboard.py`: builds the final standalone HTML dashboard.
- `Dashboard/story_streamlit_app.py`: optional Streamlit version of the story.
- `Dashboard/analyze_county_mobility.py`: county mobility modeling and feature analysis.
- `Dashboard/analyze_education_income.py`: education and income relationship analysis.
- `Dashboard/analyze_industry_composition.py`: county industry composition analysis.
- `Project Docs/presentation_story.md`: the presentation narrative and storyboard.
- `CLAUDE.md`: fuller notes on the data sources, processed outputs, and headline findings.

## Data Notes

- `Raw Data/` contains source files from BEA, Census ACS, USDA ERS, Fortune, and NCSES.
- `Clean Data/` contains processed outputs used by the dashboard and supporting analyses.
- `Archive/` holds superseded notebooks and legacy visualizations that are kept for reference but are not part of the final build.

## Notes For Future Cleanup

- The project currently tracks both source data and processed outputs, which is fine for a self-contained class project but can make diffs noisy.
- `CLAUDE.md` contains the most complete project overview; this `README.md` is the shorter, repo-facing version.
