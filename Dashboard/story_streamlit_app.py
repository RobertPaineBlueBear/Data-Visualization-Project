"""
Streamlit version of the county mobility story dashboard.

Run from the project root:
  streamlit run Dashboard/story_streamlit_app.py

This app reuses the Plotly figures from build_story_dashboard.py and does
not replace the standalone HTML dashboard.
"""

from __future__ import annotations

from pathlib import Path
import sys
import base64

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Dashboard"))

import build_story_dashboard as viz  # noqa: E402


st.set_page_config(
    page_title="Why Some County Economies Break Out",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


CSS = """
<style>
  :root {
    --ink: #171717;
    --muted: #686868;
    --paper: #fbfbfb;
    --line: #dedede;
    --red: #b43b45;
    --teal: #227c80;
    --gold: #c19a30;
  }

  html, body, [class*="css"] {
    font-family: "Avenir Next", Avenir, "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    color: var(--ink);
    background: var(--paper);
  }

  .stApp {
    background: var(--paper);
  }

  header[data-testid="stHeader"],
  div[data-testid="stToolbar"],
  div[data-testid="stDecoration"],
  div[data-testid="stStatusWidget"],
  .stDeployButton,
  #MainMenu {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
  }

  [data-testid="stAppViewContainer"] .block-container {
    max-width: 1120px;
    padding-left: clamp(32px, 8vw, 120px);
    padding-right: clamp(32px, 8vw, 120px);
    padding-top: 0;
  }

  .stMainBlockContainer,
  div[data-testid="stMainBlockContainer"] {
    max-width: 1120px;
    padding-left: clamp(32px, 8vw, 120px) !important;
    padding-right: clamp(32px, 8vw, 120px) !important;
    margin-left: auto;
    margin-right: auto;
  }

  div[data-testid="stMarkdownContainer"] p {
    font-size: 1.03rem;
    line-height: 1.55;
  }

  .story-eyebrow {
    color: var(--teal);
    text-transform: uppercase;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0;
    margin-bottom: 18px;
  }

  .site-masthead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    border-bottom: 1px solid rgba(23,23,23,0.13);
    padding: 18px 0 16px;
    margin: 0 auto 54px;
    width: min(100%, 980px);
    color: #252525;
    font-size: 13px;
  }

  .site-mark {
    font-family: "New York", Georgia, "Times New Roman", serif !important;
    font-size: 18px;
    letter-spacing: 0;
  }

  .site-links {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 16px;
    color: var(--muted);
    text-transform: uppercase;
    font-size: 11px;
    font-weight: 800;
  }

  .site-links span {
    white-space: nowrap;
  }

  .story-title {
    font-family: "New York", Georgia, "Times New Roman", serif !important;
    font-size: clamp(48px, 5.2vw, 76px);
    line-height: 0.97;
    font-weight: 400;
    letter-spacing: 0;
    margin: 0 0 22px;
    max-width: 880px;
  }

  .story-thesis {
    max-width: 780px;
    font-size: clamp(19px, 1.55vw, 23px);
    color: #303030;
    line-height: 1.35;
    margin-bottom: 32px;
    font-family: "Avenir Next", Avenir, "Helvetica Neue", Helvetica, Arial, sans-serif !important;
  }

  .hero-panel {
    position: relative;
    overflow: hidden;
    min-height: 72vh;
    width: min(100%, 980px);
    max-width: 980px;
    margin: 0 auto;
    padding: 0 0 42px;
    background-position: right -8px top 20px;
    background-repeat: no-repeat;
    background-size: min(66vw, 860px) auto;
  }

  .hero-panel::before {
    content: "";
    position: absolute;
    inset: 92px -70px 0 30%;
    z-index: 0;
    pointer-events: none;
    background:
      radial-gradient(circle at 28% 28%, rgba(193,154,48,0.22), transparent 30%),
      radial-gradient(circle at 72% 38%, rgba(34,124,128,0.20), transparent 36%),
      radial-gradient(circle at 50% 76%, rgba(180,59,69,0.12), transparent 34%);
    filter: blur(2px);
    opacity: 0.72;
  }

  .hero-panel > * {
    position: relative;
    z-index: 1;
  }

  .lens-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1px;
    background: var(--line);
    border: 1px solid var(--line);
    margin: 46px 0 8px;
    width: min(100%, 980px);
    max-width: 980px;
  }

  .lens-card {
    background: var(--paper);
    padding: 20px 20px 18px;
    min-height: 176px;
  }

  .lens-icon {
    width: 30px;
    height: 30px;
    margin-bottom: 18px;
    color: var(--teal);
  }

  .lens-icon svg {
    width: 30px;
    height: 30px;
    stroke: currentColor;
    fill: none;
    stroke-width: 1.7;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  .lens-label {
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    font-weight: 800;
    margin-bottom: 8px;
  }

  .lens-title {
    font-family: "New York", Georgia, "Times New Roman", serif !important;
    font-size: 27px;
    line-height: 1.05;
    margin-bottom: 10px;
  }

  .lens-copy {
    color: #353535;
    font-size: 15px;
    line-height: 1.38;
  }

  .lens-source {
    margin-top: 10px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.35;
  }

  .section-kicker {
    color: var(--red);
    font-size: 12px;
    text-transform: uppercase;
    font-weight: 800;
    margin: 34px 0 10px;
  }

  .section-title {
    font-family: "New York", Georgia, "Times New Roman", serif !important;
    font-size: clamp(30px, 3.5vw, 48px);
    line-height: 1.02;
    font-weight: 500;
    margin: 0 0 12px;
    max-width: 880px;
  }

  .section-note {
    color: #353535;
    font-size: 18px;
    max-width: 780px;
    line-height: 1.45;
    margin-bottom: 20px;
  }

  .metric-definition {
    border-left: 3px solid var(--teal);
    padding: 12px 16px;
    max-width: 860px;
    color: #303030;
    background: rgba(255,255,255,0.62);
    font-size: 16px;
    line-height: 1.45;
    margin: 0 0 16px;
  }

  .support-heading {
    border-top: 1px solid var(--line);
    padding-top: 36px;
    margin-top: 52px;
  }

  .support-heading .section-kicker {
    margin-top: 0;
  }

  .divider {
    border-top: 1px solid var(--line);
    margin: 42px 0 22px;
  }

  .source-note {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.45;
    margin-top: 28px;
  }

  .map-room {
    background: linear-gradient(90deg, rgba(34,124,128,0.08), rgba(255,255,255,0.78));
    color: var(--ink);
    border-left: 3px solid var(--teal);
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    padding: 18px 20px;
    margin: 40px 0 18px;
  }

  .map-room-top {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(180px, 260px);
    gap: 28px;
    align-items: start;
  }

  .map-room-kicker {
    color: var(--teal);
    text-transform: uppercase;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0;
    margin-bottom: 7px;
  }

  .map-room-title {
    font-family: "New York", Georgia, "Times New Roman", serif !important;
    font-size: clamp(28px, 3vw, 40px);
    line-height: 1.02;
    margin: 0;
  }

  .map-room-note {
    color: #353535;
    font-size: 14px;
    line-height: 1.45;
    max-width: 640px;
    margin-top: 10px;
  }

  .map-room-control {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.35;
    max-width: 260px;
    text-align: right;
    padding-top: 24px;
  }

  .map-room-caption {
    border-top: 1px solid rgba(244,241,232,0.16);
    color: var(--muted);
    font-size: 13px;
    line-height: 1.45;
    padding-top: 12px;
    margin: 10px 0 28px;
  }

  .map-room-caption strong {
    color: var(--ink);
    font-weight: 700;
  }

  button, .stButton > button {
    border-radius: 6px !important;
  }

  @media (max-width: 900px) {
    [data-testid="stAppViewContainer"] .block-container {
      padding-left: 18px;
      padding-right: 18px;
    }
    .lens-grid {
      grid-template-columns: 1fr;
    }
    .hero-panel {
      min-height: auto;
      background-size: 1100px auto;
      background-position: 30% top;
      padding-top: 36px;
    }
    .site-masthead {
      align-items: flex-start;
      flex-direction: column;
      margin-bottom: 36px;
    }
    .site-links {
      justify-content: flex-start;
    }
    .map-room {
      padding: 16px;
      margin-left: -4px;
      margin-right: -4px;
    }
    .map-room-top {
      grid-template-columns: 1fr;
    }
    .map-room-control {
      max-width: none;
      text-align: left;
    }
  }
</style>
"""


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    clean = ROOT / "Clean Data"
    raw = ROOT / "Raw Data"
    merged = pd.read_csv(clean / "merged_data.csv")
    merged["state_abbr"] = merged["state"].map(viz.STATE_ABBR)
    merged["rd_per_capita"] = merged["research_spending"] * 1000 / merged["population"]
    merged["f500_per_million"] = merged["f500_count"] / (merged["population"] / 1_000_000)

    return {
        "merged": merged,
        "latest": pd.read_csv(clean / "state_story_2023.csv"),
        "county": pd.read_csv(clean / "county_story_2023.csv", dtype={"county_fips": str}),
        "state_compounding": pd.read_csv(clean / "state_compounding_metrics.csv"),
        "county_concentration": pd.read_csv(clean / "county_gdp_concentration.csv"),
        "county_income_panel": viz.read_county_income_panel(raw / "CAINC1.zip"),
    }


def render_header(data: dict[str, pd.DataFrame]) -> None:
    bg_path = ROOT / "Dashboard" / "assets" / "economic_map_background.png"
    bg_uri = ""
    if bg_path.exists():
        bg_uri = "data:image/png;base64," + base64.b64encode(bg_path.read_bytes()).decode("ascii")

    def icon(kind: str) -> str:
        icons = {
            "income": '<svg viewBox="0 0 32 32"><path d="M5 22h22"/><path d="M8 22V11l8-4 8 4v11"/><path d="M12 22v-7"/><path d="M20 22v-7"/><path d="M11 27h10"/></svg>',
            "gdp": '<svg viewBox="0 0 32 32"><path d="M5 25h22"/><path d="M8 21l5-6 5 3 6-10"/><path d="M22 8h4v4"/></svg>',
            "education": '<svg viewBox="0 0 32 32"><path d="M4 11l12-5 12 5-12 5-12-5z"/><path d="M9 14v6c3 3 11 3 14 0v-6"/><path d="M28 11v8"/></svg>',
            "metro": '<svg viewBox="0 0 32 32"><path d="M5 25h22"/><path d="M8 25V11h6v14"/><path d="M18 25V7h6v18"/><path d="M10 15h2"/><path d="M20 12h2"/><path d="M20 17h2"/></svg>',
            "industry": '<svg viewBox="0 0 32 32"><path d="M5 25h22"/><path d="M7 25V14l6 4v-4l6 4v-7h6v14"/><path d="M10 22h2"/><path d="M16 22h2"/><path d="M22 22h2"/></svg>',
            "residual": '<svg viewBox="0 0 32 32"><path d="M5 25l22-18"/><circle cx="10" cy="20" r="2.5"/><circle cx="16" cy="16" r="2.5"/><circle cx="23" cy="10" r="2.5"/><path d="M7 8h6"/><path d="M10 5v6"/></svg>',
            "map": '<svg viewBox="0 0 32 32"><path d="M4 8l7-3 10 3 7-3v19l-7 3-10-3-7 3V8z"/><path d="M11 5v19"/><path d="M21 8v19"/></svg>',
        }
        return f'<div class="lens-icon">{icons[kind]}</div>'

    st.markdown(
        f"""
        <div class="hero-panel" style="background-image: url('{bg_uri}'); width: min(100%, 980px); max-width: 980px; margin-left: auto; margin-right: auto;">
          <div class="site-masthead">
            <div class="site-mark">County Engines</div>
            <div class="site-links">
              <span>Breakouts</span>
              <span>Atlas</span>
              <span>Metro</span>
              <span>Industry</span>
              <span>States</span>
            </div>
          </div>
          <div class="story-eyebrow">County mobility, state context, and human capital</div>
          <h1 class="story-title" style="font-family: 'New York', Georgia, 'Times New Roman', serif !important; font-weight: 400 !important; max-width: 880px;">Why some county economies break out.</h1>
          <div class="story-thesis">
            Some county economies do not rise because their state magically compounds. They rise when
            high-skill labor, command functions, and specialized industries concentrate in particular
            labor markets.
          </div>
          <div class="metric-definition">
            Mobility = change in county per-capita personal-income index from the 1969-1973 average
            to the 2020-2024 average, with the U.S. average equal to 100 each year.
          </div>
          <div class="lens-grid" style="width: min(100%, 980px); max-width: 980px;">
            <div class="lens-card">
              {icon("income")}
              <div class="lens-label">Outcome metric</div>
              <div class="lens-title">Personal income mobility</div>
              <div class="lens-copy">The scoreboard: resident income moving above or below the national line.</div>
              <div class="lens-source">BEA CAINC1 · 1969-1973 to 2020-2024</div>
            </div>
            <div class="lens-card">
              {icon("gdp")}
              <div class="lens-label">Production metrics</div>
              <div class="lens-title">County GDP and industry</div>
              <div class="lens-copy">Where production happens, and what kind of economy produces it.</div>
              <div class="lens-source">BEA CAGDP1/CAGDP2 · 2023-2024</div>
            </div>
            <div class="lens-card">
              {icon("education")}
              <div class="lens-label">Human capital metric</div>
              <div class="lens-title">Bachelor's share</div>
              <div class="lens-copy">The strongest observed signal for knowledge-intensive county economies.</div>
              <div class="lens-source">Census ACS 2023 · adults 25+ with bachelor's degree or higher</div>
            </div>
            <div class="lens-card">
              {icon("metro")}
              <div class="lens-label">Labor-market geography</div>
              <div class="lens-title">Metro class</div>
              <div class="lens-copy">Whether the county sits inside a large metro, small metro, or nonmetro economy.</div>
              <div class="lens-source">USDA ERS RUCC 2023</div>
            </div>
            <div class="lens-card">
              {icon("residual")}
              <div class="lens-label">Model metric</div>
              <div class="lens-title">Residual mobility</div>
              <div class="lens-copy">Actual mobility minus Random Forest predicted mobility: the surprise score.</div>
              <div class="lens-source">Random Forest descriptive model · R2 about 0.59</div>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )


def section(num: int, title: str, note: str, fig) -> None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="section-kicker">Visual {num}</div>
        <h2 class="section-title">{title}</h2>
        <div class="section-note">{note}</div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def support_section(title: str, note: str, fig) -> None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="section-kicker">Supporting Evidence</div>
        <h2 class="section-title">{title}</h2>
        <div class="section-note">{note}</div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    data = load_data()

    render_header(data)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-kicker">Visual 1</div>
        <h2 class="section-title">Counties Move</h2>
        <div class="section-note">
          The national state hierarchy is stable, but counties inside it can radically reorder.
          Move the year to watch local economies rise above or fall below their starting line.
        </div>
        """,
        unsafe_allow_html=True,
    )
    year = st.slider(
        "Year",
        min_value=1969,
        max_value=2024,
        value=2024,
        step=1,
        help="Per-capita personal income index, U.S. average = 100 each year.",
    )
    st.plotly_chart(
        viz.county_mobility_snapshot(data["county_income_panel"], year),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    st.markdown(
        """
        <div class="map-room">
          <div class="map-room-top">
            <div>
              <div class="map-room-kicker">Visual 2 · Breakout Atlas</div>
              <h2 class="map-room-title">The outliers have addresses.</h2>
              <div class="map-room-note">
                The biggest movers are not evenly distributed. Freeze the outliers and they become
                a geographic question: command centers, resource basins, resort counties, and places
                that missed the model.
              </div>
            </div>
            <div class="map-room-control">
              Switch the atlas between raw long-run mobility and model surprise.
            </div>
          </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    atlas_metric = st.segmented_control(
        "Atlas metric",
        ["Largest gains/losses", "Model residuals"],
        default="Largest gains/losses",
        help="Largest gains/losses uses smoothed income-index mobility; model residuals uses actual minus predicted mobility.",
    )
    st.plotly_chart(
        viz.county_breakout_atlas(
            "residual" if atlas_metric == "Model residuals" else "mobility",
            show_buttons=False,
        ),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )
    st.markdown(
        """
        <div class="map-room-caption">
          <strong>Reading the map:</strong> only selected breakout counties are colored.
          The basemap is context; the signal is the county outline that lights up.
        </div>
        """,
        unsafe_allow_html=True,
    )
    section(
        3,
        "Metro Lens",
        "Large labor markets are the main arena for upward breakout, but metro status alone is not destiny.",
        viz.metro_nonmetro_lens(),
    )
    section(
        4,
        "Industry Signature",
        "The breakout counties are disproportionately command centers: finance, information, professional services, headquarters, and research-heavy local economies.",
        viz.industry_composition_lens(),
    )
    section(
        5,
        "State Engines",
        "The best states are portfolios of county engines, not just one famous superstar county.",
        viz.state_engine_breakdown(data["county"], data["latest"]),
    )

    st.markdown(
        """
        <div class="support-heading">
          <div class="section-kicker">Appendix</div>
          <h2 class="section-title">What supports the claim?</h2>
          <div class="section-note">
            These diagnostics are useful for questions after the five-visual story. They explain why
            education, state context, and residual outliers are evidence, not the main narrative spine.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    support_section(
        "Education Gradient",
        "Education remains a strong income signal even after starting income, population, and state controls.",
        viz.education_income_diagnostics(),
    )
    support_section(
        "State Context",
        "States differ, but counties inside the same state often diverge. State context matters; it is not destiny.",
        viz.state_mobility_distributions(),
    )
    support_section(
        "Model Diagnostics",
        "The descriptive Random Forest asks whether observed county traits can explain the breakout pattern. Adding industry composition lifts R2 from about 0.57 to about 0.59.",
        viz.model_diagnostics(),
    )

    st.markdown(
        """
        <div class="source-note">
        Sources: BEA CAINC1 county personal income, BEA CAGDP1 county GDP, Census ACS 2023 profile variables,
        BEA CAGDP2 county GDP by industry, USDA ERS 2023 Rural-Urban Continuum Codes, existing cleaned state GDP dataset, and county GeoJSON from Plotly datasets. County GDP per person is
        local production divided by residents; personal income per person is income received by residents.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
