"""
Combines all chart HTML files from Dashboard/charts/ into a single
self-contained HTML file that anyone can open in a browser.

Usage:  python export_dashboard.py
Output: Dashboard/dashboard_standalone.html
"""

import os

CHARTS_DIR = 'charts'
OUTPUT_FILE = 'dashboard_standalone.html'

CHARTS = [
    ("Interactive Choropleth Map",              "choropleth.html"),
    ("Animated Bubble Chart",                   "bubble_chart.html"),
    ("Correlation Heatmap",                     "correlation_heatmap.html"),
    ("Linked Brushing Dashboard",               "linked_brushing.html"),
    ("Top vs Bottom States by GDP per Capita",  "animated_bars.html"),
]

# Build the combined HTML
sections = []
for title, filename in CHARTS:
    filepath = os.path.join(CHARTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"WARNING: {filepath} not found, skipping")
        continue
    with open(filepath, 'r') as f:
        chart_html = f.read()
    sections.append(f"""
    <div class="chart-section">
      <h2>{title}</h2>
      <iframe srcdoc="{chart_html.replace('&', '&amp;').replace('"', '&quot;')}"
              width="100%" height="700" frameborder="0"
              sandbox="allow-scripts allow-same-origin"></iframe>
    </div>""")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>State Economic Indicators Dashboard</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: #f5f5f5; color: #333; }}
    header {{ background: #1a1a2e; color: white; text-align: center;
             padding: 30px 20px; }}
    header h1 {{ font-size: 2em; margin-bottom: 8px; }}
    header p {{ color: #aaa; font-size: 1.1em; }}
    .chart-section {{ background: white; margin: 24px auto; padding: 24px;
                      max-width: 1200px; border-radius: 8px;
                      box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    .chart-section h2 {{ margin-bottom: 16px; padding-bottom: 8px;
                         border-bottom: 2px solid #eee; color: #1a1a2e; }}
    footer {{ text-align: center; padding: 20px; color: #888; font-size: 0.9em; }}
  </style>
</head>
<body>
  <header>
    <h1>State Economic Indicators Dashboard</h1>
    <p>Interactive visualizations — hover, zoom, brush, and animate</p>
  </header>
  {"".join(sections)}
  <footer>All charts are fully interactive. Scroll down to see all visualizations.</footer>
</body>
</html>"""

with open(OUTPUT_FILE, 'w') as f:
    f.write(html)

size_kb = os.path.getsize(OUTPUT_FILE) / 1024
print(f"Exported: {OUTPUT_FILE} ({size_kb:.0f} KB)")
print(f"Contains {len(sections)} charts — just open in any browser!")
