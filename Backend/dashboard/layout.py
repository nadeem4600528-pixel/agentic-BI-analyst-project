"""Layout helpers for a Power BI / Tableau style dashboard.

Groups the generated charts into named business sections and gives each
chart a span hint so the frontend can arrange a responsive grid:
  - "wide"  -> full width
  - "half"  -> half width
  - "third" -> one third
"""

from __future__ import annotations

from typing import Any, Dict, List


# Default placement rules by chart id / type.
_SECTION_RULES = [
    ("trends", {"trend_line", "trend_area", "growth_bars", "stacked_trend"}, "Trends Over Time"),
    ("performance", {"gauge", "kpi_trend"}, "Performance vs Target"),
    ("composition", {"donut", "treemap", "pareto", "bar", "category_bar"}, "Composition & Ranking"),
    ("distribution", {"box", "histogram", "scatter", "grouped_bar", "heatmap"}, "Distribution & Relationships"),
]

_DEFAULT_SPANS = {
    "trend_area": "wide",
    "trend_line": "wide",
    "growth_bars": "wide",
    "stacked_trend": "wide",
    "treemap": "wide",
    "pareto": "wide",
    "heatmap": "half",
    "donut": "third",
    "gauge": "third",
    "box": "half",
    "histogram": "half",
    "scatter": "half",
    "bar": "half",
    "category_bar": "half",
    "grouped_bar": "half",
}


def build_dashboard_layout(charts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a sectioned, responsive layout description for the frontend."""
    sections: List[Dict[str, Any]] = []
    placed = set()

    for section_id, ids, title in _SECTION_RULES:
        chart_ids = [
            chart.get("id") for chart in charts
            if chart.get("id") in ids and chart.get("id") not in placed
        ]
        # also place by explicit type fallback
        for chart in charts:
            cid = chart.get("id")
            ctype = chart.get("type")
            if cid in placed:
                continue
            if cid in ids or ctype in ids:
                if cid not in chart_ids:
                    chart_ids.append(cid)
        if chart_ids:
            sections.append({
                "id": section_id,
                "title": title,
                "chart_ids": chart_ids,
            })
            placed.update(chart_ids)

    # Any charts not captured go into a generic section.
    leftover = [chart.get("id") for chart in charts if chart.get("id") not in placed]
    if leftover:
        sections.append({"id": "more", "title": "More Analytics", "chart_ids": leftover})

    # Attach span hints per chart.
    spans = {}
    for chart in charts:
        cid = chart.get("id")
        spans[cid] = chart.get("layout_hint", {}).get("span") or _DEFAULT_SPANS.get(cid, "half")

    return {
        "sections": sections,
        "spans": spans,
        "grid": {"columns": 12, "gap": 20},
    }
