"""Layout helpers for a dashboard payload."""

from __future__ import annotations

from typing import Any, Dict, List


def build_dashboard_layout(charts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a simple responsive layout description for frontend rendering."""
    chart_ids = [chart.get("id", f"chart_{index}") for index, chart in enumerate(charts)]
    return {
        "sections": [
            {
                "id": "overview",
                "title": "Overview",
                "chart_ids": chart_ids[:2],
            },
            {
                "id": "detail",
                "title": "Details",
                "chart_ids": chart_ids[2:],
            },
        ]
    }