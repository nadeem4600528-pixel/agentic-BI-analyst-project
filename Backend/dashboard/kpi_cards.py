"""KPI card generation for the dashboard layer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def _get_first_numeric_column(df: pd.DataFrame) -> Optional[str]:
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    return numeric_columns[0] if numeric_columns else None


def build_kpi_cards(
    df: pd.DataFrame,
    analysis: Optional[Dict[str, Any]] = None,
    value_column: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build a dashboard-ready list of KPI cards."""
    cards: List[Dict[str, Any]] = []
    analysis = analysis or {}

    total_rows = len(df)
    cards.append(
        {
            "id": "total_records",
            "title": "Total Records",
            "value": int(total_rows),
            "type": "number",
            "delta": None,
        }
    )

    metric_column = value_column or _get_first_numeric_column(df)
    if metric_column and metric_column in df.columns:
        metric_series = df[metric_column].dropna()
        if not metric_series.empty:
            total_value = float(metric_series.sum())
            average_value = float(metric_series.mean())
            max_value = float(metric_series.max())

            cards.append(
                {
                    "id": "total_value",
                    "title": f"Total {metric_column}",
                    "value": round(total_value, 2),
                    "type": "currency" if "revenue" in metric_column.lower() or "sales" in metric_column.lower() else "number",
                    "delta": None,
                }
            )
            cards.append(
                {
                    "id": "average_value",
                    "title": f"Average {metric_column}",
                    "value": round(average_value, 2),
                    "type": "number",
                    "delta": None,
                }
            )
            cards.append(
                {
                    "id": "max_value",
                    "title": f"Max {metric_column}",
                    "value": round(max_value, 2),
                    "type": "number",
                    "delta": None,
                }
            )

    anomalies = analysis.get("anomaly_insights") or []
    anomaly_count = sum(int(item.get("outlier_count", 0)) for item in anomalies if isinstance(item, dict))
    cards.append(
        {
            "id": "anomaly_count",
            "title": "Anomalies",
            "value": anomaly_count,
            "type": "number",
            "delta": None,
        }
    )

    trends = analysis.get("trends") or {}
    trend_rows = trends.get("trends") or []
    if len(trend_rows) >= 2:
        first_total = float(trend_rows[0].get("sum", 0) or 0)
        last_total = float(trend_rows[-1].get("sum", 0) or 0)
        delta = ((last_total - first_total) / first_total * 100) if first_total else 0.0
        cards.append(
            {
                "id": "trend_delta",
                "title": "Trend Change",
                "value": f"{delta:.1f}%",
                "type": "percent",
                "delta": delta,
            }
        )

    return cards