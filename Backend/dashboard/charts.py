"""Automatic chart generation logic for the dashboard layer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def _infer_datetime_column(df: pd.DataFrame) -> Optional[str]:
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            return column
        name = str(column).lower()
        if "date" in name or "time" in name:
            return column
    return None


def _infer_numeric_column(df: pd.DataFrame) -> Optional[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return numeric_cols[0] if numeric_cols else None


def _infer_category_column(df: pd.DataFrame) -> Optional[str]:
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        unique_count = df[column].dropna().nunique()
        if 1 < unique_count <= min(20, max(3, len(df) // 10)):
            return column
    return None


def _to_chart_data(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"x": record.get("x"), "y": record.get("y"), "label": record.get("label")} for record in records
    ]


def _build_line_chart(df: pd.DataFrame, date_column: str, value_column: str) -> Optional[Dict[str, Any]]:
    if not date_column or not value_column or date_column not in df.columns or value_column not in df.columns:
        return None

    temp_df = df[[date_column, value_column]].copy()
    temp_df[date_column] = pd.to_datetime(temp_df[date_column], errors="coerce")
    temp_df = temp_df.dropna(subset=[date_column, value_column]).sort_values(by=date_column)

    if temp_df.empty:
        return None

    if len(temp_df) > 60:
        temp_df = temp_df.groupby(pd.Grouper(key=date_column, freq="M")).sum().reset_index()
    else:
        temp_df = temp_df.groupby(date_column, as_index=False).sum()

    return {
        "id": "trend_chart",
        "type": "line",
        "title": f"{value_column} Trend",
        "x_axis": date_column,
        "y_axis": value_column,
        "data": [
            {"x": str(row[date_column]), "y": float(row[value_column])}
            for _, row in temp_df.iterrows()
        ],
    }


def _build_bar_chart(df: pd.DataFrame, category_column: str, value_column: str) -> Optional[Dict[str, Any]]:
    if not category_column or not value_column:
        return None

    if category_column not in df.columns or value_column not in df.columns:
        return None

    grouped = df.groupby(category_column, dropna=False)[value_column].sum().reset_index()
    grouped = grouped.sort_values(by=value_column, ascending=False).head(10)

    if grouped.empty:
        return None

    return {
        "id": "category_chart",
        "type": "bar",
        "title": f"{value_column} by {category_column}",
        "x_axis": category_column,
        "y_axis": value_column,
        "data": [
            {"x": str(row[category_column]), "y": float(row[value_column])}
            for _, row in grouped.iterrows()
        ],
    }


def _build_histogram(df: pd.DataFrame, value_column: str) -> Optional[Dict[str, Any]]:
    if not value_column or value_column not in df.columns:
        return None

    series = df[value_column].dropna()
    if series.empty:
        return None

    bins = min(10, max(5, len(series) // 10))
    hist = pd.cut(series, bins=bins, include_lowest=True).value_counts().sort_index()

    return {
        "id": "distribution_chart",
        "type": "histogram",
        "title": f"Distribution of {value_column}",
        "x_axis": value_column,
        "y_axis": "count",
        "data": [
            {"x": str(interval), "y": int(count)}
            for interval, count in hist.items()
        ],
    }


def _build_heatmap_chart(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    corr = analysis.get("correlations") or {}
    matrix = corr.get("correlation_matrix") or {}
    if not matrix:
        return None

    labels = list(matrix.keys())
    if len(labels) < 2:
        return None

    z = []
    for row_label in labels:
        row = []
        for col_label in labels:
            row_value = matrix.get(row_label, {}).get(col_label, 0)
            row.append(float(row_value))
        z.append(row)

    return {
        "id": "correlation_heatmap",
        "type": "heatmap",
        "title": "Correlation Heatmap",
        "x_axis": "feature",
        "y_axis": "feature",
        "data": {"x": labels, "y": labels, "z": z},
    }


def generate_dashboard_charts(
    df: pd.DataFrame,
    analysis: Optional[Dict[str, Any]] = None,
    date_column: Optional[str] = None,
    value_column: Optional[str] = None,
    category_column: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate a dashboard-ready list of chart definitions."""
    charts: List[Dict[str, Any]] = []

    if df.empty:
        return charts

    if analysis is None:
        analysis = {}

    inferred_date = date_column or _infer_datetime_column(df)
    inferred_value = value_column or _infer_numeric_column(df)
    inferred_category = category_column or _infer_category_column(df)

    if inferred_date and inferred_value:
        line_chart = _build_line_chart(df, inferred_date, inferred_value)
        if line_chart:
            charts.append(line_chart)

    if inferred_value:
        histogram = _build_histogram(df, inferred_value)
        if histogram:
            charts.append(histogram)

    if inferred_category and inferred_value:
        bar_chart = _build_bar_chart(df, inferred_category, inferred_value)
        if bar_chart:
            charts.append(bar_chart)

    heatmap = _build_heatmap_chart(analysis)
    if heatmap:
        charts.append(heatmap)

    if not charts:
        first_numeric = _infer_numeric_column(df)
        first_category = _infer_category_column(df)
        if first_numeric and first_category:
            charts.append(_build_bar_chart(df, first_category, first_numeric) or {})

    return [chart for chart in charts if chart]