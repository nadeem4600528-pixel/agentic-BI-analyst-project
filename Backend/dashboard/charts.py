"""Automatic chart generation for the Power BI / Tableau style dashboard.

Each chart is a self-describing dict with a stable contract so the frontend
(Plotly) can render it without any business logic:

    {
        "id": str, "type": str, "title": str, "subtitle": str,
        "x_axis": str, "y_axis": str,
        "data": ...            # payload shape depends on chart type
        "layout_hint": {...},  # optional presentation hints
    }

Supported types: line, area, bar, stacked_bar, grouped_bar, donut, treemap,
pareto, box, scatter, heatmap, gauge, kpi_trend.

DataFrame-in / dict-list-out. This module never mutates the input.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column inference
# ---------------------------------------------------------------------------

def _month_freq() -> str:
    try:
        pd.date_range("2024-01-01", periods=2, freq="ME")
        return "ME"
    except (ValueError, TypeError):
        return "M"


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _infer_datetime_column(df: pd.DataFrame) -> Optional[str]:
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            return column
        name = str(column).lower()
        if "date" in name or "time" in name:
            parsed = _to_datetime(series)
            if parsed.notna().mean() >= 0.7:
                return column
    # fallback: any column that parses mostly as dates
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        parsed = _to_datetime(df[column])
        if len(df) > 0 and parsed.notna().mean() >= 0.85 and df[column].nunique() > 3:
            return column
    return None


def _infer_numeric_columns(df: pd.DataFrame, exclude: Sequence[str] = ()) -> List[str]:
    cols = []
    for column in df.select_dtypes(include=[np.number]).columns:
        if column in exclude:
            continue
        # skip id-like columns (almost all unique integers)
        if df[column].nunique() > max(20, len(df) * 0.9) and pd.api.types.is_integer_dtype(df[column]):
            continue
        cols.append(str(column))
    return cols


def _infer_numeric_column(df: pd.DataFrame, exclude: Sequence[str] = ()) -> Optional[str]:
    """Pick the primary measure: prefer money/business columns, then measures."""
    cols = _infer_numeric_columns(df, exclude=exclude)
    if not cols:
        return None
    money = [c for c in cols if _money_name(c)]
    if money:
        return money[0]
    measure_like = [c for c in cols if _is_measure(c)]
    if measure_like:
        return measure_like[0]
    return cols[0]


def _infer_category_columns(df: pd.DataFrame, max_cardinality: int = 30, exclude: Sequence[str] = ()) -> List[str]:
    result = []
    for column in df.columns:
        if column in exclude or pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_datetime64_any_dtype(df[column]):
            continue
        unique_count = df[column].dropna().nunique()
        if 1 < unique_count <= min(max_cardinality, max(2, len(df) // 5)):
            result.append(str(column))
    return result


def _infer_category_column(df: pd.DataFrame, exclude: Sequence[str] = ()) -> Optional[str]:
    cols = _infer_category_columns(df, exclude=exclude)
    if not cols:
        return None
    # prefer business-looking dimensions
    preference = ("region", "country", "city", "state", "category", "segment",
                  "department", "product", "channel", "status", "gender")
    for token in preference:
        for col in cols:
            if token in col.lower():
                return col
    return cols[0]


def _money_name(column: str) -> bool:
    token = str(column).lower()
    return any(w in token for w in (
        "sales", "revenue", "profit", "income", "price", "cost", "amount",
        "turnover", "gmv", "earnings", "expense", "spend", "budget",
    ))


def _is_measure(column: str) -> bool:
    token = str(column).lower()
    return _money_name(column) or any(w in token for w in (
        "value", "quantity", "qty", "units", "count", "total", "discount",
        "margin", "rate",
    ))


def _pick_measures(df: pd.DataFrame, value_column: Optional[str], limit: int = 4) -> List[str]:
    """Return measures ordered by business relevance.

    Priority: the chosen value column, then currency measures (sales/profit/…,
    which share a comparable scale), then other measure-like columns, then the
    rest of the numeric columns.
    """
    numeric = _infer_numeric_columns(df)
    money = [c for c in numeric if _money_name(c)]
    other_measure = [c for c in numeric if c not in money and _is_measure(c)]
    rest = [c for c in numeric if c not in money and c not in other_measure]

    ordered: List[str] = []
    if value_column and value_column in numeric:
        ordered.append(value_column)
    for group in (money, other_measure, rest):
        for col in group:
            if col not in ordered:
                ordered.append(col)
    return ordered[:limit]


# ---------------------------------------------------------------------------
# Individual chart builders. Each returns None when it cannot be built.
# ---------------------------------------------------------------------------

def _trend_chart(
    df: pd.DataFrame, date_column: str, measures: Sequence[str], kind: str = "line"
) -> Optional[Dict[str, Any]]:
    if not date_column or not measures or date_column not in df.columns:
        return None
    present = [m for m in measures if m in df.columns]
    if not present:
        return None

    temp = df[[date_column] + present].copy()
    temp[date_column] = _to_datetime(temp[date_column])
    temp = temp.dropna(subset=[date_column])
    if temp.empty:
        return None

    span_days = (temp[date_column].max() - temp[date_column].min()).days
    if span_days >= 730:
        freq, label = "YE", "Yearly"
    elif span_days >= 180:
        freq, label = "QE", "Quarterly"
    elif span_days >= 45:
        freq, label = _month_freq(), "Monthly"
    else:
        freq, label = "W", "Weekly"

    grouped = temp.set_index(date_column)[present].resample(freq).sum(min_count=1).dropna(how="all").reset_index()
    if len(grouped) < 2:
        grouped = temp.set_index(date_column)[present].resample("D").sum(min_count=1).dropna(how="all").reset_index()
        label = "Daily"
    if grouped.empty:
        return None

    x = [d.strftime("%Y-%m-%d") for d in grouped[date_column]]
    series = []
    for measure in present:
        series.append({
            "name": str(measure),
            "y": [None if pd.isna(v) else round(float(v), 2) for v in grouped[measure]],
        })

    primary = present[0]
    chart_id = "trend_area" if kind == "area" else "trend_line"
    return {
        "id": chart_id,
        "type": kind,
        "title": f"{label} Trend" + (f" — {primary.title()}" if len(present) == 1 else " — Key Measures"),
        "subtitle": f"{date_column} · {', '.join(present)}",
        "x_axis": str(date_column),
        "y_axis": ", ".join(present),
        "data": {"x": x, "series": series},
        "layout_hint": {"span": "wide", "fill": kind == "area"},
    }


def _growth_chart(df: pd.DataFrame, date_column: str, measure: str) -> Optional[Dict[str, Any]]:
    if not date_column or not measure or date_column not in df.columns or measure not in df.columns:
        return None
    temp = df[[date_column, measure]].copy()
    temp[date_column] = _to_datetime(temp[date_column])
    temp = temp.dropna(subset=[date_column, measure]).sort_values(date_column)
    monthly = temp.set_index(date_column)[measure].resample(_month_freq()).sum()
    if len(monthly) < 2:
        return None
    growth = monthly.pct_change().replace([np.inf, -np.inf], np.nan).dropna() * 100
    if growth.empty:
        return None
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in growth]
    return {
        "id": "growth_bars",
        "type": "growth",
        "title": f"Month-over-Month Growth — {measure.title()}",
        "subtitle": "Percentage change between consecutive months",
        "x_axis": str(date_column),
        "y_axis": f"{measure} growth %",
        "data": {
            "x": [d.strftime("%b %Y") for d in growth.index],
            "y": [round(float(v), 1) for v in growth.values],
            "colors": colors,
        },
        "layout_hint": {"span": "wide"},
    }


def _category_bar(
    df: pd.DataFrame, category_column: str, measure: str, top_n: int = 10
) -> Optional[Dict[str, Any]]:
    if not category_column or not measure or category_column not in df.columns or measure not in df.columns:
        return None
    grouped = (
        df.groupby(category_column, dropna=False)[measure]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    if grouped.empty:
        return None
    return {
        "id": "category_bar",
        "type": "bar",
        "title": f"{measure.title()} by {category_column.title()}",
        "subtitle": f"Top {len(grouped)} {category_column} by total {measure}",
        "x_axis": str(category_column),
        "y_axis": str(measure),
        "data": {
            "x": [str(k) for k in grouped.index],
            "y": [round(float(v), 2) for v in grouped.values],
        },
        "layout_hint": {"orientation": "v"},
    }


def _donut_chart(df: pd.DataFrame, category_column: str, measure: Optional[str], top_n: int = 8) -> Optional[Dict[str, Any]]:
    if not category_column or category_column not in df.columns:
        return None
    if measure and measure in df.columns:
        grouped = df.groupby(category_column, dropna=False)[measure].sum()
    else:
        grouped = df.groupby(category_column, dropna=False).size().rename("count")
    grouped = grouped.sort_values(ascending=False)
    if grouped.empty:
        return None
    top = grouped.head(top_n)
    other = grouped.iloc[top_n:].sum() if len(grouped) > top_n else 0
    labels = [str(k) for k in top.index]
    values = [round(float(v), 2) for v in top.values]
    if other > 0:
        labels.append("Other")
        values.append(round(float(other), 2))
    return {
        "id": "share_donut",
        "type": "donut",
        "title": f"{(measure or 'Records').title()} Share by {category_column.title()}",
        "subtitle": f"Proportion across {category_column}",
        "x_axis": str(category_column),
        "y_axis": str(measure or "count"),
        "data": {"labels": labels, "values": values},
        "layout_hint": {"orientation": "v"},
    }


def _treemap_chart(df: pd.DataFrame, hierarchy: Sequence[str], measure: str) -> Optional[Dict[str, Any]]:
    levels = [c for c in hierarchy if c in df.columns]
    if not levels or not measure or measure not in df.columns:
        return None
    grouped = df.groupby(levels, dropna=False)[measure].sum().reset_index()
    grouped = grouped.sort_values(measure, ascending=False)
    if grouped.empty:
        return None
    # Build labels + parents + values for a Plotly treemap.
    labels: List[str] = []
    parents: List[str] = []
    values: List[float] = []
    ids: List[str] = []
    root_id = "root"
    ids.append(root_id); labels.append("All"); parents.append(""); values.append(round(float(grouped[measure].sum()), 2))
    for depth, level in enumerate(levels):
        subset_cols = levels[: depth + 1]
        agg = grouped.groupby(subset_cols, dropna=False)[measure].sum().reset_index()
        for _, row in agg.iterrows():
            node_id = " / ".join(str(row[c]) for c in subset_cols)
            parent_id = root_id if depth == 0 else " / ".join(str(row[c]) for c in levels[:depth])
            if node_id in ids:
                continue
            ids.append(node_id)
            labels.append(str(row[level]))
            parents.append(parent_id)
            values.append(round(float(row[measure]), 2))
    return {
        "id": "treemap",
        "type": "treemap",
        "title": f"{measure.title()} Breakdown — {' → '.join(l.title() for l in levels)}",
        "subtitle": "Hierarchical contribution to total",
        "x_axis": " / ".join(levels),
        "y_axis": str(measure),
        "data": {"ids": ids, "labels": labels, "parents": parents, "values": values},
        "layout_hint": {"span": "wide"},
    }


def _pareto_chart(df: pd.DataFrame, category_column: str, measure: str, top_n: int = 12) -> Optional[Dict[str, Any]]:
    if not category_column or not measure or category_column not in df.columns or measure not in df.columns:
        return None
    grouped = df.groupby(category_column, dropna=False)[measure].sum().sort_values(ascending=False).head(top_n)
    if grouped.empty:
        return None
    cumulative = grouped.cumsum() / grouped.sum() * 100
    return {
        "id": "pareto",
        "type": "pareto",
        "title": f"Pareto Analysis — {measure.title()} by {category_column.title()}",
        "subtitle": "Bars: value · Line: cumulative % (80/20)",
        "x_axis": str(category_column),
        "y_axis": str(measure),
        "data": {
            "x": [str(k) for k in grouped.index],
            "y": [round(float(v), 2) for v in grouped.values],
            "cumulative": [round(float(v), 1) for v in cumulative.values],
        },
    }


def _stacked_bar_chart(
    df: pd.DataFrame, date_column: str, category_column: str, measure: str
) -> Optional[Dict[str, Any]]:
    if not date_column or not category_column or not measure:
        return None
    if date_column not in df.columns or category_column not in df.columns or measure not in df.columns:
        return None
    temp = df[[date_column, category_column, measure]].copy()
    temp[date_column] = _to_datetime(temp[date_column])
    temp = temp.dropna(subset=[date_column])
    if temp.empty:
        return None
    pivot = (
        temp.set_index(date_column)
        .groupby([pd.Grouper(freq=_month_freq()), category_column])[measure]
        .sum()
        .unstack(category_column)
        .fillna(0)
        .tail(12)
    )
    if pivot.empty or pivot.shape[1] < 1:
        return None
    series = []
    for col in pivot.columns:
        series.append({
            "name": str(col),
            "y": [round(float(v), 2) for v in pivot[col].values],
        })
    return {
        "id": "stacked_trend",
        "type": "stacked_bar",
        "title": f"{measure.title()} Trend by {category_column.title()}",
        "subtitle": "Monthly stacked composition over time",
        "x_axis": str(date_column),
        "y_axis": str(measure),
        "data": {"x": [d.strftime("%b %Y") for d in pivot.index], "series": series},
        "layout_hint": {"span": "wide"},
    }


def _grouped_bar_chart(
    df: pd.DataFrame, category_column: str, measures: Sequence[str]
) -> Optional[Dict[str, Any]]:
    present = [m for m in measures if m in df.columns]
    if not category_column or category_column not in df.columns or len(present) < 2:
        return None
    grouped = df.groupby(category_column, dropna=False)[present].sum().sort_values(present[0], ascending=False).head(8)
    if grouped.empty:
        return None
    series = []
    for m in present:
        series.append({"name": str(m), "y": [round(float(v), 2) for v in grouped[m].values]})
    return {
        "id": "grouped_bar",
        "type": "grouped_bar",
        "title": f"Measures Compared by {category_column.title()}",
        "subtitle": ", ".join(present),
        "x_axis": str(category_column),
        "y_axis": "value",
        "data": {"x": [str(k) for k in grouped.index], "series": series},
    }


def _box_chart(df: pd.DataFrame, category_column: str, measure: str, top_n: int = 8) -> Optional[Dict[str, Any]]:
    if not category_column or not measure or category_column not in df.columns or measure not in df.columns:
        return None
    top_groups = df[category_column].value_counts().head(top_n).index.tolist()
    subset = df[df[category_column].isin(top_groups)]
    traces = []
    for group in top_groups:
        values = pd.to_numeric(subset.loc[subset[category_column] == group, measure], errors="coerce").dropna()
        if values.empty:
            continue
        q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower = max(values.min(), q1 - 1.5 * iqr)
        upper = min(values.max(), q3 + 1.5 * iqr)
        traces.append({
            "label": str(group),
            "min": round(float(lower), 2),
            "q1": round(float(q1), 2),
            "median": round(float(median), 2),
            "q3": round(float(q3), 2),
            "max": round(float(upper), 2),
            "mean": round(float(values.mean()), 2),
        })
    if not traces:
        return None
    return {
        "id": "box_plot",
        "type": "box",
        "title": f"Distribution of {measure.title()} by {category_column.title()}",
        "subtitle": "Median, quartiles and range per segment",
        "x_axis": str(category_column),
        "y_axis": str(measure),
        "data": {"traces": traces},
    }


def _scatter_chart(df: pd.DataFrame, x_measure: str, y_measure: str, color_by: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not x_measure or not y_measure:
        return None
    if x_measure not in df.columns or y_measure not in df.columns:
        return None
    temp = df[[c for c in {x_measure, y_measure, color_by} if c]].copy()
    temp[x_measure] = pd.to_numeric(temp[x_measure], errors="coerce")
    temp[y_measure] = pd.to_numeric(temp[y_measure], errors="coerce")
    temp = temp.dropna(subset=[x_measure, y_measure])
    if len(temp) < 5:
        return None
    # correlation for the subtitle
    corr = float(temp[x_measure].corr(temp[y_measure]))
    if color_by and color_by in temp.columns and temp[color_by].nunique() <= 12:
        groups = []
        for name, grp in temp.groupby(color_by, dropna=False):
            groups.append({
                "name": str(name),
                "x": [round(float(v), 2) for v in grp[x_measure].head(500)],
                "y": [round(float(v), 2) for v in grp[y_measure].head(500)],
            })
        data = {"groups": groups}
    else:
        sample = temp.head(800)
        data = {
            "x": [round(float(v), 2) for v in sample[x_measure]],
            "y": [round(float(v), 2) for v in sample[y_measure]],
        }
    return {
        "id": "scatter",
        "type": "scatter",
        "title": f"{y_measure.title()} vs {x_measure.title()}",
        "subtitle": f"Correlation r = {corr:.2f}" + (f" · colored by {color_by}" if color_by and "groups" in data else ""),
        "x_axis": str(x_measure),
        "y_axis": str(y_measure),
        "data": data,
    }


def _heatmap_chart(df: pd.DataFrame, measures: Sequence[str]) -> Optional[Dict[str, Any]]:
    present = [m for m in measures if m in df.columns]
    if len(present) < 2:
        return None
    numeric = df[present].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(numeric_only=True)
    labels = [str(c) for c in corr.columns]
    z = [[None if pd.isna(v) else round(float(v), 2) for v in row] for row in corr.values]
    return {
        "id": "correlation_heatmap",
        "type": "heatmap",
        "title": "Correlation Heatmap",
        "subtitle": "Pearson correlation between numeric measures",
        "x_axis": "measure",
        "y_axis": "measure",
        "data": {"x": labels, "y": labels, "z": z},
    }


def _gauge_chart(df: pd.DataFrame, measure: str, target: Optional[float] = None) -> Optional[Dict[str, Any]]:
    if not measure or measure not in df.columns:
        return None
    series = pd.to_numeric(df[measure], errors="coerce").dropna()
    if series.empty:
        return None
    total = float(series.sum())
    if target is None:
        # heuristic target: round up the current total by ~15% to show attainment
        target = total * 1.15
    attainment = round(total / target * 100, 1) if target else 0
    return {
        "id": "gauge",
        "type": "gauge",
        "title": f"{measure.title()} Target Attainment",
        "subtitle": "Actual vs target",
        "x_axis": "",
        "y_axis": str(measure),
        "data": {"value": attainment, "actual": round(total, 2), "target": round(float(target), 2)},
    }


def _histogram_chart(df: pd.DataFrame, measure: str, bins: int = 20) -> Optional[Dict[str, Any]]:
    if not measure or measure not in df.columns:
        return None
    series = pd.to_numeric(df[measure], errors="coerce").dropna()
    if series.empty:
        return None
    counts, edges = np.histogram(series, bins=min(bins, max(5, len(series) // 5)))
    labels = [f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(len(counts))]
    return {
        "id": "distribution_histogram",
        "type": "histogram",
        "title": f"Distribution of {measure.title()}",
        "subtitle": "Frequency across value ranges",
        "x_axis": str(measure),
        "y_axis": "frequency",
        "data": {"x": labels, "y": [int(c) for c in counts]},
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_dashboard_charts(
    df: pd.DataFrame,
    analysis: Optional[Dict[str, Any]] = None,
    date_column: Optional[str] = None,
    value_column: Optional[str] = None,
    category_column: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate a full set of business dashboard charts."""
    charts: List[Dict[str, Any]] = []
    if df.empty:
        return charts

    date_col = date_column if date_column and date_column in df.columns else _infer_datetime_column(df)
    value_col = value_column if value_column and value_column in df.columns else _infer_numeric_column(df)
    category_col = category_column if category_column and category_column in df.columns else _infer_category_column(df, exclude=[date_col] if date_col else ())

    measures = _pick_measures(df, value_col, limit=3)
    category_cols = _infer_category_columns(df, exclude=[date_col] if date_col else ())
    numeric_cols = _infer_numeric_columns(df)

    def add(chart: Optional[Dict[str, Any]]) -> None:
        if chart:
            charts.append(chart)

    # --- Trend row (time based) ---
    if date_col and measures:
        add(_trend_chart(df, date_col, measures[:2], kind="area"))
        add(_growth_chart(df, date_col, measures[0]))
        if category_col:
            add(_stacked_bar_chart(df, date_col, category_col, measures[0]))

    # --- Composition row ---
    if category_col:
        add(_donut_chart(df, category_col, measures[0] if measures else None))
        add(_category_bar(df, category_col, measures[0] if measures else None))
        # hierarchy treemap if there is a second dimension
        second_dim = next((c for c in category_cols if c != category_col), None)
        if second_dim and measures:
            add(_treemap_chart(df, [category_col, second_dim], measures[0]))
        if measures:
            add(_pareto_chart(df, category_col, measures[0]))

    # --- Distribution & relationship row ---
    if category_col and measures:
        add(_box_chart(df, category_col, measures[0]))
    if measures:
        add(_histogram_chart(df, measures[0]))
    if len(numeric_cols) >= 2:
        add(_heatmap_chart(df, numeric_cols[:6]))
        y_measure = value_col or numeric_cols[0]
        x_measure = next((m for m in numeric_cols if m != y_measure), numeric_cols[0])
        add(_scatter_chart(df, x_measure, y_measure, color_by=category_col))
    # Grouped bars only make sense for measures on a comparable (currency) scale.
    money_measures = [m for m in measures if _money_name(m)]
    if len(money_measures) >= 2 and category_col:
        add(_grouped_bar_chart(df, category_col, money_measures[:3]))

    # --- KPI gauge ---
    if measures:
        add(_gauge_chart(df, measures[0]))

    # Fallback if nothing could be built
    if not charts:
        if measures:
            add(_histogram_chart(df, measures[0]))
        if category_cols:
            add(_donut_chart(df, category_cols[0], None))

    # de-duplicate by id while preserving order
    seen = set()
    unique = []
    for chart in charts:
        if chart["id"] not in seen:
            seen.add(chart["id"])
            unique.append(chart)
    return unique
