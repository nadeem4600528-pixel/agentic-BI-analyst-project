"""KPI card generation for the Power BI / Tableau style dashboard.

Every KPI card is a self-describing payload the frontend renders as a
"metric tile": a formatted headline value, an optional period-over-period
delta with direction, an optional sparkline, and a business-friendly label.

DataFrame-in / dict-list-out. This module never mutates the input.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _is_money(name: str) -> bool:
    token = str(name).lower()
    return any(word in token for word in (
        "sales", "revenue", "profit", "income", "price", "cost", "amount",
        "value", "turnover", "gmv", "earnings", "expense", "budget", "margin$",
    ))


def _is_rate(name: str) -> bool:
    token = str(name).lower()
    return any(word in token for word in (
        "rate", "ratio", "percent", "pct", "discount", "margin", "growth",
        "share", "score",
    ))


def _fmt(value: Optional[float], kind: str = "number") -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "—"
    value = float(value)
    if kind == "currency":
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:,.2f}M"
        if abs(value) >= 1_000:
            return f"${value/1_000:,.1f}K"
        return f"${value:,.0f}"
    if kind == "percent":
        return f"{value:.1f}%"
    if kind == "decimal":
        return f"{value:,.2f}"
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{value/1_000:,.1f}K"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _kind_for(column: str) -> str:
    if _is_money(column):
        return "currency"
    if _is_rate(column):
        return "percent"
    return "number"


def _pct_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None:
        return None
    try:
        current = float(current)
        previous = float(previous)
    except (TypeError, ValueError):
        return None
    if previous == 0:
        return None if current == 0 else 100.0
    return round((current - previous) / abs(previous) * 100.0, 1)


def _card(
    card_id: str,
    title: str,
    value: Any,
    display_value: str,
    delta: Optional[float] = None,
    delta_label: Optional[str] = None,
    kind: str = "number",
    sparkline: Optional[List[Dict[str, Any]]] = None,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": card_id,
        "title": title,
        "value": value,
        "display_value": display_value,
        "kind": kind,
        "delta": delta,
        "delta_label": delta_label,
        "trend_direction": (
            "up" if (delta or 0) > 0 else "down" if (delta or 0) < 0 else "flat"
        ) if delta is not None else None,
        "sparkline": sparkline or [],
        "subtitle": subtitle or "",
        "icon": icon or "",
    }


def _series_sparkline(
    df: pd.DataFrame, date_column: str, value_column: str, agg: str = "sum", points: int = 12
) -> List[Dict[str, Any]]:
    try:
        temp = df[[date_column, value_column]].copy()
        temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce")
        temp = temp.dropna(subset=[date_column])
        if temp.empty:
            return []
        freq = "ME" if pd.date_range("2020", periods=2, freq="ME") is not None else "M"
        grouped = temp.set_index(date_column)[value_column].resample(freq).agg(agg).dropna()
        if grouped.empty:
            return []
        tail = grouped.tail(points)
        return [{"x": idx.strftime("%b %y"), "y": round(float(v), 2)} for idx, v in tail.items()]
    except Exception:
        return []


def _period_split(
    df: pd.DataFrame, date_column: str
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split data into current vs previous equal-length windows.

    Returns (current_df, previous_df, grain) where grain is 'day'|'month'|'year'.
    If there is not enough history, previous window is empty.
    """
    temp = df.copy()
    temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce")
    temp = temp.dropna(subset=[date_column])
    if temp.empty:
        return temp, temp, "day"

    span_days = (temp[date_column].max() - temp[date_column].min()).days
    if span_days >= 365:
        grain = "year"
    elif span_days >= 60:
        grain = "month"
    else:
        grain = "day"

    periods = temp.set_index(date_column).sort_index()
    if grain == "year":
        current_start = periods.index.max() - pd.DateOffset(years=1)
        previous_start = current_start - pd.DateOffset(years=1)
    elif grain == "month":
        current_start = periods.index.max() - pd.DateOffset(months=1)
        previous_start = current_start - pd.DateOffset(months=1)
    else:
        current_start = periods.index.max() - pd.Timedelta(days=1)
        previous_start = current_start - pd.Timedelta(days=1)

    current = periods[periods.index > current_start]
    previous = periods[(periods.index > previous_start) & (periods.index <= current_start)]
    return current.reset_index(), previous.reset_index(), grain


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_kpi_cards(
    df: pd.DataFrame,
    analysis: Optional[Dict[str, Any]] = None,
    value_column: Optional[str] = None,
    date_column: Optional[str] = None,
    category_column: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build a set of business KPI tiles with deltas and sparklines."""
    analysis = analysis or {}
    cards: List[Dict[str, Any]] = []
    if df.empty:
        return cards

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if value_column and value_column in df.columns:
        measure = value_column
    else:
        # prefer business/money columns over counts, ids and rates
        money = [c for c in numeric_cols if _is_money(c)]
        measure = money[0] if money else (numeric_cols[0] if numeric_cols else None)

    # Determine a valid date column for period-over-period math.
    usable_date = date_column if date_column and date_column in df.columns else None
    if usable_date is None:
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.8 and df[col].nunique() > 3:
                usable_date = col
                break

    current = previous = pd.DataFrame()
    grain = "day"
    if usable_date is not None:
        current, previous, grain = _period_split(df, usable_date)

    grain_label = {"day": "vs yesterday", "month": "vs last month", "year": "vs last year"}.get(grain, "")

    def delta_for(series_agg: str, column: str) -> Optional[float]:
        if current.empty or previous.empty or column not in current.columns:
            return None
        try:
            cur = getattr(current[column], series_agg)()
            prev = getattr(previous[column], series_agg)()
            return _pct_change(cur, prev)
        except Exception:
            return None

    # 1) Total records / orders (count) -----------------------------------
    count_delta = None
    if usable_date is not None and not current.empty and not previous.empty:
        count_delta = _pct_change(len(current), len(previous))
    cards.append(_card(
        "total_records", "Total Records", int(len(df)), _fmt(len(df)),
        delta=count_delta, delta_label=grain_label, kind="number", icon="records",
        sparkline=_series_sparkline(df, usable_date, df.columns[0], agg="count") if usable_date else [],
    ))

    # 2) Primary measure total (sales / revenue / profit) ------------------
    if measure is not None:
        series = pd.to_numeric(df[measure], errors="coerce").dropna()
        kind = _kind_for(measure)
        cards.append(_card(
            "total_measure",
            f"Total {measure.title()}",
            round(float(series.sum()), 2),
            _fmt(series.sum(), kind),
            delta=delta_for("sum", measure),
            delta_label=grain_label,
            kind=kind,
            icon="sales",
            sparkline=_series_sparkline(df, usable_date, measure, agg="sum") if usable_date else [],
        ))
        # 3) Average measure
        cards.append(_card(
            "avg_measure",
            f"Average {measure.title()}",
            round(float(series.mean()), 2),
            _fmt(series.mean(), "decimal" if kind == "number" else kind),
            delta=delta_for("mean", measure),
            delta_label=grain_label,
            kind=kind,
            icon="average",
        ))

    # 4) Profit / margin (auto-detect a second money column) ---------------
    profit_col = next(
        (c for c in numeric_cols
         if c != measure and any(w in c.lower() for w in ("profit", "margin", "income"))),
        None,
    )
    if profit_col is not None:
        pseries = pd.to_numeric(df[profit_col], errors="coerce").dropna()
        pk = "percent" if "margin" in profit_col.lower() or "rate" in profit_col.lower() else "currency"
        cards.append(_card(
            "total_profit",
            f"Total {profit_col.title()}",
            round(float(pseries.sum()), 2),
            _fmt(pseries.sum() if pk == "currency" else pseries.mean(), pk),
            delta=delta_for("sum" if pk == "currency" else "mean", profit_col),
            delta_label=grain_label,
            kind=pk,
            icon="profit",
            sparkline=_series_sparkline(df, usable_date, profit_col, agg="sum") if usable_date else [],
        ))

    # 5) Distinct entities (customers / orders / products) -----------------
    id_col = next(
        (c for c in df.columns
         if any(w in c.lower() for w in ("customer", "client", "buyer", "user", "account"))
         and not pd.api.types.is_numeric_dtype(df[c])),
        None,
    )
    if id_col is not None:
        cards.append(_card(
            "unique_customers", f"Unique {id_col.title()}", int(df[id_col].nunique()),
            _fmt(df[id_col].nunique()), kind="number", icon="customers",
        ))

    # 6) Average discount / rate if present --------------------------------
    rate_col = next(
        (c for c in numeric_cols if any(w in c.lower() for w in ("discount", "rate", "margin", "ratio"))),
        None,
    )
    if rate_col is not None:
        rseries = pd.to_numeric(df[rate_col], errors="coerce").dropna()
        avg = float(rseries.mean())
        # Values stored as fractions (0..1) vs percentages (0..100)
        avg_display = avg * 100 if avg <= 1.0 else avg
        cards.append(_card(
            "avg_rate", f"Average {rate_col.title()}", round(avg_display, 2),
            _fmt(avg_display, "percent"), kind="percent", icon="rate",
        ))

    # 7) Anomalies (from analysis engine) ----------------------------------
    anomalies = analysis.get("anomaly_insights") or []
    anomaly_count = sum(int(item.get("outlier_count", 0)) for item in anomalies if isinstance(item, dict))
    cards.append(_card(
        "anomaly_count", "Anomalies Detected", int(anomaly_count), _fmt(anomaly_count),
        kind="number", icon="alert",
        subtitle="Statistical outliers (IQR)",
    ))

    # 8) Trend change over the observed window -----------------------------
    trends = analysis.get("trends") or {}
    trend_rows = trends.get("trends") or []
    if len(trend_rows) >= 2:
        first_total = float(trend_rows[0].get("sum", 0) or 0)
        last_total = float(trend_rows[-1].get("sum", 0) or 0)
        raw_delta = ((last_total - first_total) / first_total * 100) if first_total else None
        delta = round(raw_delta, 1) if raw_delta is not None else None
        cards.append(_card(
            "trend_change", "Period Trend",
            f"{delta:.1f}%" if delta is not None else "—",
            f"{delta:+.1f}%" if delta is not None else "—",
            delta=delta, delta_label="first vs latest period", kind="percent", icon="trend",
        ))

    return cards
