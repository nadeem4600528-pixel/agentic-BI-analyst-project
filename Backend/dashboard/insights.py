"""Auto-generated business insight highlights for the dashboard.

Produces a prioritized list of human-readable findings (top performers,
growth, concentration/risk, variability, correlations) similar to the
"smart narrative" / insight tiles in Power BI and Tableau.

DataFrame-in / dict-list-out. This module never mutates the input.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _money(name: str) -> bool:
    token = str(name).lower()
    return any(w in token for w in ("sales", "revenue", "profit", "income", "amount", "turnover", "gmv"))


def _fmt(value: float, money: bool = False) -> str:
    value = float(value)
    prefix = "$" if money else ""
    if abs(value) >= 1_000_000:
        return f"{prefix}{value/1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{prefix}{value/1_000:,.1f}K"
    if float(value).is_integer():
        return f"{prefix}{int(value):,}"
    return f"{prefix}{value:,.2f}"


def build_insights(
    df: pd.DataFrame,
    date_column: Optional[str] = None,
    value_column: Optional[str] = None,
    category_column: Optional[str] = None,
    charts: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return a list of {level, title, detail, icon} insight cards."""
    insights: List[Dict[str, Any]] = []
    if df.empty:
        return insights

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    measure = value_column if value_column in numeric_cols else (numeric_cols[0] if numeric_cols else None)
    is_money = bool(measure and _money(measure))

    def add(level: str, title: str, detail: str, icon: str = "insight") -> None:
        insights.append({"level": level, "title": title, "detail": detail, "icon": icon})

    # 1) Top category performer -------------------------------------------
    if category_column and measure and category_column in df.columns:
        grouped = df.groupby(category_column, dropna=False)[measure].sum().sort_values(ascending=False)
        if len(grouped) >= 2:
            top_name, top_val = grouped.index[0], grouped.iloc[0]
            share = top_val / grouped.sum() * 100 if grouped.sum() else 0
            add(
                "positive",
                f"Top {category_column.title()}: {top_name}",
                f"Leads with {_fmt(top_val, is_money)} in {measure} — {share:.0f}% of the total.",
                "trophy",
            )
            # Concentration risk (80/20)
            cumulative = grouped.cumsum() / grouped.sum()
            count_to_80 = int((cumulative <= 0.8).sum()) + 1
            if count_to_80 <= max(1, len(grouped) // 3) and len(grouped) >= 4:
                add(
                    "warning",
                    "Revenue concentration",
                    f"Just {count_to_80} of {len(grouped)} {category_column}s drive ~80% of {measure}. Diversify to reduce risk.",
                    "concentration",
                )

    # 2) Growth / decline over time ----------------------------------------
    if date_column and measure and date_column in df.columns:
        temp = df[[date_column, measure]].copy()
        temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce")
        temp = temp.dropna(subset=[date_column])
        monthly = temp.set_index(date_column)[measure].resample("ME").sum().dropna()
        if len(monthly) >= 2:
            try:
                monthly.index = monthly.index.to_timestamp()
            except Exception:
                pass
            first, last = float(monthly.iloc[0]), float(monthly.iloc[-1])
            change = ((last - first) / first * 100) if first else 0
            best_idx = monthly.idxmax()
            worst_idx = monthly.idxmin()
            if change >= 0:
                add(
                    "positive",
                    f"{change:+.0f}% growth over the period",
                    f"{measure.title()} moved from {_fmt(first, is_money)} to {_fmt(last, is_money)} (first vs latest month).",
                    "growth",
                )
            else:
                add(
                    "warning",
                    f"{change:.0f}% decline over the period",
                    f"{measure.title()} fell from {_fmt(first, is_money)} to {_fmt(last, is_money)} between the first and latest month.",
                    "decline",
                )
            add(
                "neutral",
                f"Best month: {pd.Timestamp(best_idx).strftime('%b %Y')}",
                f"Peaked at {_fmt(float(monthly.max()), is_money)}; lowest was {_fmt(float(monthly.min()), is_money)} in {pd.Timestamp(worst_idx).strftime('%b %Y')}.",
                "calendar",
            )

    # 3) Variability / outliers --------------------------------------------
    if measure:
        series = _num(df[measure]).dropna()
        if len(series) >= 5:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
            cv = float(series.std() / series.mean() * 100) if series.mean() else 0
            if len(outliers) > 0:
                add(
                    "warning",
                    f"{len(outliers)} outlier transactions",
                    f"{len(outliers)} rows in '{measure}' fall outside the normal range (IQR). Worth reviewing for data entry or big-ticket events.",
                    "alert",
                )
            add(
                "neutral",
                f"{cv:.0f}% variability in {measure}",
                f"Coefficient of variation is {cv:.0f}% — "
                + ("highly volatile; averages may mislead." if cv > 80 else "moderately stable across records."),
                "variance",
            )

    # 4) Strongest numeric relationship ------------------------------------
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].apply(_num).corr()
        best_pair = None
        best_val = 0.0
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if pd.notna(val) and abs(val) > abs(best_val):
                    best_val, best_pair = float(val), (cols[i], cols[j])
        if best_pair and abs(best_val) >= 0.5:
            direction = "positive" if best_val > 0 else "negative"
            add(
                "neutral",
                f"Strong {direction} relationship",
                f"'{best_pair[0]}' and '{best_pair[1]}' move together (r = {best_val:.2f}). Useful for forecasting and driver analysis.",
                "correlation",
            )

    # 5) Profitability hint -------------------------------------------------
    profit_col = next((c for c in numeric_cols if any(w in c.lower() for w in ("profit", "margin"))), None)
    if profit_col and measure and profit_col != measure:
        total_measure = _num(df[measure]).sum()
        total_profit = _num(df[profit_col]).sum()
        if total_measure:
            margin = total_profit / total_measure * 100
            add(
                "positive" if margin >= 15 else "warning",
                f"Overall margin: {margin:.1f}%",
                f"{profit_col.title()} is {_fmt(total_profit, _money(profit_col))} on {_fmt(total_measure, is_money)} of {measure}.",
                "margin",
            )

    return insights
