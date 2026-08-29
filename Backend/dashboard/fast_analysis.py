"""Lightweight, fast statistics / correlation helpers for the dashboard.

The full profiling engines (profiling.correlation with Cramer's V / Theil's U,
profiling.statistics with exhaustive semantics) are thorough but far too slow
for an interactive dashboard on large files. These helpers compute only the
numeric summaries a BI dashboard needs, in well under a second.

DataFrame-in / dict-out. This module never mutates the input.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def fast_numeric_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Descriptive statistics for numeric columns + a summary count."""
    numeric = df.select_dtypes(include=[np.number])
    columns: Dict[str, Any] = {}
    for col in numeric.columns:
        series = pd.to_numeric(numeric[col], errors="coerce").dropna()
        if series.empty:
            continue
        columns[str(col)] = {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
        }
    return {
        "columns": columns,
        "summary": {
            "numeric_columns_count": len(columns),
            "total_rows": int(len(df)),
        },
    }


def fast_correlation(df: pd.DataFrame, max_columns: int = 8) -> Dict[str, Any]:
    """Pearson correlation matrix for numeric columns (capped for rendering)."""
    numeric = df.select_dtypes(include=[np.number])
    # Drop id-like near-unique integer columns that carry no correlation meaning.
    keep = []
    for col in numeric.columns:
        series = numeric[col]
        if series.nunique(dropna=True) > max(20, len(df) * 0.9) and pd.api.types.is_integer_dtype(series):
            continue
        keep.append(col)
    numeric = numeric[keep[:max_columns]].apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] < 2:
        return {"correlation_matrix": {}, "strong_pairs": []}

    corr = numeric.corr(numeric_only=True)
    matrix: Dict[str, Dict[str, float]] = {}
    strong_pairs = []
    labels = [str(c) for c in corr.columns]
    for i, ci in enumerate(labels):
        matrix[ci] = {}
        for j, cj in enumerate(labels):
            value = corr.iloc[i, j]
            matrix[ci][cj] = None if pd.isna(value) else round(float(value), 3)
            if j > i and pd.notna(value) and abs(float(value)) >= 0.7:
                strong_pairs.append({"column_1": ci, "column_2": cj, "correlation": round(float(value), 3)})
    return {"correlation_matrix": matrix, "strong_pairs": strong_pairs}


def fast_trends(
    df: pd.DataFrame, date_column: Optional[str], value_column: Optional[str]
) -> Dict[str, Any]:
    """Monthly sum/mean/count trend rows for a measure over a date column."""
    if not date_column or date_column not in df.columns:
        return {"trends": []}
    measure = value_column
    if not measure or measure not in df.columns:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        measure = numeric[0] if numeric else None
    if not measure:
        return {"trends": []}

    temp = df[[date_column, measure]].copy()
    temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce")
    temp = temp.dropna(subset=[date_column]).sort_values(date_column)
    if temp.empty:
        return {"trends": []}

    span_days = (temp[date_column].max() - temp[date_column].min()).days
    freq = "YE" if span_days >= 730 else "QE" if span_days >= 180 else "ME" if span_days >= 45 else "W"
    grouped = temp.set_index(date_column)[measure].resample(freq).agg(["sum", "mean", "count"]).dropna(how="all").reset_index()
    rows = grouped.rename(columns={date_column: "period"})
    records = []
    for _, row in rows.iterrows():
        records.append({
            "period": pd.Timestamp(row["period"]).strftime("%Y-%m-%d"),
            "sum": round(float(row["sum"]), 2) if pd.notna(row["sum"]) else None,
            "mean": round(float(row["mean"]), 2) if pd.notna(row["mean"]) else None,
            "count": int(row["count"]) if pd.notna(row["count"]) else 0,
        })
    return {"date_column": date_column, "value_column": measure, "trends": records}


def fast_segmentation(
    df: pd.DataFrame, group_column: str, measure_column: str, top_n: int = 20
) -> Optional[Dict[str, Any]]:
    """Group-by aggregate for the single most informative dimension/measure."""
    if group_column not in df.columns or measure_column not in df.columns:
        return None
    grouped = df.groupby(group_column, dropna=False)[measure_column].agg(["count", "mean", "sum", "min", "max"]).reset_index()
    grouped = grouped.rename(columns={"mean": "average", "sum": "total"}).sort_values("total", ascending=False).head(top_n)
    records = grouped.to_dict(orient="records")
    for row in records:
        row["group_column"] = group_column
        row["measure_column"] = measure_column
    return {"group_column": group_column, "measure_column": measure_column, "segments": records}


def detect_anomalies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """IQR-based outlier counts per numeric column."""
    anomalies: List[Dict[str, Any]] = []
    for col in df.select_dtypes(include=[np.number]).columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 5:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        count = int(mask.sum())
        if count:
            anomalies.append({
                "column": str(col),
                "outlier_count": count,
                "percentage": round(count / len(series) * 100, 2),
                "description": f"Detected {count} statistical outlier values outside IQR bounds in '{col}'.",
            })
    return anomalies
