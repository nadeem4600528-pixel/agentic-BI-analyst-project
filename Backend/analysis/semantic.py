"""Semantic and segmentation analysis engine for datasets."""

from typing import Any, Dict

import pandas as pd

from profiling.semantic import SemanticProfiler, profile_semantics


def analyze_semantics(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze semantic types, entities, and categorical segments."""
    return profile_semantics(df)


def analyze_segmentation(df: pd.DataFrame, group_column: str, measure_column: str) -> Dict[str, Any]:
    """Group data by a categorical segment and aggregate measures."""
    if group_column not in df.columns or measure_column not in df.columns:
        return {"error": "Columns not found in DataFrame"}

    grouped = df.groupby(group_column)[measure_column].agg(["count", "mean", "sum", "min", "max"]).reset_index()
    grouped = grouped.rename(columns={"mean": "average", "sum": "total"})
    records = grouped.to_dict(orient="records")

    for row in records:
        row["group_column"] = group_column
        row["measure_column"] = measure_column

    return {
        "group_column": group_column,
        "measure_column": measure_column,
        "segments": records,
    }
