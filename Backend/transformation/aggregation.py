"""Aggregation transformations."""
from typing import Any, Mapping, Sequence
import pandas as pd


def aggregate(df: pd.DataFrame, group_by: Sequence[str], aggregations: Mapping[str, Any]) -> pd.DataFrame:
    missing = [c for c in list(group_by) + list(aggregations) if c not in df.columns]
    if missing: raise KeyError(f"Columns not found: {missing}")
    return df.groupby(list(group_by), dropna=False).agg(dict(aggregations)).reset_index()

