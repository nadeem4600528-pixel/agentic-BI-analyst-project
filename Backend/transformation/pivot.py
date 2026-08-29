"""Pivot and unpivot transformations."""
from typing import Any
import pandas as pd


def pivot(df: pd.DataFrame, index: str | list[str], columns: str, values: str | None = None, aggfunc: Any = "sum") -> pd.DataFrame:
    required = ([index] if isinstance(index, str) else index) + [columns] + ([values] if values else [])
    missing = [c for c in required if c not in df.columns]
    if missing: raise KeyError(f"Columns not found: {missing}")
    result = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc).reset_index()
    result.columns = ["_".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in result.columns]
    return result


def unpivot(df: pd.DataFrame, id_vars: list[str], value_vars: list[str] | None = None, var_name: str = "variable", value_name: str = "value") -> pd.DataFrame:
    missing = [c for c in id_vars + (value_vars or []) if c not in df.columns]
    if missing: raise KeyError(f"Columns not found: {missing}")
    return df.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)

