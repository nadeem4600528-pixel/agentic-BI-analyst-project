"""Categorical encoding transformations."""
import pandas as pd


def one_hot_encode(df: pd.DataFrame, columns: list[str], drop_first: bool = False) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing: raise KeyError(f"Columns not found: {missing}")
    return pd.get_dummies(df.copy(deep=True), columns=columns, drop_first=drop_first, dtype=int)


def map_values(df: pd.DataFrame, column: str, mapping: dict, default=None) -> pd.DataFrame:
    if column not in df.columns: raise KeyError(f"Column not found: {column}")
    result = df.copy(deep=True); mapped = result[column].map(mapping)
    result[column] = mapped if default is None else mapped.fillna(default); return result

