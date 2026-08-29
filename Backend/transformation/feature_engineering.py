"""Feature creation transformations."""
import pandas as pd


def create_feature(df: pd.DataFrame, target: str, expression: str) -> pd.DataFrame:
    if not target or not expression: raise ValueError("Feature target and expression are required.")
    result = df.copy(deep=True); result[target] = result.eval(expression); return result


def bin_numeric(df: pd.DataFrame, column: str, bins: list[float], labels: list[str] | None = None, target: str | None = None) -> pd.DataFrame:
    if column not in df.columns: raise KeyError(f"Column not found: {column}")
    result = df.copy(deep=True); result[target or f"{column}_bin"] = pd.cut(result[column], bins=bins, labels=labels); return result

