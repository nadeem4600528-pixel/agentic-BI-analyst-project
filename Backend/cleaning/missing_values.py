"""Missing-value detection and imputation techniques."""

from typing import Any, Dict, Optional

import pandas as pd


def missing_value_report(df: pd.DataFrame) -> Dict[str, Any]:
	return {
		"rows": int(len(df)),
		"columns": {
			str(column): {
				"missing_count": int(df[column].isna().sum()),
				"missing_percentage": round(float(df[column].isna().mean() * 100), 4),
				"empty_string_count": int((df[column].astype("string").str.strip() == "").sum()),
			}
			for column in df.columns
		},
	}


def impute_missing(
	df: pd.DataFrame,
	strategy: str = "median",
	columns: Optional[list[str]] = None,
	value: Any = None,
	group_by: Optional[str] = None,
) -> pd.DataFrame:
	result = df.copy(deep=True)
	selected = columns or [str(column) for column in result.columns]
	for column in selected:
		if column not in result.columns:
			continue
		series = result[column]
		if strategy == "drop":
			result = result.dropna(subset=[column])
		elif strategy == "mean" and pd.api.types.is_numeric_dtype(series):
			result[column] = series.fillna(series.mean())
		elif strategy == "median" and pd.api.types.is_numeric_dtype(series):
			result[column] = series.fillna(series.median())
		elif strategy == "mode":
			mode = series.mode(dropna=True)
			if not mode.empty:
				result[column] = series.fillna(mode.iloc[0])
		elif strategy in {"ffill", "bfill"}:
			result[column] = series.ffill() if strategy == "ffill" else series.bfill()
		elif strategy == "constant":
			result[column] = series.fillna(value)
		elif strategy == "group" and group_by and group_by in result.columns:
			result[column] = result[column].fillna(
				result.groupby(group_by, dropna=False)[column].transform("median")
			)
		elif strategy in {"interpolate", "time"} and pd.api.types.is_numeric_dtype(series):
			result[column] = series.interpolate(method="linear")
		elif strategy == "indicator":
			result[f"{column}__missing"] = series.isna().astype("int8")
	return result
