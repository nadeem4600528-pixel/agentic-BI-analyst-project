"""Outlier detection, flagging, capping, flooring, and removal."""

from typing import Any, Dict

import numpy as np
import pandas as pd


def outlier_report(df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> Dict[str, Any]:
	result: Dict[str, Any] = {"method": method, "columns": {}}
	for column in df.select_dtypes(include=[np.number]).columns:
		series = df[column].dropna()
		if series.empty:
			continue
		if method == "zscore":
			center, scale = series.mean(), series.std()
			scores = (series - center) / scale if scale else series * 0
			mask = scores.abs() > threshold
		elif method in {"mad", "modified_zscore"}:
			median = series.median()
			mad = (series - median).abs().median()
			scores = 0.6745 * (series - median) / mad if mad else series * 0
			mask = scores.abs() > threshold
		elif method == "percentile":
			low, high = series.quantile(0.01), series.quantile(0.99)
			mask = (series < low) | (series > high)
		else:
			low, high = series.quantile(0.25), series.quantile(0.75)
			spread = high - low
			mask = (series < low - threshold * spread) | (series > high + threshold * spread)
		result["columns"][str(column)] = {"count": int(mask.sum()), "percentage": round(float(mask.mean() * 100), 4)}
	return result


def treat_outliers(df: pd.DataFrame, method: str = "cap", threshold: float = 1.5) -> pd.DataFrame:
	result = df.copy(deep=True)
	for column in result.select_dtypes(include=[np.number]).columns:
		series = result[column]
		low, high = series.quantile(0.25), series.quantile(0.75)
		spread = high - low
		lower, upper = low - threshold * spread, high + threshold * spread
		mask = (series < lower) | (series > upper)
		if method in {"cap", "winsorize"}:
			result[column] = series.clip(lower, upper)
		elif method == "floor":
			result.loc[series < lower, column] = lower
		elif method == "remove":
			result = result.loc[~mask]
		elif method == "replace":
			result.loc[mask, column] = series.median()
		elif method == "flag":
			result[f"{column}__outlier"] = mask.astype("int8")
	return result.reset_index(drop=True)
