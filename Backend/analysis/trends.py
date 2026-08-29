"""Trend and pattern analysis engine for datasets."""

from typing import Any, Dict, List, Optional
import pandas as pd


def analyze_trends(df: pd.DataFrame, date_column: Optional[str] = None, value_column: Optional[str] = None) -> Dict[str, Any]:
	"""Extract time-series trends and patterns from dataset."""
	trends: Dict[str, Any] = {"trends": []}
	
	# Find potential date columns if not provided
	if not date_column:
		for col in df.columns:
			if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in str(col).lower() or "time" in str(col).lower():
				try:
					parsed = pd.to_datetime(df[col], errors="coerce")
					if parsed.notna().sum() > len(df) * 0.5:
						date_column = col
						break
				except Exception:
					pass

	if not date_column:
		return {"message": "No suitable date/time column found for trend analysis."}

	numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
	if not value_column and numeric_cols:
		value_column = numeric_cols[0]

	if not value_column:
		return {"message": "No numeric value column found for trend analysis."}

	temp_df = df[[date_column, value_column]].copy()
	temp_df[date_column] = pd.to_datetime(temp_df[date_column], errors="coerce")
	temp_df = temp_df.dropna().sort_values(by=date_column)

	if temp_df.empty:
		return {"message": "Insufficient valid date/value pairs for trends."}

	temp_df["period"] = temp_df[date_column].dt.to_period("M").astype(str)
	aggregated = temp_df.groupby("period")[value_column].agg(["sum", "mean", "count"]).reset_index()

	trends["date_column"] = date_column
	trends["value_column"] = value_column
	trends["trends"] = aggregated.to_dict(orient="records")
	return trends

