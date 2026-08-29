"""Statistical analysis engine for datasets."""

from typing import Any, Dict
import pandas as pd
from profiling.statistics import StatisticsProfiler, profile_statistics


def analyze_statistics(df: pd.DataFrame) -> Dict[str, Any]:
	"""Perform comprehensive statistical analysis on numeric features."""
	profile = profile_statistics(df)
	return {
		"statistics": profile,
		"summary": {
			"numeric_columns_count": len(df.select_dtypes(include=["number"]).columns),
			"total_rows": len(df)
		}
	}

