"""Analysis / Insight Agent Service and API endpoints."""

from typing import Any, Dict, Optional
import pandas as pd
from analysis.analysis_engine import analyze_dataset


class AnalysisService:
	"""Service layer for running analysis and insight generation."""

	@staticmethod
	def analyze_dataframe(df: pd.DataFrame, date_column: Optional[str] = None, value_column: Optional[str] = None) -> Dict[str, Any]:
		"""Execute comprehensive analysis on a cleaned dataset."""
		return analyze_dataset(df, date_column=date_column, value_column=value_column)

