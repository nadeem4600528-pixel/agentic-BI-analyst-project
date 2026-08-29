"""Correlation analysis engine for datasets."""

from typing import Any, Dict

import pandas as pd

from profiling.correlation import CorrelationProfiler, profile_correlation


def analyze_correlation(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze feature correlations and linear dependencies."""
    return profile_correlation(df)
