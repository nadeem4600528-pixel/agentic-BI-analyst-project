"""
profiler.py

Main Data Profiling Orchestrator for Agentic BI Analyst.

Provides a unified interface to profile single DataFrames or multi-table datasets
using all 50 data profiling use cases.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from .report_aggregator import DataUnderstandingReport

class DataProfiler:
    """
    Main orchestrator for data profiling operations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tables: Optional[Dict[str, pd.DataFrame]] = None
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DataProfiler requires a pandas DataFrame.")
        self.df = df
        self.tables = tables or {}

    def profile(
        self,
        include_modules: Optional[List[str]] = None,
        business_rules: Optional[List[Dict[str, Any]]] = None,
        baseline_schema: Optional[Dict[str, Any]] = None,
        source_metadata: Optional[Dict[str, Any]] = None,
        transformations: Optional[List[Dict[str, Any]]] = None,
        approval_decisions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run all data profiling modules and return unified report.
        """
        report_gen = DataUnderstandingReport(
            df=self.df,
            tables=self.tables,
            baseline_schema=baseline_schema,
            source_metadata=source_metadata,
            transformations=transformations,
            approval_decisions=approval_decisions
        )
        return report_gen.generate(
            include_modules=include_modules,
            business_rules=business_rules
        )


def profile_dataframe(
    df: pd.DataFrame,
    include_modules: Optional[List[str]] = None,
    baseline_schema: Optional[Dict[str, Any]] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
    transformations: Optional[List[Dict[str, Any]]] = None,
    approval_decisions: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Convenience function for profiling a single DataFrame.
    """
    profiler = DataProfiler(df)
    return profiler.profile(
        include_modules=include_modules,
        baseline_schema=baseline_schema,
        source_metadata=source_metadata,
        transformations=transformations,
        approval_decisions=approval_decisions
    )
