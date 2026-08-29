"""Service layer for report generation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from reports.report_generator import (
    build_comprehensive_report,
    build_profiling_report,
    build_cleaning_report,
    build_analysis_report,
)


class ReportService:
    """Generate all report artifacts for a dataset."""

    @staticmethod
    def generate_full_report(
        df: pd.DataFrame,
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
        category_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        return build_comprehensive_report(
            df,
            date_column=date_column,
            value_column=value_column,
            category_column=category_column,
        )

    @staticmethod
    def generate_profiling_report(df: pd.DataFrame) -> Dict[str, Any]:
        return build_profiling_report(df)

    @staticmethod
    def generate_cleaning_report(df: pd.DataFrame) -> Dict[str, Any]:
        return build_cleaning_report(df)

    @staticmethod
    def generate_analysis_report(
        df: pd.DataFrame,
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        return build_analysis_report(df, date_column=date_column, value_column=value_column)
