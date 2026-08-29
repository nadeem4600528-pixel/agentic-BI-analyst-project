"""Dashboard building engine for AI-driven business dashboards."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from analysis.analysis_engine import AnalysisAgent

from .charts import generate_dashboard_charts
from .kpi_cards import build_kpi_cards
from .layout import build_dashboard_layout


class DashboardBuilder:
    """Construct a dashboard-ready payload from a cleaned dataset."""

    @staticmethod
    def build_dashboard(
        df: pd.DataFrame,
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
        category_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DashboardBuilder requires a pandas DataFrame.")

        if df.empty:
            return {
                "kpis": [],
                "charts": [],
                "layout": {"sections": []},
                "filters": [],
                "insights": ["Dataset is empty."],
                "summary": {"rows": 0, "columns": 0},
            }

        date_column = date_column or next((col for col in df.columns if "date" in str(col).lower() or "time" in str(col).lower()), None)
        value_column = value_column or next((col for col in df.select_dtypes(include=["number"]).columns.tolist() if col not in {date_column}), None)
        category_column = category_column or next(
            (
                col
                for col in df.columns
                if col not in {date_column, value_column} and not pd.api.types.is_numeric_dtype(df[col])
            ),
            None,
        )

        analysis = AnalysisAgent(df).analyze(date_column=date_column, value_column=value_column)
        kpis = build_kpi_cards(df, analysis=analysis, value_column=value_column)
        charts = generate_dashboard_charts(
            df,
            analysis=analysis,
            date_column=date_column,
            value_column=value_column,
            category_column=category_column,
        )

        filters = [
            {"name": "date_column", "value": date_column, "options": list(df.columns)},
            {"name": "value_column", "value": value_column, "options": list(df.select_dtypes(include=["number"]).columns)},
            {"name": "category_column", "value": category_column, "options": [col for col in df.columns if col != value_column]},
        ]

        table_columns = [str(column) for column in df.columns]
        table_rows = df.head(100).where(pd.notna(df.head(100)), None).to_dict(orient="records")

        return {
            "kpis": kpis,
            "charts": charts,
            "table": {"columns": table_columns, "rows": table_rows, "total_rows": int(len(df))},
            "layout": build_dashboard_layout(charts),
            "filters": filters,
            "insights": analysis.get("business_insights", []),
            "summary": {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "value_column": value_column,
                "date_column": date_column,
                "category_column": category_column,
            },
        }


def build_dashboard(
    df: pd.DataFrame,
    date_column: Optional[str] = None,
    value_column: Optional[str] = None,
    category_column: Optional[str] = None,
) -> Dict[str, Any]:
    return DashboardBuilder.build_dashboard(
        df,
        date_column=date_column,
        value_column=value_column,
        category_column=category_column,
    )