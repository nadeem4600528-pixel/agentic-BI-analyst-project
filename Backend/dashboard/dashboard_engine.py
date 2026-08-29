"""Dashboard building engine — a Power BI / Tableau style business dashboard.

Given a (cleaned) DataFrame it auto-detects the date, measure and dimension
columns and produces a self-contained payload the frontend renders as an
interactive dashboard: KPI tiles with deltas/sparklines, a rich chart pack
(trends, composition, distribution, relationships), auto-generated business
insights, filter options, a data preview, and a sectioned layout.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .charts import (
    _infer_category_column,
    _infer_datetime_column,
    _infer_numeric_column,
    _infer_numeric_columns,
    generate_dashboard_charts,
)
from .fast_analysis import (
    detect_anomalies,
    fast_correlation,
    fast_numeric_stats,
    fast_segmentation,
    fast_trends,
)
from .insights import build_insights
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
                "insights": [],
                "layout": {"sections": [], "spans": {}},
                "filters": [],
                "table": {"columns": [], "rows": [], "total_rows": 0},
                "summary": {"rows": 0, "columns": 0},
            }

        # --- Auto-detect role columns (explicit args win) -----------------
        date_col = date_column or _infer_datetime_column(df)
        value_col = (
            value_column if value_column and value_column in df.columns
            else _infer_numeric_column(df, exclude=[date_col] if date_col else ())
        )
        numeric_cols = _infer_numeric_columns(df, exclude=[date_col] if date_col else ())
        category_col = category_column or _infer_category_column(df, exclude=[date_col, value_col])

        # --- Lightweight analysis (fast; no heavyweight profilers) --------
        trends = fast_trends(df, date_col, value_col)
        segmentation = (
            fast_segmentation(df, category_col, value_col)
            if category_col and value_col else None
        )
        analysis = {
            "statistics": fast_numeric_stats(df),
            "correlations": fast_correlation(df),
            "trends": trends,
            "anomaly_insights": detect_anomalies(df),
            "segmentation": segmentation,
            "business_insights": build_insights(
                df, date_column=date_col, value_column=value_col, category_column=category_col
            ),
        }

        # --- KPI tiles -----------------------------------------------------
        kpis = build_kpi_cards(
            df,
            analysis=analysis,
            value_column=value_col,
            date_column=date_col,
            category_column=category_col,
        )

        # --- Chart pack ----------------------------------------------------
        charts = generate_dashboard_charts(
            df,
            analysis=analysis,
            date_column=date_col,
            value_column=value_col,
            category_column=category_col,
        )

        # --- Auto business insights ---------------------------------------
        insights = build_insights(
            df,
            date_column=date_col,
            value_column=value_col,
            category_column=category_col,
            charts=charts,
        )

        # --- Interactive filter options -----------------------------------
        filters = DashboardBuilder._build_filters(df, date_col, value_col, category_col)

        # --- Data preview table -------------------------------------------
        table_columns = [str(column) for column in df.columns]
        preview = df.head(100).where(pd.notna(df.head(100)), None)
        table_rows = preview.to_dict(orient="records")

        layout = build_dashboard_layout(charts)

        return {
            "kpis": kpis,
            "charts": charts,
            "insights": insights,
            "table": {"columns": table_columns, "rows": table_rows, "total_rows": int(len(df))},
            "layout": layout,
            "filters": filters,
            # keep legacy keys for backwards compatibility
            "insights_legacy": analysis.get("business_insights", []),
            "summary": {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "value_column": value_col,
                "date_column": date_col,
                "category_column": category_col,
                "numeric_columns": numeric_cols,
            },
        }

    @staticmethod
    def _build_filters(
        df: pd.DataFrame,
        date_col: Optional[str],
        value_col: Optional[str],
        category_col: Optional[str],
    ) -> List[Dict[str, Any]]:
        filters: List[Dict[str, Any]] = []

        # Column role selectors
        filters.append({
            "name": "date_column",
            "label": "Date column",
            "type": "role",
            "value": date_col,
            "options": [None] + [str(c) for c in df.columns],
        })
        filters.append({
            "name": "value_column",
            "label": "Measure",
            "type": "role",
            "value": value_col,
            "options": [None] + [str(c) for c in df.select_dtypes(include=[np.number]).columns],
        })
        filters.append({
            "name": "category_column",
            "label": "Dimension",
            "type": "role",
            "value": category_col,
            "options": [None] + [str(c) for c in df.columns if c != value_col],
        })

        # Value filters for low-cardinality dimensions (like Power BI slicers)
        for column in df.columns:
            if column in {date_col, value_col}:
                continue
            if pd.api.types.is_numeric_dtype(df[column]):
                continue
            unique = df[column].dropna().astype(str).unique().tolist()
            if 1 < len(unique) <= 30:
                filters.append({
                    "name": f"slicer_{column}",
                    "label": str(column),
                    "type": "slicer",
                    "column": str(column),
                    "value": [],
                    "options": sorted(unique)[:50],
                })
        return filters


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
