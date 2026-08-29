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
    _contains_unhashable,
    _infer_category_column,
    _infer_datetime_column,
    _infer_numeric_column,
    _infer_numeric_columns,
    _to_datetime,
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

# Cap the rows we analyse so very large uploads build instantly and never OOM.
MAX_DASHBOARD_ROWS = 200_000


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

        # --- Bound the work so very large uploads stay fast and never OOM ---
        original_rows = int(len(df))
        truncated = False
        if original_rows > MAX_DASHBOARD_ROWS:
            truncated = True
            df = df.sample(n=MAX_DASHBOARD_ROWS, random_state=42).sort_index()

        # --- Auto-detect role columns (explicit args win) -----------------
        # Honor an explicit date column only if it actually parses as dates.
        date_col = _infer_datetime_column(df)
        if date_column and date_column in df.columns:
            probe = _to_datetime(df[date_column].dropna().head(200))
            if len(probe) > 0 and probe.notna().mean() >= 0.7:
                date_col = date_column
        value_col = (
            value_column if value_column and value_column in df.columns
            else _infer_numeric_column(df, exclude=[date_col] if date_col else [])
        )
        numeric_cols = _infer_numeric_columns(df, exclude=[date_col] if date_col else [])
        exclude = [c for c in [date_col, value_col] if c is not None]
        category_col = (
            category_column if category_column and category_column in df.columns
            and not _contains_unhashable(df[category_column])
            else _infer_category_column(df, exclude=exclude)
        )

        # --- Lightweight analysis (fast; no heavyweight profilers) --------
        # Every step is isolated so one bad column can never abort the dashboard.
        empty_analysis = {
            "statistics": {}, "correlations": {}, "trends": {"trends": []},
            "anomaly_insights": [], "segmentation": None, "business_insights": [],
        }
        try:
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
                "business_insights": [],
            }
        except Exception:
            analysis = dict(empty_analysis)

        # --- KPI tiles -----------------------------------------------------
        try:
            kpis = build_kpi_cards(
                df,
                analysis=analysis,
                value_column=value_col,
                date_column=date_col,
                category_column=category_col,
            )
        except Exception:
            kpis = []

        # --- Chart pack ----------------------------------------------------
        try:
            charts = generate_dashboard_charts(
                df,
                analysis=analysis,
                date_column=date_col,
                value_column=value_col,
                category_column=category_col,
            )
        except Exception:
            charts = []

        # --- Auto business insights ---------------------------------------
        try:
            insights = build_insights(
                df,
                date_column=date_col,
                value_column=value_col,
                category_column=category_col,
                charts=charts,
            )
        except Exception:
            insights = []

        # --- Interactive filter options -----------------------------------
        try:
            filters = DashboardBuilder._build_filters(df, date_col, value_col, category_col)
        except Exception:
            filters = []

        # --- Data preview table -------------------------------------------
        table_columns = [str(column) for column in df.columns]
        try:
            preview = df.head(100).where(pd.notna(df.head(100)), None)
            # Nested (list/dict) cells are valid JSON but not human-friendly;
            # stringify them so the preview table always renders cleanly.
            preview = preview.applymap(
                lambda v: str(v) if isinstance(v, (list, dict, set)) else v
            )
            table_rows = preview.to_dict(orient="records")
        except Exception:
            table_rows = []

        layout = build_dashboard_layout(charts)

        return {
            "kpis": kpis,
            "charts": charts,
            "insights": insights,
            "table": {"columns": table_columns, "rows": table_rows, "total_rows": original_rows},
            "layout": layout,
            "filters": filters,
            "truncated": truncated,
            "truncated_note": (
                f"Dashboard built from a {MAX_DASHBOARD_ROWS:,}-row sample "
                f"(dataset has {original_rows:,} rows)."
                if truncated else None
            ),
            # keep legacy keys for backwards compatibility
            "insights_legacy": insights,
            "summary": {
                "rows": original_rows,
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
            if _contains_unhashable(df[column]):
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
