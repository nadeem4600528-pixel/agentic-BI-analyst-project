"""Service layer for dashboard generation.

The interactive dashboard must feel instant, so the default response is built
from the fast dashboard engine (KPI cards, chart pack, smart insights, layout
and filters). The heavyweight comprehensive profiling/cleaning report — which
runs the full semantic, statistics and business-rule profilers — is opt-in via
``include_report=True`` for callers that also want the governance narrative.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from dashboard.charts import _contains_unhashable
from dashboard.dashboard_engine import DashboardBuilder


def _lightweight_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Fast, always-available data-quality snapshot (no heavy profilers)."""
    # Only hashable columns can be factorized (duplicate detection). Nested
    # list/dict cells would raise "unhashable type", so skip them.
    hashable_cols = [c for c in df.columns if not _contains_unhashable(df[c])]
    work = df[hashable_cols] if hashable_cols else df.iloc[:, :0]
    n_rows = max(1, len(df))
    total_cells = max(1, work.shape[0] * work.shape[1])
    missing = int(work.isna().sum().sum())
    completeness = round((1 - missing / total_cells) * 100, 1)
    duplicate_rows = int(work.duplicated().sum())
    numeric = work.select_dtypes(include=[np.number])
    outlier_columns = 0
    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) < 5:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr and ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).any():
            outlier_columns += 1
    score = completeness
    if duplicate_rows:
        score -= min(5, round(duplicate_rows / n_rows * 100, 1))
    return {
        "score": round(max(0, min(100, score)), 1),
        "level": "High" if score >= 85 else "Medium" if score >= 70 else "Needs review",
        "completeness": completeness,
        "missing_cells": missing,
        "duplicate_rows": duplicate_rows,
        "outlier_columns": outlier_columns,
    }


class DashboardService:
    """Generate dashboard payloads for the frontend from raw data."""

    @staticmethod
    def build_dashboard(
        df: pd.DataFrame,
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
        category_column: Optional[str] = None,
        include_report: bool = False,
    ) -> Dict[str, Any]:
        try:
            dashboard = DashboardBuilder.build_dashboard(
                df,
                date_column=date_column,
                value_column=value_column,
                category_column=category_column,
            )
        except Exception:
            dashboard = {
                "kpis": [],
                "charts": [],
                "insights": [],
                "table": {"columns": [str(c) for c in df.columns], "rows": [], "total_rows": int(len(df))},
                "layout": {"sections": [], "spans": {}},
                "filters": [],
                "summary": {"rows": int(len(df)), "columns": int(len(df.columns))},
            }

        # Fast, always-present quality summary.
        try:
            quality = _lightweight_quality(df)
        except Exception:
            quality = {
                "score": 0.0, "level": "Needs review", "completeness": 0.0,
                "missing_cells": 0, "duplicate_rows": 0, "outlier_columns": 0,
            }
        bi_agent = {
            "status": "ready",
            "role": "Professional BI Analyst",
            "workflow": ["profile", "assess_quality", "analyze", "visualize", "report"],
            "quality": {
                "score": quality["score"],
                "level": quality["level"],
                "issues": quality["missing_cells"] + quality["duplicate_rows"],
            },
            "recommendations": [insight["detail"] for insight in dashboard.get("insights", [])],
            "profiling": {
                "modules": ["kpis", "charts", "insights", "quality"],
                "findings": [i["title"] for i in dashboard.get("insights", [])],
                "critical_issues": [
                    i["detail"] for i in dashboard.get("insights", []) if i.get("level") == "warning"
                ],
            },
        }

        # Opt-in heavyweight governance report (kept for the reports/BI narrative).
        if include_report:
            from reports.report_generator import build_comprehensive_report

            report = build_comprehensive_report(
                df,
                date_column=date_column,
                value_column=value_column,
                category_column=category_column,
            )
            profile = report.get("profiling_report", {})
            qs = profile.get("quality_score", {})
            executive = profile.get("executive_summary", {})
            bi_agent.update({
                "business_summary": report.get("business_summary", {}),
                "quality": {
                    "score": qs.get("overall_quality_score", quality["score"]),
                    "level": qs.get("overall_quality_level", quality["level"]),
                    "issues": qs.get("summary", {}).get("total_quality_issues", 0),
                },
                "cleaning_plan": report.get("cleaning_report", {}).get("decision_plan", {}),
                "recommendations": report.get("recommendations", []),
                "profiling": {
                    "modules": profile.get("report_metadata", {}).get("modules_included", []),
                    "findings": executive.get("key_findings", []),
                    "critical_issues": executive.get("critical_issues", []),
                },
                "analysis": report.get("analysis_report", {}),
            })

        return {**dashboard, "bi_agent": bi_agent, "quality": quality}

    @staticmethod
    def build_dashboard_from_records(
        records: list[dict[str, Any]],
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
        category_column: Optional[str] = None,
        include_report: bool = False,
    ) -> Dict[str, Any]:
        if not records:
            raise ValueError("No records provided to build dashboard.")
        df = pd.DataFrame(records)
        return DashboardService.build_dashboard(
            df,
            date_column=date_column,
            value_column=value_column,
            category_column=category_column,
            include_report=include_report,
        )
