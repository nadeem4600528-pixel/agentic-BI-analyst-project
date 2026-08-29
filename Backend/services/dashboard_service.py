"""Service layer for dashboard generation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from dashboard.dashboard_engine import DashboardBuilder
from reports.report_generator import build_comprehensive_report


class DashboardService:
    """Generate dashboard payloads for the frontend from raw data."""

    @staticmethod
    def build_dashboard(
        df: pd.DataFrame,
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
        category_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        dashboard = DashboardBuilder.build_dashboard(
            df,
            date_column=date_column,
            value_column=value_column,
            category_column=category_column,
        )
        # One governed BI-agent response: the visual model and the explainable
        # profiling, cleaning, analysis, and recommendation context use the same data.
        report = build_comprehensive_report(
            df,
            date_column=date_column,
            value_column=value_column,
            category_column=category_column,
        )
        summary = report.get("business_summary", {})
        profile = report.get("profiling_report", {})
        quality = profile.get("quality_score", {})
        executive = profile.get("executive_summary", {})
        return {
            **dashboard,
            "bi_agent": {
                "status": "ready",
                "role": "Professional BI Analyst",
                "workflow": ["profile", "assess_quality", "recommend_cleaning", "analyze", "visualize", "report"],
                "business_summary": summary,
                "quality": {
                    "score": quality.get("overall_quality_score", executive.get("quality_overview", {}).get("overall_score", 0)),
                    "level": quality.get("overall_quality_level", executive.get("quality_overview", {}).get("level", "Unknown")),
                    "issues": quality.get("summary", {}).get("total_quality_issues", 0),
                },
                "cleaning_plan": report.get("cleaning_report", {}).get("decision_plan", {}),
                "recommendations": report.get("recommendations", []),
                "profiling": {
                    "modules": profile.get("report_metadata", {}).get("modules_included", []),
                    "findings": executive.get("key_findings", []),
                    "critical_issues": executive.get("critical_issues", []),
                },
                "analysis": report.get("analysis_report", {}),
            },
        }

    @staticmethod
    def build_dashboard_from_records(
        records: list[dict[str, Any]],
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
        category_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not records:
            raise ValueError("No records provided to build dashboard.")
        df = pd.DataFrame(records)
        return DashboardService.build_dashboard(
            df,
            date_column=date_column,
            value_column=value_column,
            category_column=category_column,
        )