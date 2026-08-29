"""Report generation agent for profiling, cleaning, analysis, and recommendations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from analysis.analysis_engine import analyze_dataset
from cleaning.cleaning_agent import CleaningAgent
from profiling.profiler import DataProfiler


def build_profiling_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a profiling-only report."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("build_profiling_report requires a pandas DataFrame.")
    return DataProfiler(df).profile()


def build_cleaning_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a cleaning plan and audit based on profiling evidence."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("build_cleaning_report requires a pandas DataFrame.")

    profiling_report = DataProfiler(df).profile()
    agent = CleaningAgent(df)
    plan = agent.plan(profiling_report)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profiling_summary": {
            "quality_score": profiling_report.get("quality_score", {}).get("overall_quality_score"),
            "quality_level": profiling_report.get("quality_score", {}).get("overall_quality_level"),
            "rows": profiling_report.get("report_metadata", {}).get("dataset_shape", {}).get("rows"),
            "columns": profiling_report.get("report_metadata", {}).get("dataset_shape", {}).get("columns"),
        },
        "decision_plan": plan,
        "recommendations": [
            item.get("reason", "Review cleaning recommendation")
            for item in plan.get("decisions", [])
        ],
    }


def build_analysis_report(
    df: pd.DataFrame,
    date_column: Optional[str] = None,
    value_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate analysis-only insights for the dataset."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("build_analysis_report requires a pandas DataFrame.")
    return analyze_dataset(df, date_column=date_column, value_column=value_column)


def build_business_summary(
    profiling_report: Optional[Dict[str, Any]] = None,
    cleaning_report: Optional[Dict[str, Any]] = None,
    analysis_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct a business-ready summary from profiling, cleaning, and analysis results."""
    insights: List[str] = []
    summary: Dict[str, Any] = {
        "headline": "Data quality and business performance are ready for review.",
        "dataset_overview": {},
        "quality_overview": {},
        "key_insights": insights,
        "risk_level": "medium",
    }

    if profiling_report:
        exec_summary = profiling_report.get("executive_summary", {})
        summary["dataset_overview"] = exec_summary.get("dataset_overview", {})
        summary["quality_overview"] = exec_summary.get("quality_overview", {})
        if exec_summary.get("key_findings"):
            insights.extend(exec_summary["key_findings"][:3])
        if exec_summary.get("critical_issues"):
            summary["risk_level"] = "high" if exec_summary["critical_issues"] else summary["risk_level"]

    if analysis_report:
        business_insights = analysis_report.get("business_insights", [])
        if business_insights:
            insights.extend(business_insights[:3])

    if cleaning_report:
        cleaning_plan = cleaning_report.get("decision_plan", {})
        decisions = cleaning_plan.get("decisions", [])
        if decisions:
            high_risk_count = sum(1 for item in decisions if item.get("risk") == "high")
            summary["cleaning_actions"] = {
                "total": len(decisions),
                "high_risk": high_risk_count,
            }
            if high_risk_count:
                insights.append(f"{high_risk_count} high-risk cleaning actions require approval before transformation.")

    summary["key_insights"] = insights
    summary["headline"] = (
        "The dataset is operationally usable but requires review on data quality and transformational risk."
        if summary["risk_level"] == "high"
        else "The dataset is in good shape for reporting and analysis."
    )
    return summary


def build_ml_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a placeholder ML report. This project does not yet have a production ML module."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("build_ml_report requires a pandas DataFrame.")

    return {
        "status": "not_configured",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "none",
        "summary": "No ML pipeline is currently configured for this project.",
        "recommendations": [
            "Add a supervised or unsupervised ML stage after the data is cleaned and profiled.",
            "Define target variable, feature set, and evaluation metrics before deployment.",
        ],
        "rows": len(df),
        "columns": len(df.columns),
    }


def build_comprehensive_report(
    df: pd.DataFrame,
    date_column: Optional[str] = None,
    value_column: Optional[str] = None,
    category_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a full report payload including profiling, cleaning, analysis, ML, summary, and recommendations."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("build_comprehensive_report requires a pandas DataFrame.")

    profiling_report = build_profiling_report(df)
    cleaning_report = build_cleaning_report(df)
    analysis_report = build_analysis_report(df, date_column=date_column, value_column=value_column)
    ml_report = build_ml_report(df)
    business_summary = build_business_summary(profiling_report, cleaning_report, analysis_report)

    recommendations: List[str] = []
    recommendations.extend(
        profiling_report.get("executive_summary", {}).get("recommendations", [])
    )
    recommendations.extend(cleaning_report.get("recommendations", []))
    recommendations.extend(analysis_report.get("business_insights", []))
    recommendations.extend(ml_report.get("recommendations", []))

    deduplicated: List[str] = []
    seen = set()
    for item in recommendations:
        key = str(item)
        if key not in seen:
            deduplicated.append(key)
            seen.add(key)

    return {
        "report_type": "comprehensive_data_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_summary": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "date_column": date_column,
            "value_column": value_column,
            "category_column": category_column,
        },
        "profiling_report": profiling_report,
        "cleaning_report": cleaning_report,
        "analysis_report": analysis_report,
        "ml_report": ml_report,
        "business_summary": business_summary,
        "recommendations": deduplicated,
        "sections": [
            "profiling_report",
            "cleaning_report",
            "analysis_report",
            "ml_report",
            "business_summary",
            "recommendations",
        ],
    }
