"""Comprehensive Analysis and Insight Agent orchestrating KPIs, statistics, trends, correlations, and anomalies."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .correlation import analyze_correlation
from .kpi import generate_kpis
from .semantic import analyze_segmentation, analyze_semantics
from .statistics import analyze_statistics
from .trends import analyze_trends


class AnalysisAgent:
    """Orchestrates comprehensive data analysis, insight generation, and anomaly detection."""

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("AnalysisAgent requires a pandas DataFrame.")
        self.df = df.copy()

    def analyze(
        self,
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run complete analysis suite including KPIs, statistics, trends, correlations, and insights."""
        kpis = generate_kpis(self.df)
        stats = analyze_statistics(self.df)
        correlations = analyze_correlation(self.df)
        semantics = analyze_semantics(self.df)
        trends = analyze_trends(self.df, date_column=date_column, value_column=value_column)

        segmentation = self._infer_segmentation(value_column=value_column)
        insights = self._generate_business_insights(
            kpis=kpis,
            stats=stats,
            correlations=correlations,
            segmentation=segmentation,
            trends=trends,
        )
        anomalies = self._detect_anomalies()

        return {
            "kpis": kpis,
            "statistics": stats,
            "correlations": correlations,
            "semantics": semantics,
            "trends": trends,
            "business_insights": insights,
            "anomaly_insights": anomalies,
        }

    def _infer_segmentation(self, value_column: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Infer a sensible group-by segment and measure for business summaries."""
        if self.df.empty:
            return None

        metric_column = value_column
        if metric_column not in self.df.columns:
            numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
            metric_column = numeric_cols[0] if numeric_cols else None

        if not metric_column or metric_column not in self.df.columns:
            return None

        if not pd.api.types.is_numeric_dtype(self.df[metric_column]):
            return None

        categorical_candidates = []
        for column in self.df.columns:
            if column == metric_column:
                continue
            series = self.df[column]
            if pd.api.types.is_numeric_dtype(series):
                continue
            unique_count = series.dropna().nunique()
            if unique_count > 0 and unique_count <= min(20, max(2, len(self.df) // 10)):
                categorical_candidates.append(column)

        if not categorical_candidates:
            return None

        group_column = categorical_candidates[0]
        return analyze_segmentation(self.df, group_column=group_column, measure_column=metric_column)

    def _generate_business_insights(
        self,
        kpis: list,
        stats: dict,
        correlations: dict,
        segmentation: Optional[Dict[str, Any]] = None,
        trends: Optional[Dict[str, Any]] = None,
    ) -> list[str]:
        """Generate concise, actionable business summary insights."""
        insights: list[str] = []

        if self.df.empty:
            return ["Dataset is empty; no meaningful insights can be generated."]

        total_records = len(self.df)
        insights.append(f"Dataset contains {total_records:,} records across {len(self.df.columns)} features.")

        if isinstance(stats, dict):
            summary = stats.get("summary", {})
            numeric_count = summary.get("numeric_columns_count", 0)
            insights.append(f"{numeric_count} numeric columns are available for KPI and statistical analysis.")

        if kpis:
            total_records_kpi = next((item for item in kpis if item.get("type") == "count"), None)
            if total_records_kpi:
                insights.append(f"The dataset holds {total_records_kpi.get('value', 0):,} total rows for analysis.")

        if segmentation and isinstance(segmentation, dict):
            segments = segmentation.get("segments") or []
            if segments:
                top_segment = max(segments, key=lambda item: float(item.get("total", item.get("sum", 0)) or 0))
                average_value = float(top_segment.get("average", top_segment.get("mean", 0)) or 0)
                insights.append(
                    f"Top segment '{top_segment.get('group_column', 'segment')}' has {top_segment.get('count', 0)} records "
                    f"with an average of {average_value:,.2f} for '{top_segment.get('measure_column')}'."
                )

        strong_pairs = []
        corr_matrix = correlations.get("correlation_matrix", {}) if isinstance(correlations, dict) else {}
        for col1, targets in corr_matrix.items():
            if not isinstance(targets, dict):
                continue
            for col2, val in targets.items():
                if col1 == col2 or not isinstance(val, (int, float)):
                    continue
                if abs(float(val)) >= 0.75:
                    strong_pairs.append(f"'{col1}' and '{col2}' (r={float(val):.2f})")

        if strong_pairs:
            unique_pairs = list(dict.fromkeys(strong_pairs))[:3]
            insights.append(f"Strong linear relationships detected between: {', '.join(unique_pairs)}.")

        if trends and isinstance(trends, dict):
            trend_rows = trends.get("trends") or []
            if trend_rows and len(trend_rows) >= 2:
                first = trend_rows[0]
                last = trend_rows[-1]
                if first.get("sum") is not None and last.get("sum") is not None:
                    first_total = float(first.get("sum", 0))
                    last_total = float(last.get("sum", 0))
                    change = (last_total - first_total) / first_total if first_total else 0.0
                    insights.append(
                        f"Trend analysis shows a {change * 100:.1f}% change in '{trends.get('value_column')}' over the observed time range."
                    )

        return insights

    def _detect_anomalies(self) -> list[dict[str, Any]]:
        """Identify obvious statistical anomalies using IQR bounds."""
        anomalies: list[dict[str, Any]] = []
        numeric_cols = self.df.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            series = self.df[col].dropna()
            if len(series) < 5:
                continue

            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (series < lower) | (series > upper)
            outlier_count = int(outlier_mask.sum())

            if outlier_count > 0:
                anomalies.append(
                    {
                        "column": col,
                        "outlier_count": outlier_count,
                        "percentage": round(outlier_count / len(series) * 100, 2),
                        "description": f"Detected {outlier_count} statistical outlier values outside IQR bounds in '{col}'.",
                    }
                )

        return anomalies


def analyze_dataset(
    df: pd.DataFrame,
    date_column: Optional[str] = None,
    value_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience entry point for AnalysisAgent."""
    return AnalysisAgent(df).analyze(date_column=date_column, value_column=value_column)

