"""
report_aggregator.py

Unified Data Understanding Report Generator for Agentic BI Analyst.

Responsibilities:
- Aggregate results from all profiling modules
- Generate executive summary
- Provide actionable insights
- Export to multiple formats (JSON, HTML, PDF-ready)

This module DOES NOT modify the input DataFrame.
"""

from typing import Any, Dict, List, Optional, Union
import json
from datetime import datetime

import pandas as pd
from .metadata import profile_metadata
from .structure import profile_structure
from .schema import profile_schema
from .statistics import profile_statistics
from .semantic import profile_semantics
from .completeness import profile_completeness
from .freshness import profile_freshness
from .quality_score import calculate_quality_score
from .correlation import profile_correlation
from .relationships import profile_relationships
from .business_rules import BusinessRuleEngine
from .governance import profile_governance


class DataUnderstandingReport:
    """
    Generates comprehensive data understanding report by aggregating
    all profiling modules.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tables: Optional[Dict[str, pd.DataFrame]] = None,
        metadata_profile: Optional[Dict[str, Any]] = None,
        statistics_profile: Optional[Dict[str, Any]] = None,
        reference_time: Optional[pd.Timestamp] = None,
        mandatory_columns: Optional[List[str]] = None,
        baseline_schema: Optional[Dict[str, Any]] = None,
        source_metadata: Optional[Dict[str, Any]] = None,
        transformations: Optional[List[Dict[str, Any]]] = None,
        approval_decisions: Optional[Dict[str, str]] = None
    ):
        """
        Initialize report generator.

        Parameters
        ----------
        df : Main DataFrame to profile
        tables : Optional dict of table_name -> DataFrame for multi-table analysis
        metadata_profile : Pre-computed metadata profile (optional)
        statistics_profile : Pre-computed statistics profile (optional)
        reference_time : Reference time for freshness analysis
        mandatory_columns : Columns that must be 100% complete
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DataUnderstandingReport requires a pandas DataFrame.")

        self.df = df
        self.tables = tables or {}
        self.metadata_profile = metadata_profile
        self.statistics_profile = statistics_profile
        self.reference_time = reference_time or pd.Timestamp.now()
        self.mandatory_columns = set(mandatory_columns or [])
        self.baseline_schema = baseline_schema
        self.source_metadata = source_metadata
        self.transformations = transformations
        self.approval_decisions = approval_decisions

    def generate(
        self,
        include_modules: Optional[List[str]] = None,
        business_rules: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete data understanding report.

        Parameters
        ----------
        include_modules : List of modules to include (None = all)
        business_rules : Optional pre-defined business rules

        Returns
        -------
        Complete report dictionary
        """
        all_modules = [
            "metadata", "structure", "schema", "statistics",
            "semantics", "completeness", "freshness",
            "quality_score", "correlations", "relationships",
            "business_rules", "governance", "executive_summary"
        ]

        modules_to_run = include_modules or all_modules

        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "reference_time": str(self.reference_time),
                "dataset_shape": {
                    "rows": int(len(self.df)),
                    "columns": int(len(self.df.columns))
                },
                "tables_analyzed": list(self.tables.keys()) if self.tables else ["main"],
                "modules_included": modules_to_run
            }
        }

        # Run profiling modules
        if "metadata" in modules_to_run:
            report["metadata"] = self.metadata_profile or profile_metadata(self.df)

        if "structure" in modules_to_run:
            report["structure"] = profile_structure(self.df)

        if "schema" in modules_to_run:
            report["schema"] = profile_schema(self.df)

        if "statistics" in modules_to_run:
            self.statistics_profile = self.statistics_profile or profile_statistics(self.df)
            report["statistics"] = self.statistics_profile

        if "semantics" in modules_to_run:
            report["semantics"] = profile_semantics(self.df)

        if "completeness" in modules_to_run:
            report["completeness"] = profile_completeness(
                self.df,
                mandatory_columns=list(self.mandatory_columns)
            )

        if "freshness" in modules_to_run:
            report["freshness"] = profile_freshness(self.df, reference_time=self.reference_time)

        if "quality_score" in modules_to_run:
            meta = self.metadata_profile or profile_metadata(self.df)
            stats = self.statistics_profile or profile_statistics(self.df)
            report["quality_score"] = calculate_quality_score(self.df, meta, stats)

        if "correlations" in modules_to_run:
            report["correlations"] = profile_correlation(self.df)

        if "relationships" in modules_to_run and self.tables:
            report["relationships"] = profile_relationships(self.tables)

        if "business_rules" in modules_to_run:
            engine = BusinessRuleEngine(self.df)
            if business_rules:
                for _ in business_rules:
                    # Rules should be BusinessRule objects or dicts
                    pass  # User can add rules via engine directly
            report["business_rules"] = engine.execute()

        if "governance" in modules_to_run:
            report.update(profile_governance(
                self.df,
                report=report,
                tables=self.tables,
                baseline_schema=self.baseline_schema,
                source_metadata=self.source_metadata,
                transformations=self.transformations,
                approval_decisions=self.approval_decisions
            ))

        # Executive summary (always generated)
        report["executive_summary"] = self._generate_executive_summary(report)

        return report

    def _generate_executive_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level executive summary from all profile results."""

        summary = {
            "dataset_overview": {},
            "quality_overview": {},
            "key_findings": [],
            "critical_issues": [],
            "recommendations": [],
            "risk_score": 0.0
        }

        # Dataset Overview
        shape = report["report_metadata"]["dataset_shape"]
        summary["dataset_overview"] = {
            "rows": shape["rows"],
            "columns": shape["columns"],
            "size_category": self._categorize_size(shape["rows"], shape["columns"])
        }

        # Quality Overview
        if "quality_score" in report:
            qs = report["quality_score"]
            summary["quality_overview"] = {
                "overall_score": qs.get("overall_quality_score", 0),
                "level": qs.get("overall_quality_level", "Unknown"),
                "total_issues": qs.get("summary", {}).get("total_quality_issues", 0),
                "critical_columns": qs.get("summary", {}).get("columns_critical", 0),
                "poor_columns": qs.get("summary", {}).get("columns_poor", 0)
            }
            summary["risk_score"] = max(0, 100 - qs.get("overall_quality_score", 0))

        # Completeness
        if "completeness" in report:
            comp = report["completeness"]["column_completeness"]["summary"]
            summary["dataset_overview"]["avg_completeness"] = comp.get("avg_completeness", 100)
            summary["dataset_overview"]["complete_columns"] = comp.get("complete_columns", 0)
            summary["dataset_overview"]["incomplete_columns"] = comp.get("incomplete_columns", 0)

            # Critical missing
            for col, info in report["completeness"]["column_completeness"]["columns"].items():
                if info.get("mandatory_violation"):
                    summary["critical_issues"].append(
                        f"Mandatory column '{col}' has {info['null_count']} missing values"
                    )

        # Duplicate Analysis
        if "structure" in report:
            dup_info = report["structure"].get("duplicate_rows", {})
            if dup_info.get("duplicate_row_count", 0) > 0:
                pct = dup_info.get("duplicate_row_percentage", 0)
                summary["key_findings"].append(
                    f"Found {dup_info['duplicate_row_count']} exact duplicate rows ({pct:.1f}%)"
                )
                if pct > 5:
                    summary["critical_issues"].append(
                        f"High duplicate rate: {pct:.1f}% - investigate deduplication"
                    )
                summary["recommendations"].append("Remove or investigate exact duplicate rows")

            entity_dup = report["structure"].get("entity_duplicates", {})
            if entity_dup.get("groups_found", 0) > 0:
                summary["key_findings"].append(
                    f"Found {entity_dup['groups_found']} potential entity duplicate groups"
                )

        # Outliers
        if "statistics" in report:
            outlier_cols = []
            for col, info in report["statistics"].get("outliers", {}).items():
                if info.get("outlier_percentage", 0) > 5:
                    outlier_cols.append(f"{col} ({info['outlier_percentage']:.1f}%)")
            if outlier_cols:
                summary["key_findings"].append(
                    f"High outlier rates in: {', '.join(outlier_cols[:5])}"
                )
                summary["recommendations"].append("Review outliers for data entry errors or valid extremes")

        # Constant/Near-constant columns
        if "statistics" in report:
            const_cols = []
            for col, info in report["statistics"].get("constant_columns", {}).items():
                if info.get("is_constant"):
                    const_cols.append(f"{col} (constant)")
                elif info.get("is_near_constant"):
                    const_cols.append(f"{col} ({info['top_frequency']:.1%} same value)")
            if const_cols:
                summary["key_findings"].append(
                    f"Constant/near-constant columns: {', '.join(const_cols[:5])}"
                )
                summary["recommendations"].append("Consider dropping constant columns")

        # PII Detection
        if "semantics" in report:
            pii_cols = []
            for col, info in report["semantics"].get("pii_detection", {}).items():
                if info.get("pii_detected"):
                    types = [p["type"] for p in info.get("pii_types", [])]
                    pii_cols.append(f"{col} ({', '.join(types)})")
            if pii_cols:
                summary["critical_issues"].append(
                    f"PII detected in: {', '.join(pii_cols[:5])} - ensure compliance"
                )
                summary["recommendations"].append("Mask or encrypt PII columns before sharing")

        # Semantic Classification
        if "semantics" in report:
            class_counts = {}
            for col, info in report["semantics"].get("column_classification", {}).items():
                primary = info.get("primary_semantic_type", "unknown")
                class_counts[primary] = class_counts.get(primary, 0) + 1
            summary["dataset_overview"]["semantic_distribution"] = class_counts

        # Freshness
        if "freshness" in report:
            fresh = report["freshness"].get("freshness_score", {})
            summary["dataset_overview"]["freshness_score"] = fresh.get("overall_score", 0)
            summary["dataset_overview"]["freshness_level"] = fresh.get("level", "unknown")
            if fresh.get("overall_score", 100) < 40:
                summary["critical_issues"].append("Data is stale - verify refresh pipeline")

        # Relationships
        if "relationships" in report:
            rels = report["relationships"].get("foreign_keys", {}).get("relationships", [])
            if rels:
                summary["key_findings"].append(
                    f"Detected {len(rels)} foreign key relationships across tables"
                )
            integrity = report["relationships"].get("referential_integrity", {})
            failed = integrity.get("summary", {}).get("failed", 0)
            if failed > 0:
                summary["critical_issues"].append(
                    f"{failed} referential integrity violations found"
                )

        # Correlations
        if "correlations" in report:
            strong = report["correlations"].get("pearson", {}).get("strong_pairs", [])
            if strong:
                top_corr = strong[0]
                summary["key_findings"].append(
                    f"Strong correlation: {top_corr['column_1']} ↔ {top_corr['column_2']} "
                    f"({top_corr['correlation']:.2f})"
                )

        # Business Rules
        if "business_rules" in report:
            br = report["business_rules"]
            if br.get("summary", {}).get("failed", 0) > 0:
                summary["critical_issues"].append(
                    f"{br['summary']['failed']} business rule violations"
                )
                if br["summary"].get("errors", 0) > 0:
                    summary["recommendations"].append("Fix business rule violations before analysis")

        # Schema Drift
        if "schema" in report:
            fingerprint = report["schema"].get("schema_fingerprint")
            if fingerprint:
                summary["dataset_overview"]["schema_fingerprint"] = fingerprint[:16] + "..."

        return summary

    def _categorize_size(self, rows: int, cols: int) -> str:
        """Categorize dataset size."""
        if rows == 0:
            return "empty"
        if rows < 1000 and cols < 20:
            return "small"
        if rows < 100000 and cols < 100:
            return "medium"
        if rows < 1000000 and cols < 500:
            return "large"
        return "very_large"

    def to_json(self, report: Dict[str, Any], filepath: Optional[str] = None) -> str:
        """Export report to JSON."""
        json_str = json.dumps(report, indent=2, default=str)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        return json_str

    def to_html(self, report: Dict[str, Any], filepath: Optional[str] = None) -> str:
        """Export report to HTML."""
        html = self._generate_html(report)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(html)
        return html

    def _generate_html(self, report: Dict[str, Any]) -> str:
        """Generate HTML report."""
        summary = report.get("executive_summary", {})
        meta = report.get("report_metadata", {})

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Understanding Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        .metric {{ display: inline-block; background: #ecf0f1; padding: 15px; margin: 10px; border-radius: 5px; min-width: 150px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        .critical {{ color: #e74c3c; }}
        .warning {{ color: #f39c12; }}
        .good {{ color: #27ae60; }}
        .finding {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }}
        .issue {{ background: #f8d7da; border-left: 4px solid #dc3545; padding: 10px; margin: 10px 0; }}
        .rec {{ background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 10px; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #dee2e6; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <h1>Data Understanding Report</h1>
    <p><strong>Generated:</strong> {meta.get('generated_at', 'N/A')}</p>
    <p><strong>Dataset:</strong> {meta.get('dataset_shape', {}).get('rows', 0):,} rows × {meta.get('dataset_shape', {}).get('columns', 0)} columns</p>

    <h2>Executive Summary</h2>

    <div class="metric">
        <div class="metric-value">{summary.get('quality_overview', {}).get('overall_score', 0):.1f}</div>
        <div class="metric-label">Quality Score</div>
    </div>
    <div class="metric">
        <div class="metric-value">{summary.get('dataset_overview', {}).get('avg_completeness', 100):.1f}%</div>
        <div class="metric-label">Completeness</div>
    </div>
    <div class="metric">
        <div class="metric-value">{summary.get('dataset_overview', {}).get('freshness_score', 100):.1f}</div>
        <div class="metric-label">Freshness</div>
    </div>
    <div class="metric">
        <div class="metric-value">{summary.get('risk_score', 0):.1f}</div>
        <div class="metric-label">Risk Score</div>
    </div>

    <h3>Dataset Overview</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Rows</td><td>{summary.get('dataset_overview', {}).get('rows', 0):,}</td></tr>
        <tr><td>Columns</td><td>{summary.get('dataset_overview', {}).get('columns', 0)}</td></tr>
        <tr><td>Size Category</td><td>{summary.get('dataset_overview', {}).get('size_category', 'N/A')}</td></tr>
        <tr><td>Avg Completeness</td><td>{summary.get('dataset_overview', {}).get('avg_completeness', 100):.1f}%</td></tr>
        <tr><td>Freshness Level</td><td>{summary.get('dataset_overview', {}).get('freshness_level', 'N/A')}</td></tr>
    </table>

    <h3>Quality Overview</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Overall Score</td><td>{summary.get('quality_overview', {}).get('overall_score', 0):.1f}</td></tr>
        <tr><td>Quality Level</td><td>{summary.get('quality_overview', {}).get('level', 'N/A')}</td></tr>
        <tr><td>Total Issues</td><td>{summary.get('quality_overview', {}).get('total_issues', 0)}</td></tr>
        <tr><td>Critical Columns</td><td>{summary.get('quality_overview', {}).get('critical_columns', 0)}</td></tr>
        <tr><td>Poor Columns</td><td>{summary.get('quality_overview', {}).get('poor_columns', 0)}</td></tr>
    </table>

    <h3>Key Findings</h3>
"""

        for finding in summary.get("key_findings", []):
            html += f'<div class="finding">{finding}</div>\n'

        html += '<h3>Critical Issues</h3>\n'
        for issue in summary.get("critical_issues", []):
            html += f'<div class="issue">{issue}</div>\n'

        html += '<h3>Recommendations</h3>\n'
        for rec in summary.get("recommendations", []):
            html += f'<div class="rec">{rec}</div>\n'

        # Semantic distribution
        if "semantic_distribution" in summary.get("dataset_overview", {}):
            html += '<h3>Semantic Column Distribution</h3>\n<table><tr><th>Type</th><th>Count</th></tr>\n'
            for k, v in summary["dataset_overview"]["semantic_distribution"].items():
                html += f'<tr><td>{k}</td><td>{v}</td></tr>\n'
            html += '</table>\n'

        html += """
</body>
</html>
"""
        return html


def generate_data_understanding_report(
    df: pd.DataFrame,
    tables: Optional[Dict[str, pd.DataFrame]] = None,
    metadata_profile: Optional[Dict[str, Any]] = None,
    statistics_profile: Optional[Dict[str, Any]] = None,
    reference_time: Optional[pd.Timestamp] = None,
    mandatory_columns: Optional[List[str]] = None,
    baseline_schema: Optional[Dict[str, Any]] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
    transformations: Optional[List[Dict[str, Any]]] = None,
    approval_decisions: Optional[Dict[str, str]] = None,
    include_modules: Optional[List[str]] = None,
    business_rules: Optional[List[Dict[str, Any]]] = None,
    output_format: str = "json",
    output_path: Optional[str] = None
) -> Union[Dict[str, Any], str]:
    """
    Convenience function to generate complete data understanding report.

    Parameters
    ----------
    df : Main DataFrame
    tables : Optional dict for multi-table analysis
    metadata_profile : Pre-computed metadata
    statistics_profile : Pre-computed statistics
    reference_time : Time reference for freshness
    mandatory_columns : Required columns
    include_modules : Modules to include
    business_rules : Pre-defined rules
    output_format : 'json', 'html', or 'dict'
    output_path : File path to save output

    Returns
    -------
    Report as dict, JSON string, or HTML string
    """
    generator = DataUnderstandingReport(
        df=df,
        tables=tables,
        metadata_profile=metadata_profile,
        statistics_profile=statistics_profile,
        reference_time=reference_time,
        mandatory_columns=mandatory_columns,
        baseline_schema=baseline_schema,
        source_metadata=source_metadata,
        transformations=transformations,
        approval_decisions=approval_decisions
    )

    report = generator.generate(include_modules=include_modules, business_rules=business_rules)

    if output_format == "json":
        return generator.to_json(report, output_path)
    elif output_format == "html":
        return generator.to_html(report, output_path)
    else:
        return report