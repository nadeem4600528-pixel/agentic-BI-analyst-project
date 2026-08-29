"""
governance.py

Governance profiling for the data-profiling use cases that need explicit
baselines, quality evidence, lineage declarations, and human controls.

This module never changes input data or performs cleaning actions.
"""

from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from .schema import SchemaProfiler
from .statistics import StatisticsProfiler


class GovernanceProfiler:
    """Profiles governance and remediation metadata for a DataFrame."""

    def __init__(
        self,
        df: pd.DataFrame,
        tables: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("GovernanceProfiler requires a pandas DataFrame.")
        self.df = df
        self.tables = tables or {}

    def schema_drift(
        self,
        baseline_schema: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compare the current schema with a supplied baseline schema."""
        current = {
            str(column): str(dtype)
            for column, dtype in self.df.dtypes.items()
        }
        current_fingerprint = SchemaProfiler(self.df).schema_fingerprint()

        if baseline_schema is None:
            return {
                "status": "baseline_required",
                "drift_detected": False,
                "current_schema": current,
                "current_fingerprint": current_fingerprint,
                "message": "Provide baseline_schema to evaluate schema drift.",
            }

        baseline_value: Any = baseline_schema
        if "columns" in baseline_schema and isinstance(baseline_schema["columns"], Mapping):
            baseline_value = baseline_schema["columns"]
        baseline = {str(column): str(dtype) for column, dtype in baseline_value.items()}

        added = sorted(set(current) - set(baseline))
        removed = sorted(set(baseline) - set(current))
        changed = sorted(
            column for column in set(current) & set(baseline)
            if current[column] != baseline[column]
        )

        return {
            "status": "drift_detected" if added or removed or changed else "unchanged",
            "drift_detected": bool(added or removed or changed),
            "added_columns": added,
            "removed_columns": removed,
            "changed_columns": [
                {
                    "column": column,
                    "baseline_dtype": baseline[column],
                    "current_dtype": current[column],
                }
                for column in changed
            ],
            "baseline_schema": baseline,
            "current_schema": current,
            "current_fingerprint": current_fingerprint,
        }

    def consistency(self) -> Dict[str, Any]:
        """Detect common representation inconsistencies in categorical text."""
        columns: Dict[str, Any] = {}
        total_issues = 0

        for column in self.df.columns:
            series = self.df[column].dropna()
            if not (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
            ):
                continue

            values = series.astype(str)
            normalized = values.str.strip().str.casefold()
            issues: Dict[str, Any] = {}
            if values.nunique() != normalized.nunique():
                issues["case_or_whitespace_variants"] = int(
                    values.nunique() - normalized.nunique()
                )
            whitespace_count = int((values != values.str.strip()).sum())
            if whitespace_count:
                issues["leading_or_trailing_whitespace"] = whitespace_count
            if issues:
                total_issues += len(issues)
            columns[str(column)] = {
                "consistent": not bool(issues),
                "issues": issues,
                "unique_values": int(values.nunique()),
            }

        return {
            "consistent": total_issues == 0,
            "total_issue_types": total_issues,
            "columns": columns,
        }

    def validity(
        self,
        expected_ranges: Optional[Dict[str, Dict[str, Any]]] = None,
        validation_rules: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Combine type/range and invalid-value evidence into one result."""
        schema = SchemaProfiler(self.df)
        statistics = StatisticsProfiler(self.df)
        range_results = schema.validate_ranges(expected_ranges)
        invalid_results = statistics.invalid_values(validation_rules)
        invalid_columns = [
            column for column, result in invalid_results.items()
            if result.get("invalid_count", 0) > 0
        ]
        range_columns = [
            column for column, result in range_results.items()
            if not result.get("valid", True)
        ]
        return {
            "valid": not invalid_columns and not range_columns,
            "invalid_columns": invalid_columns,
            "range_violation_columns": range_columns,
            "invalid_values": invalid_results,
            "range_validation": range_results,
        }

    def integrity(
        self,
        structure_profile: Optional[Dict[str, Any]] = None,
        relationship_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Summarize row/key and optional cross-table integrity evidence."""
        structure = structure_profile or {}
        duplicates = structure.get("duplicate_rows", {})
        duplicate_count = int(duplicates.get("duplicate_row_count", 0))
        result: Dict[str, Any] = {
            "integrity_ok": duplicate_count == 0,
            "duplicate_row_count": duplicate_count,
            "duplicate_rows": duplicates,
        }
        if relationship_profile is not None:
            referential = relationship_profile.get("referential_integrity", {})
            failed = int(referential.get("summary", {}).get("failed", 0))
            result["referential_integrity"] = referential
            result["integrity_ok"] = result["integrity_ok"] and failed == 0
        else:
            result["referential_integrity"] = {
                "status": "not_applicable",
                "message": "No multi-table relationship profile supplied.",
            }
        return result

    def confidence_scoring(
        self,
        semantics: Optional[Dict[str, Any]] = None,
        relationships: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Expose confidence evidence from semantic and relationship profiling."""
        semantic_scores: List[float] = []
        for result in (semantics or {}).get("column_classification", {}).values():
            score = result.get("confidence")
            if isinstance(score, (int, float)):
                semantic_scores.append(float(score))

        relationship_scores = [
            float(result["strength_score"])
            for result in (relationships or {}).get("relationship_strength", {}).get(
                "scored_relationships", []
            )
            if isinstance(result.get("strength_score"), (int, float))
        ]
        scores = semantic_scores + relationship_scores
        return {
            "available": bool(scores),
            "overall_confidence": round(sum(scores) / len(scores), 4) if scores else None,
            "semantic_scores_count": len(semantic_scores),
            "relationship_scores_count": len(relationship_scores),
        }

    def lineage(
        self,
        source_metadata: Optional[Dict[str, Any]] = None,
        transformations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return declared lineage metadata without guessing undocumented sources."""
        return {
            "status": "declared" if source_metadata or transformations else "unknown",
            "source": source_metadata or {},
            "transformations": transformations or [],
            "target": {
                "rows": int(len(self.df)),
                "columns": [str(column) for column in self.df.columns],
            },
        }

    def cleaning_recommendations(
        self,
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create prioritized, non-destructive cleaning recommendations."""
        recommendations: List[Dict[str, Any]] = []
        statistics = report.get("statistics", {})
        semantics = report.get("semantics", {})

        for column, result in statistics.get("missing_values", {}).items():
            if result.get("missing_count", 0) > 0:
                recommendations.append({
                    "action": "review_missing_values",
                    "column": column,
                    "priority": "high" if result.get("missing_percentage", 0) >= 20 else "medium",
                    "evidence": result,
                    "risk": "medium",
                })
        for column, result in statistics.get("hidden_nulls", {}).items():
            if result.get("hidden_null_count", 0) > 0:
                recommendations.append({
                    "action": "standardize_hidden_nulls",
                    "column": column,
                    "priority": "high",
                    "evidence": result,
                    "risk": "medium",
                })
        for column, result in statistics.get("outliers", {}).items():
            if result.get("outlier_count", 0) > 0:
                recommendations.append({
                    "action": "review_outliers",
                    "column": column,
                    "priority": "medium",
                    "evidence": result,
                    "risk": "high",
                })
        for column, result in semantics.get("standardization_detection", {}).items():
            if result.get("needs_standardization"):
                recommendations.append({
                    "action": "standardize_column_values",
                    "column": column,
                    "priority": "medium",
                    "evidence": result,
                    "risk": "high",
                })
        return {
            "recommendations": recommendations,
            "total": len(recommendations),
            "requires_approval": any(item["risk"] == "high" for item in recommendations),
        }

    def approval_gate(
        self,
        recommendations: Dict[str, Any],
        decisions: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        """Apply explicit human decisions to recommendations; default is pending."""
        decisions = decisions or {}
        actions = []
        for item in recommendations.get("recommendations", []):
            action_id = f"{item['action']}:{item.get('column', '*')}"
            decision = decisions.get(action_id, "pending")
            if decision not in {"pending", "approved", "rejected"}:
                decision = "pending"
            actions.append({**item, "id": action_id, "approval": decision})
        return {
            "status": "approved" if actions and all(item["approval"] == "approved" for item in actions) else "pending",
            "automatic_changes_performed": False,
            "actions": actions,
            "pending_count": sum(item["approval"] == "pending" for item in actions),
        }


def profile_governance(
    df: pd.DataFrame,
    report: Optional[Dict[str, Any]] = None,
    tables: Optional[Dict[str, pd.DataFrame]] = None,
    baseline_schema: Optional[Mapping[str, Any]] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
    transformations: Optional[List[Dict[str, Any]]] = None,
    approval_decisions: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Generate governance outputs for a completed profiling report."""
    profiler = GovernanceProfiler(df, tables)
    report = report or {}
    recommendations = profiler.cleaning_recommendations(report)
    return {
        "schema_drift": profiler.schema_drift(baseline_schema),
        "data_consistency": profiler.consistency(),
        "data_validity": profiler.validity(),
        "data_integrity": profiler.integrity(
            report.get("structure"), report.get("relationships")
        ),
        "confidence_scoring": profiler.confidence_scoring(
            report.get("semantics"), report.get("relationships")
        ),
        "data_lineage": profiler.lineage(source_metadata, transformations),
        "cleaning_recommendation_plan": recommendations,
        "human_approval": profiler.approval_gate(
            recommendations, approval_decisions
        ),
    }
