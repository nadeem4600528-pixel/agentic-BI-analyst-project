"""
business_rules.py

Business Rule Validation & Aggregation Safety module for Agentic BI Analyst.

Responsibilities:
- Custom business rule definition and validation
- Rule types: range, pattern, cross-field, conditional, aggregation
- Rule engine with pass/fail/warning outcomes
- Aggregation safety checks (grain alignment, fan-out detection)
- Safe division, null handling in aggregations
- Rollup validation (subtotal = detail)
- Time-series aggregation consistency
- Rule violation reporting and scoring

This module DOES NOT modify the input DataFrame.
"""

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import pandas as pd


class RuleSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class RuleType(Enum):
    RANGE = "range"
    PATTERN = "pattern"
    CROSS_FIELD = "cross_field"
    CONDITIONAL = "conditional"
    AGGREGATION = "aggregation"
    CUSTOM = "custom"


@dataclass
class BusinessRule:
    """Represents a single business rule."""
    name: str
    rule_type: RuleType
    severity: RuleSeverity = RuleSeverity.ERROR
    description: str = ""
    columns: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[Callable] = None
    message: str = "Rule violation"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.rule_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "columns": self.columns,
            "parameters": self.parameters,
            "message": self.message
        }


class BusinessRuleEngine:
    """
    Engine for defining, managing, and executing business rules.
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("BusinessRuleEngine requires a pandas DataFrame.")
        self.df = df
        self.rules: List[BusinessRule] = []

    # =========================================================
    # RULE DEFINITION HELPERS
    # =========================================================

    def add_range_rule(
        self,
        name: str,
        column: str,
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
        severity: RuleSeverity = RuleSeverity.ERROR,
        description: str = ""
    ) -> "BusinessRuleEngine":
        """Add a numeric range validation rule."""
        params = {}
        if min_val is not None:
            params["min"] = min_val
        if max_val is not None:
            params["max"] = max_val

        rule = BusinessRule(
            name=name,
            rule_type=RuleType.RANGE,
            severity=severity,
            description=description or f"{column} must be between {min_val} and {max_val}",
            columns=[column],
            parameters=params,
            message=f"{column} value outside allowed range"
        )
        self.rules.append(rule)
        return self

    def add_pattern_rule(
        self,
        name: str,
        column: str,
        pattern: str,
        severity: RuleSeverity = RuleSeverity.ERROR,
        description: str = ""
    ) -> "BusinessRuleEngine":
        """Add a regex pattern validation rule."""
        rule = BusinessRule(
            name=name,
            rule_type=RuleType.PATTERN,
            severity=severity,
            description=description or f"{column} must match pattern {pattern}",
            columns=[column],
            parameters={"pattern": pattern},
            message=f"{column} does not match required pattern"
        )
        self.rules.append(rule)
        return self

    def add_cross_field_rule(
        self,
        name: str,
        columns: List[str],
        condition: Callable[[pd.Series], pd.Series],
        severity: RuleSeverity = RuleSeverity.ERROR,
        description: str = "",
        message: str = "Cross-field validation failed"
    ) -> "BusinessRuleEngine":
        """Add a cross-field validation rule using a custom function."""
        rule = BusinessRule(
            name=name,
            rule_type=RuleType.CROSS_FIELD,
            severity=severity,
            description=description or f"Cross-field rule on {', '.join(columns)}",
            columns=columns,
            condition=condition,
            message=message
        )
        self.rules.append(rule)
        return self

    def add_conditional_rule(
        self,
        name: str,
        if_column: str,
        if_condition: Callable[[pd.Series], pd.Series],
        then_column: str,
        then_condition: Callable[[pd.Series], pd.Series],
        severity: RuleSeverity = RuleSeverity.ERROR,
        description: str = "",
        message: str = "Conditional rule violated"
    ) -> "BusinessRuleEngine":
        """Add a conditional rule: IF condition THEN condition."""
        def combined_condition(row):
            if_met = if_condition(row[if_column])
            then_met = then_condition(row[then_column])
            return (~if_met) | then_met  # If not IF, or IF and THEN

        rule = BusinessRule(
            name=name,
            rule_type=RuleType.CONDITIONAL,
            severity=severity,
            description=description or f"If {if_column} condition met, then {then_column} must satisfy condition",
            columns=[if_column, then_column],
            parameters={"if_column": if_column, "then_column": then_column},
            condition=combined_condition,
            message=message
        )
        self.rules.append(rule)
        return self

    def add_aggregation_rule(
        self,
        name: str,
        group_by: List[str],
        measure_column: str,
        aggregation: str,
        expected_value: Optional[Union[int, float]] = None,
        tolerance: float = 0.01,
        severity: RuleSeverity = RuleSeverity.WARNING,
        description: str = ""
    ) -> "BusinessRuleEngine":
        """Add an aggregation validation rule (e.g., sum of amounts = expected total)."""
        rule = BusinessRule(
            name=name,
            rule_type=RuleType.AGGREGATION,
            severity=severity,
            description=description or f"Aggregation of {measure_column} by {group_by}",
            columns=[measure_column] + group_by,
            parameters={
                "group_by": group_by,
                "measure": measure_column,
                "aggregation": aggregation,
                "expected": expected_value,
                "tolerance": tolerance
            },
            message=f"Aggregation mismatch for {measure_column}"
        )
        self.rules.append(rule)
        return self

    def add_custom_rule(
        self,
        name: str,
        condition: Callable[[pd.DataFrame], pd.Series],
        columns: List[str],
        severity: RuleSeverity = RuleSeverity.ERROR,
        description: str = "",
        message: str = "Custom rule violation"
    ) -> "BusinessRuleEngine":
        """Add a completely custom rule function."""
        rule = BusinessRule(
            name=name,
            rule_type=RuleType.CUSTOM,
            severity=severity,
            description=description,
            columns=columns,
            condition=condition,
            message=message
        )
        self.rules.append(rule)
        return self

    # =========================================================
    # COMMON PRE-BUILT RULES
    # =========================================================

    def add_positive_value_rule(
        self,
        column: str,
        severity: RuleSeverity = RuleSeverity.ERROR
    ) -> "BusinessRuleEngine":
        """Ensure column has only positive values."""
        return self.add_range_rule(
            name=f"{column}_positive",
            column=column,
            min_val=0,
            severity=severity,
            description=f"{column} must be positive"
        )

    def add_non_null_rule(
        self,
        column: str,
        severity: RuleSeverity = RuleSeverity.ERROR
    ) -> "BusinessRuleEngine":
        """Ensure column has no null values."""
        def check_not_null(df):
            return df[column].notna()

        return self.add_custom_rule(
            name=f"{column}_not_null",
            condition=check_not_null,
            columns=[column],
            severity=severity,
            description=f"{column} must not be null",
            message=f"{column} contains null values"
        )

    def add_unique_rule(
        self,
        column: str,
        severity: RuleSeverity = RuleSeverity.ERROR
    ) -> "BusinessRuleEngine":
        """Ensure column values are unique."""
        def check_unique(df):
            return ~df[column].duplicated(keep=False)

        return self.add_custom_rule(
            name=f"{column}_unique",
            condition=check_unique,
            columns=[column],
            severity=severity,
            description=f"{column} must have unique values",
            message=f"{column} contains duplicate values"
        )

    def add_date_order_rule(
        self,
        start_column: str,
        end_column: str,
        severity: RuleSeverity = RuleSeverity.ERROR
    ) -> "BusinessRuleEngine":
        """Ensure start date <= end date."""
        def check_dates(df):
            return df[start_column] <= df[end_column]

        return self.add_cross_field_rule(
            name=f"{start_column}_before_{end_column}",
            columns=[start_column, end_column],
            condition=check_dates,
            severity=severity,
            description=f"{start_column} must be before or equal to {end_column}",
            message=f"{start_column} is after {end_column}"
        )

    def add_email_format_rule(
        self,
        column: str,
        severity: RuleSeverity = RuleSeverity.WARNING
    ) -> "BusinessRuleEngine":
        """Validate email format."""
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return self.add_pattern_rule(
            name=f"{column}_email_format",
            column=column,
            pattern=email_pattern,
            severity=severity,
            description=f"{column} must be valid email format"
        )

    # =========================================================
    # RULE EXECUTION
    # =========================================================

    def execute(self) -> Dict[str, Any]:
        """
        Execute all rules and return results.

        Returns
        -------
        Dict with rule results, violations, and summary.
        """
        results = {
            "rule_results": [],
            "violations": [],
            "summary": {
                "total_rules": len(self.rules),
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "errors": 0
            }
        }

        for rule in self.rules:
            rule_result = self._execute_rule(rule)
            results["rule_results"].append(rule_result)

            if rule_result["status"] == "PASSED":
                results["summary"]["passed"] += 1
            elif rule_result["status"] == "FAILED":
                results["summary"]["failed"] += 1
                if rule.severity == RuleSeverity.ERROR:
                    results["summary"]["errors"] += 1
                else:
                    results["summary"]["warnings"] += 1

            # Collect violations
            for violation in rule_result.get("violations", []):
                violation["rule_name"] = rule.name
                violation["rule_severity"] = rule.severity.value
                results["violations"].append(violation)

        return results

    def _execute_rule(self, rule: BusinessRule) -> Dict[str, Any]:
        """Execute a single rule."""
        try:
            if rule.rule_type == RuleType.RANGE:
                return self._execute_range_rule(rule)
            elif rule.rule_type == RuleType.PATTERN:
                return self._execute_pattern_rule(rule)
            elif rule.rule_type == RuleType.CROSS_FIELD:
                return self._execute_cross_field_rule(rule)
            elif rule.rule_type == RuleType.CONDITIONAL:
                return self._execute_conditional_rule(rule)
            elif rule.rule_type == RuleType.AGGREGATION:
                return self._execute_aggregation_rule(rule)
            elif rule.rule_type == RuleType.CUSTOM:
                return self._execute_custom_rule(rule)
            else:
                return {"rule": rule.name, "status": "ERROR", "message": "Unknown rule type"}
        except Exception as e:
            return {
                "rule": rule.name,
                "status": "ERROR",
                "message": f"Rule execution failed: {str(e)}",
                "violations": []
            }

    def _execute_range_rule(self, rule: BusinessRule) -> Dict[str, Any]:
        col = rule.columns[0]
        series = self.df[col].dropna()

        if len(series) == 0:
            return {"rule": rule.name, "status": "PASSED", "violations": [], "checked_rows": 0}

        violations = []
        min_val = rule.parameters.get("min")
        max_val = rule.parameters.get("max")

        if min_val is not None:
            mask = series < min_val
            for idx in series[mask].index:
                violations.append({
                    "row_index": int(idx),
                    "column": col,
                    "value": series[idx],
                    "expected": f">= {min_val}",
                    "message": rule.message
                })

        if max_val is not None:
            mask = series > max_val
            for idx in series[mask].index:
                violations.append({
                    "row_index": int(idx),
                    "column": col,
                    "value": series[idx],
                    "expected": f"<= {max_val}",
                    "message": rule.message
                })

        status = "PASSED" if len(violations) == 0 else "FAILED"
        return {
            "rule": rule.name,
            "type": rule.rule_type.value,
            "status": status,
            "checked_rows": len(series),
            "violation_count": len(violations),
            "violations": violations[:100]  # Limit output
        }

    def _execute_pattern_rule(self, rule: BusinessRule) -> Dict[str, Any]:
        col = rule.columns[0]
        pattern = rule.parameters["pattern"]
        regex = re.compile(pattern)

        series = self.df[col].dropna().astype(str)
        if len(series) == 0:
            return {"rule": rule.name, "status": "PASSED", "violations": [], "checked_rows": 0}

        matches = series.apply(lambda x: bool(regex.match(x)))
        violation_mask = ~matches

        violations = []
        for idx in series[violation_mask].index:
            violations.append({
                "row_index": int(idx),
                "column": col,
                "value": series[idx],
                "expected": f"match pattern {pattern}",
                "message": rule.message
            })

        status = "PASSED" if len(violations) == 0 else "FAILED"
        return {
            "rule": rule.name,
            "type": rule.rule_type.value,
            "status": status,
            "checked_rows": len(series),
            "violation_count": len(violations),
            "violations": violations[:100]
        }

    def _execute_cross_field_rule(self, rule: BusinessRule) -> Dict[str, Any]:
        cols = rule.columns
        df_clean = self.df[cols].dropna()

        if len(df_clean) == 0:
            return {"rule": rule.name, "status": "PASSED", "violations": [], "checked_rows": 0}

        if rule.condition is None:
            return {"rule": rule.name, "status": "ERROR", "message": "No condition function provided"}

        try:
            results = rule.condition(df_clean)
            violation_mask = ~results

            violations = []
            for idx in df_clean[violation_mask].index:
                violations.append({
                    "row_index": int(idx),
                    "columns": cols,
                    "values": df_clean.loc[idx].to_dict(),
                    "message": rule.message
                })

            status = "PASSED" if len(violations) == 0 else "FAILED"
            return {
                "rule": rule.name,
                "type": rule.rule_type.value,
                "status": status,
                "checked_rows": len(df_clean),
                "violation_count": len(violations),
                "violations": violations[:100]
            }
        except Exception as e:
            return {"rule": rule.name, "status": "ERROR", "message": str(e)}

    def _execute_conditional_rule(self, rule: BusinessRule) -> Dict[str, Any]:
        # Similar to cross-field but with IF-THEN logic
        return self._execute_cross_field_rule(rule)

    def _execute_aggregation_rule(self, rule: BusinessRule) -> Dict[str, Any]:
        params = rule.parameters
        group_by = params["group_by"]
        measure = params["measure"]
        agg_func = params["aggregation"]
        expected = params.get("expected")
        tolerance = params.get("tolerance", 0.01)

        # Perform aggregation
        grouped = self.df.groupby(group_by)[measure].agg(agg_func).reset_index()

        violations = []

        if expected is not None:
            # Compare total aggregation to expected
            if agg_func == "sum":
                actual = self.df[measure].sum()
            elif agg_func == "mean":
                actual = self.df[measure].mean()
            elif agg_func == "count":
                actual = self.df[measure].count()
            else:
                actual = grouped[measure].sum()

            diff = abs(actual - expected)
            rel_diff = diff / abs(expected) if expected != 0 else diff

            if rel_diff > tolerance:
                violations.append({
                    "aggregation": agg_func,
                    "measure": measure,
                    "group_by": group_by,
                    "actual": actual,
                    "expected": expected,
                    "difference": diff,
                    "relative_difference": rel_diff,
                    "message": rule.message
                })

        # Check for fan-out (one-to-many causing inflation)
        if len(group_by) > 0:
            detail_count = len(self.df)
            grouped_count = len(grouped)
            if detail_count > grouped_count:
                fanout_ratio = detail_count / grouped_count
                if fanout_ratio > 10:  # Arbitrary threshold
                    violations.append({
                        "type": "fanout_warning",
                        "detail_rows": detail_count,
                        "grouped_rows": grouped_count,
                        "fanout_ratio": round(fanout_ratio, 2),
                        "message": f"High fan-out ratio ({fanout_ratio:.1f}:1) may indicate grain mismatch"
                    })

        status = "PASSED" if len(violations) == 0 else "FAILED"
        return {
            "rule": rule.name,
            "type": rule.rule_type.value,
            "status": status,
            "checked_rows": len(self.df),
            "violation_count": len(violations),
            "violations": violations
        }

    def _execute_custom_rule(self, rule: BusinessRule) -> Dict[str, Any]:
        if rule.condition is None:
            return {"rule": rule.name, "status": "ERROR", "message": "No condition function provided"}

        try:
            results = rule.condition(self.df)
            violation_mask = ~results

            violations = []
            for idx in self.df[violation_mask].index:
                violations.append({
                    "row_index": int(idx),
                    "columns": rule.columns,
                    "values": self.df.loc[idx, rule.columns].to_dict() if rule.columns else {},
                    "message": rule.message
                })

            status = "PASSED" if len(violations) == 0 else "FAILED"
            return {
                "rule": rule.name,
                "type": rule.rule_type.value,
                "status": status,
                "checked_rows": len(self.df),
                "violation_count": len(violations),
                "violations": violations[:100]
            }
        except Exception as e:
            return {"rule": rule.name, "status": "ERROR", "message": str(e)}

    # =========================================================
    # AGGREGATION SAFETY
    # =========================================================

    def check_aggregation_safety(
        self,
        group_by: List[str],
        measure_columns: List[str],
        aggregations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check if aggregation is safe (no grain issues, no fan-out, etc.).

        Parameters
        ----------
        group_by : Columns to group by
        measure_columns : Numeric columns to aggregate
        aggregations : List of aggregation functions (sum, mean, count, etc.)

        Returns
        -------
        Safety analysis results.
        """
        if aggregations is None:
            aggregations = ["sum", "mean", "count", "min", "max"]

        results = {
            "grain_check": self._check_grain_alignment(group_by),
            "fanout_analysis": self._analyze_fanout(group_by, measure_columns),
            "null_impact": self._analyze_null_impact(measure_columns),
            "aggregation_results": {},
            "warnings": []
        }

        # Perform aggregations
        for measure in measure_columns:
            for agg in aggregations:
                try:
                    grouped = self.df.groupby(group_by)[measure].agg(agg).reset_index()
                    results["aggregation_results"][f"{measure}_{agg}"] = {
                        "input_rows": len(self.df),
                        "output_rows": len(grouped),
                        "reduction_ratio": round(len(grouped) / len(self.df), 4) if len(self.df) > 0 else 0,
                        "sample_values": grouped[measure].head(10).tolist()
                    }
                except Exception as e:
                    results["aggregation_results"][f"{measure}_{agg}"] = {"error": str(e)}

        # Add warnings
        grain = results["grain_check"]
        if not grain["is_unique"]:
            results["warnings"].append(f"Group-by columns {group_by} do not form a unique key (uniqueness: {grain['uniqueness']:.2%})")

        fanout = results["fanout_analysis"]
        if fanout["max_fanout"] > 10:
            results["warnings"].append(f"High fan-out detected (max: {fanout['max_fanout']:.1f}) - check for grain mismatch")

        null_impact = results["null_impact"]
        for measure, info in null_impact.items():
            if info["null_percentage"] > 10:
                results["warnings"].append(f"{measure} has {info['null_percentage']:.1f}% nulls - aggregations may be skewed")

        return results

    def _check_grain_alignment(self, group_by: List[str]) -> Dict[str, Any]:
        """Check if group-by columns uniquely identify rows."""
        if not group_by:
            return {"is_unique": True, "uniqueness": 1.0, "message": "No group-by (grand total)"}

        subset = self.df[group_by].dropna()
        if len(subset) == 0:
            return {"is_unique": False, "uniqueness": 0.0, "message": "All group-by values are null"}

        unique_rows = subset.drop_duplicates().shape[0]
        total_rows = len(subset)
        uniqueness = unique_rows / total_rows if total_rows > 0 else 0

        return {
            "is_unique": uniqueness == 1.0,
            "uniqueness": uniqueness,
            "unique_combinations": unique_rows,
            "total_combinations": total_rows,
            "duplicate_combinations": total_rows - unique_rows
        }

    def _analyze_fanout(self, group_by: List[str], measure_columns: List[str]) -> Dict[str, Any]:
        """Analyze fan-out (one-to-many) in the data."""
        if not group_by:
            return {"max_fanout": 1.0, "avg_fanout": 1.0}

        grouped = self.df.groupby(group_by).size().reset_index(name="count")
        max_fanout = grouped["count"].max()
        avg_fanout = grouped["count"].mean()

        # Check measure-specific fanout
        measure_fanout = {}
        for measure in measure_columns:
            if pd.api.types.is_numeric_dtype(self.df[measure]):
                measure_grouped = self.df.groupby(group_by)[measure].sum().reset_index()
                measure_fanout[measure] = {
                    "rows_before": len(self.df),
                    "rows_after": len(measure_grouped),
                    "reduction": round(len(measure_grouped) / len(self.df), 4)
                }

        return {
            "max_fanout": float(max_fanout),
            "avg_fanout": round(float(avg_fanout), 2),
            "total_groups": len(grouped),
            "measure_fanout": measure_fanout
        }

    def _analyze_null_impact(self, measure_columns: List[str]) -> Dict[str, Any]:
        """Analyze impact of null values on aggregations."""
        results = {}

        for measure in measure_columns:
            series = self.df[measure]
            null_count = series.isna().sum()
            null_pct = (null_count / len(series)) * 100 if len(series) > 0 else 0

            # Calculate with and without nulls
            with_nulls = series.sum()  # pandas skips nulls by default
            without_nulls = series.dropna().sum()

            results[measure] = {
                "null_count": int(null_count),
                "null_percentage": round(null_pct, 2),
                "sum_with_nulls": float(with_nulls),
                "sum_without_nulls": float(without_nulls),
                "difference": float(without_nulls - with_nulls),
                "impact": "high" if null_pct > 20 else "medium" if null_pct > 5 else "low"
            }

        return results


class AggregationSafetyChecker:
    """
    Specialized checker for aggregation safety across tables.
    """

    def __init__(self, tables: Dict[str, pd.DataFrame]):
        self.tables = tables

    def validate_rollup(
        self,
        detail_table: str,
        summary_table: str,
        group_by: List[str],
        measure: str
    ) -> Dict[str, Any]:
        """
        Validate that summary table matches detail table rollup.
        """
        if detail_table not in self.tables or summary_table not in self.tables:
            return {"error": "Table not found"}

        detail_df = self.tables[detail_table]
        summary_df = self.tables[summary_table]

        # Rollup detail table
        rolled_up = detail_df.groupby(group_by)[measure].sum().reset_index()
        rolled_up.columns = group_by + [f"{measure}_detail"]

        # Merge with summary
        merged = rolled_up.merge(summary_df, on=group_by, how="outer", indicator=True)

        # Find mismatches
        both = merged[merged["_merge"] == "both"]
        detail_only = merged[merged["_merge"] == "left_only"]
        summary_only = merged[merged["_merge"] == "right_only"]

        mismatches = []
        for _, row in both.iterrows():
            detail_val = row[f"{measure}_detail"]
            summary_val = row[measure]
            if abs(detail_val - summary_val) > 0.01:  # Tolerance
                mismatches.append({
                    "group_keys": {k: row[k] for k in group_by},
                    "detail_value": detail_val,
                    "summary_value": summary_val,
                    "difference": detail_val - summary_val
                })

        return {
            "detail_table": detail_table,
            "summary_table": summary_table,
            "measure": measure,
            "group_by": group_by,
            "matching_groups": len(both),
            "detail_only_groups": len(detail_only),
            "summary_only_groups": len(summary_only),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:50],
            "is_valid": len(mismatches) == 0 and len(detail_only) == 0 and len(summary_only) == 0
        }

    def check_time_series_consistency(
        self,
        table: str,
        date_column: str,
        measure_column: str,
        frequency: str = "D"
    ) -> Dict[str, Any]:
        """
        Check time series aggregation consistency (no gaps, no overlaps).
        """
        if table not in self.tables:
            return {"error": "Table not found"}

        df = self.tables[table].copy()

        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        df = df.dropna(subset=[date_column, measure_column]).sort_values(date_column)

        # Check for gaps
        df["next_date"] = df[date_column].shift(-1)
        df["gap"] = (df["next_date"] - df[date_column]).dt.total_seconds() / (24 * 3600)
        expected_gap = {"D": 1, "W": 7, "M": 30, "H": 1/24}.get(frequency, 1)

        large_gaps = df[df["gap"] > expected_gap * 1.5]

        # Check for duplicates (same date)
        dup_dates = df[df.duplicated(subset=[date_column], keep=False)]

        return {
            "table": table,
            "date_column": date_column,
            "measure_column": measure_column,
            "frequency": frequency,
            "total_points": len(df),
            "date_range": {
                "start": str(df[date_column].min()),
                "end": str(df[date_column].max())
            },
            "gaps_detected": len(large_gaps),
            "gap_details": large_gaps[[date_column, "next_date", "gap"]].head(20).to_dict("records"),
            "duplicate_dates": len(dup_dates),
            "duplicate_details": dup_dates[[date_column, measure_column]].head(20).to_dict("records"),
            "is_consistent": len(large_gaps) == 0 and len(dup_dates) == 0
        }


def create_business_rule_engine(df: pd.DataFrame) -> BusinessRuleEngine:
    """Convenience function to create a business rule engine."""
    return BusinessRuleEngine(df)


def profile_business_rules(
    df: pd.DataFrame,
    rules: Optional[List[BusinessRule]] = None
) -> Dict[str, Any]:
    """
    Profile data against business rules.

    Parameters
    ----------
    df : DataFrame to validate
    rules : Optional list of pre-defined rules

    Returns
    -------
    Validation results.
    """
    engine = BusinessRuleEngine(df)

    if rules:
        engine.rules = rules

    return engine.execute()