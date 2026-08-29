"""
completeness.py

Data Completeness Profiling module for Agentic BI Analyst.

Responsibilities:
- Column-level completeness (null percentages)
- Row-level completeness (record completeness)
- Conditional completeness (completeness given conditions)
- Expected vs actual record counts
- Completeness by time periods
- Completeness by groups/segments
- Mandatory field validation
- Completeness scoring and trending

This module DOES NOT modify the input DataFrame.
"""

from typing import Any, Dict, List, Optional, Union, cast
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd


class CompletenessLevel(Enum):
    COMPLETE = "complete"       # 100%
    NEAR_COMPLETE = "near_complete"  # 99-100%
    HIGH = "high"               # 95-99%
    MEDIUM = "medium"           # 80-95%
    LOW = "low"                 # 50-80%
    CRITICAL = "critical"       # < 50%


@dataclass
class CompletenessThresholds:
    """Configurable thresholds for completeness levels."""
    near_complete: float = 99.0
    high: float = 95.0
    medium: float = 80.0
    low: float = 50.0
    # critical is < low


class CompletenessProfiler:
    """
    Profiles data completeness at multiple levels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        thresholds: Optional[CompletenessThresholds] = None,
        mandatory_columns: Optional[List[str]] = None
    ):
        """
        Initialize completeness profiler.

        Parameters
        ----------
        df : DataFrame to profile
        thresholds : Custom completeness thresholds
        mandatory_columns : Columns that must be 100% complete
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("CompletenessProfiler requires a pandas DataFrame.")

        self.df = df
        self.thresholds = thresholds or CompletenessThresholds()
        self.mandatory_columns = set(mandatory_columns or [])

    # =========================================================
    # COLUMN-LEVEL COMPLETENESS
    # =========================================================

    def column_completeness(self) -> Dict[str, Any]:
        """
        Calculate completeness for each column.
        """
        results = {
            "columns": {},
            "summary": {
                "total_columns": len(self.df.columns),
                "complete_columns": 0,
                "near_complete_columns": 0,
                "incomplete_columns": 0,
                "avg_completeness": 0.0
            }
        }

        completeness_scores = []

        for col in self.df.columns:
            total = len(self.df)
            non_null = self.df[col].notna().sum()
            null_count = total - non_null
            completeness = (non_null / total * 100) if total > 0 else 100.0

            level = self._classify_completeness(completeness)
            is_mandatory = col in self.mandatory_columns

            col_result = {
                "total_rows": int(total),
                "non_null_count": int(non_null),
                "null_count": int(null_count),
                "completeness_percentage": round(completeness, 2),
                "level": level.value,
                "is_mandatory": is_mandatory,
                "mandatory_violation": is_mandatory and completeness < 100
            }

            results["columns"][col] = col_result
            completeness_scores.append(completeness)

            if level == CompletenessLevel.COMPLETE:
                results["summary"]["complete_columns"] += 1
            elif level == CompletenessLevel.NEAR_COMPLETE:
                results["summary"]["near_complete_columns"] += 1
            else:
                results["summary"]["incomplete_columns"] += 1

        results["summary"]["avg_completeness"] = round(np.mean(completeness_scores), 2) if completeness_scores else 100.0
        results["summary"]["overall_level"] = self._classify_completeness(results["summary"]["avg_completeness"]).value

        return results

    def _classify_completeness(self, percentage: float) -> CompletenessLevel:
        """Classify completeness percentage into level."""
        t = self.thresholds
        if percentage == 100:
            return CompletenessLevel.COMPLETE
        elif percentage >= t.near_complete:
            return CompletenessLevel.NEAR_COMPLETE
        elif percentage >= t.high:
            return CompletenessLevel.HIGH
        elif percentage >= t.medium:
            return CompletenessLevel.MEDIUM
        elif percentage >= t.low:
            return CompletenessLevel.LOW
        else:
            return CompletenessLevel.CRITICAL

    # =========================================================
    # ROW-LEVEL COMPLETENESS
    # =========================================================

    def row_completeness(
        self,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate completeness for each row (record).
        """
        work_cols = columns or self.df.columns.tolist()
        work_df = self.df[work_cols]

        total_cols = len(work_cols)
        non_null_per_row = work_df.notna().sum(axis=1)
        if total_cols > 0:
            completeness_per_row = non_null_per_row.astype(float).div(total_cols).mul(100)
        else:
            completeness_per_row = pd.Series(100.0, index=self.df.index)

        complete_rows = (completeness_per_row == 100).sum()
        incomplete_rows = (completeness_per_row < 100).sum()

        # Distribution
        bins = [0, 25, 50, 75, 90, 95, 99, 100]
        hist, _ = np.histogram(completeness_per_row, bins=bins)
        distribution = {
            f"{bins[i]}-{bins[i+1]}%": int(hist[i]) for i in range(len(bins)-1)
        }

        return {
            "total_rows": len(self.df),
            "columns_analyzed": work_cols,
            "complete_rows": int(complete_rows),
            "incomplete_rows": int(incomplete_rows),
            "complete_percentage": round(
                complete_rows / len(self.df) * 100, 2
            ) if len(self.df) > 0 else 100.0,
            "row_completeness_stats": {
                "mean": round(completeness_per_row.mean(), 2),
                "median": round(completeness_per_row.median(), 2),
                "min": round(completeness_per_row.min(), 2),
                "max": round(completeness_per_row.max(), 2),
                "std": round(completeness_per_row.std(), 2)
            },
            "distribution": distribution,
            "least_complete_rows": completeness_per_row.nsmallest(10).to_dict(),
            "most_complete_rows": completeness_per_row.nlargest(10).to_dict()
        }

    # =========================================================
    # CONDITIONAL COMPLETENESS
    # =========================================================

    def conditional_completeness(
        self,
        target_columns: List[str],
        condition_column: str,
        condition_values: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze completeness of target columns when condition is met.

        Example: When status='completed', completion_date should be complete.
        """
        if condition_column not in self.df.columns:
            return {"error": f"Condition column '{condition_column}' not found"}

        results = {}

        # Determine condition values
        if condition_values is None:
            condition_values = self.df[condition_column].dropna().unique().tolist()

        for cond_val in condition_values:
            mask = self.df[condition_column] == cond_val
            subset = self.df[mask]

            if len(subset) == 0:
                results[str(cond_val)] = {"records": 0, "message": "No records match condition"}
                continue

            cond_results = {}
            for target in target_columns:
                if target not in self.df.columns:
                    cond_results[target] = {"error": "Column not found"}
                    continue

                non_null = subset[target].notna().sum()
                total = len(subset)
                completeness = (non_null / total * 100) if total > 0 else 100.0

                cond_results[target] = {
                    "total_records": int(total),
                    "non_null": int(non_null),
                    "completeness_percentage": round(completeness, 2),
                    "level": self._classify_completeness(completeness).value
                }

            results[str(cond_val)] = cond_results

        return {
            "condition_column": condition_column,
            "target_columns": target_columns,
            "results": results
        }

    # =========================================================
    # EXPECTED VS ACTUAL COUNTS
    # =========================================================

    def expected_vs_actual(
        self,
        expected_counts: Dict[str, Union[int, Dict[str, int]]],
        group_by: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare actual record counts against expected counts.

        Parameters
        ----------
        expected_counts : Dict of table-level or group-level expected counts
            - Table-level: {"total": 1000}
            - Group-level: {"group_col": {"A": 100, "B": 200}}
        group_by : Columns to group by for group-level comparison
        """
        results = {
            "table_level": {},
            "group_level": {}
        }

        # Table-level
        if "total" in expected_counts:
            expected_value = expected_counts["total"]
            if isinstance(expected_value, dict):
                expected_value = 0
            actual = len(self.df)
            results["table_level"]["total"] = {
                "expected": expected_value,
                "actual": actual,
                "difference": actual - expected_value,
                "completeness_percentage": round(
                    actual / expected_value * 100, 2
                ) if expected_value > 0 else 0
            }

        # Group-level
        if group_by:
            for group_col, expected_dict in expected_counts.items():
                if group_col == "total":
                    continue

                if group_col not in self.df.columns:
                    results["group_level"][group_col] = {"error": "Column not found"}
                    continue

                if not isinstance(expected_dict, dict):
                    results["group_level"][group_col] = {
                        "error": "Group-level expected counts must be a dictionary"
                    }
                    continue

                actual_counts = {
                    str(key): int(value)
                    for key, value in self.df[group_col].value_counts().items()
                }
                group_results = {}

                all_keys = set(expected_dict) | set(actual_counts)
                for key in all_keys:
                    exp = expected_dict.get(key, 0)
                    act = actual_counts.get(key, 0)
                    group_results[str(key)] = {
                        "expected": exp,
                        "actual": act,
                        "difference": act - exp,
                        "completeness_percentage": round(act / exp * 100, 2) if exp > 0 else 0
                    }

                results["group_level"][group_col] = group_results

        return results

    # =========================================================
    # TIME-BASED COMPLETENESS
    # =========================================================

    def completeness_by_time(
        self,
        date_column: str,
        target_columns: Optional[List[str]] = None,
        freq: str = "D"
    ) -> Dict[str, Any]:
        """
        Analyze completeness trends over time.
        """
        if date_column not in self.df.columns:
            return {"error": f"Date column '{date_column}' not found"}

        df_work = self.df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_work[date_column]):
            df_work[date_column] = pd.to_datetime(df_work[date_column], errors="coerce")

        df_work = df_work.dropna(subset=[date_column])
        if len(df_work) == 0:
            return {"error": "No valid dates"}

        target_cols = target_columns or [c for c in self.df.columns if c != date_column]

        # Group by time period
        df_work["period"] = df_work[date_column].dt.to_period(freq)
        grouped = df_work.groupby("period")

        period_results = {}
        for period, group in grouped:
            period_str = str(period)
            period_results[period_str] = {
                "record_count": len(group),
                "columns": {}
            }
            for col in target_cols:
                if col in group.columns:
                    non_null = group[col].notna().sum()
                    total = len(group)
                    completeness = (non_null / total * 100) if total > 0 else 100.0
                    period_results[period_str]["columns"][col] = {
                        "completeness": round(completeness, 2),
                        "non_null": int(non_null),
                        "total": int(total)
                    }

        # Overall trend
        avg_completeness = []
        for period, data in period_results.items():
            col_scores = [v["completeness"] for v in data["columns"].values()]
            avg_completeness.append({
                "period": period,
                "avg_completeness": round(np.mean(col_scores), 2) if col_scores else 100,
                "record_count": data["record_count"]
            })

        return {
            "date_column": date_column,
            "frequency": freq,
            "periods_analyzed": len(period_results),
            "period_details": period_results,
            "trend": avg_completeness
        }

    # =========================================================
    # GROUP/SEGMENT COMPLETENESS
    # =========================================================

    def completeness_by_group(
        self,
        group_columns: List[str],
        target_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze completeness segmented by group columns.
        """
        missing_cols = [c for c in group_columns if c not in self.df.columns]
        if missing_cols:
            return {"error": f"Columns not found: {missing_cols}"}

        target_cols = target_columns or [c for c in self.df.columns if c not in group_columns]

        grouped = self.df.groupby(group_columns)

        results = {
            "group_columns": group_columns,
            "target_columns": target_cols,
            "groups": {}
        }

        for group_key, group_df in grouped:
            key_str = str(group_key) if not isinstance(group_key, tuple) else " | ".join(map(str, group_key))
            group_results = {
                "record_count": len(group_df),
                "columns": {}
            }

            for col in target_cols:
                if col in group_df.columns:
                    non_null = group_df[col].notna().sum()
                    total = len(group_df)
                    completeness = (non_null / total * 100) if total > 0 else 100.0
                    group_results["columns"][col] = {
                        "completeness": round(completeness, 2),
                        "level": self._classify_completeness(completeness).value
                    }

            results["groups"][key_str] = group_results

        # Summary: find groups with lowest completeness
        group_scores = []
        for group_key, data in results["groups"].items():
            scores = [v["completeness"] for v in data["columns"].values()]
            if scores:
                group_scores.append({
                    "group": group_key,
                    "avg_completeness": round(np.mean(scores), 2),
                    "record_count": data["record_count"]
                })

        group_scores.sort(key=lambda x: x["avg_completeness"])
        results["worst_groups"] = group_scores[:10]
        results["best_groups"] = group_scores[-10:]

        return results

    # =========================================================
    # PATTERN-BASED COMPLETENESS
    # =========================================================

    def pattern_completeness(
        self,
        pattern_column: str,
        target_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze completeness based on patterns in a column.
        E.g., different completeness for different categories.
        """
        if pattern_column not in self.df.columns:
            return {"error": f"Pattern column '{pattern_column}' not found"}

        target_cols = target_columns or [c for c in self.df.columns if c != pattern_column]

        patterns = self.df[pattern_column].dropna().unique()
        results = {
            "pattern_column": pattern_column,
            "patterns": {}
        }

        for pattern in patterns:
            mask = self.df[pattern_column] == pattern
            subset = self.df[mask]

            pattern_results = {
                "record_count": len(subset),
                "columns": {}
            }

            for col in target_cols:
                if col in subset.columns:
                    non_null = subset[col].notna().sum()
                    total = len(subset)
                    completeness = (non_null / total * 100) if total > 0 else 100.0
                    pattern_results["columns"][col] = {
                        "completeness": round(completeness, 2),
                        "level": self._classify_completeness(completeness).value
                    }

            results["patterns"][str(pattern)] = pattern_results

        return results

    # =========================================================
    # MISSING VALUE PATTERNS
    # =========================================================

    def missing_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns of missing values across columns.
        """
        missing_matrix = self.df.isna()

        results = {
            "column_missingness": {},
            "row_missingness": {},
            "correlated_missingness": [],
            "systematic_patterns": []
        }

        # Column missingness
        for col in self.df.columns:
            missing = missing_matrix[col].sum()
            total = len(self.df)
            results["column_missingness"][col] = {
                "missing_count": int(missing),
                    "missing_percentage": round(
                        missing / total * 100, 2
                    ) if total > 0 else 0.0
            }

        # Row missingness
        missing_per_row = missing_matrix.sum(axis=1)
        results["row_missingness"] = {
            "rows_with_any_missing": int((missing_per_row > 0).sum()),
            "rows_completely_missing": int((missing_per_row == len(self.df.columns)).sum()),
            "avg_missing_per_row": round(missing_per_row.mean(), 2),
            "max_missing_per_row": int(missing_per_row.max()),
            "distribution": missing_per_row.value_counts().to_dict()
        }

        # Correlated missingness (columns that are missing together)
        numeric_missing = missing_matrix.astype(int)
        if numeric_missing.shape[1] > 1:
            corr = numeric_missing.corr()
            for i, col1 in enumerate(corr.columns):
                for col2 in corr.columns[i+1:]:
                    c = cast(float, corr.loc[col1, col2])
                    if abs(c) > 0.7:
                        results["correlated_missingness"].append({
                            "column_1": col1,
                            "column_2": col2,
                            "correlation": round(float(c), 4),
                            "type": "missing_together" if c > 0 else "mutually_exclusive"
                        })

        # Systematic patterns (same missing pattern across multiple columns)
        if len(self.df.columns) <= 20:
            pattern_counts = missing_matrix.groupby(list(self.df.columns)).size().reset_index(name='count')
            for _, row in pattern_counts.iterrows():
                missing_cols = [c for c in self.df.columns if row[c] == True]
                if len(missing_cols) >= 2:
                    results["systematic_patterns"].append({
                        "missing_columns": missing_cols,
                        "row_count": int(row['count']),
                        "percentage": round(row['count'] / len(self.df) * 100, 2)
                    })

        return results

    # =========================================================
    # COMPLETENESS SCORE
    # =========================================================

    def calculate_completeness_score(
        self,
        column_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate weighted completeness score (0-100).
        """
        col_completeness = self.column_completeness()

        if column_weights is None:
            # Default: mandatory columns weight 2x, others 1x
            column_weights = {}
            for col in self.df.columns:
                column_weights[col] = 2.0 if col in self.mandatory_columns else 1.0

        scores = []
        weights = []
        for col, weight in column_weights.items():
            if col in col_completeness["columns"]:
                scores.append(col_completeness["columns"][col]["completeness_percentage"])
                weights.append(weight)

        if not scores:
            return {"score": 0, "level": "unknown"}

        weighted_score = np.average(scores, weights=weights)
        level = self._classify_completeness(weighted_score).value

        return {
            "overall_score": round(weighted_score, 2),
            "level": level,
            "column_scores": {col: col_completeness["columns"][col]["completeness_percentage"] for col in column_weights if col in col_completeness["columns"]},
            "weights_used": column_weights
        }

    # =========================================================
    # COMPREHENSIVE PROFILE
    # =========================================================

    def profile(self) -> Dict[str, Any]:
        """Generate complete completeness profile."""
        return {
            "column_completeness": self.column_completeness(),
            "row_completeness": self.row_completeness(),
            "missing_patterns": self.missing_patterns(),
            "completeness_score": self.calculate_completeness_score()
        }


def profile_completeness(
    df: pd.DataFrame,
    thresholds: Optional[CompletenessThresholds] = None,
    mandatory_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Convenience function for completeness profiling."""
    profiler = CompletenessProfiler(df, thresholds, mandatory_columns)
    return profiler.profile()