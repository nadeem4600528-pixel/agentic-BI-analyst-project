"""
schema.py

Schema profiling module for Agentic BI Analyst.

Responsibilities:
- Discover dataset schema
- Identify column names
- Identify pandas data types
- Classify columns into broad data categories
- Detect nullable columns
- Generate schema summary
- Generate schema fingerprint
- Composite key detection
- Range/domain validation
- Cross-column validation

This module DOES NOT modify the input DataFrame.
"""

from typing import Any, Dict, Optional
import hashlib
import itertools

import pandas as pd


class SchemaProfiler:
    """
    Performs schema-level profiling on a pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "SchemaProfiler requires a pandas DataFrame."
            )

        self.df = df

    # =========================================================
    # COLUMN TYPE CLASSIFICATION
    # =========================================================

    def classify_column(
        self,
        column: str
    ) -> str:
        """
        Classify a column into a broad data category.

        Possible results:
        - numeric
        - categorical
        - boolean
        - datetime
        - text
        - object
        - unknown
        """

        series = self.df[column]

        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        if isinstance(series.dtype, pd.CategoricalDtype):
            return "categorical"

        if pd.api.types.is_string_dtype(series):
            return "text"

        if pd.api.types.is_string_dtype(series):

            # Object columns containing text are treated as text.
            if series.dtype == "object":
                return "text"

            return "text"

        if series.dtype == "object":
            return "object"

        return "unknown"

    # =========================================================
    # COLUMN SCHEMA
    # =========================================================

    def column_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate detailed schema information for every column.
        """

        results = {}

        total_rows = len(self.df)

        for column in self.df.columns:

            series = self.df[column]

            missing_count = int(
                series.isna().sum()
            )

            non_null_count = int(
                series.notna().sum()
            )

            if total_rows > 0:
                missing_percentage = (
                    missing_count / total_rows
                ) * 100
            else:
                missing_percentage = 0.0

            results[column] = {
                "column_name": column,
                "dtype": str(series.dtype),
                "category": self.classify_column(column),
                "nullable": bool(missing_count > 0),
                "missing_count": missing_count,
                "missing_percentage": round(
                    missing_percentage,
                    2
                ),
                "non_null_count": non_null_count,
                "unique_count": int(
                    series.nunique(dropna=True)
                )
            }

        return results

    # =========================================================
    # DATASET SCHEMA SUMMARY
    # =========================================================

    def summary(self) -> Dict[str, Any]:
        """
        Generate high-level schema summary.
        """

        numeric_columns = []
        categorical_columns = []
        datetime_columns = []
        boolean_columns = []
        text_columns = []
        object_columns = []
        unknown_columns = []

        for column in self.df.columns:

            category = self.classify_column(column)

            if category == "numeric":
                numeric_columns.append(column)

            elif category == "categorical":
                categorical_columns.append(column)

            elif category == "datetime":
                datetime_columns.append(column)

            elif category == "boolean":
                boolean_columns.append(column)

            elif category == "text":
                text_columns.append(column)

            elif category == "object":
                object_columns.append(column)

            else:
                unknown_columns.append(column)

        return {
            "row_count": int(len(self.df)),
            "column_count": int(len(self.df.columns)),
            "columns": list(self.df.columns),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": datetime_columns,
            "boolean_columns": boolean_columns,
            "text_columns": text_columns,
            "object_columns": object_columns,
            "unknown_columns": unknown_columns
        }

    # =========================================================
    # NULLABLE COLUMN ANALYSIS
    # =========================================================

    def nullable_columns(self) -> Dict[str, bool]:
        """
        Identify columns containing missing values.
        """

        results = {}

        for column in self.df.columns:

            results[column] = bool(
                self.df[column].isna().any()
            )

        return results

    # =========================================================
    # SCHEMA FINGERPRINT
    # =========================================================

    def schema_fingerprint(self) -> str:
        """
        Generate a deterministic hash representing
        the current column structure and data types.

        This can later be used for schema drift detection.
        """

        schema_definition = []

        for column in self.df.columns:

            schema_definition.append(
                f"{column}:{self.df[column].dtype}"
            )

        schema_string = "|".join(
            schema_definition
        )

        fingerprint = hashlib.sha256(
            schema_string.encode("utf-8")
        ).hexdigest()

        return fingerprint

    # =========================================================
    # COMPOSITE KEY DETECTION
    # =========================================================

    def detect_composite_keys(
        self,
        max_key_size: int = 4,
        uniqueness_threshold: float = 0.99
    ) -> Dict[str, Any]:
        """
        Detect potential composite primary keys.

        A composite key is a combination of columns that uniquely
        identifies each row.

        Parameters
        ----------
        max_key_size: Maximum number of columns in a composite key.
        uniqueness_threshold: Minimum uniqueness ratio to qualify.

        Returns
        -------
        Dictionary with candidate composite keys and their uniqueness.
        """

        if len(self.df) == 0:
            return {"composite_keys": [], "message": "Empty dataset"}

        results = {
            "composite_keys": [],
            "analyzed_combinations": 0
        }

        # Get columns that could be part of a key (exclude high-cardinality text)
        candidate_columns = []
        for column in self.df.columns:
            series = self.df[column]
            non_null = series.dropna()
            if len(non_null) == 0:
                continue
            unique_ratio = non_null.nunique() / len(non_null)
            if unique_ratio > 0.01:  # At least some variation
                candidate_columns.append(column)

        # Limit combinations for performance
        candidate_columns = candidate_columns[:15]

        for key_size in range(2, min(max_key_size + 1, len(candidate_columns) + 1)):
            for combo in itertools.combinations(candidate_columns, key_size):
                results["analyzed_combinations"] += 1

                # Check uniqueness of combination
                combo_df = self.df[list(combo)].dropna()
                if len(combo_df) == 0:
                    continue

                unique_count = combo_df.drop_duplicates().shape[0]
                total_count = len(combo_df)
                uniqueness = unique_count / total_count

                if uniqueness >= uniqueness_threshold:
                    results["composite_keys"].append({
                        "columns": list(combo),
                        "key_size": key_size,
                        "uniqueness_ratio": round(uniqueness, 4),
                        "unique_count": int(unique_count),
                        "total_count": int(total_count),
                        "is_unique": uniqueness == 1.0
                    })

        # Sort by key size (smaller preferred) then by uniqueness
        results["composite_keys"].sort(
            key=lambda x: (x["key_size"], -x["uniqueness_ratio"])
        )

        return results

    # =========================================================
    # RANGE / DOMAIN VALIDATION
    # =========================================================

    def validate_ranges(
        self,
        expected_ranges: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Validate column values against expected ranges/domains.

        Parameters
        ----------
        expected_ranges: Dict mapping column names to expected ranges.
            Example: {"age": {"min": 0, "max": 120}, "status": {"values": ["active", "inactive"]}}

        Returns
        -------
        Validation results per column.
        """

        results = {}

        for column in self.df.columns:
            series = self.df[column].dropna()

            if len(series) == 0:
                results[column] = {
                    "valid": True,
                    "violations": 0,
                    "violation_percentage": 0.0,
                    "details": "Column is empty"
                }
                continue

            column_result = {
                "valid": True,
                "violations": 0,
                "violation_percentage": 0.0,
                "details": []
            }

            # Check user-provided expected ranges
            if expected_ranges and column in expected_ranges:
                expected = expected_ranges[column]

                if "min" in expected and "max" in expected:
                    # Numeric range validation
                    if pd.api.types.is_numeric_dtype(series):
                        violations = (
                            (series < expected["min"]) |
                            (series > expected["max"])
                        ).sum()
                        column_result["violations"] = int(violations)
                        column_result["violation_percentage"] = round(
                            (violations / len(series)) * 100, 2
                        )
                        if violations > 0:
                            column_result["valid"] = False
                            column_result["details"].append(
                                f"Values outside range [{expected['min']}, {expected['max']}]"
                            )

                if "values" in expected:
                    # Categorical domain validation
                    allowed = set(expected["values"])
                    actual = set(series.astype(str).unique())
                    invalid = actual - allowed
                    if invalid:
                        violations = series.astype(str).isin(invalid).sum()
                        column_result["violations"] = int(violations)
                        column_result["violation_percentage"] = round(
                            (violations / len(series)) * 100, 2
                        )
                        column_result["valid"] = False
                        column_result["details"].append(
                            f"Invalid values found: {list(invalid)[:10]}"
                        )

            # Auto-detect reasonable ranges for numeric columns
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) and not expected_ranges:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr

                violations = ((series < lower) | (series > upper)).sum()
                if violations > 0:
                    column_result["violations"] = int(violations)
                    column_result["violation_percentage"] = round(
                        (violations / len(series)) * 100, 2
                    )
                    column_result["valid"] = False
                    column_result["details"].append(
                        f"Values outside 3*IQR range [{lower:.2f}, {upper:.2f}]"
                    )

            # Check for negative values in columns that should be positive
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                col_lower = column.lower()
                if any(kw in col_lower for kw in ["age", "count", "quantity", "amount", "price", "salary", "id", "key"]):
                    negative_count = (series < 0).sum()
                    if negative_count > 0:
                        column_result["violations"] += int(negative_count)
                        column_result["violation_percentage"] = round(
                            (column_result["violations"] / len(series)) * 100, 2
                        )
                        column_result["valid"] = False
                        column_result["details"].append(
                            f"Negative values found in positive-expected column"
                        )

            results[column] = column_result

        return results

    # =========================================================
    # CROSS-COLUMN VALIDATION
    # =========================================================

    def cross_column_validation(self) -> Dict[str, Any]:
        """
        Validate relationships between columns.

        Checks:
        - Start date <= End date
        - Logical column dependencies
        - Consistent categorical combinations
        """

        results = {
            "date_order_checks": [],
            "logical_dependencies": [],
            "inconsistent_combinations": []
        }

        columns = self.df.columns.tolist()
        # Date order validation
        date_columns = [
            c for c in columns
            if pd.api.types.is_datetime64_any_dtype(self.df[c])
        ]

        # Look for start/end date pairs
        for i, col1 in enumerate(date_columns):
            for col2 in date_columns[i+1:]:
                c1, c2 = col1.lower(), col2.lower()

                start_keywords = ["start", "begin", "from", "created", "opened"]
                end_keywords = ["end", "finish", "to", "closed", "completed", "expired"]

                is_start1 = any(kw in c1 for kw in start_keywords)
                is_end1 = any(kw in c1 for kw in end_keywords)
                is_start2 = any(kw in c2 for kw in start_keywords)
                is_end2 = any(kw in c2 for kw in end_keywords)

                if (is_start1 and is_end2) or (is_start2 and is_end1):
                    start_col = col1 if is_start1 else col2
                    end_col = col2 if is_end2 else col1

                    valid_rows = self.df[[start_col, end_col]].dropna()
                    if len(valid_rows) > 0:
                        violations = (valid_rows[start_col] > valid_rows[end_col]).sum()
                        if violations > 0:
                            results["date_order_checks"].append({
                                "start_column": start_col,
                                "end_column": end_col,
                                "violations": int(violations),
                                "violation_percentage": round(
                                    (violations / len(valid_rows)) * 100, 2
                                )
                            })

        # Logical dependency checks (e.g., if status=completed, completion_date should not be null)
        status_cols = [c for c in columns if "status" in c.lower() or "state" in c.lower()]
        date_cols = [c for c in columns if "date" in c.lower() or "time" in c.lower()]

        for status_col in status_cols:
            for date_col in date_cols:
                status_vals = self.df[status_col].dropna().astype(str).str.lower().unique()
                completed_states = {"completed", "done", "finished", "closed", "shipped", "delivered"}

                if any(s in completed_states for s in status_vals):
                    # Check if completed rows have date
                    mask = self.df[status_col].astype(str).str.lower().isin(completed_states)
                    completed_rows = self.df[mask]
                    missing_dates = completed_rows[date_col].isna().sum()

                    if missing_dates > 0:
                        results["logical_dependencies"].append({
                            "condition_column": status_col,
                            "condition_values": list(completed_states),
                            "dependent_column": date_col,
                            "missing_count": int(missing_dates),
                            "message": f"Rows with completed status missing {date_col}"
                        })

        return results

    # =========================================================
    # COMPLETE SCHEMA PROFILE
    # =========================================================

    def profile(self) -> Dict[str, Any]:
        """
        Generate the complete schema profile.
        """

        return {
            "summary": self.summary(),

            "columns": self.column_schema(),

            "nullable_columns": (
                self.nullable_columns()
            ),

            "schema_fingerprint": (
                self.schema_fingerprint()
            ),

            "composite_keys": self.detect_composite_keys(),

            "range_validation": self.validate_ranges(),

            "cross_column_validation": self.cross_column_validation()
        }


# =============================================================
# CONVENIENCE FUNCTION
# =============================================================

def profile_schema(
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Convenience function for schema profiling.

    Example:
        schema = profile_schema(df)
    """

    profiler = SchemaProfiler(df)

    return profiler.profile()