"""
metadata.py

Metadata and structural profiling for Agentic BI Analyst.

Responsibilities:
- Analyze dataset shape
- Identify columns and data types
- Classify columns by basic data nature
- Detect potential identifier columns
- Detect date/datetime candidates
- Calculate memory usage
- Generate basic dataset metadata

This module DOES NOT modify the input DataFrame.
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np


class MetadataProfiler:
    """
    Performs metadata and structural profiling on a pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("MetadataProfiler requires a pandas DataFrame.")

        self.df = df

    # ---------------------------------------------------------
    # DATASET OVERVIEW
    # ---------------------------------------------------------

    def get_dataset_overview(self) -> Dict[str, Any]:
        """
        Return basic information about the dataset.
        """

        rows, columns = self.df.shape

        memory_usage = self.df.memory_usage(
            index=True,
            deep=True
        ).sum()

        return {
            "rows": int(rows),
            "columns": int(columns),
            "shape": [int(rows), int(columns)],
            "memory_usage_bytes": int(memory_usage),
            "memory_usage_mb": round(
                memory_usage / (1024 ** 2), 4
            ),
            "empty_dataset": bool(self.df.empty),
        }

    # ---------------------------------------------------------
    # COLUMN NAMES
    # ---------------------------------------------------------

    def get_column_names(self) -> List[str]:
        """
        Return all column names.
        """

        return self.df.columns.tolist()

    # ---------------------------------------------------------
    # DATA TYPES
    # ---------------------------------------------------------

    def get_data_types(self) -> Dict[str, str]:
        """
        Return pandas data type for every column.
        """

        return {
            str(column): str(dtype)
            for column, dtype in self.df.dtypes.items()
        }

    # ---------------------------------------------------------
    # NUMERICAL COLUMNS
    # ---------------------------------------------------------

    def get_numeric_columns(self) -> List[str]:
        """
        Identify numerical columns.
        """

        return self.df.select_dtypes(
            include=np.number
        ).columns.tolist()

    # ---------------------------------------------------------
    # CATEGORICAL COLUMNS
    # ---------------------------------------------------------

    def get_categorical_columns(self) -> List[str]:
        """
        Identify categorical/object columns.

        Object and category columns are initially considered
        categorical candidates. Further semantic analysis will
        be performed later.
        """

        return self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    # ---------------------------------------------------------
    # BOOLEAN COLUMNS
    # ---------------------------------------------------------

    def get_boolean_columns(self) -> List[str]:
        """
        Identify boolean columns.
        """

        return self.df.select_dtypes(
            include=["bool"]
        ).columns.tolist()

    # ---------------------------------------------------------
    # DATETIME COLUMNS
    # ---------------------------------------------------------

    def get_datetime_columns(self) -> List[str]:
        """
        Identify columns already stored as datetime types.
        """

        return self.df.select_dtypes(
            include=["datetime", "datetimetz"]
        ).columns.tolist()

    # ---------------------------------------------------------
    # TEXT COLUMNS
    # ---------------------------------------------------------

    def get_text_columns(self) -> List[str]:
        """
        Identify object columns that appear to contain text.

        This is a preliminary classification.
        More advanced semantic detection will be added later.
        """

        text_columns = []

        for column in self.df.select_dtypes(
            include=["object"]
        ).columns:

            non_null_values = self.df[column].dropna()

            if len(non_null_values) == 0:
                continue

            if non_null_values.map(
                lambda value: isinstance(value, str)
            ).all():

                text_columns.append(column)

        return text_columns

    # ---------------------------------------------------------
    # DATE CANDIDATES
    # ---------------------------------------------------------

    def get_date_candidates(self) -> List[str]:
        """
        Detect object/string columns that may contain dates.

        This is only a candidate detector.
        Actual date validation will be handled later.
        """

        candidates = []

        for column in self.df.select_dtypes(
            include=["object"]
        ).columns:

            non_null_values = self.df[column].dropna()

            if len(non_null_values) == 0:
                continue

            # Limit conversion sample for performance
            sample = non_null_values.head(1000)

            try:
                converted = pd.to_datetime(
                    sample,
                    errors="coerce",
                    format="mixed"
                )

                valid_ratio = converted.notna().mean()

                if valid_ratio >= 0.80:
                    candidates.append(column)

            except (ValueError, TypeError):
                continue

        return candidates

    # ---------------------------------------------------------
    # IDENTIFIER CANDIDATES
    # ---------------------------------------------------------

    def get_identifier_candidates(self) -> List[str]:
        """
        Detect potential identifier columns.

        Heuristics:
        - Column name contains ID / CODE / KEY
        - High uniqueness ratio
        """

        candidates = []

        for column in self.df.columns:

            column_name = str(column).lower()

            non_null = self.df[column].dropna()

            if len(non_null) == 0:
                continue

            unique_ratio = (
                non_null.nunique(dropna=True)
                / len(non_null)
            )

            name_based_candidate = any(
                keyword in column_name
                for keyword in [
                    "id",
                    "identifier",
                    "code",
                    "key"
                ]
            )

            high_uniqueness = unique_ratio >= 0.95

            if name_based_candidate or high_uniqueness:
                candidates.append(column)

        return candidates

    # ---------------------------------------------------------
    # COLUMN SUMMARY
    # ---------------------------------------------------------

    def get_column_summary(self) -> List[Dict[str, Any]]:
        """
        Generate structural metadata for every column.
        """

        summary = []

        for column in self.df.columns:

            series = self.df[column]

            non_null_count = int(
                series.notna().sum()
            )

            unique_count = int(
                series.nunique(dropna=True)
            )

            total_count = len(series)

            if total_count > 0:
                uniqueness_ratio = (
                    unique_count / total_count
                )
            else:
                uniqueness_ratio = 0.0

            summary.append({
                "column": column,
                "dtype": str(series.dtype),
                "non_null_count": non_null_count,
                "null_count": int(series.isna().sum()),
                "unique_count": unique_count,
                "uniqueness_ratio": round(
                    uniqueness_ratio,
                    4
                ),
                "memory_usage_bytes": int(
                    series.memory_usage(
                        index=True,
                        deep=True
                    )
                ),
            })

        return summary

    # ---------------------------------------------------------
    # COMPLETE METADATA PROFILE
    # ---------------------------------------------------------

    def profile(self) -> Dict[str, Any]:
        """
        Generate the complete metadata profile.
        """

        numeric_columns = self.get_numeric_columns()
        categorical_columns = self.get_categorical_columns()
        boolean_columns = self.get_boolean_columns()
        datetime_columns = self.get_datetime_columns()
        text_columns = self.get_text_columns()
        date_candidates = self.get_date_candidates()
        identifier_candidates = self.get_identifier_candidates()

        return {
            "dataset": self.get_dataset_overview(),

            "columns": self.get_column_names(),

            "data_types": self.get_data_types(),

            "column_categories": {
                "numeric": numeric_columns,
                "categorical": categorical_columns,
                "boolean": boolean_columns,
                "datetime": datetime_columns,
                "text": text_columns,
                "date_candidates": date_candidates,
                "identifier_candidates": identifier_candidates,
            },

            "column_summary": self.get_column_summary(),
        }


# -------------------------------------------------------------
# CONVENIENCE FUNCTION
# -------------------------------------------------------------

def profile_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function for metadata profiling.

    Example:
        metadata = profile_metadata(df)
    """

    profiler = MetadataProfiler(df)

    return profiler.profile()