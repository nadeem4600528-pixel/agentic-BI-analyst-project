"""
structure.py

Dataset structure profiling module for Agentic BI Analyst.

Responsibilities:
- Dataset shape analysis
- Row and column counts
- Empty dataset detection
- Empty row detection
- Empty column detection
- Fully populated column detection
- Column type composition
- Dataset width/size classification
- Row-to-column ratio
- Duplicate row statistics
- Memory usage estimation
- Structural warnings

This module DOES NOT modify the input DataFrame.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from difflib import SequenceMatcher


class StructureProfiler:
    """
    Performs structural profiling on a pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "StructureProfiler requires a pandas DataFrame."
            )

        self.df = df

    # =========================================================
    # DATASET SHAPE
    # =========================================================

    def dataset_shape(self) -> Dict[str, Any]:
        """
        Return basic dataset dimensions.
        """

        rows, columns = self.df.shape

        return {
            "rows": int(rows),
            "columns": int(columns),
            "total_cells": int(rows * columns),
            "is_empty": bool(rows == 0 or columns == 0)
        }

    # =========================================================
    # EMPTY ROW ANALYSIS
    # =========================================================

    def empty_rows(self) -> Dict[str, Any]:
        """
        Detect completely empty rows.

        A row is considered empty when all values are missing.
        """

        total_rows = len(self.df)

        if total_rows == 0:
            return {
                "empty_row_count": 0,
                "empty_row_percentage": 0.0
            }

        empty_row_mask = self.df.isna().all(axis=1)

        empty_count = int(empty_row_mask.sum())

        percentage = (
            empty_count / total_rows
        ) * 100

        return {
            "empty_row_count": empty_count,
            "empty_row_percentage": round(
                percentage,
                2
            )
        }

    # =========================================================
    # EMPTY COLUMN ANALYSIS
    # =========================================================

    def empty_columns(self) -> Dict[str, Any]:
        """
        Detect columns containing only missing values.
        """

        total_rows = len(self.df)

        columns = []

        for column in self.df.columns:

            missing_count = int(
                self.df[column].isna().sum()
            )

            if missing_count == total_rows:
                columns.append(column)

        return {
            "empty_column_count": len(columns),
            "empty_columns": columns
        }

    # =========================================================
    # FULLY POPULATED COLUMNS
    # =========================================================

    def fully_populated_columns(self) -> Dict[str, Any]:
        """
        Detect columns containing no missing values.
        """

        columns = []

        for column in self.df.columns:

            if not self.df[column].isna().any():
                columns.append(column)

        return {
            "fully_populated_column_count": len(columns),
            "fully_populated_columns": columns
        }

    # =========================================================
    # COLUMN TYPE COMPOSITION
    # =========================================================

    def column_composition(self) -> Dict[str, Any]:
        """
        Classify columns into broad data-type groups.

        Categories:
        - numeric
        - categorical
        - datetime
        - boolean
        - text
        - other

        Uses pandas dtype detection first.
        For object columns, performs safe content-based
        detection so that columns containing numeric,
        boolean, datetime, or text values are classified
        correctly.
        """

        numeric_columns = []
        categorical_columns = []
        datetime_columns = []
        boolean_columns = []
        text_columns = []
        other_columns = []

        for column in self.df.columns:

            series = self.df[column]

            # -------------------------------------------------
            # Remove missing values for content-based checks
            # -------------------------------------------------

            non_null = series.dropna()

            # -------------------------------------------------
            # Empty column
            # -------------------------------------------------

            if len(non_null) == 0:
                text_columns.append(column)
                continue

            # -------------------------------------------------
            # Native Boolean
            # -------------------------------------------------

            if pd.api.types.is_bool_dtype(series):
                boolean_columns.append(column)
                continue

            # -------------------------------------------------
            # Native Numeric
            # -------------------------------------------------

            if pd.api.types.is_numeric_dtype(series):
                numeric_columns.append(column)
                continue

            # -------------------------------------------------
            # Native Datetime
            # -------------------------------------------------

            if pd.api.types.is_datetime64_any_dtype(series):
                datetime_columns.append(column)
                continue

            # -------------------------------------------------
            # Native Categorical
            # -------------------------------------------------

            if isinstance(
                series.dtype,
                pd.CategoricalDtype
            ):
                categorical_columns.append(column)
                continue

            # -------------------------------------------------
            # Object columns
            # -------------------------------------------------

            if pd.api.types.is_object_dtype(series):

                # ---------------------------------------------
                # Boolean-like values
                # ---------------------------------------------

                boolean_values = {
                    True,
                    False,
                    "True",
                    "False",
                    "true",
                    "false",
                    "TRUE",
                    "FALSE"
                }

                if set(non_null.unique()).issubset(
                    boolean_values
                ):
                    boolean_columns.append(column)
                    continue

                # ---------------------------------------------
                # Numeric-like values
                # ---------------------------------------------

                numeric_converted = pd.to_numeric(
                    non_null,
                    errors="coerce"
                )

                numeric_ratio = (
                    numeric_converted.notna().mean()
                )

                if numeric_ratio == 1.0:
                    numeric_columns.append(column)
                    continue

                # ---------------------------------------------
                # Datetime-like values
                # ---------------------------------------------

                datetime_converted = pd.to_datetime(
                    non_null,
                    errors="coerce"
                )

                datetime_ratio = (
                    datetime_converted.notna().mean()
                )

                if datetime_ratio == 1.0:
                    datetime_columns.append(column)
                    continue

                # ---------------------------------------------
                # Otherwise treat as text
                # ---------------------------------------------

                text_columns.append(column)
                continue

            # -------------------------------------------------
            # Unknown / other dtype
            # -------------------------------------------------

            other_columns.append(column)

        return {
            "numeric": numeric_columns,
            "categorical": categorical_columns,
            "datetime": datetime_columns,
            "boolean": boolean_columns,
            "text": text_columns,
            "other": other_columns,

            "numeric_count": len(numeric_columns),
            "categorical_count": len(categorical_columns),
            "datetime_count": len(datetime_columns),
            "boolean_count": len(boolean_columns),
            "text_count": len(text_columns),
            "other_count": len(other_columns)
        }

    # =========================================================
    # DATASET WIDTH CLASSIFICATION
    # =========================================================

    def width_classification(self) -> Dict[str, Any]:
        """
        Classify dataset based on number of columns.

        Classification:
        - empty
        - narrow
        - medium
        - wide
        - very_wide
        """

        column_count = len(self.df.columns)

        if column_count == 0:
            classification = "empty"

        elif column_count <= 10:
            classification = "narrow"

        elif column_count <= 50:
            classification = "medium"

        elif column_count <= 200:
            classification = "wide"

        else:
            classification = "very_wide"

        return {
            "column_count": int(column_count),
            "classification": classification
        }

    # =========================================================
    # DATASET SIZE CLASSIFICATION
    # =========================================================

    def size_classification(self) -> Dict[str, Any]:
        """
        Classify dataset based on row count.

        Classification:
        - empty
        - small
        - medium
        - large
        - very_large
        """

        row_count = len(self.df)

        if row_count == 0:
            classification = "empty"

        elif row_count <= 10_000:
            classification = "small"

        elif row_count <= 100_000:
            classification = "medium"

        elif row_count <= 1_000_000:
            classification = "large"

        else:
            classification = "very_large"

        return {
            "row_count": int(row_count),
            "classification": classification
        }

    # =========================================================
    # ROW / COLUMN RATIO
    # =========================================================

    def row_column_ratio(self) -> Dict[str, Any]:
        """
        Calculate the ratio between rows and columns.
        """

        rows = len(self.df)
        columns = len(self.df.columns)

        if columns == 0:
            ratio = None
        else:
            ratio = rows / columns

        return {
            "rows": int(rows),
            "columns": int(columns),
            "row_column_ratio": (
                round(ratio, 2)
                if ratio is not None
                else None
            )
        }

    # =========================================================
    # DUPLICATE ROW ANALYSIS
    # =========================================================

    def duplicate_rows(self) -> Dict[str, Any]:
        """
        Detect completely duplicated rows.

        This only profiles duplicates.
        It does NOT remove them.
        """

        total_rows = len(self.df)

        if total_rows == 0:
            return {
                "duplicate_row_count": 0,
                "duplicate_row_percentage": 0.0
            }

        duplicate_count = int(
            self.df.duplicated().sum()
        )

        percentage = (
            duplicate_count / total_rows
        ) * 100

        return {
            "duplicate_row_count": duplicate_count,
            "duplicate_row_percentage": round(
                percentage,
                2
            )
        }

    # =========================================================
    # ENTITY DUPLICATE DETECTION (Fuzzy Matching)
    # =========================================================

    def entity_duplicates(
        self,
        key_columns: Optional[List[str]] = None,
        similarity_threshold: float = 0.85,
        max_comparisons: int = 10000
    ) -> Dict[str, Any]:
        """
        Detect near-duplicate entities using fuzzy matching.

        Unlike exact duplicates, entity duplicates represent the same
        real-world entity with slight variations (typos, abbreviations, etc.).

        Parameters
        ----------
        key_columns: Columns to use for comparison. If None, uses identifier candidates.
        similarity_threshold: Minimum similarity ratio (0-1) to consider as duplicate.
        max_comparisons: Maximum pairwise comparisons for performance.

        Returns
        -------
        Potential entity duplicate groups.
        """

        if len(self.df) < 2:
            return {
                "entity_duplicates": [],
                "message": "Insufficient rows for entity duplicate detection"
            }

        # Determine columns to compare
        if key_columns is None:
            # Use identifier candidates or low-cardinality text columns
            key_columns = []
            for column in self.df.columns:
                series = self.df[column].dropna()
                if len(series) == 0:
                    continue
                unique_ratio = series.nunique() / len(series)
                if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and unique_ratio < 0.5:
                    key_columns.append(column)
                elif unique_ratio > 0.95:  # High uniqueness = identifier
                    key_columns.append(column)

            # Limit to top 5 key columns
            key_columns = key_columns[:5]

        if not key_columns:
            return {
                "entity_duplicates": [],
                "message": "No suitable key columns found for comparison"
            }

        results = {
            "key_columns_used": key_columns,
            "similarity_threshold": similarity_threshold,
            "entity_duplicate_groups": []
        }

        # Sample for performance if too many rows
        compare_df = self.df
        if len(self.df) > 1000:
            compare_df = self.df.sample(n=1000, random_state=42)

        # Create composite key for comparison
        def make_composite(row):
            parts = []
            for col in key_columns:
                val = row[col]
                if pd.notna(val):
                    parts.append(str(val).strip().lower())
            return " | ".join(parts)

        composites = compare_df[key_columns].apply(make_composite, axis=1)

        # Compare pairs
        n = len(composites)
        comparisons = 0
        duplicate_groups = []

        for i in range(n):
            if comparisons >= max_comparisons:
                break
            for j in range(i + 1, n):
                if comparisons >= max_comparisons:
                    break

                comp1 = composites.iloc[i]
                comp2 = composites.iloc[j]

                if not comp1 or not comp2:
                    continue

                similarity = SequenceMatcher(None, comp1, comp2).ratio()

                if similarity >= similarity_threshold:
                    duplicate_groups.append({
                        "row_index_1": int(composites.index[i]),
                        "row_index_2": int(composites.index[j]),
                        "similarity": round(similarity, 4),
                        "values_1": {col: str(compare_df.iloc[i][col]) for col in key_columns},
                        "values_2": {col: str(compare_df.iloc[j][col]) for col in key_columns}
                    })

                comparisons += 1

        # Group by connected components (transitive closure)
        if duplicate_groups:
            # Simple grouping: merge overlapping pairs
            groups = []

            for dup in duplicate_groups:
                idx1 = dup["row_index_1"]
                idx2 = dup["row_index_2"]

                # Find existing group containing either index
                group_idx = None
                for g_idx, group in enumerate(groups):
                    if idx1 in group["row_indices"] or idx2 in group["row_indices"]:
                        group_idx = g_idx
                        break

                if group_idx is not None:
                    groups[group_idx]["row_indices"].update([idx1, idx2])
                    groups[group_idx]["pairs"].append(dup)
                else:
                    groups.append({
                        "row_indices": {idx1, idx2},
                        "pairs": [dup]
                    })

            # Convert to output format
            for group in groups:
                if len(group["row_indices"]) >= 2:
                    results["entity_duplicate_groups"].append({
                        "row_indices": sorted(list(group["row_indices"])),
                        "group_size": len(group["row_indices"]),
                        "pair_details": group["pairs"][:5]  # Limit details
                    })

        results["total_comparisons"] = comparisons
        results["groups_found"] = len(results["entity_duplicate_groups"])

        return results

    # =========================================================
    # DATASET GRAIN ANALYSIS
    # =========================================================

    def dataset_grain(self) -> Dict[str, Any]:
        """
        Analyze the grain (level of detail) of the dataset.

        Determines:
        - What a single row represents
        - Whether grain is consistent
        - Potential grain issues (multiple grains mixed)
        """

        results = {
            "grain_description": "",
            "grain_confidence": 0.0,
            "grain_columns": [],
            "issues": []
        }

        # Check for identifier columns (single column unique)
        single_col_keys = []
        for column in self.df.columns:
            series = self.df[column].dropna()
            if len(series) > 0:
                uniqueness = series.nunique() / len(series)
                if uniqueness == 1.0:
                    single_col_keys.append(column)

        if single_col_keys:
            results["grain_columns"] = single_col_keys
            results["grain_description"] = f"Row grain defined by: {', '.join(single_col_keys)}"
            results["grain_confidence"] = 0.95
        else:
            # Check composite keys
            composite = self.detect_composite_keys()
            if composite.get("composite_keys"):
                best_key = composite["composite_keys"][0]
                results["grain_columns"] = best_key["columns"]
                results["grain_description"] = f"Row grain defined by composite key: {', '.join(best_key['columns'])}"
                results["grain_confidence"] = 0.85
            else:
                results["grain_description"] = "No clear grain identified - may have duplicate entities"
                results["grain_confidence"] = 0.3
                results["issues"].append("No unique identifier found - dataset grain unclear")

        # Check for potential multiple grains
        # If there are columns with very different cardinalities, might be mixed grain
        cardinalities = {}
        for column in self.df.columns:
            series = self.df[column].dropna()
            if len(series) > 0:
                cardinalities[column] = series.nunique() / len(series)

        high_card = [c for c, v in cardinalities.items() if v > 0.9]
        low_card = [c for c, v in cardinalities.items() if v < 0.1 and v > 0]

        if len(high_card) > 1 and len(low_card) > 3:
            results["issues"].append(
                f"Potential mixed grain: {len(high_card)} high-cardinality columns "
                f"and {len(low_card)} low-cardinality columns"
            )
            results["grain_confidence"] *= 0.7

        return results

    # =========================================================
    # COMPOSITE KEY DETECTION (Simplified)
    # =========================================================

    def detect_composite_keys(
        self,
        max_key_size: int = 3,
        uniqueness_threshold: float = 0.99
    ) -> Dict[str, Any]:
        """
        Simplified composite key detection for grain analysis.
        """

        import itertools

        if len(self.df) == 0:
            return {"composite_keys": [], "message": "Empty dataset"}

        results = {"composite_keys": []}

        # Get candidate columns (not all object columns)
        candidate_columns = []
        for column in self.df.columns:
            series = self.df[column].dropna()
            if len(series) == 0:
                continue
            unique_ratio = series.nunique() / len(series)
            if unique_ratio > 0.1:  # Some variation
                candidate_columns.append(column)

        candidate_columns = candidate_columns[:10]  # Limit for performance

        for key_size in range(2, min(max_key_size + 1, len(candidate_columns) + 1)):
            for combo in itertools.combinations(candidate_columns, key_size):
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
                        "uniqueness_ratio": round(uniqueness, 4)
                    })

        results["composite_keys"].sort(
            key=lambda x: (x["key_size"], -x["uniqueness_ratio"])
        )

        return results

    # =========================================================
    # MEMORY USAGE
    # =========================================================

    def memory_usage(self) -> Dict[str, Any]:
        """
        Estimate DataFrame memory usage.

        Returns memory usage in:
        - bytes
        - kilobytes
        - megabytes

        More precision is retained for small datasets so that
        memory usage does not incorrectly appear as 0.0 MB.
        """

        memory_bytes = int(
            self.df.memory_usage(
                deep=True
            ).sum()
        )

        memory_kb = memory_bytes / 1024

        memory_mb = memory_bytes / (1024 ** 2)

        return {
            "memory_bytes": memory_bytes,
            "memory_kb": round(
                memory_kb,
                2
            ),
            "memory_mb": round(
                memory_mb,
                4
            )
        }

    # =========================================================
    # STRUCTURAL WARNINGS
    # =========================================================

    def structural_warnings(self) -> List[str]:
        """
        Generate basic structural warnings.
        """

        warnings = []

        rows = len(self.df)
        columns = len(self.df.columns)

        # -----------------------------------------------------
        # Empty dataset
        # -----------------------------------------------------

        if rows == 0:

            warnings.append(
                "Dataset contains zero rows."
            )

        if columns == 0:

            warnings.append(
                "Dataset contains zero columns."
            )

        # -----------------------------------------------------
        # Empty rows
        # -----------------------------------------------------

        empty_rows = self.df.isna().all(axis=1).sum()

        if empty_rows > 0:

            warnings.append(
                f"Dataset contains {int(empty_rows)} completely "
                "empty row(s)."
            )

        # -----------------------------------------------------
        # Empty columns
        # -----------------------------------------------------

        for column in self.df.columns:

            if self.df[column].isna().all():

                warnings.append(
                    f"Column '{column}' contains only missing values."
                )

        # -----------------------------------------------------
        # Very wide dataset
        # -----------------------------------------------------

        if columns > 200:

            warnings.append(
                "Dataset is very wide and contains more than "
                "200 columns."
            )

        # -----------------------------------------------------
        # Very small dataset
        # -----------------------------------------------------

        if 0 < rows < 10:

            warnings.append(
                "Dataset contains fewer than 10 rows; "
                "statistical conclusions may be unreliable."
            )

        return warnings

    # =========================================================
    # COMPLETE STRUCTURE PROFILE
    # =========================================================

    def profile(self) -> Dict[str, Any]:
        """
        Generate complete structural profile.
        """

        return {
            "dataset_shape": self.dataset_shape(),

            "empty_rows": self.empty_rows(),

            "empty_columns": self.empty_columns(),

            "fully_populated_columns": (
                self.fully_populated_columns()
            ),

            "column_composition": (
                self.column_composition()
            ),

            "width_classification": (
                self.width_classification()
            ),

            "size_classification": (
                self.size_classification()
            ),

            "row_column_ratio": (
                self.row_column_ratio()
            ),

            "duplicate_rows": (
                self.duplicate_rows()
            ),

            "entity_duplicates": (
                self.entity_duplicates()
            ),

            "dataset_grain": (
                self.dataset_grain()
            ),

            "memory_usage": (
                self.memory_usage()
            ),

            "structural_warnings": (
                self.structural_warnings()
            )
        }


# =============================================================
# CONVENIENCE FUNCTION
# =============================================================

def profile_structure(
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Convenience function for structural profiling.

    Example:

        structure = profile_structure(df)
    """

    profiler = StructureProfiler(df)

    return profiler.profile()