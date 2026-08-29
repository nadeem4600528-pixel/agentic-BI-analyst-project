"""
statistics.py

Statistical profiling module for Agentic BI Analyst.

Responsibilities:
- Missing-value analysis
- Unique-value analysis
- Numerical statistics
- Categorical statistics
- Distribution analysis
- Skewness and kurtosis
- Outlier detection
- Constant / near-constant column detection
- Basic statistical anomalies
- Hidden NULL detection
- Missingness pattern analysis
- Temporal gap detection
- Invalid value detection
- Text quality analysis
- Date/Time profiling

This module DOES NOT modify the input DataFrame.
"""

from typing import Any, Dict, List, Optional, cast
import re
import pandas as pd
import numpy as np


class StatisticsProfiler:
    """
    Performs statistical profiling on a pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "StatisticsProfiler requires a pandas DataFrame."
            )

        self.df = df

    # =========================================================
    # MISSING VALUE ANALYSIS
    # =========================================================

    def missing_values(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze missing values for every column.
        """

        results = {}

        total_rows = len(self.df)

        for column in self.df.columns:

            missing_count = int(
                self.df[column].isna().sum()
            )

            if total_rows > 0:
                missing_percentage = (
                    missing_count / total_rows
                ) * 100
            else:
                missing_percentage = 0.0

            results[column] = {
                "missing_count": missing_count,
                "missing_percentage": round(
                    missing_percentage,
                    2
                ),
                "complete_count": int(
                    total_rows - missing_count
                )
            }

        return results

    # =========================================================
    # UNIQUE VALUE ANALYSIS
    # =========================================================

    def unique_values(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze unique and duplicate values per column.
        """

        results = {}

        for column in self.df.columns:

            series = self.df[column]

            unique_count = int(
                series.nunique(dropna=True)
            )

            non_null_count = int(
                series.notna().sum()
            )

            if non_null_count > 0:
                uniqueness_percentage = (
                    unique_count / non_null_count
                ) * 100
            else:
                uniqueness_percentage = 0.0

            results[column] = {
                "unique_count": unique_count,
                "uniqueness_percentage": round(
                    uniqueness_percentage,
                    2
                )
            }

        return results

    # =========================================================
    # NUMERICAL STATISTICS
    # =========================================================

    def numerical_statistics(
        self
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate descriptive statistics for numerical columns.
        """

        results = {}

        numeric_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        for column in numeric_columns:

            series = self.df[column].dropna()

            if len(series) == 0:
                continue

            results[column] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "variance": float(series.var()),
                "min": float(series.min()),
                "max": float(series.max()),
                "range": float(
                    series.max() - series.min()
                ),
                "q1": float(
                    series.quantile(0.25)
                ),
                "q2": float(
                    series.quantile(0.50)
                ),
                "q3": float(
                    series.quantile(0.75)
                ),
                "iqr": float(
                    series.quantile(0.75)
                    - series.quantile(0.25)
                ),
                "skewness": float(
                    cast(Any, series.skew())
                ),
                "kurtosis": float(
                    cast(Any, series.kurtosis())
                )
            }

        return results

    # =========================================================
    # CATEGORICAL STATISTICS
    # =========================================================

    def categorical_statistics(
        self
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze categorical columns.

        Includes:
        - Number of unique categories
        - Most frequent category
        - Frequency of most frequent category
        - Category distribution
        """

        results = {}

        categorical_columns = self.df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        for column in categorical_columns:

            series = self.df[column].dropna()

            if len(series) == 0:
                results[column] = {
                    "unique_categories": 0,
                    "top_category": None,
                    "top_category_count": 0,
                    "distribution": {}
                }
                continue

            value_counts = series.value_counts()

            distribution = {
                str(value): int(count)
                for value, count in value_counts.items()
            }

            results[column] = {
                "unique_categories": int(
                    series.nunique()
                ),
                "top_category": str(
                    value_counts.index[0]
                ),
                "top_category_count": int(
                    value_counts.iloc[0]
                ),
                "distribution": distribution
            }

        return results

    # =========================================================
    # OUTLIER DETECTION
    # =========================================================

    def outliers(
        self
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect statistical outliers using the IQR method.

        IMPORTANT:
        This method only identifies outliers.
        It does NOT remove them.
        """

        results = {}

        numeric_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        for column in numeric_columns:

            series = self.df[column].dropna()

            if len(series) == 0:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            outlier_mask = (
                (series < lower_bound)
                | (series > upper_bound)
            )

            outlier_count = int(
                outlier_mask.sum()
            )

            total_count = len(series)

            if total_count > 0:
                outlier_percentage = (
                    outlier_count / total_count
                ) * 100
            else:
                outlier_percentage = 0.0

            results[column] = {
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(
                    lower_bound
                ),
                "upper_bound": float(
                    upper_bound
                ),
                "outlier_count": outlier_count,
                "outlier_percentage": round(
                    outlier_percentage,
                    2
                )
            }

        return results

    # =========================================================
    # CONSTANT / NEAR-CONSTANT COLUMNS
    # =========================================================

    def constant_columns(
        self,
        near_constant_threshold: float = 0.95
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect constant and near-constant columns.

        A column is:
        - Constant if it has only one unique value.
        - Near-constant if one value represents at least
          `near_constant_threshold` of non-null records.
        """

        results = {}

        for column in self.df.columns:

            series = self.df[column].dropna()

            if len(series) == 0:
                continue

            value_counts = series.value_counts()

            unique_count = len(value_counts)

            top_frequency = (
                value_counts.iloc[0] / len(series)
            )

            is_constant = unique_count == 1

            is_near_constant = (
                not is_constant
                and round(top_frequency, 10)
                >= round(near_constant_threshold, 10)
            )
            results[column] = {
                "unique_count": int(unique_count),
                "top_value": str(
                    value_counts.index[0]
                ),
                "top_frequency": round(
                    float(top_frequency), 4
                ),
                "is_constant": bool(is_constant),
                "is_near_constant": bool(is_near_constant),
            }

        return results

    # =========================================================
    # HIDDEN NULL DETECTION
    # =========================================================

    def hidden_nulls(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect hidden NULL values - values that represent missing data
        but are not recognized as NaN by pandas.

        Examples: "NULL", "N/A", "NA", "null", "none", "", " ", "-", "unknown", "missing"
        """

        hidden_null_patterns = {
            "empty_string": r"^\s*$",
            "null_keywords": r"^(null|none|nil|n/a|na|missing|unknown|undefined|-)$",
            "placeholder": r"^(---|===|\.\.\.|n\.a\.|tbd|to be determined)$",
            "zero_length": r"^$"
        }

        results = {}

        for column in self.df.columns:
            series = self.df[column]

            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
                # Numeric and datetime columns don't typically have hidden nulls
                results[column] = {
                    "hidden_null_count": 0,
                    "hidden_null_percentage": 0.0,
                    "detected_patterns": []
                }
                continue

            non_null_mask = series.notna()
            non_null_series = series[non_null_mask].astype(str)

            if len(non_null_series) == 0:
                results[column] = {
                    "hidden_null_count": 0,
                    "hidden_null_percentage": 0.0,
                    "detected_patterns": []
                }
                continue

            total_hidden = 0
            detected_patterns = []

            for pattern_name, pattern in hidden_null_patterns.items():
                matches = non_null_series.str.match(pattern, case=False)
                match_count = int(matches.sum())

                if match_count > 0:
                    total_hidden += match_count
                    detected_patterns.append({
                        "pattern": pattern_name,
                        "count": match_count,
                        "percentage": round((match_count / len(non_null_series)) * 100, 2)
                    })

            # Also check for whitespace-only strings
            if series.dtype == "object":
                whitespace_mask = non_null_series.apply(lambda x: isinstance(x, str) and x.strip() == "" and x != "")
                ws_count = int(whitespace_mask.sum())
                if ws_count > 0:
                    total_hidden += ws_count
                    detected_patterns.append({
                        "pattern": "whitespace_only",
                        "count": ws_count,
                        "percentage": round((ws_count / len(non_null_series)) * 100, 2)
                    })

            total_non_null = len(non_null_series)
            hidden_percentage = (total_hidden / total_non_null * 100) if total_non_null > 0 else 0

            results[column] = {
                "hidden_null_count": total_hidden,
                "hidden_null_percentage": round(hidden_percentage, 2),
                "detected_patterns": detected_patterns
            }

        return results

    # =========================================================
    # MISSINGNESS PATTERN ANALYSIS
    # =========================================================

    def missingness_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns of missing values across columns and rows.

        Identifies:
        - Columns that are missing together
        - Rows with systematic missingness
        - Missing completely at random (MCAR) vs not
        """

        missing_matrix = self.df.isna()

        results = {
            "column_missingness_correlation": {},
            "row_missingness_profile": {},
            "systematic_missing_patterns": []
        }

        # Column-to-column missingness correlation
        numeric_missing = missing_matrix.astype(int)
        if numeric_missing.shape[1] > 1:
            corr_matrix = numeric_missing.corr()
            high_corr_pairs = []

            for i, col1 in enumerate(corr_matrix.columns):
                for col2 in corr_matrix.columns[i+1:]:
                    corr = cast(float, corr_matrix.loc[col1, col2])
                    if abs(corr) > 0.7:
                        high_corr_pairs.append({
                            "column_1": col1,
                            "column_2": col2,
                            "correlation": round(float(corr), 4),
                            "interpretation": "Missing together" if corr > 0 else "Mutually exclusive missingness"
                        })

            results["column_missingness_correlation"] = {
                "high_correlation_pairs": high_corr_pairs
            }

        # Row-level missingness
        missing_per_row = missing_matrix.sum(axis=1)
        total_cols = len(self.df.columns)

        results["row_missingness_profile"] = {
            "rows_with_any_missing": int((missing_per_row > 0).sum()),
            "rows_completely_missing": int((missing_per_row == total_cols).sum()),
            "avg_missing_per_row": round(float(missing_per_row.mean()), 2),
            "max_missing_per_row": int(missing_per_row.max()),
            "missing_distribution": missing_per_row.value_counts().to_dict()
        }

        # Systematic patterns - columns that are always missing together
        if len(self.df.columns) <= 20:  # Limit for performance
            missing_patterns = missing_matrix.groupby(
                list(self.df.columns)
            ).size().reset_index(name='count')

            # Find patterns where multiple columns are missing together
            for _, row in missing_patterns.iterrows():
                missing_cols = [c for c in self.df.columns if row[c] == True]
                if len(missing_cols) >= 2:
                    results["systematic_missing_patterns"].append({
                        "missing_columns": missing_cols,
                        "row_count": int(row['count']),
                        "percentage": round((row['count'] / len(self.df)) * 100, 2)
                    })

        return results

    # =========================================================
    # TEMPORAL GAP DETECTION
    # =========================================================

    def temporal_gaps(
        self,
        datetime_columns: Optional[List[str]] = None,
        freq: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect gaps in datetime sequences.

        Parameters
        ----------
        datetime_columns: Specific datetime columns to analyze.
        freq: Expected frequency (e.g., 'D', 'H', 'M'). Auto-detected if None.
        """

        results = {}

        if datetime_columns is None:
            datetime_columns = [
                c for c in self.df.columns
                if pd.api.types.is_datetime64_any_dtype(self.df[c])
            ]

        for column in datetime_columns:
            series = self.df[column].dropna().sort_values()

            if len(series) < 2:
                results[column] = {
                    "gaps_detected": False,
                    "message": "Insufficient data for gap analysis"
                }
                continue

            # Calculate differences
            diffs = series.diff().dropna()

            if freq is None:
                # Auto-detect frequency
                median_diff: Any = diffs.median()
                if median_diff <= pd.Timedelta(hours=1):
                    freq = "H"
                elif median_diff <= pd.Timedelta(days=1):
                    freq = "D"
                elif median_diff <= pd.Timedelta(weeks=1):
                    freq = "W"
                else:
                    freq = "M"

            # Expected frequency as timedelta
            freq_map = {
                "H": pd.Timedelta(hours=1),
                "D": pd.Timedelta(days=1),
                "W": pd.Timedelta(weeks=1),
                "M": pd.Timedelta(days=30)
            }
            expected_delta = freq_map.get(freq, pd.Timedelta(days=1))

            # Detect gaps (differences > 1.5x expected)
            gap_threshold = expected_delta * 1.5
            gaps = diffs[diffs > gap_threshold]

            results[column] = {
                "gaps_detected": len(gaps) > 0,
                "gap_count": int(len(gaps)),
                "largest_gap": str(gaps.max()) if len(gaps) > 0 else None,
                "avg_gap": str(gaps.mean()) if len(gaps) > 0 else None,
                "expected_frequency": freq,
                "data_range": {
                    "start": str(series.min()),
                    "end": str(series.max()),
                    "span": str(series.max() - series.min())
                },
                "gap_details": [
                    {
                        "gap_start": str(series.iloc[i]),
                        "gap_end": str(series.iloc[i+1]),
                        "gap_size": str(diffs.iloc[i])
                    }
                    for i in gaps.index[:10]  # Limit to first 10 gaps
                ]
            }

        return results

    # =========================================================
    # INVALID VALUE DETECTION
    # =========================================================

    def invalid_values(
        self,
        validation_rules: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect invalid values based on data type expectations and custom rules.

        Parameters
        ----------
        validation_rules: Custom validation rules per column.
        """

        results = {}

        for column in self.df.columns:
            series = self.df[column].dropna()

            if len(series) == 0:
                results[column] = {
                    "invalid_count": 0,
                    "invalid_percentage": 0.0,
                    "issues": []
                }
                continue

            issues = []
            invalid_count = 0

            # Type-specific validation
            if pd.api.types.is_numeric_dtype(series):
                # Check for infinity
                inf_count = int(np.isinf(series).sum())
                if inf_count > 0:
                    issues.append(f"Infinite values: {inf_count}")
                    invalid_count += inf_count

                # Check for NaN in non-null series (shouldn't happen but safety)
                nan_count = int(series.isna().sum())
                if nan_count > 0:
                    issues.append(f"NaN values in non-null series: {nan_count}")
                    invalid_count += nan_count

            elif pd.api.types.is_datetime64_any_dtype(series):
                # Check for dates far in future/past
                now = pd.Timestamp.now()
                future_count = int((series > now + pd.Timedelta(days=365*10)).sum())
                if future_count > 0:
                    issues.append(f"Dates far in future: {future_count}")
                    invalid_count += future_count

                past_count = int((series < pd.Timestamp("1900-01-01")).sum())
                if past_count > 0:
                    issues.append(f"Dates before 1900: {past_count}")
                    invalid_count += past_count

            elif series.dtype == "object":
                # Check for mixed types in object column
                types = series.apply(cast(Any, type)).unique()
                if len(types) > 1:
                    issues.append(f"Mixed types detected: {[str(t) for t in types]}")
                    # Not counting as invalid, just warning

                # Check for extremely long strings (potential data corruption)
                str_lengths = series.astype(str).str.len()
                max_len = str_lengths.max()
                if max_len > 10000:
                    issues.append(f"Extremely long strings detected (max: {max_len})")

            # Custom validation rules
            if validation_rules and column in validation_rules:
                rules = validation_rules[column]

                if "regex" in rules:
                    pattern = re.compile(rules["regex"])
                    matches = series.astype(str).apply(lambda x: bool(pattern.match(x)))
                    non_matches = (~matches).sum()
                    if non_matches > 0:
                        issues.append(f"Regex validation failed: {int(non_matches)} values")
                        invalid_count += int(non_matches)

                if "min_length" in rules:
                    short = (series.astype(str).str.len() < rules["min_length"]).sum()
                    if short > 0:
                        issues.append(f"Below min length: {int(short)}")
                        invalid_count += int(short)

                if "max_length" in rules:
                    long = (series.astype(str).str.len() > rules["max_length"]).sum()
                    if long > 0:
                        issues.append(f"Above max length: {int(long)}")
                        invalid_count += int(long)

                if "allowed_values" in rules:
                    allowed = set(rules["allowed_values"])
                    invalid = (~series.astype(str).isin(allowed)).sum()
                    if invalid > 0:
                        issues.append(f"Values not in allowed set: {int(invalid)}")
                        invalid_count += int(invalid)

            total = len(series)
            results[column] = {
                "invalid_count": invalid_count,
                "invalid_percentage": round((invalid_count / total * 100) if total > 0 else 0, 2),
                "issues": issues
            }

        return results

    # =========================================================
    # TEXT QUALITY ANALYSIS
    # =========================================================

    def text_quality(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze text quality for string columns.

        Checks:
        - Whitespace issues (leading/trailing/multiple spaces)
        - Case consistency
        - Special characters
        - Encoding issues
        - Duplicate spaces
        """

        results = {}

        for column in self.df.columns:
            series = self.df[column].dropna()

            if len(series) == 0:
                results[column] = {
                    "text_quality_score": 100.0,
                    "issues": {}
                }
                continue

            if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
                results[column] = {
                    "text_quality_score": 100.0,
                    "issues": {"note": "Not a text column"}
                }
                continue

            str_series = series.astype(str)
            total = len(str_series)

            issues = {}

            # Leading/trailing whitespace
            leading = str_series.str.startswith(' ').sum()
            trailing = str_series.str.endswith(' ').sum()
            if leading > 0 or trailing > 0:
                issues["whitespace"] = {
                    "leading_count": int(leading),
                    "trailing_count": int(trailing),
                    "percentage": round(((leading + trailing) / total) * 100, 2)
                }

            # Multiple consecutive spaces
            multi_space = str_series.str.contains(r'  +').sum()
            if multi_space > 0:
                issues["multiple_spaces"] = {
                    "count": int(multi_space),
                    "percentage": round((multi_space / total) * 100, 2)
                }

            # Case inconsistency (for categorical-like columns)
            if total > 10:
                lower = str_series.str.lower()
                unique_lower = lower.nunique()
                unique_original = str_series.nunique()
                if unique_lower < unique_original:
                    issues["case_inconsistency"] = {
                        "unique_original": int(unique_original),
                        "unique_case_normalized": int(unique_lower),
                        "potential_duplicates": int(unique_original - unique_lower)
                    }

            # Special characters
            special_char = str_series.str.contains(r'[^\w\s\-.,@]').sum()
            if special_char > 0:
                issues["special_characters"] = {
                    "count": int(special_char),
                    "percentage": round((special_char / total) * 100, 2)
                }

            # Control characters / encoding issues
            control_chars = str_series.apply(lambda x: any(ord(c) < 32 for c in x)).sum()
            if control_chars > 0:
                issues["control_characters"] = {
                    "count": int(control_chars),
                    "percentage": round((control_chars / total) * 100, 2)
                }

            # Calculate quality score (100 = perfect)
            issue_penalty = sum(
                v.get("percentage", 0) if isinstance(v, dict) else 0
                for v in issues.values()
            )
            quality_score = max(0, 100 - issue_penalty)

            results[column] = {
                "text_quality_score": round(quality_score, 2),
                "issues": issues
            }

        return results

    # =========================================================
    # DATE/TIME PROFILING
    # =========================================================

    def datetime_profiling(
        self,
        datetime_columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Comprehensive datetime column profiling.

        Analyzes:
        - Date ranges and spans
        - Timezone information
        - Granularity (year, month, day, hour, minute, second)
        - Business day vs weekend distribution
        - Seasonal patterns
        """

        results = {}

        if datetime_columns is None:
            datetime_columns = [
                c for c in self.df.columns
                if pd.api.types.is_datetime64_any_dtype(self.df[c])
            ]

        # Also check object columns that might be parseable
        for column in self.df.columns:
            if column in datetime_columns:
                continue
            if pd.api.types.is_object_dtype(self.df[column]):
                sample = self.df[column].dropna().head(100)
                if len(sample) > 0:
                    try:
                        parsed = pd.to_datetime(sample, errors="coerce")
                        if parsed.notna().mean() > 0.8:
                            datetime_columns.append(column)
                    except Exception:
                        pass

        for column in datetime_columns:
            series = self.df[column].dropna()

            if len(series) == 0:
                results[column] = {"error": "No valid datetime values"}
                continue

            # Ensure datetime type
            if not pd.api.types.is_datetime64_any_dtype(series):
                series = pd.to_datetime(series, errors="coerce").dropna()

            if len(series) == 0:
                results[column] = {"error": "Could not parse as datetime"}
                continue

            # Basic range
            min_dt = series.min()
            max_dt = series.max()
            span = max_dt - min_dt

            # Granularity analysis
            granularity = {}
            for freq, name in [("Y", "year"), ("M", "month"), ("D", "day"),
                               ("h", "hour"), ("min", "minute"), ("s", "second")]:
                unique_count = series.dt.to_period(freq).nunique()
                granularity[name] = int(unique_count)

            # Timezone
            tz = str(series.dt.tz) if hasattr(series.dt, 'tz') and series.dt.tz else "naive"

            # Day of week distribution
            dow_dist = series.dt.dayofweek.value_counts().to_dict()
            dow_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday",
                         3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
            dow_dist = {dow_names.get(cast(int, k), str(k)): v for k, v in dow_dist.items()}

            # Month distribution
            month_dist = series.dt.month.value_counts().to_dict()
            month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                           7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
            month_dist = {month_names.get(cast(int, k), str(k)): v for k, v in month_dist.items()}

            # Hour distribution (if applicable)
            hour_dist = series.dt.hour.value_counts().to_dict()

            # Business day vs weekend
            is_weekend = series.dt.dayofweek >= 5
            weekend_pct = (is_weekend.sum() / len(series)) * 100

            results[column] = {
                "count": int(len(series)),
                "range": {
                    "min": str(min_dt),
                    "max": str(max_dt),
                    "span_days": span.days,
                    "span_years": round(span.days / 365.25, 2)
                },
                "timezone": tz,
                "granularity": granularity,
                "day_of_week_distribution": dow_dist,
                "month_distribution": month_dist,
                "hour_distribution": hour_dist,
                "weekend_percentage": round(weekend_pct, 2),
                "business_day_percentage": round(100 - weekend_pct, 2),
                "has_time_component": bool((series.dt.hour > 0).any() or
                                            (series.dt.minute > 0).any() or
                                            (series.dt.second > 0).any())
            }

        return results

    # =========================================================
    # COMPLETE STATISTICAL PROFILE
    # =========================================================

    def profile(self) -> Dict[str, Any]:
        """
        Generate complete statistical profile.
        """

        return {
            "missing_values": self.missing_values(),

            "unique_values": self.unique_values(),

            "numerical_statistics": (
                self.numerical_statistics()
            ),

            "categorical_statistics": (
                self.categorical_statistics()
            ),

            "outliers": self.outliers(),

            "constant_columns": (
                self.constant_columns()
            ),

            "hidden_nulls": self.hidden_nulls(),

            "missingness_patterns": self.missingness_patterns(),

            "temporal_gaps": self.temporal_gaps(),

            "invalid_values": self.invalid_values(),

            "text_quality": self.text_quality(),

            "datetime_profiling": self.datetime_profiling()
        }


# =============================================================
# CONVENIENCE FUNCTION
# =============================================================

def profile_statistics(
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Convenience function for statistical profiling.

    Example:
        statistics = profile_statistics(df)
    """

    profiler = StatisticsProfiler(df)

    return profiler.profile()