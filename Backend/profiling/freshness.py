"""
freshness.py

Data Freshness Profiling module for Agentic BI Analyst.

Responsibilities:
- Timestamp analysis (created, updated, loaded)
- Data age calculation
- Staleness detection
- Refresh frequency analysis
- Time since last update
- Freshness scoring
- Batch/load identification
- Incremental vs full load detection

This module DOES NOT modify the input DataFrame.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd


class FreshnessLevel(Enum):
    FRESH = "fresh"           # < 1 day
    RECENT = "recent"         # 1-7 days
    AGING = "aging"           # 1-30 days
    STALE = "stale"           # 30-90 days
    VERY_STALE = "very_stale" # > 90 days


@dataclass
class FreshnessThresholds:
    """Configurable thresholds for freshness levels (in days)."""
    fresh: float = 1.0
    recent: float = 7.0
    aging: float = 30.0
    stale: float = 90.0
    # very_stale is > stale


class FreshnessProfiler:
    """
    Profiles data freshness using timestamp columns.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        thresholds: Optional[FreshnessThresholds] = None,
        reference_time: Optional[pd.Timestamp] = None
    ):
        """
        Initialize freshness profiler.

        Parameters
        ----------
        df : DataFrame to profile
        thresholds : Custom freshness thresholds
        reference_time : Time to calculate age from (default: now)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("FreshnessProfiler requires a pandas DataFrame.")

        self.df = df
        self.thresholds = thresholds or FreshnessThresholds()
        self.reference_time = reference_time or pd.Timestamp.now()

    # =========================================================
    # TIMESTAMP DETECTION
    # =========================================================

    def detect_timestamp_columns(self) -> Dict[str, Any]:
        """
        Detect columns that represent timestamps (created, updated, loaded, etc.).
        """
        results = {
            "datetime_columns": [],
            "created_candidates": [],
            "updated_candidates": [],
            "loaded_candidates": [],
            "business_date_candidates": [],
            "other_datetime": []
        }

        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                results["datetime_columns"].append(col)
                col_lower = col.lower()

                # Categorize by name
                if any(kw in col_lower for kw in ["created", "insert", "load", "ingest", "import"]):
                    results["created_candidates"].append(col)
                elif any(kw in col_lower for kw in ["updated", "modified", "changed", "edit", "last_"]):
                    results["updated_candidates"].append(col)
                elif any(kw in col_lower for kw in ["batch", "load_id", "etl", "snapshot", "as_of", "reporting"]):
                    results["loaded_candidates"].append(col)
                elif any(kw in col_lower for kw in ["date", "dt_", "_dt", "day", "month", "year", "period"]):
                    results["business_date_candidates"].append(col)
                else:
                    results["other_datetime"].append(col)

        return results

    # =========================================================
    # DATA AGE ANALYSIS
    # =========================================================

    def analyze_data_age(
        self,
        timestamp_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate age of data for each timestamp column.
        """
        if timestamp_columns is None:
            detected = self.detect_timestamp_columns()
            timestamp_columns = detected["datetime_columns"]

        if not timestamp_columns:
            return {"error": "No datetime columns found", "columns_analyzed": []}

        results = {
            "reference_time": str(self.reference_time),
            "columns": {}
        }

        for col in timestamp_columns:
            series = self.df[col].dropna()
            if len(series) == 0:
                results["columns"][col] = {"error": "All values are null"}
                continue

            ages = (self.reference_time - series).dt.total_seconds() / (24 * 3600)  # days

            results["columns"][col] = {
                "row_count": int(len(series)),
                "null_count": int(self.df[col].isna().sum()),
                "age_days": {
                    "min": float(ages.min()),
                    "max": float(ages.max()),
                    "mean": float(ages.mean()),
                    "median": float(ages.median()),
                    "std": float(ages.std()),
                    "p25": float(ages.quantile(0.25)),
                    "p75": float(ages.quantile(0.75)),
                    "p90": float(ages.quantile(0.90)),
                    "p99": float(ages.quantile(0.99))
                },
                "freshness_distribution": self._categorize_freshness(ages),
                "oldest_record": str(series.min()),
                "newest_record": str(series.max())
            }

        return results

    def _categorize_freshness(self, ages: pd.Series) -> Dict[str, Any]:
        """Categorize ages into freshness levels."""
        t = self.thresholds

        fresh = (ages <= t.fresh).sum()
        recent = ((ages > t.fresh) & (ages <= t.recent)).sum()
        aging = ((ages > t.recent) & (ages <= t.aging)).sum()
        stale = ((ages > t.aging) & (ages <= t.stale)).sum()
        very_stale = (ages > t.stale).sum()
        total = len(ages)

        return {
            "fresh": {"count": int(fresh), "percentage": round(fresh / total * 100, 2)},
            "recent": {"count": int(recent), "percentage": round(recent / total * 100, 2)},
            "aging": {"count": int(aging), "percentage": round(aging / total * 100, 2)},
            "stale": {"count": int(stale), "percentage": round(stale / total * 100, 2)},
            "very_stale": {"count": int(very_stale), "percentage": round(very_stale / total * 100, 2)},
            "overall_level": self._overall_freshness_level(ages)
        }

    def _overall_freshness_level(self, ages: pd.Series) -> str:
        """Determine overall freshness level based on median age."""
        median_age = ages.median()
        t = self.thresholds

        if median_age <= t.fresh:
            return FreshnessLevel.FRESH.value
        elif median_age <= t.recent:
            return FreshnessLevel.RECENT.value
        elif median_age <= t.aging:
            return FreshnessLevel.AGING.value
        elif median_age <= t.stale:
            return FreshnessLevel.STALE.value
        else:
            return FreshnessLevel.VERY_STALE.value

    # =========================================================
    # STALENESS DETECTION
    # =========================================================

    def detect_staleness(
        self,
        timestamp_column: str,
        max_age_days: Optional[float] = None,
        group_by: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect stale records based on timestamp.

        Parameters
        ----------
        timestamp_column : Column to check for staleness
        max_age_days : Maximum allowed age (uses threshold if None)
        group_by : Optional grouping for per-group staleness
        """
        if timestamp_column not in self.df.columns:
            return {"error": f"Column '{timestamp_column}' not found"}

        max_age = max_age_days or self.thresholds.stale
        series = self.df[timestamp_column].dropna()

        if len(series) == 0:
            return {"error": "All values are null"}

        ages = (self.reference_time - series).dt.total_seconds() / (24 * 3600)
        stale_mask = ages > max_age

        results = {
            "column": timestamp_column,
            "max_age_threshold_days": max_age,
            "total_records": int(len(series)),
            "stale_records": int(stale_mask.sum()),
            "stale_percentage": round(stale_mask.sum() / len(series) * 100, 2),
            "freshest_record": str(series.max()),
            "oldest_record": str(series.min()),
            "oldest_age_days": float(ages.max())
        }

        if group_by:
            group_results = {}
            for group_val, group_df in self.df.groupby(group_by):
                group_series = group_df[timestamp_column].dropna()
                if len(group_series) > 0:
                    group_ages = (self.reference_time - group_series).dt.total_seconds() / (24 * 3600)
                    group_stale = (group_ages > max_age).sum()
                    group_results[str(group_val)] = {
                        "records": len(group_series),
                        "stale_count": int(group_stale),
                        "stale_percentage": round(group_stale / len(group_series) * 100, 2),
                        "max_age_days": float(group_ages.max())
                    }
            results["by_group"] = group_results

        return results

    # =========================================================
    # REFRESH FREQUENCY
    # =========================================================

    def analyze_refresh_frequency(
        self,
        timestamp_column: str,
        group_by: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze how frequently data is refreshed/updated.
        """
        if timestamp_column not in self.df.columns:
            return {"error": f"Column '{timestamp_column}' not found"}

        series = self.df[timestamp_column].dropna().sort_values()
        if len(series) < 2:
            return {"error": "Insufficient data for frequency analysis"}

        diffs = series.diff().dropna()
        diff_days = diffs.dt.total_seconds() / (24 * 3600)

        results = {
            "column": timestamp_column,
            "total_records": len(series),
            "date_range": {
                "start": str(series.min()),
                "end": str(series.max()),
                "span_days": float((series.max() - series.min()).days)
            },
            "intervals": {
                "min_days": float(diff_days.min()),
                "max_days": float(diff_days.max()),
                "mean_days": float(diff_days.mean()),
                "median_days": float(diff_days.median()),
                "std_days": float(diff_days.std())
            },
            "inferred_frequency": self._infer_frequency(diff_days.median()),
            "gap_analysis": self._analyze_gaps(diff_days, expected_days=self._expected_gap_days(diff_days.median()))
        }

        if group_by:
            group_results = {}
            for group_val, group_df in self.df.groupby(group_by):
                group_series = group_df[timestamp_column].dropna().sort_values()
                if len(group_series) >= 2:
                    group_diffs = group_series.diff().dropna()
                    group_diff_days = group_diffs.dt.total_seconds() / (24 * 3600)
                    group_results[str(group_val)] = {
                        "records": len(group_series),
                        "mean_interval_days": float(group_diff_days.mean()),
                        "median_interval_days": float(group_diff_days.median()),
                        "inferred_frequency": self._infer_frequency(group_diff_days.median())
                    }
            results["by_group"] = group_results

        return results

    def _infer_frequency(self, median_interval: float) -> str:
        """Infer refresh frequency from median interval."""
        if median_interval <= 1/24:  # < 1 hour
            return "hourly"
        elif median_interval <= 1:   # < 1 day
            return "daily"
        elif median_interval <= 7:   # < 1 week
            return "weekly"
        elif median_interval <= 31:  # < 1 month
            return "monthly"
        elif median_interval <= 93:  # < 1 quarter
            return "quarterly"
        elif median_interval <= 366: # < 1 year
            return "yearly"
        else:
            return "irregular"

    def _expected_gap_days(self, median_interval: float) -> float:
        """Get expected gap in days for frequency."""
        freq = self._infer_frequency(median_interval)
        return {"hourly": 1/24, "daily": 1, "weekly": 7, "monthly": 30, "quarterly": 90, "yearly": 365}.get(freq, 1)

    def _analyze_gaps(self, diff_days: pd.Series, expected_days: float) -> Dict[str, Any]:
        """Analyze gaps in refresh schedule."""
        gap_threshold = expected_days * 1.5
        gaps = diff_days[diff_days > gap_threshold]

        return {
            "expected_interval_days": expected_days,
            "gap_threshold_days": gap_threshold,
            "gaps_found": int(len(gaps)),
            "largest_gap_days": float(gaps.max()) if len(gaps) > 0 else 0,
            "average_gap_days": float(gaps.mean()) if len(gaps) > 0 else 0
        }

    # =========================================================
    # BATCH/LOAD ANALYSIS
    # =========================================================

    def analyze_batches(
        self,
        batch_column: Optional[str] = None,
        timestamp_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze batch/load patterns in the data.
        """
        results: Dict[str, Any] = {"batches_detected": False}

        # Try to find batch column
        if batch_column is None:
            for col in self.df.columns:
                col_lower = col.lower()
                if any(kw in col_lower for kw in ["batch", "load_id", "etl_run", "snapshot_id", "file_id"]):
                    batch_column = col
                    break

        if batch_column and batch_column in self.df.columns:
            results["batches_detected"] = True
            results["batch_column"] = batch_column

            batch_series = self.df[batch_column].dropna()
            batch_counts = batch_series.value_counts()

            results["batch_summary"] = {
                "total_batches": int(batch_counts.nunique()),
                "total_records": int(len(batch_series)),
                "avg_batch_size": float(batch_counts.mean()),
                "batch_size_std": float(batch_counts.std()),
                "min_batch_size": int(batch_counts.min()),
                "max_batch_size": int(batch_counts.max()),
                "batch_distribution": batch_counts.head(20).to_dict()
            }

            # If timestamp column provided, analyze batch timing
            if timestamp_column and timestamp_column in self.df.columns:
                batch_times = self.df.groupby(batch_column)[timestamp_column].agg(["min", "max", "count"])
                batch_times["duration"] = (batch_times["max"] - batch_times["min"]).dt.total_seconds()
                results["batch_timing"] = batch_times.describe().to_dict()

        # Also check for incremental vs full load patterns
        if timestamp_column and timestamp_column in self.df.columns:
            results["load_pattern"] = self._detect_load_pattern(timestamp_column)

        return results

    def _detect_load_pattern(self, timestamp_column: str) -> Dict[str, Any]:
        """Detect if loads are incremental or full refresh."""
        series = self.df[timestamp_column].dropna().sort_values()
        if len(series) < 2:
            return {"pattern": "unknown", "reason": "insufficient data"}

        # Check if timestamps are clustered (batch loads) or continuous
        diffs = series.diff().dropna()
        diff_seconds = diffs.dt.total_seconds()

        # If most diffs are very small (seconds) but occasional large gaps -> batch
        # If diffs are relatively uniform -> continuous/incremental
        small_diffs = (diff_seconds < 60).sum()  # < 1 minute
        large_gaps = (diff_seconds > 3600).sum()  # > 1 hour

        total = len(diff_seconds)
        small_pct = small_diffs / total
        large_pct = large_gaps / total

        if small_pct > 0.8 and large_pct > 0.1:
            pattern = "batch_incremental"
        elif large_pct > 0.5:
            pattern = "batch_full"
        elif small_pct > 0.9:
            pattern = "continuous_streaming"
        else:
            pattern = "mixed"

        return {
            "pattern": pattern,
            "small_intervals_pct": round(small_pct * 100, 2),
            "large_gaps_pct": round(large_pct * 100, 2),
            "median_interval_seconds": float(diff_seconds.median())
        }

    # =========================================================
    # FRESHNESS SCORE
    # =========================================================

    def calculate_freshness_score(
        self,
        timestamp_columns: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate overall freshness score (0-100).

        Higher score = fresher data.
        """
        if timestamp_columns is None:
            detected = self.detect_timestamp_columns()
            timestamp_columns = detected["updated_candidates"] or detected["datetime_columns"]

        if not timestamp_columns:
            return {"score": 0, "level": "unknown", "message": "No timestamp columns found"}

        if weights is None:
            weights = {col: 1.0 for col in timestamp_columns}

        scores = {}
        for col in timestamp_columns:
            if col not in self.df.columns:
                continue

            series = self.df[col].dropna()
            if len(series) == 0:
                scores[col] = 0
                continue

            ages = (self.reference_time - series).dt.total_seconds() / (24 * 3600)
            median_age = ages.median()

            # Score: 100 at age 0, decays exponentially
            # Half-life at aging threshold (30 days)
            half_life = self.thresholds.aging
            score = 100 * np.exp(-np.log(2) * median_age / half_life)
            scores[col] = round(max(0, min(100, score)), 2)

        # Weighted average
        total_weight = sum(weights.get(c, 0) for c in scores)
        if total_weight > 0:
            weighted_score = sum(scores[c] * weights.get(c, 0) for c in scores) / total_weight
        else:
            weighted_score = np.mean(list(scores.values())) if scores else 0

        return {
            "overall_score": round(weighted_score, 2),
            "level": self._score_to_level(weighted_score),
            "column_scores": scores,
            "weights": weights,
            "reference_time": str(self.reference_time)
        }

    def _score_to_level(self, score: float) -> str:
        """Convert score to freshness level."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        elif score >= 20:
            return "poor"
        else:
            return "critical"

    # =========================================================
    # COMPREHENSIVE PROFILE
    # =========================================================

    def profile(self) -> Dict[str, Any]:
        """Generate complete freshness profile."""
        detected = self.detect_timestamp_columns()
        all_dt = detected["datetime_columns"]

        return {
            "timestamp_detection": detected,
            "data_age_analysis": self.analyze_data_age(all_dt),
            "staleness_checks": {
                col: self.detect_staleness(col) for col in all_dt[:3]  # Limit to first 3
            },
            "refresh_frequency": {
                col: self.analyze_refresh_frequency(col) for col in all_dt[:3]
            },
            "batch_analysis": self.analyze_batches(),
            "freshness_score": self.calculate_freshness_score(all_dt)
        }


def profile_freshness(
    df: pd.DataFrame,
    thresholds: Optional[FreshnessThresholds] = None,
    reference_time: Optional[pd.Timestamp] = None
) -> Dict[str, Any]:
    """Convenience function for freshness profiling."""
    profiler = FreshnessProfiler(df, thresholds, reference_time)
    return profiler.profile()