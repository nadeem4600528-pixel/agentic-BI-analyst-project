"""
relationships.py

Multi-Table Relationship Discovery module for Agentic BI Analyst.

Responsibilities:
- Primary Key detection (single & composite)
- Foreign Key detection
- Referential Integrity validation
- Cross-table relationship discovery
- One-to-One, One-to-Many, Many-to-Many classification
- Join cardinality estimation
- Relationship strength scoring
- Orphan record detection
- Circular reference detection

This module operates on multiple DataFrames (dictionary of table_name -> DataFrame).
"""

from typing import Any, Dict, List, Optional, Union
from collections import defaultdict
import itertools
import numpy as np
import pandas as pd


class RelationshipProfiler:
    """
    Performs multi-table relationship analysis on a dictionary of DataFrames.
    """

    def __init__(self, tables: Dict[str, pd.DataFrame]):
        """
        Initialize with a dictionary of table_name -> DataFrame.

        Parameters
        ----------
        tables : Dict[str, pd.DataFrame]
            Dictionary mapping table names to DataFrames.
        """
        if not isinstance(tables, dict):
            raise TypeError("RelationshipProfiler requires a dict of table_name -> DataFrame.")

        for name, df in tables.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Table '{name}' is not a pandas DataFrame.")

        self.tables = tables
        self._table_profiles = {}
        self._build_table_profiles()

    def _build_table_profiles(self):
        """Build basic profiles for each table."""
        for name, df in self.tables.items():
            self._table_profiles[name] = {
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "row_count": len(df),
                "col_count": len(df.columns),
                "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
                "cat_cols": df.select_dtypes(include=["object", "category", "bool"]).columns.tolist(),
                "datetime_cols": df.select_dtypes(include=["datetime64"]).columns.tolist(),
            }

    # =========================================================
    # PRIMARY KEY DETECTION
    # =========================================================

    def detect_primary_keys(
        self,
        uniqueness_threshold: float = 0.99,
        max_key_size: int = 3
    ) -> Dict[str, Any]:
        """
        Detect primary key candidates for each table.

        Returns
        -------
        Dict with table names as keys, each containing candidate PKs.
        """
        results = {}

        for table_name, df in self.tables.items():
            candidates = []

            # Single column candidates
            for col in df.columns:
                series = df[col].dropna()
                if len(series) == 0:
                    continue

                uniqueness = series.nunique() / len(series)
                if uniqueness >= uniqueness_threshold:
                    candidates.append({
                        "columns": [col],
                        "key_size": 1,
                        "uniqueness_ratio": round(uniqueness, 4),
                        "is_unique": uniqueness == 1.0,
                        "null_count": int(df[col].isna().sum()),
                        "suitable": df[col].isna().sum() == 0
                    })

            # Composite key candidates
            candidate_cols = [c for c in df.columns if df[c].dropna().nunique() / len(df[c].dropna()) > 0.1]
            candidate_cols = candidate_cols[:10]  # Limit for performance

            for key_size in range(2, min(max_key_size + 1, len(candidate_cols) + 1)):
                for combo in itertools.combinations(candidate_cols, key_size):
                    combo_df = df[list(combo)].dropna()
                    if len(combo_df) == 0:
                        continue

                    unique_count = combo_df.drop_duplicates().shape[0]
                    total_count = len(combo_df)
                    uniqueness = unique_count / total_count

                    if uniqueness >= uniqueness_threshold:
                        candidates.append({
                            "columns": list(combo),
                            "key_size": key_size,
                            "uniqueness_ratio": round(uniqueness, 4),
                            "is_unique": uniqueness == 1.0,
                            "null_rows": int(len(df) - total_count),
                            "suitable": total_count == len(df)
                        })

            # Sort by key size then uniqueness
            candidates.sort(key=lambda x: (x["key_size"], -x["uniqueness_ratio"]))

            results[table_name] = {
                "candidates": candidates[:20],  # Top 20
                "best_candidate": candidates[0] if candidates else None
            }

        return results

    # =========================================================
    # FOREIGN KEY DETECTION
    # =========================================================

    def detect_foreign_keys(
        self,
        pk_results: Optional[Dict] = None,
        sample_size: int = 10000
    ) -> Dict[str, Any]:
        """
        Detect foreign key relationships between tables.

        Parameters
        ----------
        pk_results : Optional primary key results from detect_primary_keys()
        sample_size : Max rows to sample for performance

        Returns
        -------
        Dict with detected FK relationships.
        """
        if pk_results is None:
            pk_results = self.detect_primary_keys()

        results = {
            "relationships": [],
            "potential_fks": []
        }

        table_names = list(self.tables.keys())

        for parent_table in table_names:
            pk_info = pk_results.get(parent_table, {})
            best_pk = pk_info.get("best_candidate")

            if not best_pk:
                continue

            pk_cols = best_pk["columns"]
            parent_df = self.tables[parent_table]
            parent_keys = set(zip(*[parent_df[c].dropna() for c in pk_cols]))

            for child_table in table_names:
                if child_table == parent_table:
                    continue

                child_df = self.tables[child_table]

                # Find matching column names or similar names
                for child_col in child_df.columns:
                    for pk_col in pk_cols:
                        if self._columns_match(child_col, pk_col):
                            # Check referential integrity
                            child_vals = child_df[child_col].dropna()
                            if len(child_vals) == 0:
                                continue

                            # Sample for performance
                            if len(child_vals) > sample_size:
                                child_vals = child_vals.sample(n=sample_size, random_state=42)

                            child_keys = set(child_vals.unique())
                            matching = child_keys & parent_keys
                            coverage = len(matching) / len(child_keys) if child_keys else 0

                            if coverage > 0.5:  # At least 50% match
                                results["relationships"].append({
                                    "parent_table": parent_table,
                                    "parent_columns": pk_cols,
                                    "child_table": child_table,
                                    "child_columns": [child_col],
                                    "match_coverage": round(coverage, 4),
                                    "orphan_count": int(len(child_keys) - len(matching)),
                                    "relationship_type": self._classify_relationship(
                                        parent_df, child_df, pk_cols, [child_col]
                                    )
                                })

        # Also check all column pairs for potential FKs (name-based)
        results["potential_fks"] = self._find_potential_fks_by_name()

        return results

    def _columns_match(self, col1: str, col2: str) -> bool:
        """Check if two column names likely represent the same entity."""
        c1, c2 = col1.lower(), col2.lower()

        # Exact match
        if c1 == c2:
            return True

        # Common FK patterns
        fk_suffixes = ["_id", "id_", "key", "_key", "code", "_code", "num", "_num", "no", "_no"]

        # Check if one is the other with FK suffix
        for suffix in fk_suffixes:
            if c1 == c2 + suffix or c2 == c1 + suffix:
                return True

        # Check if both end with same identifier
        for suffix in fk_suffixes:
            if c1.endswith(suffix) and c2.endswith(suffix):
                base1 = c1[:-len(suffix)]
                base2 = c2[:-len(suffix)]
                if base1 == base2:
                    return True

        return False

    def _find_potential_fks_by_name(self) -> List[Dict[str, Any]]:
        """Find potential FK relationships based on column name similarity."""
        potential = []
        table_names = list(self.tables.keys())

        for t1 in table_names:
            for t2 in table_names:
                if t1 >= t2:
                    continue

                df1 = self.tables[t1]
                df2 = self.tables[t2]

                for c1 in df1.columns:
                    for c2 in df2.columns:
                        if self._columns_match(c1, c2):
                            # Check data type compatibility
                            dt1 = str(df1[c1].dtype)
                            dt2 = str(df2[c2].dtype)

                            type_compatible = (
                                (pd.api.types.is_numeric_dtype(df1[c1]) and pd.api.types.is_numeric_dtype(df2[c2])) or
                                (pd.api.types.is_string_dtype(df1[c1]) and pd.api.types.is_string_dtype(df2[c2])) or
                                (pd.api.types.is_datetime64_any_dtype(df1[c1]) and pd.api.types.is_datetime64_any_dtype(df2[c2]))
                            )

                            if type_compatible:
                                potential.append({
                                    "table_1": t1,
                                    "column_1": c1,
                                    "table_2": t2,
                                    "column_2": c2,
                                    "dtype_1": dt1,
                                    "dtype_2": dt2,
                                    "name_match": True
                                })

        return potential

    def _classify_relationship(
        self,
        parent_df: pd.DataFrame,
        child_df: pd.DataFrame,
        parent_cols: List[str],
        child_cols: List[str]
    ) -> str:
        """Classify relationship as 1:1, 1:N, N:1, or N:M."""
        # Check uniqueness in parent
        parent_unique = parent_df[parent_cols].drop_duplicates().shape[0] == len(parent_df.dropna(subset=parent_cols))

        # Check uniqueness in child
        child_unique = child_df[child_cols].drop_duplicates().shape[0] == len(child_df.dropna(subset=child_cols))

        if parent_unique and child_unique:
            return "one_to_one"
        elif parent_unique and not child_unique:
            return "one_to_many"
        elif not parent_unique and child_unique:
            return "many_to_one"
        else:
            return "many_to_many"

    # =========================================================
    # REFERENTIAL INTEGRITY
    # =========================================================

    def validate_referential_integrity(
        self,
        relationships: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Validate referential integrity for detected or provided relationships.

        Parameters
        ----------
        relationships : List of relationship dicts from detect_foreign_keys()

        Returns
        -------
        Validation results per relationship.
        """
        if relationships is None:
            fk_results = self.detect_foreign_keys()
            relationships = fk_results.get("relationships", [])
        relationships = relationships or []

        results = {
            "validations": [],
            "summary": {
                "total_relationships": len(relationships),
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }

        for rel in relationships:
            parent_table = rel["parent_table"]
            child_table = rel["child_table"]
            parent_cols = rel["parent_columns"]
            child_cols = rel["child_columns"]

            parent_df = self.tables[parent_table]
            child_df = self.tables[child_table]

            # Build parent key set
            parent_keys = set()
            for _, row in parent_df[parent_cols].dropna().iterrows():
                key = tuple(row[c] for c in parent_cols)
                parent_keys.add(key)

            # Check child keys
            child_clean = child_df[child_cols].dropna()
            total_child = len(child_clean)

            if total_child == 0:
                results["validations"].append({
                    "relationship": f"{child_table} -> {parent_table}",
                    "status": "SKIPPED",
                    "message": "No non-null child keys",
                    "orphan_count": 0,
                    "orphan_percentage": 0.0
                })
                results["summary"]["warnings"] += 1
                continue

            orphan_keys = []
            for _, row in child_clean.iterrows():
                key = tuple(row[c] for c in child_cols)
                if key not in parent_keys:
                    orphan_keys.append(key)

            orphan_count = len(orphan_keys)
            orphan_pct = (orphan_count / total_child) * 100

            if orphan_count == 0:
                status = "PASSED"
                results["summary"]["passed"] += 1
            elif orphan_pct < 1:
                status = "WARNING"
                results["summary"]["warnings"] += 1
            else:
                status = "FAILED"
                results["summary"]["failed"] += 1

            results["validations"].append({
                "relationship": f"{child_table}({','.join(child_cols)}) -> {parent_table}({','.join(parent_cols)})",
                "status": status,
                "total_child_rows": total_child,
                "orphan_count": orphan_count,
                "orphan_percentage": round(orphan_pct, 2),
                "sample_orphans": [str(k) for k in orphan_keys[:10]]
            })

        return results

    def find_orphan_records(
        self,
        child_table: str,
        child_columns: List[str],
        parent_table: str,
        parent_columns: List[str]
    ) -> Dict[str, Any]:
        """
        Find orphan records in child table (FK values not in parent PK).
        """
        if child_table not in self.tables or parent_table not in self.tables:
            return {"error": "Table not found"}

        child_df = self.tables[child_table]
        parent_df = self.tables[parent_table]

        parent_keys = set()
        for _, row in parent_df[parent_columns].dropna().iterrows():
            key = tuple(row[c] for c in parent_columns)
            parent_keys.add(key)

        orphans = child_df[child_df[child_columns].apply(
            lambda row: tuple(row) not in parent_keys, axis=1
        )]

        return {
            "child_table": child_table,
            "parent_table": parent_table,
            "orphan_count": len(orphans),
            "orphan_percentage": round((len(orphans) / len(child_df)) * 100, 2),
            "orphan_sample": orphans.head(20).to_dict("records")
        }

    # =========================================================
    # JOIN CARDINALITY & ANALYSIS
    # =========================================================

    def estimate_join_cardinality(
        self,
        left_table: str,
        right_table: str,
        left_on: Union[str, List[str]],
        right_on: Union[str, List[str]],
        how: str = "inner"
    ) -> Dict[str, Any]:
        """
        Estimate join cardinality without performing the join.

        Parameters
        ----------
        left_table, right_table : Table names
        left_on, right_on : Join columns
        how : Join type (inner, left, right, outer)

        Returns
        -------
        Estimated row count and join characteristics.
        """
        if left_table not in self.tables or right_table not in self.tables:
            return {"error": "Table not found"}

        left_df = self.tables[left_table]
        right_df = self.tables[right_table]

        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        # Unique counts
        left_keys = left_df[left_on].dropna().drop_duplicates()
        right_keys = right_df[right_on].dropna().drop_duplicates()

        left_unique = len(left_keys)
        right_unique = len(right_keys)

        # Overlap estimation
        left_key_set = set(zip(*[left_keys[c] for c in left_on]))
        right_key_set = set(zip(*[right_keys[c] for c in right_on]))

        overlap = len(left_key_set & right_key_set)
        left_only = len(left_key_set - right_key_set)
        right_only = len(right_key_set - left_key_set)

        # Cardinality estimation
        left_rows = len(left_df)
        right_rows = len(right_df)

        left_dup_factor = left_rows / left_unique if left_unique > 0 else 1
        right_dup_factor = right_rows / right_unique if right_unique > 0 else 1

        if how == "inner":
            est_rows = overlap * left_dup_factor * right_dup_factor
        elif how == "left":
            est_rows = left_rows
        elif how == "right":
            est_rows = right_rows
        elif how == "outer":
            est_rows = (overlap * left_dup_factor * right_dup_factor) + \
                       (left_only * left_dup_factor) + (right_only * right_dup_factor)
        else:
            return {"error": f"Unknown join type: {how}"}

        return {
            "left_table": left_table,
            "right_table": right_table,
            "join_type": how,
            "left_keys": left_unique,
            "right_keys": right_unique,
            "overlapping_keys": overlap,
            "left_only_keys": left_only,
            "right_only_keys": right_only,
            "estimated_rows": int(est_rows),
            "left_duplication_factor": round(left_dup_factor, 2),
            "right_duplication_factor": round(right_dup_factor, 2),
            "selectivity": round(overlap / max(left_unique, 1), 4)
        }

    # =========================================================
    # CIRCULAR REFERENCES
    # =========================================================

    def detect_circular_references(
        self,
        relationships: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Detect circular foreign key references.
        e.g., A -> B -> C -> A
        """
        if relationships is None:
            fk_results = self.detect_foreign_keys()
            relationships = fk_results.get("relationships", [])
        relationships = relationships or []

        # Build adjacency graph
        graph = defaultdict(list)
        for rel in relationships:
            parent = rel["parent_table"]
            child = rel["child_table"]
            graph[parent].append(child)

        # Find cycles using DFS
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    idx = path.index(neighbor)
                    cycle = path[idx:] + [neighbor]
                    if cycle not in cycles:
                        cycles.append(cycle)
                    return True

            rec_stack.remove(node)
            path.pop()
            return False

        for node in graph:
            if node not in visited:
                dfs(node)

        return {
            "circular_references": cycles,
            "has_cycles": len(cycles) > 0,
            "graph": {k: v for k, v in graph.items()}
        }

    # =========================================================
    # RELATIONSHIP STRENGTH
    # =========================================================

    def score_relationship_strength(
        self,
        relationships: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Score the strength/confidence of each relationship.
        """
        if relationships is None:
            fk_results = self.detect_foreign_keys()
            relationships = fk_results.get("relationships", [])
        relationships = relationships or []

        scored = []

        for rel in relationships:
            parent_table = rel["parent_table"]
            child_table = rel["child_table"]
            parent_cols = rel["parent_columns"]
            child_cols = rel["child_columns"]

            parent_df = self.tables[parent_table]
            child_df = self.tables[child_table]

            # Coverage score
            coverage = rel.get("match_coverage", 0)

            # Uniqueness score (parent should be unique)
            parent_unique = parent_df[parent_cols].drop_duplicates().shape[0] == len(parent_df[parent_cols].dropna())

            # Null score (fewer nulls in FK is better)
            child_null_pct = child_df[child_cols[0]].isna().mean() if child_cols else 1

            # Name similarity score
            name_score = 1.0 if self._columns_match(child_cols[0], parent_cols[0]) else 0.5

            # Combined score
            strength = (
                coverage * 0.4 +
                (1.0 if parent_unique else 0.3) * 0.3 +
                (1.0 - child_null_pct) * 0.2 +
                name_score * 0.1
            )

            rel_copy = rel.copy()
            rel_copy["strength_score"] = round(strength, 3)
            rel_copy["strength_level"] = (
                "Strong" if strength >= 0.8 else
                "Medium" if strength >= 0.5 else
                "Weak"
            )
            scored.append(rel_copy)

        return {
            "scored_relationships": sorted(scored, key=lambda x: x["strength_score"], reverse=True)
        }

    # =========================================================
    # COMPREHENSIVE PROFILE
    # =========================================================

    def profile(self) -> Dict[str, Any]:
        """Generate complete multi-table relationship profile."""
        pk_results = self.detect_primary_keys()
        fk_results = self.detect_foreign_keys(pk_results)
        relationships = fk_results.get("relationships", [])

        return {
            "tables": {name: {"rows": len(df), "cols": len(df.columns)} for name, df in self.tables.items()},
            "primary_keys": pk_results,
            "foreign_keys": fk_results,
            "referential_integrity": self.validate_referential_integrity(relationships),
            "relationship_strength": self.score_relationship_strength(relationships),
            "circular_references": self.detect_circular_references(relationships),
            "join_analysis": self._analyze_all_joins(relationships)
        }

    def _analyze_all_joins(self, relationships: List[Dict]) -> Dict[str, Any]:
        """Analyze join cardinality for all relationships."""
        join_analyses = []

        for rel in relationships:
            analysis = self.estimate_join_cardinality(
                rel["parent_table"],
                rel["child_table"],
                rel["parent_columns"],
                rel["child_columns"],
                "inner"
            )
            analysis["relationship_type"] = rel.get("relationship_type", "unknown")
            join_analyses.append(analysis)

        return {
            "join_cardinalities": join_analyses
        }


def profile_relationships(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Convenience function for multi-table relationship profiling."""
    profiler = RelationshipProfiler(tables)
    return profiler.profile()