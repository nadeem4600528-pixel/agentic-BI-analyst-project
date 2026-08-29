"""
correlation.py

Correlation and Dependency Analysis module for Agentic BI Analyst.

Responsibilities:
- Pearson correlation for numeric columns
- Spearman correlation for monotonic relationships
- Kendall's Tau for ordinal data
- Cramér's V for categorical associations
- Theil's U for asymmetric categorical relationships
- Mutual Information for non-linear dependencies
- Conditional dependency analysis
- Correlation network/clustering
- Partial correlation (controlling for confounders)
- Distance correlation for non-linear relationships

This module DOES NOT modify the input DataFrame.
"""

from typing import Any, Dict, List, Optional, cast
from scipy.stats import chi2_contingency, entropy
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd


class CorrelationProfiler:
    """
    Performs correlation and dependency analysis on pandas DataFrames.
    Supports both single DataFrame and multi-DataFrame analysis.
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("CorrelationProfiler requires a pandas DataFrame.")
        self.df = df

    # =========================================================
    # NUMERIC CORRELATIONS
    # =========================================================

    def pearson_correlation(
        self,
        columns: Optional[List[str]] = None,
        min_periods: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate Pearson correlation matrix for numeric columns.
        Measures linear relationships.
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        if columns:
            numeric_df = numeric_df[columns]

        if numeric_df.shape[1] < 2:
            return {"correlation_matrix": {}, "message": "Insufficient numeric columns"}

        corr_matrix = numeric_df.corr(method="pearson", min_periods=min_periods)

        return {
            "method": "pearson",
            "correlation_matrix": corr_matrix.round(4).to_dict(),
            "columns": list(corr_matrix.columns),
            "strong_pairs": self._find_strong_correlations(corr_matrix, threshold=0.7)
        }

    def spearman_correlation(
        self,
        columns: Optional[List[str]] = None,
        min_periods: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate Spearman rank correlation for monotonic relationships.
        Works with ordinal and non-normally distributed data.
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        if columns:
            numeric_df = numeric_df[columns]

        if numeric_df.shape[1] < 2:
            return {"correlation_matrix": {}, "message": "Insufficient numeric columns"}

        corr_matrix = numeric_df.corr(method="spearman", min_periods=min_periods)

        return {
            "method": "spearman",
            "correlation_matrix": corr_matrix.round(4).to_dict(),
            "columns": list(corr_matrix.columns),
            "strong_pairs": self._find_strong_correlations(corr_matrix, threshold=0.7)
        }

    def kendall_tau(
        self,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate Kendall's Tau for ordinal data.
        More robust for small samples and ties.
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        if columns:
            numeric_df = numeric_df[columns]

        if numeric_df.shape[1] < 2:
            return {"correlation_matrix": {}, "message": "Insufficient numeric columns"}

        corr_matrix = numeric_df.corr(method="kendall")

        return {
            "method": "kendall",
            "correlation_matrix": corr_matrix.round(4).to_dict(),
            "columns": list(corr_matrix.columns),
            "strong_pairs": self._find_strong_correlations(corr_matrix, threshold=0.5)
        }

    def distance_correlation(
        self,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate distance correlation - detects non-linear relationships.
        Range: [0, 1], where 0 = independent, 1 = dependent.
        """
        numeric_df = self.df.select_dtypes(include=[np.number]).dropna()
        if columns:
            numeric_df = numeric_df[columns]

        if numeric_df.shape[1] < 2:
            return {"correlation_matrix": {}, "message": "Insufficient numeric columns"}

        cols = numeric_df.columns.tolist()
        n = len(cols)
        dist_corr = np.zeros((n, n))

        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i <= j:
                    # NEW
                    dcor = self._distance_correlation(
                        np.asarray(numeric_df[col1].values, dtype=float),
                        np.asarray(numeric_df[col2].values, dtype=float)
)
                    dist_corr[i, j] = dcor
                    dist_corr[j, i] = dcor

        corr_df = pd.DataFrame(dist_corr, index=cols, columns=cols)

        return {
            "method": "distance_correlation",
            "correlation_matrix": corr_df.round(4).to_dict(),
            "columns": cols,
            "strong_pairs": self._find_strong_correlations(corr_df, threshold=0.5)
        }

    def _distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute distance correlation between two arrays."""
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        a = self._distance_matrix(x)
        b = self._distance_matrix(y)

        A = self._double_center(a)
        B = self._double_center(b)

        dcov2_xy = np.mean(A * B)
        dcov2_xx = np.mean(A * A)
        dcov2_yy = np.mean(B * B)

        if dcov2_xx == 0 or dcov2_yy == 0:
            return 0.0

        return float(np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy)))
    def _distance_matrix(self, x: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix."""
        n = x.shape[0]
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = np.sqrt(np.sum((x[i] - x[j]) ** 2))
        return dist

    def _double_center(self, mat: np.ndarray) -> np.ndarray:
        """Double center a matrix."""
        row_means = np.mean(mat, axis=1, keepdims=True)
        col_means = np.mean(mat, axis=0, keepdims=True)
        grand_mean = np.mean(mat)
        return mat - row_means - col_means + grand_mean

    # =========================================================
    # CATEGORICAL ASSOCIATIONS
    # =========================================================

    def cramers_v(
        self,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate Cramér's V for categorical-categorical associations.
        Range: [0, 1], corrected for bias.
        """
        cat_columns = self.df.select_dtypes(include=["object", "category", "bool"]).columns
        if columns:
            cat_columns = [c for c in columns if c in cat_columns]

        if len(cat_columns) < 2:
            return {"association_matrix": {}, "message": "Insufficient categorical columns"}

        cols = list(cat_columns)
        n = len(cols)
        v_matrix = np.zeros((n, n))

        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i <= j:
                    v = self._cramers_v(self.df[col1], self.df[col2])
                    v_matrix[i, j] = v
                    v_matrix[j, i] = v

        v_df = pd.DataFrame(v_matrix, index=cols, columns=cols)

        return {
            "method": "cramers_v",
            "association_matrix": v_df.round(4).to_dict(),
            "columns": cols,
            "strong_pairs": self._find_strong_correlations(v_df, threshold=0.5)
        }

    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate bias-corrected Cramér's V."""
        confusion = pd.crosstab(x, y)
        if confusion.shape[0] < 2 or confusion.shape[1] < 2:
            return 0.0

        chi2 = chi2_contingency(confusion, correction=False)[0]
        n = confusion.sum().sum()
        phi2 = chi2 / n
        r, k = confusion.shape

        # Bias correction
        phi2_corrected = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        r_corrected = r - ((r - 1) ** 2) / (n - 1)
        k_corrected = k - ((k - 1) ** 2) / (n - 1)

        if min(r_corrected, k_corrected) <= 0:
            return 0.0

        return np.sqrt(phi2_corrected / min(r_corrected - 1, k_corrected - 1))

    def theils_u(
        self,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate Theil's U (Uncertainty Coefficient) for asymmetric categorical relationships.
        U(X|Y) = how well Y predicts X. Range: [0, 1].
        """
        cat_columns = self.df.select_dtypes(include=["object", "category", "bool"]).columns
        if columns:
            cat_columns = [c for c in columns if c in cat_columns]

        if len(cat_columns) < 2:
            return {"association_matrix": {}, "message": "Insufficient categorical columns"}

        cols = list(cat_columns)
        n = len(cols)
        u_matrix = np.zeros((n, n))

        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i != j:
                    u = self._theils_u(self.df[col1], self.df[col2])
                    u_matrix[i, j] = u

        u_df = pd.DataFrame(u_matrix, index=cols, columns=cols)

        return {
            "method": "theils_u",
            "association_matrix": u_df.round(4).to_dict(),
            "columns": cols,
            "note": "Asymmetric: row predicts column. U(row|col) = how well col predicts row."
        }

    def _theils_u(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate Theil's U (Uncertainty Coefficient) U(x|y)."""


        # Joint and marginal distributions
        joint = pd.crosstab(x, y, normalize=True)
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)

        # H(X)
        hx = entropy(px, base=2)
        if hx == 0:
            return 0.0

        # H(X|Y)
        hx_given_y = 0
        for y_val in py.index:
            if py[y_val] > 0:
                conditional = joint[y_val] / py[y_val]
                hx_given_y += py[y_val] * entropy(conditional, base=2)

        return float((hx - hx_given_y) / hx)

    def mutual_information(
        self,
        columns: Optional[List[str]] = None,
        bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate Mutual Information for numeric and categorical columns.
        Detects any statistical dependency (non-linear included).
        """
        if columns:
            work_df = self.df[columns]
        else:
            work_df = self.df

        cols = work_df.columns.tolist()
        if len(cols) < 2:
            return {"mi_matrix": {}, "message": "Insufficient columns"}

        n = len(cols)
        mi_matrix = np.zeros((n, n))

        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i <= j:
                    mi = self._mutual_information(work_df[col1], work_df[col2], bins)
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi

        mi_df = pd.DataFrame(mi_matrix, index=cols, columns=cols)

        return {
            "method": "mutual_information",
            "mi_matrix": mi_df.round(4).to_dict(),
            "columns": cols
        }

    def _mutual_information(self, x: pd.Series, y: pd.Series, bins: int = 10) -> float:
        """Calculate mutual information between two series."""


        x_clean = x.dropna()
        y_clean = y.dropna()

        # Align indices
        common_idx = x_clean.index.intersection(y_clean.index)
        if len(common_idx) < 2:
            return 0.0

        x_vals = x_clean.loc[common_idx]
        y_vals = y_clean.loc[common_idx]

        # Discretize numeric columns
        if pd.api.types.is_numeric_dtype(x_vals):
            x_vals = pd.cut(x_vals, bins=bins, duplicates="drop")
        if pd.api.types.is_numeric_dtype(y_vals):
            y_vals = pd.cut(y_vals, bins=bins, duplicates="drop")

        # Joint distribution
        joint = pd.crosstab(x_vals, y_vals, normalize=True)
        if joint.shape[0] < 2 or joint.shape[1] < 2:
            return 0.0

        px = joint.sum(axis=1)
        py = joint.sum(axis=0)

        hx = entropy(px, base=2)
        hy = entropy(py, base=2)

        # Joint entropy
        hxy = entropy(joint.values.flatten(), base=2)

        return float(hx + hy - hxy)

    # =========================================================
    # MIXED TYPE CORRELATIONS
    # =========================================================

    def correlation_ratio(
        self,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate Correlation Ratio (eta-squared) for numeric vs categorical.
        Measures how much variance in numeric is explained by categorical.
        """
        num_cols = numeric_columns or self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = categorical_columns or self.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if not num_cols or not cat_cols:
            return {"correlation_ratios": {}, "message": "Need both numeric and categorical columns"}

        results = {}
        for num_col in num_cols:
            for cat_col in cat_cols:
                eta2 = self._correlation_ratio(self.df[num_col], self.df[cat_col])
                key = f"{num_col} ~ {cat_col}"
                results[key] = round(eta2, 4)

        return {
            "method": "correlation_ratio (eta-squared)",
            "correlation_ratios": results,
            "interpretation": "Proportion of variance in numeric explained by categorical"
        }

    def _correlation_ratio(self, numeric: pd.Series, categorical: pd.Series) -> float:
        """Calculate eta-squared (correlation ratio)."""
        df_clean = pd.DataFrame({"num": numeric, "cat": categorical}).dropna()
        if len(df_clean) < 2 or df_clean["cat"].nunique() < 2:
            return 0.0

        groups = df_clean.groupby("cat")["num"]
        overall_mean = df_clean["num"].mean()

        ss_between = float(sum(len(g) * (g.mean() - overall_mean) ** 2 for _, g in groups))
        ss_total = float(sum((df_clean["num"] - overall_mean) ** 2))

        if ss_total == 0:
            return 0.0

        return ss_between / ss_total

    # =========================================================
    # PARTIAL & CONDITIONAL CORRELATION
    # =========================================================

    def partial_correlation(
        self,
        target_columns: List[str],
        control_columns: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate partial correlation controlling for confounding variables.
        Uses regression residuals approach.
        """
        all_cols = target_columns + control_columns
        df_clean = self.df[all_cols].dropna()

        if len(df_clean) < len(all_cols) + 1:
            return {"partial_correlations": {}, "message": "Insufficient data after dropping NaN"}

        results = {}
        for i, col1 in enumerate(target_columns):
            for col2 in target_columns[i+1:]:
                # Regress both on control variables
                X_control = df_clean[control_columns]
                if X_control.shape[1] == 0:
                    continue

                # Add constant for statsmodels-like regression
                reg = LinearRegression()

                # Residuals of col1 after controlling
                reg.fit(X_control, df_clean[col1])
                res1 = df_clean[col1] - reg.predict(X_control)

                # Residuals of col2 after controlling
                reg.fit(X_control, df_clean[col2])
                res2 = df_clean[col2] - reg.predict(X_control)

                # Correlation of residuals
                partial_corr = np.corrcoef(res1, res2)[0, 1]
                results[f"{col1} | {col2}"] = round(partial_corr, 4)

        return {
            "method": "partial_correlation",
            "control_variables": control_columns,
            "partial_correlations": results
        }

    def conditional_dependency(
        self,
        x: str,
        y: str,
        z: List[str]
    ) -> Dict[str, Any]:
        """
        Test conditional independence X _||_ Y | Z.
        Uses mutual information: I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
        """
        if x not in self.df.columns or y not in self.df.columns:
            return {"error": "Columns not found"}

        cols = [x, y] + z
        df_clean = self.df[cols].dropna()

        if len(df_clean) < 2:
            return {"error": "Insufficient data"}

        # I(X;Y|Z) ≈ I(X;Y,Z) - I(X;Z)
        mi_xyz = self._mutual_information(df_clean[x], df_clean[[y] + z].apply(tuple, axis=1))
        mi_xz = self._mutual_information(df_clean[x], df_clean[z].apply(tuple, axis=1) if z else df_clean[x])
        mi_xy = self._mutual_information(df_clean[x], df_clean[y])

        cond_mi = mi_xyz - mi_xz

        return {
            "x": x,
            "y": y,
            "z": z,
            "mutual_info_xy": round(mi_xy, 4),
            "mutual_info_xyz": round(mi_xyz, 4),
            "mutual_info_xz": round(mi_xz, 4),
            "conditional_mutual_info": round(max(0, cond_mi), 4),
            "conditionally_independent": cond_mi < 0.01
        }

    # =========================================================
    # CORRELATION NETWORKS & CLUSTERING
    # =========================================================

    def correlation_network(
        self,
        threshold: float = 0.5,
        method: str = "pearson"
    ) -> Dict[str, Any]:
        """
        Build correlation network graph.
        Nodes = columns, Edges = correlations above threshold.
        """
        if method == "pearson":
            corr_result = self.pearson_correlation()
        elif method == "spearman":
            corr_result = self.spearman_correlation()
        elif method == "kendall":
            corr_result = self.kendall_tau()
        elif method == "distance":
            corr_result = self.distance_correlation()
        else:
            return {"error": "Unknown method"}

        if "correlation_matrix" not in corr_result:
            return corr_result

        corr_matrix = pd.DataFrame(corr_result["correlation_matrix"])
        edges = []

        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:
# NEW
                    corr_val = float(cast(Any, corr_matrix.loc[col1, col2]))
                    if abs(corr_val) >= threshold:
                        edges.append({
                            "source": col1,
                            "target": col2,
                            "weight": round(corr_val, 4),
                            "type": "positive" if corr_val > 0 else "negative"
                        })

        # Find connected components (clusters)
        clusters = self._find_clusters(corr_matrix.columns.tolist(), edges)

        return {
            "method": method,
            "threshold": threshold,
            "nodes": corr_matrix.columns.tolist(),
            "edges": edges,
            "clusters": clusters,
            "edge_count": len(edges)
        }

    def _find_clusters(self, nodes: List[str], edges: List[Dict]) -> List[List[str]]:
        """Find connected components in correlation network."""
        from collections import defaultdict

        adj = defaultdict(set)
        for e in edges:
            adj[e["source"]].add(e["target"])
            adj[e["target"]].add(e["source"])

        visited = set()
        clusters = []

        for node in nodes:
            if node not in visited:
                cluster = []
                stack = [node]
                while stack:
                    n = stack.pop()
                    if n not in visited:
                        visited.add(n)
                        cluster.append(n)
                        stack.extend(adj[n] - visited)
                clusters.append(cluster)

        return clusters

    def correlation_clustering(
        self,
        method: str = "pearson",
        n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Hierarchical clustering of correlation matrix.
        Groups highly correlated variables together.
        """
        if method == "pearson":
            corr_result = self.pearson_correlation()
        elif method == "spearman":
            corr_result = self.spearman_correlation()
        else:
            return {"error": "Only pearson/spearman supported for clustering"}

        if "correlation_matrix" not in corr_result:
            return corr_result

        corr_matrix = pd.DataFrame(corr_result["correlation_matrix"])
        cols = corr_matrix.columns.tolist()

        if len(cols) < 2:
            return {
                "method": method,
                "n_clusters": len(cols),
                "clusters": [cols] if cols else [],
                "cluster_labels": {column: 1 for column in cols},
                "message": "At least two columns are required for hierarchical clustering"
            }

        # Convert correlation to distance
        dist_matrix = 1 - np.abs(corr_matrix.values)
        np.fill_diagonal(dist_matrix, 0)

        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        linkage_matrix = linkage(squareform(dist_matrix, checks=False), method="average")

        if n_clusters is None:
            # Auto-determine: cut at distance 0.5 (correlation 0.5)
            cluster_labels = fcluster(linkage_matrix, t=0.5, criterion="distance")
        else:
            cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

        clusters = {}
        for col, label in zip(cols, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(col)

        return {
            "method": method,
            "n_clusters": len(clusters),
            "clusters": list(clusters.values()),
            "cluster_labels": dict(zip(cols, cluster_labels.tolist()))
        }

    # =========================================================
    # HELPER METHODS
    # =========================================================

    def _find_strong_correlations(
        self,
        matrix: pd.DataFrame,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find pairs with correlation above threshold."""
        strong = []
        cols = matrix.columns.tolist()

        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i < j:
                    val = float(cast(Any, matrix.loc[col1, col2]))
                    if abs(val) >= threshold:
                        strong.append({
                            "column_1": col1,
                            "column_2": col2,
                            "correlation": round(val, 4),
                            "strength": "very strong" if abs(val) > 0.9 else "strong"
                        })

        return sorted(strong, key=lambda x: abs(x["correlation"]), reverse=True)

    # =========================================================
    # COMPREHENSIVE PROFILE
    # =========================================================

    def profile(self) -> Dict[str, Any]:
        """Generate complete correlation and dependency profile."""
        return {
            "pearson": self.pearson_correlation(),
            "spearman": self.spearman_correlation(),
            "kendall": self.kendall_tau(),
            "distance_correlation": self.distance_correlation(),
            "cramers_v": self.cramers_v(),
            "theils_u": self.theils_u(),
            "mutual_information": self.mutual_information(),
            "correlation_ratio": self.correlation_ratio(),
            "correlation_network": self.correlation_network(threshold=0.5),
            "correlation_clustering": self.correlation_clustering()
        }


def profile_correlation(df: pd.DataFrame) -> Dict[str, Any]:
    """Convenience function for correlation profiling."""
    profiler = CorrelationProfiler(df)
    return profiler.profile()