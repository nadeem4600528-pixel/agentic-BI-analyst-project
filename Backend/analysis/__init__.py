"""Analysis / Insight Agent module."""

from .analysis_engine import AnalysisAgent, analyze_dataset
from .correlation import analyze_correlation
from .kpi import generate_kpis
from .semantic import analyze_segmentation, analyze_semantics
from .statistics import analyze_statistics
from .trends import analyze_trends

__all__ = [
    "AnalysisAgent",
    "analyze_dataset",
    "generate_kpis",
    "analyze_statistics",
    "analyze_correlation",
    "analyze_semantics",
    "analyze_segmentation",
    "analyze_trends",
]
