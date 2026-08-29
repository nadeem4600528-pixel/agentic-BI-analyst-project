"""KPI generation engine for datasets."""

from typing import Any, Dict, List
import numpy as np
import pandas as pd


def generate_kpis(df: pd.DataFrame) -> List[Dict[str, Any]]:
	"""Generate automated business KPIs from numeric and date columns."""
	kpis: List[Dict[str, Any]] = []
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

	for col in numeric_cols:
		series = df[col].dropna()
		if series.empty:
			continue
		
		total = float(series.sum())
		mean_val = float(series.mean())
		max_val = float(series.max())

		kpis.append({
			"metric": f"Total {col}",
			"value": round(total, 4),
			"type": "sum",
			"column": col,
			"description": f"Sum total across {len(series)} records."
		})
		kpis.append({
			"metric": f"Average {col}",
			"value": round(mean_val, 4),
			"type": "average",
			"column": col,
			"description": f"Mean value for {col}."
		})
		kpis.append({
			"metric": f"Max {col}",
			"value": round(max_val, 4),
			"type": "maximum",
			"column": col,
			"description": f"Peak value for {col}."
		})

	kpis.insert(0, {
		"metric": "Total Records",
		"value": int(len(df)),
		"type": "count",
		"column": None,
		"description": "Total number of rows in the dataset."
	})

	return kpis

