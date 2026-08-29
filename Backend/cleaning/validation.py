"""Validation, constraints, keys, contact fields, and quarantine techniques."""

import re
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import pandas as pd


def validate_dataframe(
	df: pd.DataFrame,
	rules: Optional[Mapping[str, Callable[[Any], bool]]] = None,
	ranges: Optional[Mapping[str, tuple[float, float]]] = None,
	required: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
	errors = []
	for column in required or []:
		if column not in df.columns:
			errors.append({"type": "missing_column", "column": column})
	for column, rule in (rules or {}).items():
		if column in df.columns:
			invalid = ~df[column].map(rule).fillna(False)
			errors.extend({"type": "rule_violation", "column": column, "index": str(index)} for index in df.index[invalid])
	for column, bounds in (ranges or {}).items():
		if column in df.columns:
			invalid = (df[column] < bounds[0]) | (df[column] > bounds[1])
			errors.extend({"type": "range_violation", "column": column, "index": str(index)} for index in df.index[invalid.fillna(False)])
	return {"valid": not errors, "error_count": len(errors), "errors": errors}


def validate_common_formats(df: pd.DataFrame) -> Dict[str, Any]:
	result = {}
	email_pattern = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
	for column in df.columns:
		name = str(column).lower()
		if "email" in name:
			result[str(column)] = {"invalid_count": int((~df[column].astype("string").str.match(email_pattern, na=False)).sum())}
	return result


def quarantine(df: pd.DataFrame, invalid_mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
	return df.loc[~invalid_mask].copy(deep=True), df.loc[invalid_mask].copy(deep=True)
