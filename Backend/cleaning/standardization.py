"""Text, category, identifier, and PII standardization techniques."""

import re
import unicodedata
from typing import Any, Dict, Mapping, Optional

import pandas as pd


def normalize_text(value: Any, case: Optional[str] = None) -> Any:
	if pd.isna(value):
		return value
	text = unicodedata.normalize("NFKC", str(value))
	text = re.sub(r"\s+", " ", text.strip())
	text = re.sub(r"[^\w\s@.+:/-]", "", text)
	if case == "lower":
		return text.lower()
	if case == "upper":
		return text.upper()
	if case == "title":
		return text.title()
	return text


def standardize_columns(
	df: pd.DataFrame,
	columns: Optional[list[str]] = None,
	case: Optional[str] = None,
	synonyms: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
	result = df.copy(deep=True)
	for column in columns or [str(item) for item in result.columns]:
		if column not in result.columns:
			continue
		result[column] = result[column].map(lambda value: normalize_text(value, case))
		if synonyms:
			result[column] = result[column].replace(dict(synonyms))
	return result


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
	result = df.copy(deep=True)
	names = []
	used: Dict[str, int] = {}
	for column in result.columns:
		name = re.sub(r"[^a-zA-Z0-9]+", "_", str(column).strip()).strip("_").lower() or "column"
		used[name] = used.get(name, 0) + 1
		names.append(name if used[name] == 1 else f"{name}_{used[name]}")
	result.columns = names
	return result


def mask_pii(df: pd.DataFrame, columns: list[str], replacement: str = "[REDACTED]") -> pd.DataFrame:
	result = df.copy(deep=True)
	for column in columns:
		if column in result.columns:
			result[column] = result[column].map(lambda value: replacement if pd.notna(value) else value)
	return result
