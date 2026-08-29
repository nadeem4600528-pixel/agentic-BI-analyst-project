"""Extended cleaning techniques used by the decision-driven cleaning agent.

Functions are deliberately DataFrame-in/DataFrame-out and never mutate input.
Validation functions return evidence dictionaries. This module does not profile
data and does not duplicate the profiling layer.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def _copy(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True)


def knn_imputation(df: pd.DataFrame, columns: Optional[list[str]] = None, neighbors: int = 5) -> pd.DataFrame:
    from sklearn.impute import KNNImputer
    result = _copy(df)
    selected = columns or result.select_dtypes(include=[np.number]).columns.tolist()
    if selected:
        result[selected] = KNNImputer(n_neighbors=neighbors).fit_transform(result[selected])
    return result


def normalize_empty_strings(df: pd.DataFrame, replacement: Any = np.nan) -> pd.DataFrame:
    result = _copy(df)
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(lambda value: replacement if isinstance(value, str) and not value.strip() else value)
    return result


def normalize_punctuation(value: Any) -> Any:
    if pd.isna(value):
        return value
    return re.sub(r"[^\w\s]", "", unicodedata.normalize("NFKC", str(value)))


def clean_tokens(value: Any) -> Any:
    if pd.isna(value):
        return value
    return " ".join(str(value).split())


def clean_text(df: pd.DataFrame, columns: Optional[list[str]] = None, case: Optional[str] = None) -> pd.DataFrame:
    result = _copy(df)
    for column in columns or result.select_dtypes(include=["object", "string"]).columns.tolist():
        if column not in result.columns:
            continue
        values = result[column].map(clean_tokens).map(normalize_punctuation)
        if case == "lower":
            values = values.str.lower()
        elif case == "upper":
            values = values.str.upper()
        elif case == "title":
            values = values.str.title()
        result[column] = values
    return result


def validate_string_lengths(df: pd.DataFrame, limits: Mapping[str, tuple[int, int]]) -> dict[str, Any]:
    errors = []
    for column, (minimum, maximum) in limits.items():
        if column in df.columns:
            lengths = df[column].astype("string").str.len()
            for index in df.index[(lengths < minimum) | (lengths > maximum)]:
                errors.append({"column": column, "index": str(index)})
    return {"valid": not errors, "errors": errors, "error_count": len(errors)}


def numeric_clean(df: pd.DataFrame, columns: Optional[list[str]] = None, decimals: Optional[int] = None) -> pd.DataFrame:
    result = _copy(df)
    for column in columns or result.columns.tolist():
        if column not in result.columns:
            continue
        values = result[column].astype("string").str.replace(r"[$€£,]", "", regex=True).str.replace("%", "", regex=False)
        result[column] = pd.to_numeric(values, errors="coerce")
        if decimals is not None:
            result[column] = result[column].round(decimals)
    return result


def normalize_units(df: pd.DataFrame, column: str, factors: Mapping[str, float], unit_column: Optional[str] = None) -> pd.DataFrame:
    result = _copy(df)
    if column not in result.columns:
        return result
    if unit_column and unit_column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce") * result[unit_column].map(factors)
    else:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def validate_numeric(df: pd.DataFrame, ranges: Optional[Mapping[str, tuple[float, float]]] = None, non_negative: Optional[Iterable[str]] = None) -> dict[str, Any]:
    errors = []
    for column in non_negative or []:
        if column in df.columns:
            errors.extend({"column": column, "index": str(index), "reason": "negative"} for index in df.index[df[column] < 0])
    for column, (low, high) in (ranges or {}).items():
        if column in df.columns:
            mask = (df[column] < low) | (df[column] > high)
            errors.extend({"column": column, "index": str(index), "reason": "out_of_range"} for index in df.index[mask.fillna(False)])
    return {"valid": not errors, "errors": errors, "error_count": len(errors)}


def parse_dates(df: pd.DataFrame, columns: Optional[list[str]] = None, utc: bool = False) -> pd.DataFrame:
    result = _copy(df)
    for column in columns or result.columns.tolist():
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce", utc=utc)
    return result


def validate_dates(df: pd.DataFrame, columns: list[str], minimum: Any = None, maximum: Any = None, reject_future: bool = False) -> dict[str, Any]:
    errors = []
    now = pd.Timestamp.now(tz="UTC") if reject_future else None
    for column in columns:
        if column not in df.columns:
            continue
        dates = pd.to_datetime(df[column], errors="coerce", utc=reject_future)
        mask = dates.isna()
        if minimum is not None:
            mask |= dates < pd.Timestamp(minimum)
        if maximum is not None:
            mask |= dates > pd.Timestamp(maximum)
        if now is not None:
            mask |= dates > now
        errors.extend({"column": column, "index": str(index)} for index in df.index[mask.fillna(True)])
    return {"valid": not errors, "errors": errors, "error_count": len(errors)}


def extract_date_components(df: pd.DataFrame, column: str) -> pd.DataFrame:
    result = _copy(df)
    dates = pd.to_datetime(result[column], errors="coerce")
    result[f"{column}_year"] = dates.dt.year
    result[f"{column}_month"] = dates.dt.month
    result[f"{column}_day"] = dates.dt.day
    result[f"{column}_weekday"] = dates.dt.dayofweek
    return result


def normalize_booleans(df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
    result = _copy(df)
    true_values = {"true", "t", "yes", "y", "1", "on"}
    false_values = {"false", "f", "no", "n", "0", "off"}
    for column in columns or result.columns.tolist():
        if column in result.columns:
            result[column] = result[column].map(lambda value: True if str(value).strip().casefold() in true_values else False if str(value).strip().casefold() in false_values else pd.NA)
    return result


def normalize_categories(df: pd.DataFrame, column: str, mapping: Optional[Mapping[str, str]] = None, rare_threshold: Optional[float] = None, other: str = "Other") -> pd.DataFrame:
    result = _copy(df)
    if column not in result.columns:
        return result
    values = result[column].astype("string").str.strip().str.casefold()
    if mapping:
        values = values.replace({str(key).casefold(): value for key, value in mapping.items()})
    if rare_threshold is not None and len(values):
        counts = values.value_counts(normalize=True)
        values = values.where(values.map(counts).fillna(0) >= rare_threshold, other)
    result[column] = values
    return result


def validate_categories(df: pd.DataFrame, allowed: Mapping[str, set[Any]]) -> dict[str, Any]:
    errors = []
    for column, values in allowed.items():
        if column in df.columns:
            errors.extend({"column": column, "index": str(index), "value": value} for index, value in df[column].items() if pd.notna(value) and value not in values)
    return {"valid": not errors, "errors": errors, "error_count": len(errors)}


def normalize_contact_fields(df: pd.DataFrame, email_columns: Optional[list[str]] = None, phone_columns: Optional[list[str]] = None) -> pd.DataFrame:
    result = _copy(df)
    for column in email_columns or []:
        if column in result.columns:
            result[column] = result[column].astype("string").str.strip().str.lower()
    for column in phone_columns or []:
        if column in result.columns:
            result[column] = result[column].astype("string").str.replace(r"\D", "", regex=True)
    return result


def validate_contacts(df: pd.DataFrame, email_columns: Optional[list[str]] = None, url_columns: Optional[list[str]] = None) -> dict[str, Any]:
    errors = []
    email = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    url = re.compile(r"^https?://[^\s]+$")
    for column in email_columns or []:
        if column in df.columns:
            errors.extend({"column": column, "index": str(index)} for index, value in df[column].items() if pd.notna(value) and not email.match(str(value)))
    for column in url_columns or []:
        if column in df.columns:
            errors.extend({"column": column, "index": str(index)} for index, value in df[column].items() if pd.notna(value) and not url.match(str(value)))
    return {"valid": not errors, "errors": errors, "error_count": len(errors)}


def mask_sensitive(df: pd.DataFrame, columns: list[str], mode: str = "mask") -> pd.DataFrame:
    result = _copy(df)
    for column in columns:
        if column not in result.columns:
            continue
        if mode == "hash":
            result[column] = result[column].map(lambda value: hashlib.sha256(str(value).encode()).hexdigest() if pd.notna(value) else value)
        else:
            result[column] = result[column].map(lambda value: "[REDACTED]" if pd.notna(value) else value)
    return result


def remove_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return _copy(df).drop(columns=[column for column in columns if column in df.columns], errors="ignore")


def column_report(df: pd.DataFrame) -> dict[str, Any]:
    duplicate_flags = df.columns.duplicated(keep="first")
    return {
        "empty": [str(column) for column in df.columns if df[column].isna().all()],
        "constant": [str(column) for column in df.columns if df[column].nunique(dropna=False) <= 1],
        "duplicate": [str(column) for position, column in enumerate(df.columns) if duplicate_flags[position]],
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
    }


def enforce_schema(df: pd.DataFrame, dtypes: Mapping[str, str]) -> pd.DataFrame:
    from .datatype import convert_types
    return convert_types(df, dtypes)


def row_validation(df: pd.DataFrame, predicate: Callable[[pd.Series], bool]) -> dict[str, Any]:
    mask = ~df.apply(predicate, axis=1)
    return {"valid": not bool(mask.any()), "invalid_indices": [str(index) for index in df.index[mask]], "invalid_count": int(mask.sum())}


def mapping_correction(df: pd.DataFrame, column: str, mapping: Mapping[Any, Any]) -> pd.DataFrame:
    result = _copy(df)
    if column in result.columns:
        result[column] = result[column].replace(dict(mapping))
    return result


def fuzzy_match(values: Sequence[Any], candidates: Sequence[Any], threshold: float = 0.85) -> dict[Any, Any]:
    result = {}
    for value in values:
        match = max(candidates, key=lambda candidate: SequenceMatcher(None, str(value).casefold(), str(candidate).casefold()).ratio(), default=None)
        if match is not None and SequenceMatcher(None, str(value).casefold(), str(match).casefold()).ratio() >= threshold:
            result[value] = match
    return result


def reconcile_frames(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> dict[str, Any]:
    merged = left.merge(right, on=keys, how="outer", indicator=True)
    return {"left_only": int((merged["_merge"] == "left_only").sum()), "right_only": int((merged["_merge"] == "right_only").sum()), "matched": int((merged["_merge"] == "both").sum())}


def temporal_consistency(df: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    starts, ends = pd.to_datetime(df[start], errors="coerce"), pd.to_datetime(df[end], errors="coerce")
    mask = starts > ends
    return {"valid": not bool(mask.any()), "invalid_indices": [str(index) for index in df.index[mask]]}


def flag_records(df: pd.DataFrame, mask: pd.Series, column: str = "__data_quality_flag") -> pd.DataFrame:
    result = _copy(df)
    result[column] = mask.astype("int8")
    return result


def quarantine_records(df: pd.DataFrame, mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _copy(df.loc[~mask]), _copy(df.loc[mask])


def preserve_lineage(df: pd.DataFrame, source: str) -> pd.DataFrame:
    result = _copy(df)
    result["__source"] = source
    result["__source_row"] = result.index
    return result


def before_after_quality(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_score = before.get("quality_score", {}).get("overall_quality_score")
    after_score = after.get("quality_score", {}).get("overall_quality_score")
    return {"before": before_score, "after": after_score, "improved": after_score is not None and before_score is not None and after_score >= before_score}


def execute_technique(name: str, df: pd.DataFrame, **parameters: Any) -> Any:
    """Execute a named cleaning use case through the common technique registry."""
    registry = {
        "knn_imputation": knn_imputation, "empty_string_normalization": normalize_empty_strings,
        "text_token_cleanup": clean_text, "numeric_formatting_normalization": numeric_clean,
        "currency_format_normalization": numeric_clean, "percentage_format_normalization": numeric_clean,
        "unit_normalization": normalize_units, "scale_normalization": numeric_clean,
        "date_parsing": parse_dates, "datetime_conversion": parse_dates,
        "date_component_extraction": extract_date_components, "boolean_normalization": normalize_booleans,
        "category_normalization": normalize_categories, "category_synonym_mapping": normalize_categories,
        "email_phone_normalization": normalize_contact_fields, "pii_anonymization": mask_sensitive,
        "sensitive_column_removal": remove_columns, "schema_normalization": enforce_schema,
        "data_standardization": clean_text, "data_normalization": clean_text,
        "data_harmonization": mapping_correction, "reference_data_mapping": mapping_correction,
        "lookup_based_correction": mapping_correction, "master_data_standardization": normalize_categories,
        "entity_resolution": mapping_correction, "data_reconciliation": reconcile_frames,
        "invalid_record_flagging": flag_records, "suspicious_record_flagging": flag_records,
        "quarantine_problematic_records": quarantine_records,
    }
    if name not in registry:
        raise ValueError(f"Unknown cleaning technique: {name}")
    return registry[name](df, **parameters)