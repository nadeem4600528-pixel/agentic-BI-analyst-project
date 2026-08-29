"""Date and time transformations."""
from __future__ import annotations

from typing import Literal, Optional

import pandas as pd


ErrorsMode = Literal["raise", "coerce"]
_ALLOWED_ERRORS = {"raise", "coerce"}


def convert_datetime(
    df: pd.DataFrame,
    column: str,
    errors: ErrorsMode = "coerce",
    format: Optional[str] = None,
) -> pd.DataFrame:
    """Convert one column to pandas datetime without mutating the input.

    Note: pandas 3.0 removed the errors="ignore" option from pd.to_datetime.
    Only "raise" and "coerce" are supported here. Use "coerce" to turn
    unparseable values into NaT, or "raise" to fail loudly on bad data.
    """
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")
    if errors not in _ALLOWED_ERRORS:
        raise ValueError("errors must be 'raise' or 'coerce'.")
    result = df.copy(deep=True)
    result[column] = pd.to_datetime(result[column], errors=errors, format=format)
    return result


def extract_datetime_parts(
    df: pd.DataFrame,
    column: str,
    parts: list[str],
    errors: ErrorsMode = "coerce",
    format: Optional[str] = None,
) -> pd.DataFrame:
    """Convert a datetime column and add standard date/time component columns."""
    if not parts:
        raise ValueError("At least one datetime part is required.")
    result = convert_datetime(df, column, errors=errors, format=format)
    series = result[column]
    accessors = {
        "year": series.dt.year,
        "quarter": series.dt.quarter,
        "month": series.dt.month,
        "week": series.dt.isocalendar().week,
        "day": series.dt.day,
        "dayofweek": series.dt.dayofweek,
        "day_of_week": series.dt.dayofweek,
        "dayofyear": series.dt.dayofyear,
        "day_of_year": series.dt.dayofyear,
        "hour": series.dt.hour,
        "minute": series.dt.minute,
        "second": series.dt.second,
        "date": series.dt.date,
        "time": series.dt.time,
    }
    for part in parts:
        key = str(part).strip().lower()
        if key not in accessors:
            raise ValueError(f"Unsupported datetime part: {part}")
        result[f"{column}_{key}"] = accessors[key]
    return result