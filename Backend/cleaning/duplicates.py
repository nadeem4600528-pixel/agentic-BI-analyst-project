"""Exact, key-based, composite, and fuzzy duplicate techniques."""

from typing import Any, Dict, Optional, Literal
from difflib import SequenceMatcher

import pandas as pd


DropKeep = Literal["first", "last", False]


def duplicate_report(
    df: pd.DataFrame,
    keys: Optional[list[str]] = None
) -> Dict[str, Any]:
    """
    Generate a duplicate-data report.

    Supports:
    - Exact duplicate detection
    - Key-based duplicate detection
    - Composite-key duplicate detection
    """

    mask = df.duplicated(
        subset=keys,
        keep=False
    )

    if mask.any():
        group_columns = keys or list(df.columns)

        groups = int(
            df.loc[mask]
            .groupby(
                group_columns,
                dropna=False
            )
            .ngroups
        )
    else:
        groups = 0

    return {
        "duplicate_count": int(mask.sum()),
        "duplicate_percentage": (
            round(float(mask.mean() * 100), 4)
            if len(df)
            else 0.0
        ),
        "keys": keys,
        "groups": groups,
    }


def remove_duplicates(
    df: pd.DataFrame,
    keys: Optional[list[str]] = None,
    keep: DropKeep = "first"
) -> pd.DataFrame:
    """
    Remove exact/key-based duplicates.

    keep:
        "first" -> keep first occurrence
        "last"  -> keep last occurrence
        False   -> remove all duplicated records
    """

    return (
        df.drop_duplicates(
            subset=keys,
            keep=keep
        )
        .reset_index(drop=True)
    )


def fuzzy_duplicate_candidates(
    df: pd.DataFrame,
    columns: list[str],
    threshold: float = 0.9
) -> list[Dict[str, Any]]:
    """
    Find potential fuzzy duplicates using
    string similarity.

    Returns candidate row pairs whose similarity
    is greater than or equal to the threshold.
    """

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if not columns:
        raise ValueError(
            "At least one column is required."
        )

    missing_columns = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Columns not found in DataFrame: "
            f"{missing_columns}"
        )

    candidates: list[Dict[str, Any]] = []

    values = (
        df[columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .tolist()
    )

    for left in range(len(values)):

        for right in range(left + 1, len(values)):

            score = SequenceMatcher(
                None,
                values[left].casefold(),
                values[right].casefold()
            ).ratio()

            if score >= threshold:

                candidates.append(
                    {
                        "left_index": left,
                        "right_index": right,
                        "similarity": round(
                            score,
                            4
                        )
                    }
                )

    return candidates