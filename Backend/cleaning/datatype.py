"""Data type conversion and schema normalization techniques."""

from typing import Any, Dict, Mapping

import pandas as pd


def datatype_report(
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generate a report of the current data types.
    """

    return {
        "dtypes": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        }
    }


def convert_types(
    df: pd.DataFrame,
    conversions: Mapping[str, str]
) -> pd.DataFrame:
    """
    Convert DataFrame columns to requested data types.

    Supported semantic targets:
        datetime
        date
        numeric
        number
        string
        text
        bool
        boolean

    Any other target is treated as a pandas dtype and
    converted using astype().
    """

    result = df.copy(deep=True)

    for column, target in conversions.items():

        # -----------------------------------------------------
        # Skip columns that do not exist
        # -----------------------------------------------------

        if column not in result.columns:
            continue

        target_normalized = str(target).strip().lower()

        # -----------------------------------------------------
        # Datetime
        # -----------------------------------------------------

        if target_normalized in {"datetime", "date"}:

            result[column] = pd.to_datetime(
                result[column],
                errors="coerce"
            )

        # -----------------------------------------------------
        # Numeric
        # -----------------------------------------------------

        elif target_normalized in {
            "numeric",
            "number"
        }:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            )

        # -----------------------------------------------------
        # String
        # -----------------------------------------------------

        elif target_normalized in {
            "string",
            "text"
        }:

            result[column] = result[column].astype(
                "string"
            )

        # -----------------------------------------------------
        # Boolean
        # -----------------------------------------------------

        elif target_normalized in {
            "bool",
            "boolean"
        }:

            boolean_mapping = {
                True: True,
                False: False,
                "true": True,
                "false": False,
                "True": True,
                "False": False,
                "TRUE": True,
                "FALSE": False,
                "yes": True,
                "no": False,
                "Yes": True,
                "No": False,
                "YES": True,
                "NO": False,
                "1": True,
                "0": False,
                1: True,
                0: False,
            }

            result[column] = (
                result[column]
                .map(boolean_mapping)
                .astype("boolean")
            )

        # -----------------------------------------------------
        # Generic pandas dtype
        # -----------------------------------------------------

        else:

            try:
                result[column] = result[column].astype(
                    target  # type: ignore[arg-type]
                )

            except (TypeError, ValueError) as exc:

                raise ValueError(
                    f"Unable to convert column "
                    f"'{column}' to dtype '{target}'."
                ) from exc

    return result