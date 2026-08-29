"""Data transformation service."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import pandas as pd

from .aggregation import aggregate
from .datetime import convert_datetime, extract_datetime_parts
from .encoding import map_values, one_hot_encode
from .feature_engineering import create_feature
from .pivot import pivot, unpivot


def transform_dataframe(df: pd.DataFrame, operations: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("transform_dataframe requires a pandas DataFrame.")
    result = df.copy(deep=True)
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise TypeError("Each transformation operation must be an object.")
        name = str(operation.get("operation", "")).strip().lower()
        if not name:
            raise ValueError("Each transformation operation requires an operation name.")
        if name == "rename_columns":
            result = result.rename(columns=dict(operation.get("mapping", {})))
        elif name == "merge_columns":
            columns = list(operation.get("columns", [])); target = operation.get("target", "merged")
            if not columns or any(column not in result.columns for column in columns): raise ValueError("Merge columns are invalid.")
            result[target] = result[columns].fillna("").astype(str).agg(str(operation.get("separator", " ")).join, axis=1)
        elif name == "split_column":
            source = operation["column"]
            targets = list(operation["targets"])
            if source not in result.columns or not targets:
                raise ValueError("Split column and at least one target are required.")
            parts = result[source].astype("string").str.split(operation.get("separator", " "), expand=True)
            for index, target in enumerate(targets): result[target] = parts[index] if index < parts.shape[1] else None
        elif name == "date_conversion":
            result = convert_datetime(result, operation["column"], operation.get("errors", "coerce"), operation.get("format"))
        elif name == "date_component_extraction":
            result = extract_datetime_parts(result, operation["column"], operation["parts"], operation.get("errors", "coerce"), operation.get("format"))
        elif name == "feature_creation":
            result = create_feature(result, operation["target"], str(operation["expression"]))
        elif name == "aggregate":
            result = aggregate(result, operation["group_by"], operation["aggregations"])
        elif name == "pivot":
            result = pivot(result, operation["index"], operation["columns"], operation.get("values"), operation.get("aggfunc", "sum"))
        elif name == "unpivot":
            result = unpivot(result, operation["id_vars"], operation.get("value_vars"), operation.get("var_name", "variable"), operation.get("value_name", "value"))
        elif name == "one_hot_encode":
            result = one_hot_encode(result, operation["columns"], operation.get("drop_first", False))
        elif name == "map_values":
            result = map_values(result, operation["column"], operation["mapping"], operation.get("default"))
        else:
            raise ValueError(f"Unsupported transformation operation: {name}")
    return result


class TransformationService:
    @staticmethod
    def transform(df: pd.DataFrame, operations: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        return transform_dataframe(df, operations)