"""Transformation API endpoints."""
from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from transformation.transformation_engine import TransformationService

router = APIRouter(prefix="/transformation", tags=["transformation"])


class TransformationRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., min_length=1)
    operations: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def validate_operations(cls, operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for operation in operations:
            if not isinstance(operation, dict) or not str(operation.get("operation", "")).strip():
                raise ValueError("Each transformation operation requires an operation name.")
        return operations


@router.post("/")
def transform_data(payload: TransformationRequest) -> dict[str, Any]:
    """Apply ordered transformation operations to tabular records."""
    try:
        TransformationRequest.validate_operations(payload.operations)
        result = TransformationService.transform(
            pd.DataFrame(payload.data), payload.operations
        )
        records = result.astype(object).where(pd.notna(result), None).to_dict(orient="records")
        return {
            "data": records,
            "rows": int(len(result)),
            "columns": [str(column) for column in result.columns],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
