"""Dashboard API endpoints."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from reports.dashboard_exports import export_dashboard_json
from pydantic import BaseModel, Field

from services.dashboard_service import DashboardService


router = APIRouter(prefix="/dashboard", tags=["dashboard"])


class DashboardRequest(BaseModel):
    data: list[dict[str, Any]]
    date_column: Optional[str] = None
    value_column: Optional[str] = None
    category_column: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)


def _filtered_frame(payload: DashboardRequest) -> pd.DataFrame:
    if not payload.data:
        raise ValueError("Data payload is empty.")
    frame = pd.DataFrame(payload.data)
    for column, value in getattr(payload, "filters", {}).items():
        if column not in frame.columns:
            raise ValueError(f"Filter column not found: {column}")
        values = value if isinstance(value, list) else [value]
        frame = frame[frame[column].isin(values)]
    return frame


@router.post("/")
async def dashboard_data(payload: DashboardRequest) -> Dict[str, Any]:
    """Build a dashboard-ready payload from a dataset."""
    try:
        df = _filtered_frame(payload)
        return DashboardService.build_dashboard(
            df,
            date_column=payload.date_column,
            value_column=payload.value_column,
            category_column=payload.category_column,
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/export/json")
async def export_dashboard(payload: DashboardRequest) -> Response:
    """Download the filtered dashboard payload as JSON."""
    dashboard = await dashboard_data(payload)
    return Response(
        content=export_dashboard_json(dashboard),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=dashboard.json"},
    )
