"""Workflow API routes orchestrating the full analytics pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.workflow_service import WorkflowService


router = APIRouter(prefix="/workflow", tags=["workflow"])


class WorkflowRequest(BaseModel):
    data: list[dict[str, Any]]
    date_column: Optional[str] = None
    value_column: Optional[str] = None
    category_column: Optional[str] = None


@router.post("/run")
async def run_workflow(payload: WorkflowRequest) -> Dict[str, Any]:
    """Execute the full upload -> profile -> clean -> analyze -> visualize pipeline."""
    try:
        if not payload.data:
            raise HTTPException(status_code=400, detail="Data payload is empty.")
        job = WorkflowService.run_pipeline(
            records=payload.data,
            date_column=payload.date_column,
            value_column=payload.value_column,
            category_column=payload.category_column,
        )
        return job
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/jobs")
async def list_workflows() -> List[Dict[str, Any]]:
    """List all in-memory workflow jobs."""
    return WorkflowService.list_statuses()


@router.get("/jobs/{job_id}")
async def get_workflow(job_id: str) -> Dict[str, Any]:
    """Get status of a workflow execution."""
    try:
        return WorkflowService.get_status(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error