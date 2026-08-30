"""Report-generation API endpoints."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from reports.excel_reports import export_excel
from reports.pdf_reports import export_html, export_pdf
from pydantic import BaseModel

from services.report_service import ReportService
from services import dataset_store


router = APIRouter(prefix="/report", tags=["report"])


class ReportRequest(BaseModel):
    data: list[dict[str, Any]]
    date_column: Optional[str] = None
    value_column: Optional[str] = None
    category_column: Optional[str] = None

class ReportByIdRequest(BaseModel):
    dataset_id: str
    kind: str = "full"  # "profiling" | "cleaning" | "analysis" | "full"
    date_column: Optional[str] = None
    value_column: Optional[str] = None
    category_column: Optional[str] = None

@router.post("/")
async def generate_report(payload: ReportRequest) -> Dict[str, Any]:
    """Generate a full report with profiling, cleaning, analysis, and business recommendations."""
    try:
        if not payload.data:
            raise HTTPException(status_code=400, detail="Data payload is empty.")

        df = pd.DataFrame(payload.data)
        return ReportService.generate_full_report(
            df,
            date_column=payload.date_column,
            value_column=payload.value_column,
            category_column=payload.category_column,
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/profiling")
async def profiling_report(payload: ReportRequest) -> Dict[str, Any]:
    """Return profiling-only results."""
    try:
        if not payload.data:
            raise HTTPException(status_code=400, detail="Data payload is empty.")
        return ReportService.generate_profiling_report(pd.DataFrame(payload.data))
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/cleaning")
async def cleaning_report(payload: ReportRequest) -> Dict[str, Any]:
    """Return cleaning recommendation insights."""
    try:
        if not payload.data:
            raise HTTPException(status_code=400, detail="Data payload is empty.")
        return ReportService.generate_cleaning_report(pd.DataFrame(payload.data))
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/analysis")
async def analysis_report(payload: ReportRequest) -> Dict[str, Any]:
    """Return analysis insight results."""
    try:
        if not payload.data:
            raise HTTPException(status_code=400, detail="Data payload is empty.")
        return ReportService.generate_analysis_report(
            pd.DataFrame(payload.data),
            date_column=payload.date_column,
            value_column=payload.value_column,
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error



def _full_report(payload: ReportRequest) -> Dict[str, Any]:
    if not payload.data:
        raise ValueError("Data payload is empty.")
    return ReportService.generate_full_report(
        pd.DataFrame(payload.data),
        date_column=payload.date_column,
        value_column=payload.value_column,
        category_column=payload.category_column,
    )

@router.post("/by-id")
async def report_by_id(payload: ReportByIdRequest) -> Dict[str, Any]:
    """Generate a single report section from a dataset already stored
    server-side, so the frontend never re-uploads the full dataset."""
    df = dataset_store.get(payload.dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found or expired. Please re-upload.")
    try:
        if payload.kind == "profiling":
            return ReportService.generate_profiling_report(df)
        if payload.kind == "cleaning":
            return ReportService.generate_cleaning_report(df)
        if payload.kind == "analysis":
            return ReportService.generate_analysis_report(
                df, date_column=payload.date_column, value_column=payload.value_column
            )
        return ReportService.generate_full_report(
            df,
            date_column=payload.date_column,
            value_column=payload.value_column,
            category_column=payload.category_column,
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/export/excel")
async def export_report_excel(payload: ReportRequest) -> Response:
    """Download the consolidated report as an Excel workbook."""
    try:
        workbook = export_excel(_full_report(payload))
        return Response(
            content=workbook.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=report.xlsx"},
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Report export failed: {error}") from error


@router.post("/export/html")
async def export_report_html(payload: ReportRequest) -> Response:
    """Download the consolidated report as printable HTML."""
    try:
        return Response(
            content=export_html(_full_report(payload)),
            media_type="text/html",
            headers={"Content-Disposition": "attachment; filename=report.html"},
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Report export failed: {error}") from error


@router.post("/export/pdf")
async def export_report_pdf(payload: ReportRequest) -> Response:
    """Download the consolidated report as a native PDF document."""
    try:
        return Response(
            content=export_pdf(_full_report(payload)),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=report.pdf"},
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Report export failed: {error}") from error
