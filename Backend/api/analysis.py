"""Analysis API endpoints."""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
import pandas as pd

from services.analysis_service import AnalysisService


router = APIRouter(prefix="/analysis", tags=["analysis"])


class AnalysisRequest(BaseModel):
	data: list[dict[str, Any]]
	date_column: Optional[str] = None
	value_column: Optional[str] = None


@router.post("/")
async def analyze_data(payload: AnalysisRequest) -> Dict[str, Any]:
	"""Run comprehensive analysis on provided dataset records."""
	try:
		if not payload.data:
			raise HTTPException(status_code=400, detail="Data payload is empty.")
		df = pd.DataFrame(payload.data)
		return AnalysisService.analyze_dataframe(df, date_column=payload.date_column, value_column=payload.value_column)
	except Exception as error:
		raise HTTPException(status_code=400, detail=str(error)) from error

