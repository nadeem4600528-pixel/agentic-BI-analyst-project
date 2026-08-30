"""Demo dataset endpoint — lets the UI showcase the dashboard without an upload."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter

from services.upload_service import _json_safe
from services.dataset_store import put as store_dataset

router = APIRouter(prefix="/demo", tags=["demo"])

_SAMPLE_PATH = Path(__file__).resolve().parents[1] / "test_data" / "sample_sales.csv"


@router.get("/dataset")
def demo_dataset() -> Dict[str, Any]:
    """Return the bundled sample sales dataset for instant dashboard demos."""
    if not _SAMPLE_PATH.exists():
        return {"data": [], "rows": 0, "columns": [], "message": "Demo dataset not found."}

    df = pd.read_csv(_SAMPLE_PATH)
    records = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    payload = {
        "data": records,
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "suggested": {
            "date_column": next((c for c in df.columns if "date" in c.lower()), None),
            "value_column": next((c for c in df.columns if c.lower() in ("sales", "revenue", "profit")), "sales") if "sales" in df.columns else None,
            "category_column": next((c for c in ("region", "category", "segment") if c in df.columns), None),
        },
        "filename": "sample_sales.csv",
        "dataset_id": store_dataset(df),
    }
    return _json_safe(payload)
