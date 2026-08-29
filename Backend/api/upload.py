"""Upload endpoints for automatic loading and data profiling."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from services.upload_service import profile_file, profile_sql_server


router = APIRouter(prefix="/upload", tags=["upload"])
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}


class SQLServerProfileRequest(BaseModel):
	server: str
	database: str
	table_name: str | None = None
	query: str | None = None
	driver: str = "ODBC Driver 18 for SQL Server"


@router.post("/profile")
async def upload_and_profile(file: UploadFile = File(...)) -> Dict[str, Any]:
	"""Load an uploaded file and return one complete profiling report."""
	extension = Path(file.filename or "").suffix.lower()
	if extension not in SUPPORTED_EXTENSIONS:
		raise HTTPException(
			status_code=415,
			detail="Supported uploads are CSV, Excel, JSON, and Parquet files.",
		)

	temporary_path = None
	try:
		with NamedTemporaryFile(suffix=extension, delete=False) as temporary_file:
			temporary_path = Path(temporary_file.name)
			while chunk := await file.read(1024 * 1024):
				temporary_file.write(chunk)
		return profile_file(str(temporary_path))
	except (ValueError, RuntimeError, FileNotFoundError) as error:
		raise HTTPException(status_code=400, detail=str(error)) from error
	finally:
		await file.close()
		if temporary_path is not None:
			temporary_path.unlink(missing_ok=True)


@router.post("/profile/sql-server")
def profile_sql_server_data(request: SQLServerProfileRequest) -> Dict[str, Any]:
	"""Read a SQL Server table or SELECT query and return its profile report."""
	try:
		return profile_sql_server(
			server=request.server,
			database=request.database,
			table_name=request.table_name,
			query=request.query,
			driver=request.driver,
		)
	except (ConnectionError, ValueError, RuntimeError) as error:
		raise HTTPException(status_code=400, detail=str(error)) from error
