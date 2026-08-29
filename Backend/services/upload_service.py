"""Load uploaded data sources and run the complete profiling pipeline."""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from connectors.connector_factory import ConnectorFactory


def _json_safe(value: Any) -> Any:
	"""Convert pandas/NumPy values in profiling output to JSON-safe values."""
	if isinstance(value, dict):
		return {str(key): _json_safe(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_safe(item) for item in value]
	if isinstance(value, np.ndarray):
		return [_json_safe(item) for item in value.tolist()]
	if isinstance(value, np.generic):
		return _json_safe(value.item())
	if isinstance(value, (pd.Timestamp, pd.Timedelta)):
		return value.isoformat()
	return value
from connectors.sqlserver_connector import SQLServerConnector
from profiling.profiler import DataProfiler


def load_dataframe(file_path: str) -> pd.DataFrame:
	"""Load a supported file into a DataFrame through the connector factory."""
	path = Path(file_path)
	if not path.exists() or not path.is_file():
		raise FileNotFoundError(f"File not found: {file_path}")
	connector = ConnectorFactory.get_connector(str(path))
	if isinstance(connector, SQLServerConnector):
		raise ValueError("SQL Server connector cannot be used to read a file path.")
	return connector.read(str(path))


def profile_file(
	file_path: str,
	profiling_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""Load a supported file and return its complete profiling report."""
	path = Path(file_path)
	if not path.exists() or not path.is_file():
		raise FileNotFoundError(f"File not found: {file_path}")
	dataframe = load_dataframe(str(path))
	# Upload must return promptly so the UI can load the dataset. The full
	# profiling suite is intentionally reserved for the profiling/report views.
	options = profiling_options if profiling_options is not None else [
		"metadata", "structure", "schema", "statistics", "completeness", "quality_score"
	]
	table_name = path.stem or "uploaded_data"

	profiler = DataProfiler(dataframe, tables={table_name: dataframe})
	if isinstance(options, list):
		report = profiler.profile(include_modules=options)
	else:
		report = profiler.profile(**options)
	report["report_metadata"]["source"] = {
		"filename": path.name,
		"extension": path.suffix.lower(),
		"connector": type(ConnectorFactory.get_connector(str(path))).__name__,
	}
	# Return the loaded rows so the frontend can use this dataset for
	# dashboards, analysis, transformations, and report exports.
	report["data"] = dataframe.astype(object).where(pd.notna(dataframe), None).to_dict(orient="records")
	report["rows"] = int(len(dataframe))
	report["columns"] = [str(column) for column in dataframe.columns]
	return _json_safe(report)


def profile_sql_server(
	server: str,
	database: str,
	table_name: Optional[str] = None,
	query: Optional[str] = None,
	driver: str = "ODBC Driver 18 for SQL Server",
	profiling_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""Read a SQL Server table or SELECT query and return its profile report."""
	if bool(table_name) == bool(query):
		raise ValueError("Provide exactly one of table_name or query.")

	connector = ConnectorFactory.get_connector("sqlserver")
	if not isinstance(connector, SQLServerConnector):
		raise TypeError("Expected SQL Server connector instance.")
	connector.connect(server=server, database=database, driver=driver)
	try:
		if table_name:
			dataframe = connector.read_table(table_name)
		else:
			assert query is not None  # guaranteed by the XOR check above
			dataframe = connector.read_query(query)
		options = profiling_options or {}
		source_name = table_name or "sql_query"
		report = DataProfiler(dataframe, tables={source_name: dataframe}).profile(**options)
		report["report_metadata"]["source"] = {
			"connector": type(connector).__name__,
			"server": server,
			"database": database,
			"table": table_name,
			"query": query,
		}
		return report
	finally:
		connector.close()
