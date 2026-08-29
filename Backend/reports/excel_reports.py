"""Excel report export helpers."""
from io import BytesIO
from typing import Any, Mapping
import pandas as pd


def export_excel(report: Mapping[str, Any]) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, value in report.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                pd.DataFrame(value).to_excel(writer, sheet_name=str(name)[:31], index=False)
            elif isinstance(value, dict):
                pd.DataFrame([value]).to_excel(writer, sheet_name=str(name)[:31], index=False)
    output.seek(0)
    return output

