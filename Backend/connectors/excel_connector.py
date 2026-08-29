import pandas as pd
from pathlib import Path


class ExcelConnector:
    """
    Connector responsible for reading Excel files
    and converting them into pandas DataFrames.
    """

    SUPPORTED_EXTENSIONS = {".xlsx", ".xls"}

    def read(self, file_path: str, sheet_name=0) -> pd.DataFrame:
        """
        Read an Excel file and return it as a pandas DataFrame.

        Args:
            file_path: Path to the Excel file.
            sheet_name: Excel sheet to read.
                        Default is the first sheet (0).

        Returns:
            pandas.DataFrame containing the Excel data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is unsupported or contains no data.
            RuntimeError: If the Excel file cannot be read.
        """

        path = Path(file_path)

        # Check whether file exists
        if not path.exists():
            raise FileNotFoundError(
                f"Excel file not found: {file_path}"
            )

        # Check file extension
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Expected: .xlsx or .xls"
            )

        try:
            df = pd.read_excel(
                path,
                sheet_name=sheet_name
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to read Excel file: {e}"
            ) from e

        # Check whether the sheet contains data
        if df.empty:
            raise ValueError(
                "The Excel sheet is empty or contains no readable data."
            )

        return df