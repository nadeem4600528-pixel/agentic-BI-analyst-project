import pandas as pd
from pathlib import Path


class CSVConnector:
    """
    Connector responsible for reading CSV files
    and converting them into pandas DataFrames.
    """

    SUPPORTED_EXTENSIONS = {".csv"}

    def read(self, file_path: str) -> pd.DataFrame:
        """
        Read a CSV file and return it as a pandas DataFrame.

        Args:
            file_path: Path to the CSV file.

        Returns:
            pandas.DataFrame containing the CSV data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a CSV file or is empty.
            RuntimeError: If the CSV cannot be read.
        """

        path = Path(file_path)

        # Check whether file exists
        if not path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {file_path}"
            )

        # Check file extension
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Expected: .csv"
            )

        try:
            df = pd.read_csv(path)

        except Exception as e:
            raise RuntimeError(
                f"Failed to read CSV file: {e}"
            ) from e

        # Check whether the file contains data
        if df.empty:
            raise ValueError(
                "The CSV file is empty or contains no readable data."
            )

        return df