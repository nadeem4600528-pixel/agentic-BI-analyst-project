import pandas as pd
from pathlib import Path


class ParquetConnector:
    """
    Connector responsible for reading Parquet files
    and converting them into pandas DataFrames.
    """

    SUPPORTED_EXTENSIONS = {".parquet"}

    def read(self, file_path: str) -> pd.DataFrame:
        """
        Read a Parquet file and return it as a pandas DataFrame.

        Args:
            file_path: Path to the Parquet file.

        Returns:
            pandas.DataFrame containing the Parquet data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is unsupported or contains no data.
            RuntimeError: If the Parquet file cannot be read.
        """

        path = Path(file_path)

        # Check whether file exists
        if not path.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {file_path}"
            )

        # Check file extension
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Expected: .parquet"
            )

        try:
            df = pd.read_parquet(path)

        except Exception as e:
            raise RuntimeError(
                f"Failed to read Parquet file: {e}"
            ) from e

        # Check whether the Parquet file contains data
        if df.empty:
            raise ValueError(
                "The Parquet file is empty or contains no readable data."
            )

        return df