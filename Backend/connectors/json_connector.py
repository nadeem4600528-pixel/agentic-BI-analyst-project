import pandas as pd
from pathlib import Path


class JSONConnector:
    """
    Connector responsible for reading JSON files
    and converting them into pandas DataFrames.
    """

    SUPPORTED_EXTENSIONS = {".json"}

    def read(self, file_path: str) -> pd.DataFrame:
        """
        Read a JSON file and return it as a pandas DataFrame.

        Args:
            file_path: Path to the JSON file.

        Returns:
            pandas.DataFrame containing the JSON data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is unsupported or contains no data.
            RuntimeError: If the JSON file cannot be read.
        """

        path = Path(file_path)

        # Check whether file exists
        if not path.exists():
            raise FileNotFoundError(
                f"JSON file not found: {file_path}"
            )

        # Check file extension
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Expected: .json"
            )

        try:
            df = pd.read_json(path)

        except ValueError as e:
            raise RuntimeError(
                f"Invalid JSON format: {e}"
            ) from e

        except Exception as e:
            raise RuntimeError(
                f"Failed to read JSON file: {e}"
            ) from e

        # Check whether the JSON contains data
        if df.empty:
            raise ValueError(
                "The JSON file is empty or contains no readable data."
            )

        return df