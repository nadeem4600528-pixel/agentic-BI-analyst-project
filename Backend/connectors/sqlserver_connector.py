import pandas as pd

try:  # pyodbc needs the unixODBC/ODBC Driver system libraries at import time.
    import pyodbc  # type: ignore
except Exception:  # pragma: no cover - depends on host ODBC runtime
    pyodbc = None  # type: ignore


class SQLServerConnector:
    """
    Connector for Microsoft SQL Server using Windows Authentication.
    """

    def __init__(self):
        self.connection = None

    @staticmethod
    def _require_pyodbc():
        if pyodbc is None:
            raise ConnectionError(
                "SQL Server support is unavailable: the 'pyodbc' package or the "
                "ODBC Driver for SQL Server is not installed/configured on this host. "
                "File-based sources (CSV, Excel, JSON, Parquet) work without it."
            )

    def read(self, source: str) -> pd.DataFrame:
        """Compatibility method for the generic connector contract.

        For SQL Server, `source` is interpreted as either a table name or a SELECT query.
        """
        if not source:
            raise ValueError("Source cannot be empty.")

        source_text = source.strip()
        if source_text.lower().startswith("select"):
            return self.read_query(source_text)
        return self.read_table(source_text)

    def connect(
        self,
        server: str,
        database: str,
        driver: str = "ODBC Driver 18 for SQL Server"
    ):
        """
        Connect to SQL Server using Windows Authentication.

        Args:
            server: SQL Server instance name.
            database: Database name.
            driver: Installed SQL Server ODBC driver.

        Returns:
            Active pyodbc connection.
        """

        if not server:
            raise ValueError("Server name cannot be empty.")

        if not database:
            raise ValueError("Database name cannot be empty.")

        self._require_pyodbc()

        connection_string = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
        )

        try:
            self.connection = pyodbc.connect(connection_string)

            print("SQL Server connection successful!")

            return self.connection

        except pyodbc.Error as e:
            raise ConnectionError(
                f"Failed to connect to SQL Server: {e}"
            ) from e

    def read_table(self, table_name: str) -> pd.DataFrame:
        """
        Read a complete SQL Server table into a DataFrame.
        """

        if self.connection is None:
            raise ConnectionError(
                "No active SQL Server connection."
            )

        if not table_name:
            raise ValueError(
                "Table name cannot be empty."
            )

        try:
            query = f"SELECT * FROM [{table_name}]"

            df = pd.read_sql(
                query,
                self.connection
            )

            if df.empty:
                raise ValueError(
                    f"The table '{table_name}' contains no data."
                )

            return df

        except Exception as e:
            raise RuntimeError(
                f"Failed to read table '{table_name}': {e}"
            ) from e

    def read_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SELECT query and return the result as DataFrame.
        """

        if self.connection is None:
            raise ConnectionError(
                "No active SQL Server connection."
            )

        if not query:
            raise ValueError(
                "SQL query cannot be empty."
            )

        # Basic safety guard
        if not query.strip().lower().startswith("select"):
            raise ValueError(
                "Only SELECT queries are allowed."
            )

        try:
            df = pd.read_sql(
                query,
                self.connection
            )

            if df.empty:
                raise ValueError(
                    "The query returned no data."
                )

            return df

        except Exception as e:
            raise RuntimeError(
                f"Failed to execute SQL query: {e}"
            ) from e

    def close(self):
        """
        Close the SQL Server connection.
        """

        if self.connection is not None:

            self.connection.close()
            self.connection = None

            print("SQL Server connection closed.")