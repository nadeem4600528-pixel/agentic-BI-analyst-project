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

    @staticmethod
    def _quote_table_name(table_name: str) -> str:
        """
        Safely quote a (possibly schema-qualified) table name for interpolation
        into a SQL statement, e.g. 'dbo.Orders' -> '[dbo].[Orders]'.
        """
        table_name = table_name.strip()
        parts = table_name.split(".")

        quoted_parts = []
        for part in parts:
            part = part.strip()
            if part.startswith("[") and part.endswith("]"):
                part = part[1:-1]
            # Escape any closing brackets already in the identifier.
            part = part.replace("]", "]]")
            quoted_parts.append(f"[{part}]")

        return ".".join(quoted_parts)

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
        driver: str = "ODBC Driver 18 for SQL Server",
        username: str | None = None,
        password: str | None = None,
    ):
        """
        Connect to SQL Server using Windows Authentication (or SQL auth if
        username/password are supplied).

        Args:
            server: SQL Server instance name.
            database: Database name.
            driver: Installed SQL Server ODBC driver.
            username: Optional SQL Server username for SQL authentication.
            password: Optional SQL Server password for SQL authentication.

        Returns:
            Active pyodbc connection.
        """

        if not server or not server.strip():
            raise ValueError("Server name cannot be empty.")

        if not database or not database.strip():
            raise ValueError("Database name cannot be empty.")

        self._require_pyodbc()
        assert pyodbc is not None  # narrows the type after the runtime check above

        driver = driver.strip() if driver else "ODBC Driver 18 for SQL Server"
        server = server.strip()
        database = database.strip()

        connection_string = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            "TrustServerCertificate=yes;"
        )

        if username and password:
            connection_string += f"UID={username};PWD={password};"
        else:
            connection_string += "Trusted_Connection=yes;"

        try:
            connection = pyodbc.connect(connection_string)
            self.connection = connection
            return self.connection
        except Exception as e:  # pyodbc can be absent or connection can fail.
            raise ConnectionError(
                f"Failed to connect to SQL Server: {e}"
            ) from e

    def read_table(self, table_name: str) -> pd.DataFrame:
        """
        Read a complete SQL Server table into a DataFrame.
        """

        if self.connection is None:
            raise ConnectionError("No active SQL Server connection.")

        if not table_name or not table_name.strip():
            raise ValueError("Table name cannot be empty.")

        try:
            safe_table_name = self._quote_table_name(table_name)
            query = f"SELECT * FROM {safe_table_name}"
            df = pd.read_sql(query, self.connection)
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
            raise ConnectionError("No active SQL Server connection.")

        if not query or not query.strip():
            raise ValueError("SQL query cannot be empty.")

        normalized_query = query.strip()
        if not normalized_query.lower().lstrip().startswith(("select", "with")):
            raise ValueError("Only SELECT queries are allowed.")

        try:
            df = pd.read_sql(normalized_query, self.connection)
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