from pathlib import Path

from connectors.csv_connector import CSVConnector
from connectors.excel_connector import ExcelConnector
from connectors.json_connector import JSONConnector
from connectors.parquet_connector import ParquetConnector
from connectors.sqlserver_connector import SQLServerConnector


class ConnectorFactory:
    """
    Factory responsible for selecting the appropriate
    data connector based on the data source type.
    """

    FILE_CONNECTORS = {
        ".csv": CSVConnector,
        ".xlsx": ExcelConnector,
        ".xls": ExcelConnector,
        ".json": JSONConnector,
        ".parquet": ParquetConnector,
    }

    DATABASE_CONNECTORS = {
        "sqlserver": SQLServerConnector,
        "mssql": SQLServerConnector,
    }

    @staticmethod
    def get_connector(source: str):
        """
        Return the appropriate connector for a file or database.

        Examples:
            ConnectorFactory.get_connector("sample.csv")
            ConnectorFactory.get_connector("sample.xlsx")
            ConnectorFactory.get_connector("sqlserver")
        """

        if not source:
            raise ValueError("Data source cannot be empty.")

        source = str(source).strip()

        # Check whether source is a database type
        database_type = source.lower()

        if database_type in ConnectorFactory.DATABASE_CONNECTORS:
            connector_class = ConnectorFactory.DATABASE_CONNECTORS[
                database_type
            ]

            return connector_class()

        # Check whether source is a file
        extension = Path(source).suffix.lower()

        if extension in ConnectorFactory.FILE_CONNECTORS:
            connector_class = ConnectorFactory.FILE_CONNECTORS[
                extension
            ]

            return connector_class()

        raise ValueError(
            f"Unsupported data source: {source}"
        )