from connectors.connector_factory import ConnectorFactory


def _test_file_connector(file_path):
    connector = ConnectorFactory.get_connector(file_path)

    print(
        f"{file_path} -> "
        f"{connector.__class__.__name__}"
    )


def _test_database_connector(database_type):
    connector = ConnectorFactory.get_connector(database_type)

    print(
        f"{database_type} -> "
        f"{connector.__class__.__name__}"
    )


def main():

    print("\n--- File Connector Tests ---\n")

    _test_file_connector("test_data/sample.csv")

    _test_file_connector("test_data/sample.xlsx")

    _test_file_connector("test_data/sample.json")

    _test_file_connector("test_data/sample.parquet")

    print("\n--- Database Connector Tests ---\n")

    _test_database_connector("sqlserver")

    print("\nConnector Factory test completed!")


if __name__ == "__main__":
    main()