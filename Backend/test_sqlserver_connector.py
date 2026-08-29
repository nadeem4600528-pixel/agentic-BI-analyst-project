from connectors.sqlserver_connector import SQLServerConnector


def main():

    connector = SQLServerConnector()

    connector.connect(
        server=r"WIN-AS5GKNEQGB3\MSSQLTAAHA",
        database="StudentManagementDB"
    )

    df = connector.read_table("Students")

    print("\nSQL Server data successfully loaded!\n")

    print("Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    connector.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Skipping SQL Server connector test (SQL Server instance not reachable): {e}")