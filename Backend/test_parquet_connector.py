from connectors.parquet_connector import ParquetConnector


def main():
    connector = ParquetConnector()

    df = connector.read("test_data/sample.parquet")

    print("\nParquet successfully loaded!\n")

    print("Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData:")
    print(df)


if __name__ == "__main__":
    main()