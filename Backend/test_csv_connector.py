from connectors.csv_connector import CSVConnector


def main():
    connector = CSVConnector()

    df = connector.read("test_data/sample.csv")

    print("\nCSV successfully loaded!\n")

    print("Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData:")
    print(df)


if __name__ == "__main__":
    main()