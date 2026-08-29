from connectors.json_connector import JSONConnector


def main():
    connector = JSONConnector()

    df = connector.read("test_data/sample.json")

    print("\nJSON successfully loaded!\n")

    print("Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData:")
    print(df)


if __name__ == "__main__":
    main()