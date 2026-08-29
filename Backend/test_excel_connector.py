from connectors.excel_connector import ExcelConnector


def main():
    connector = ExcelConnector()

    df = connector.read("test_data/sample.xlsx")

    print("\nExcel successfully loaded!\n")

    print("Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData:")
    print(df)


if __name__ == "__main__":
    main()