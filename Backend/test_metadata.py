"""
test_metadata.py

Test file for the Metadata Profiling module
of Agentic BI Analyst.
"""

import pandas as pd

from profiling.metadata import MetadataProfiler


def main():
    # ---------------------------------------------------------
    # CREATE SAMPLE DATA
    # ---------------------------------------------------------

    data = {
        "CustomerID": [101, 102, 103, 104],
        "Name": ["Ali", "Sara", "Ahmed", "Fatima"],
        "Age": [25, 28, 31, 26],
        "Department": ["IT", "HR", "Finance", "IT"],
        "Salary": [75000, 65000, 85000, 72000],
        "IsActive": [True, True, False, True],
        "JoinDate": pd.to_datetime([
            "2024-01-10",
            "2024-02-15",
            "2024-03-20",
            "2024-04-05"
        ])
    }

    df = pd.DataFrame(data)

    print("=" * 60)
    print("DATA PROFILING - METADATA TEST")
    print("=" * 60)

    print("\nInput Data:")
    print(df)

    # ---------------------------------------------------------
    # CREATE PROFILER
    # ---------------------------------------------------------

    profiler = MetadataProfiler(df)

    # ---------------------------------------------------------
    # DATASET OVERVIEW
    # ---------------------------------------------------------

    overview = profiler.get_dataset_overview()

    print("\nDataset Overview:")
    print(overview)

    # ---------------------------------------------------------
    # COLUMN NAMES
    # ---------------------------------------------------------

    columns = profiler.get_column_names()

    print("\nColumns:")
    print(columns)

    # ---------------------------------------------------------
    # DATA TYPES
    # ---------------------------------------------------------

    data_types = profiler.get_data_types()

    print("\nData Types:")

    for column, dtype in data_types.items():
        print(f"{column}: {dtype}")

    # ---------------------------------------------------------
    # COLUMN CLASSIFICATION
    # ---------------------------------------------------------

    print("\nNumeric Columns:")
    print(profiler.get_numeric_columns())

    print("\nCategorical Columns:")
    print(profiler.get_categorical_columns())

    print("\nBoolean Columns:")
    print(profiler.get_boolean_columns())

    print("\nDatetime Columns:")
    print(profiler.get_datetime_columns())

    print("\nText Columns:")
    print(profiler.get_text_columns())

    print("\nDate Candidates:")
    print(profiler.get_date_candidates())

    print("\nIdentifier Candidates:")
    print(profiler.get_identifier_candidates())

    # ---------------------------------------------------------
    # COLUMN SUMMARY
    # ---------------------------------------------------------

    print("\nColumn Summary:")

    column_summary = profiler.get_column_summary()

    for column_info in column_summary:
        print(column_info)

    # ---------------------------------------------------------
    # COMPLETE PROFILE
    # ---------------------------------------------------------

    print("\nComplete Metadata Profile:")

    profile = profiler.profile()

    for section, value in profile.items():
        print(f"\n--- {section.upper()} ---")
        print(value)

    # ---------------------------------------------------------
    # BASIC TEST ASSERTIONS
    # ---------------------------------------------------------

    assert overview["rows"] == 4
    assert overview["columns"] == 7

    assert "Age" in profiler.get_numeric_columns()
    assert "Salary" in profiler.get_numeric_columns()

    assert "Department" in profiler.get_categorical_columns()
    assert "Name" in profiler.get_categorical_columns()

    assert "IsActive" in profiler.get_boolean_columns()

    assert "JoinDate" in profiler.get_datetime_columns()

    assert "CustomerID" in profiler.get_identifier_candidates()

    print("\n" + "=" * 60)
    print("METADATA PROFILING TEST PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()