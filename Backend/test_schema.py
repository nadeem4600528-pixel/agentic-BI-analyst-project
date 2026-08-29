"""
test_schema.py

Test file for the Schema Profiling module
of Agentic BI Analyst.

Tests every public function of schema.py.
"""

import pandas as pd

from profiling.schema import SchemaProfiler, profile_schema


def main():

    # =========================================================
    # CREATE TEST DATA
    # =========================================================

    data = {
        "CustomerID": [
            101, 102, 103, 104, 105,
            106, 107, 108, 109, 110
        ],

        "Name": [
            "Ali",
            "Sara",
            "Ahmed",
            "Fatima",
            "Hassan",
            "Ayesha",
            "Bilal",
            "Zain",
            "Usman",
            "Hina"
        ],

        "Age": [
            25, 28, 31, 26, 29,
            35, 27, 30, 32, 26
        ],

        "Salary": [
            75000,
            65000,
            85000,
            72000,
            68000,
            90000,
            71000,
            73000,
            82000,
            None
        ],

        "Department": [
            "IT",
            "HR",
            "Finance",
            "IT",
            "IT",
            "HR",
            "IT",
            "IT",
            "Finance",
            None
        ],

        "IsActive": [
            True,
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            True
        ]
    }

    df = pd.DataFrame(data)

    print("=" * 70)
    print("DATA PROFILING - SCHEMA TEST")
    print("=" * 70)

    print("\nInput Data:")
    print(df)

    print("\nDataFrame Shape:")
    print(df.shape)

    # =========================================================
    # CREATE PROFILER
    # =========================================================

    profiler = SchemaProfiler(df)

    # =========================================================
    # TEST 1 - classify_column()
    # =========================================================

    print("\n" + "=" * 70)
    print("TEST 1 - COLUMN CLASSIFICATION")
    print("=" * 70)

    for column in df.columns:

        classification = profiler.classify_column(column)

        print(
            f"{column:15} -> {classification}"
        )

    # Verify important classifications

    assert profiler.classify_column(
        "CustomerID"
    ) is not None

    assert profiler.classify_column(
        "Name"
    ) is not None

    assert profiler.classify_column(
        "Age"
    ) is not None

    assert profiler.classify_column(
        "Salary"
    ) is not None

    assert profiler.classify_column(
        "Department"
    ) is not None

    assert profiler.classify_column(
        "IsActive"
    ) is not None

    # =========================================================
    # TEST 2 - column_schema()
    # =========================================================

    print("\n" + "=" * 70)
    print("TEST 2 - COLUMN SCHEMA")
    print("=" * 70)

    column_schema = profiler.column_schema()

    for column, result in column_schema.items():

        print(f"\n{column}:")
        print(result)

    # Verify all columns are present

    assert len(column_schema) == len(df.columns)

    for column in df.columns:

        assert column in column_schema

        assert isinstance(
            column_schema[column],
            dict
        )

    # Verify expected metadata fields exist

    for column in df.columns:

        result = column_schema[column]

        assert "dtype" in result

    # =========================================================
    # TEST 3 - summary()
    # =========================================================

    print("\n" + "=" * 70)
    print("TEST 3 - SCHEMA SUMMARY")
    print("=" * 70)

    summary = profiler.summary()

    print("\nSchema Summary:")
    print(summary)

    # Summary must be a dictionary

    assert isinstance(summary, dict)

    # Dataset should contain 10 rows

    assert summary["row_count"] == 10

    # Dataset should contain 6 columns

    assert summary["column_count"] == 6

    # =========================================================
    # TEST 4 - nullable_columns()
    # =========================================================

    print("\n" + "=" * 70)
    print("TEST 4 - NULLABLE COLUMN ANALYSIS")
    print("=" * 70)

    nullable = profiler.nullable_columns()

    print("\nNullable Columns:")

    for column, value in nullable.items():

        print(
            f"{column:15} -> {value}"
        )

    # Verify all columns are present

    assert len(nullable) == len(df.columns)

    for column in df.columns:

        assert column in nullable

        assert isinstance(
            nullable[column],
            bool
        )

    # Salary contains a NULL

    assert nullable["Salary"] is True

    # Department contains a NULL

    assert nullable["Department"] is True

    # CustomerID has no NULL

    assert nullable["CustomerID"] is False

    # Name has no NULL

    assert nullable["Name"] is False

    # =========================================================
    # TEST 5 - schema_fingerprint()
    # =========================================================

    print("\n" + "=" * 70)
    print("TEST 5 - SCHEMA FINGERPRINT")
    print("=" * 70)

    fingerprint = profiler.schema_fingerprint()

    print("\nSchema Fingerprint:")
    print(fingerprint)

    # Fingerprint should exist

    assert fingerprint is not None

    # Fingerprint should be a string

    assert isinstance(
        fingerprint,
        str
    )

    # Fingerprint should not be empty

    assert len(fingerprint) > 0

    # Running it again should produce
    # the same fingerprint

    fingerprint_again = (
        profiler.schema_fingerprint()
    )

    assert fingerprint == fingerprint_again

    # =========================================================
    # TEST 6 - profile()
    # =========================================================

    print("\n" + "=" * 70)
    print("TEST 6 - COMPLETE SCHEMA PROFILE")
    print("=" * 70)

    profile = profiler.profile()

    print("\nComplete Schema Profile:")

    for section, result in profile.items():

        print(
            f"\n--- {section.upper()} ---"
        )

        print(result)

    # Profile should be dictionary

    assert isinstance(
        profile,
        dict
    )

    # Expected sections should exist

    assert "columns" in profile

    assert "summary" in profile

    assert "nullable_columns" in profile

    assert "schema_fingerprint" in profile

    # =========================================================
    # TEST 7 - profile_schema()
    # =========================================================

    print("\n" + "=" * 70)
    print("TEST 7 - CONVENIENCE FUNCTION")
    print("=" * 70)

    standalone_profile = profile_schema(df)

    print("\nprofile_schema() result:")

    for section, result in standalone_profile.items():

        print(
            f"\n--- {section.upper()} ---"
        )

        print(result)

    # Verify result

    assert isinstance(
        standalone_profile,
        dict
    )

    assert "columns" in standalone_profile

    assert "summary" in standalone_profile

    assert "nullable_columns" in standalone_profile

    assert "schema_fingerprint" in standalone_profile

    # =========================================================
    # TEST 8 - INPUT DATA WAS NOT MODIFIED
    # =========================================================

    print("\n" + "=" * 70)
    print("TEST 8 - DATAFRAME INTEGRITY")
    print("=" * 70)

    assert df.shape == (10, 6)

    assert list(df.columns) == [
        "CustomerID",
        "Name",
        "Age",
        "Salary",
        "Department",
        "IsActive"
    ]

    print("\nDataFrame was not modified.")

    # =========================================================
    # SUCCESS
    # =========================================================

    print("\n" + "=" * 70)
    print("SCHEMA PROFILING TEST PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()