"""
test_structure.py

Comprehensive test file for the Dataset Structure Profiling module
of Agentic BI Analyst.

Tests:
- dataset_shape()
- empty_rows()
- empty_columns()
- fully_populated_columns()
- column_composition()
- width_classification()
- size_classification()
- row_column_ratio()
- duplicate_rows()
- memory_usage()
- structural_warnings()
- profile()
- profile_structure()

The tests also verify that the input DataFrame is NOT modified.
"""

import pandas as pd
import numpy as np

from profiling.structure import (
    StructureProfiler,
    profile_structure
)


def main():

    # =========================================================
    # CREATE TEST DATA
    # =========================================================

    data = {
        "CustomerID": [
            101, 102, 103, 104, 105,
            106, 107, 108, 109, 110
        ],

        "Age": [
            25, 28, 31, 26, 29,
            35, 27, 30, 100, 26
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
            1000000,
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

        "Status": [
            "Active",
            "Active",
            "Active",
            "Active",
            "Active",
            "Active",
            "Active",
            "Active",
            "Active",
            "Inactive"
        ],

        "IsVerified": [
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
            True
        ],

        "JoinDate": pd.to_datetime([
            "2024-01-10",
            "2024-01-15",
            "2024-02-01",
            "2024-02-10",
            "2024-03-01",
            "2024-03-15",
            "2024-04-01",
            "2024-04-10",
            "2024-05-01",
            "2024-05-15"
        ]),

        "EmptyColumn": [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        ]
    }

    df = pd.DataFrame(data)

    # =========================================================
    # ADD A COMPLETELY EMPTY ROW
    # =========================================================

    empty_row = pd.DataFrame({
        "CustomerID": [None],
        "Age": [None],
        "Salary": [None],
        "Department": [None],
        "Status": [None],
        "IsVerified": [None],
        "JoinDate": [None],
        "EmptyColumn": [None]
    })

    df = pd.concat(
        [df, empty_row],
        ignore_index=True
    )

    # =========================================================
    # ADD DUPLICATE ROW
    # =========================================================

    duplicate_row = df.iloc[[0]].copy()

    df = pd.concat(
        [df, duplicate_row],
        ignore_index=True
    )

    # =========================================================
    # STORE ORIGINAL DATAFRAME INFORMATION
    # =========================================================

    original_shape = df.shape
    original_columns = df.columns.tolist()
    original_data = df.copy(deep=True)

    # =========================================================
    # DISPLAY INPUT
    # =========================================================

    print("=" * 70)
    print("DATA PROFILING - STRUCTURE TEST")
    print("=" * 70)

    print("\nInput Data:")
    print(df)

    print("\nDataset Shape:")
    print(df.shape)

    # =========================================================
    # CREATE PROFILER
    # =========================================================

    profiler = StructureProfiler(df)

    # =========================================================
    # 1. DATASET SHAPE
    # =========================================================

    print("\n" + "=" * 70)
    print("1. DATASET SHAPE")
    print("=" * 70)

    shape = profiler.dataset_shape()

    print(shape)

    assert shape["rows"] == 12
    assert shape["columns"] == 8
    assert shape["total_cells"] == 96
    assert shape["is_empty"] is False

    print("Dataset shape test: PASSED")

    # =========================================================
    # 2. EMPTY ROWS
    # =========================================================

    print("\n" + "=" * 70)
    print("2. EMPTY ROW ANALYSIS")
    print("=" * 70)

    empty_rows = profiler.empty_rows()

    print(empty_rows)

    assert empty_rows["empty_row_count"] == 1
    assert empty_rows["empty_row_percentage"] == round(
        (1 / 12) * 100,
        2
    )

    print("Empty row test: PASSED")

    # =========================================================
    # 3. EMPTY COLUMNS
    # =========================================================

    print("\n" + "=" * 70)
    print("3. EMPTY COLUMN ANALYSIS")
    print("=" * 70)

    empty_columns = profiler.empty_columns()

    print(empty_columns)

    assert empty_columns["empty_column_count"] == 1
    assert "EmptyColumn" in empty_columns["empty_columns"]

    print("Empty column test: PASSED")

    # =========================================================
    # 4. FULLY POPULATED COLUMNS
    # =========================================================

    print("\n" + "=" * 70)
    print("4. FULLY POPULATED COLUMN ANALYSIS")
    print("=" * 70)

    fully_populated = profiler.fully_populated_columns()

    print(fully_populated)

    # Because the test dataset intentionally contains
    # one completely empty row, every column has at least
    # one missing value.

    assert fully_populated[
        "fully_populated_column_count"
    ] == 0

    assert fully_populated[
        "fully_populated_columns"
    ] == []

    print("Fully populated column test: PASSED")

    # =========================================================
    # 5. COLUMN COMPOSITION
    # =========================================================

    print("\n" + "=" * 70)
    print("5. COLUMN COMPOSITION")
    print("=" * 70)

    composition = profiler.column_composition()

    print(composition)

    # Numeric columns
    assert "CustomerID" in composition["numeric"]
    assert "Age" in composition["numeric"]
    assert "Salary" in composition["numeric"]

    # Text columns
    assert "Department" in composition["text"]
    assert "Status" in composition["text"]

    # Boolean columns
    assert "IsVerified" in composition["boolean"]

    # Datetime columns
    assert "JoinDate" in composition["datetime"]

    # EmptyColumn is object dtype and therefore text
    assert "EmptyColumn" in composition["text"]

    assert composition["numeric_count"] == 3
    assert composition["text_count"] == 3
    assert composition["boolean_count"] == 1
    assert composition["datetime_count"] == 1

    print("Column composition test: PASSED")

    # =========================================================
    # 6. WIDTH CLASSIFICATION
    # =========================================================

    print("\n" + "=" * 70)
    print("6. WIDTH CLASSIFICATION")
    print("=" * 70)

    width = profiler.width_classification()

    print(width)

    assert width["column_count"] == 8
    assert width["classification"] == "narrow"

    print("Width classification test: PASSED")

    # =========================================================
    # 7. SIZE CLASSIFICATION
    # =========================================================

    print("\n" + "=" * 70)
    print("7. SIZE CLASSIFICATION")
    print("=" * 70)

    size = profiler.size_classification()

    print(size)

    assert size["row_count"] == 12
    assert size["classification"] == "small"

    print("Size classification test: PASSED")

    # =========================================================
    # 8. ROW / COLUMN RATIO
    # =========================================================

    print("\n" + "=" * 70)
    print("8. ROW / COLUMN RATIO")
    print("=" * 70)

    ratio = profiler.row_column_ratio()

    print(ratio)

    assert ratio["rows"] == 12
    assert ratio["columns"] == 8
    assert ratio["row_column_ratio"] == 1.5

    print("Row/column ratio test: PASSED")

    # =========================================================
    # 9. DUPLICATE ROW ANALYSIS
    # =========================================================

    print("\n" + "=" * 70)
    print("9. DUPLICATE ROW ANALYSIS")
    print("=" * 70)

    duplicates = profiler.duplicate_rows()

    print(duplicates)

    # The duplicated first row should be detected.
    assert duplicates["duplicate_row_count"] >= 1

    assert duplicates[
        "duplicate_row_percentage"
    ] > 0

    print("Duplicate row test: PASSED")

    # =========================================================
    # 10. MEMORY USAGE
    # =========================================================

    print("\n" + "=" * 70)
    print("10. MEMORY USAGE")
    print("=" * 70)

    memory = profiler.memory_usage()

    print(memory)

    assert "memory_bytes" in memory
    assert "memory_kb" in memory
    assert "memory_mb" in memory

    assert memory["memory_bytes"] > 0
    assert memory["memory_kb"] > 0
    assert memory["memory_mb"] > 0

    print("Memory usage test: PASSED")

    # =========================================================
    # 11. STRUCTURAL WARNINGS
    # =========================================================

    print("\n" + "=" * 70)
    print("11. STRUCTURAL WARNINGS")
    print("=" * 70)

    warnings = profiler.structural_warnings()

    print(warnings)

    # Empty row should produce a warning.
    assert any(
        "empty row" in warning.lower()
        for warning in warnings
    )

    # EmptyColumn should produce a warning.
    assert any(
        "EmptyColumn" in warning
        for warning in warnings
    )

    print("Structural warnings test: PASSED")

    # =========================================================
    # 12. COMPLETE PROFILE
    # =========================================================

    print("\n" + "=" * 70)
    print("12. COMPLETE STRUCTURAL PROFILE")
    print("=" * 70)

    profile = profiler.profile()

    print(profile)

    # Verify every expected section exists.

    expected_sections = [
        "dataset_shape",
        "empty_rows",
        "empty_columns",
        "fully_populated_columns",
        "column_composition",
        "width_classification",
        "size_classification",
        "row_column_ratio",
        "duplicate_rows",
        "memory_usage",
        "structural_warnings"
    ]

    for section in expected_sections:

        assert section in profile

    print("Complete profile test: PASSED")

    # =========================================================
    # 13. CONVENIENCE FUNCTION
    # =========================================================

    print("\n" + "=" * 70)
    print("13. CONVENIENCE FUNCTION")
    print("=" * 70)

    convenience_profile = profile_structure(df)

    assert isinstance(
        convenience_profile,
        dict
    )

    assert "dataset_shape" in convenience_profile
    assert "column_composition" in convenience_profile
    assert "memory_usage" in convenience_profile

    assert (
        convenience_profile["dataset_shape"]["rows"]
        == 12
    )

    assert (
        convenience_profile["dataset_shape"]["columns"]
        == 8
    )

    print("Convenience function test: PASSED")

    # =========================================================
    # 14. DATAFRAME IMMUTABILITY
    # =========================================================

    print("\n" + "=" * 70)
    print("14. DATAFRAME IMMUTABILITY")
    print("=" * 70)

    # The profiling module must NOT modify the input DataFrame.

    assert df.shape == original_shape

    assert df.columns.tolist() == original_columns

    pd.testing.assert_frame_equal(
        df,
        original_data
    )

    print("DataFrame immutability test: PASSED")

    # =========================================================
    # FINAL SUCCESS
    # =========================================================

    print("\n" + "=" * 70)
    print("ALL STRUCTURE PROFILING TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()