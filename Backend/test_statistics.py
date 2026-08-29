"""
test_statistics.py

Test file for the Statistics Profiling module
of Agentic BI Analyst.
"""

import pandas as pd

from profiling.statistics import StatisticsProfiler


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

        # 9 Active + 1 Inactive = 10 values
        "Status": ["Active"] * 9 + ["Inactive"]
    }

    # =========================================================
    # CREATE DATAFRAME
    # =========================================================

    print("Status values:", data["Status"])
    print("Status length:", len(data["Status"]))

    df = pd.DataFrame(data)

    print("=" * 70)
    print("DATA PROFILING - STATISTICS TEST")
    print("=" * 70)

    print("\nInput Data:")
    print(df)

    # =========================================================
    # CREATE PROFILER
    # =========================================================

    profiler = StatisticsProfiler(df)

    # =========================================================
    # MISSING VALUE ANALYSIS
    # =========================================================

    print("\n" + "=" * 70)
    print("MISSING VALUE ANALYSIS")
    print("=" * 70)

    missing = profiler.missing_values()

    for column, result in missing.items():
        print(f"\n{column}:")
        print(result)

    # =========================================================
    # UNIQUE VALUE ANALYSIS
    # =========================================================

    print("\n" + "=" * 70)
    print("UNIQUE VALUE ANALYSIS")
    print("=" * 70)

    unique = profiler.unique_values()

    for column, result in unique.items():
        print(f"\n{column}:")
        print(result)

    # =========================================================
    # NUMERICAL STATISTICS
    # =========================================================

    print("\n" + "=" * 70)
    print("NUMERICAL STATISTICS")
    print("=" * 70)

    numerical = profiler.numerical_statistics()

    for column, result in numerical.items():
        print(f"\n{column}:")
        print(result)

    # =========================================================
    # CATEGORICAL STATISTICS
    # =========================================================

    print("\n" + "=" * 70)
    print("CATEGORICAL STATISTICS")
    print("=" * 70)

    categorical = profiler.categorical_statistics()

    for column, result in categorical.items():
        print(f"\n{column}:")
        print(result)

    # =========================================================
    # OUTLIER DETECTION
    # =========================================================

    print("\n" + "=" * 70)
    print("OUTLIER DETECTION")
    print("=" * 70)

    outliers = profiler.outliers()

    for column, result in outliers.items():
        print(f"\n{column}:")
        print(result)

    # =========================================================
    # CONSTANT / NEAR-CONSTANT ANALYSIS
    # =========================================================

    print("\n" + "=" * 70)
    print("CONSTANT / NEAR-CONSTANT ANALYSIS")
    print("=" * 70)

    # Status:
    # Active = 9
    # Inactive = 1
    # Dominant percentage = 90%
    #
    # We use 0.89 for this small test dataset.

    constants = profiler.constant_columns(
        near_constant_threshold=0.89
    )

    print("\nStatus result:")
    print(constants["Status"])

    for column, result in constants.items():
        print(f"\n{column}:")
        print(result)

    # =========================================================
    # COMPLETE STATISTICAL PROFILE
    # =========================================================

    print("\n" + "=" * 70)
    print("COMPLETE STATISTICAL PROFILE")
    print("=" * 70)

    profile = profiler.profile()

    for section, result in profile.items():

        print(f"\n--- {section.upper()} ---")

        print(result)

    # =========================================================
    # ASSERTIONS
    # =========================================================

    # ---------------------------------------------------------
    # Missing value tests
    # ---------------------------------------------------------

    assert missing["Salary"]["missing_count"] == 1

    assert missing["Department"]["missing_count"] == 1

    # ---------------------------------------------------------
    # Unique value tests
    # ---------------------------------------------------------

    assert unique["CustomerID"]["unique_count"] == 10

    # ---------------------------------------------------------
    # Numerical statistics tests
    # ---------------------------------------------------------

    assert "Age" in numerical

    assert "Salary" in numerical

    assert numerical["Age"]["count"] == 10

    # ---------------------------------------------------------
    # Categorical statistics tests
    # ---------------------------------------------------------

    assert "Department" in categorical

    assert "Status" in categorical

    # ---------------------------------------------------------
    # Outlier detection tests
    # ---------------------------------------------------------

    assert "Salary" in outliers

    assert outliers["Salary"]["outlier_count"] >= 1

    # ---------------------------------------------------------
    # Near-constant column test
    # ---------------------------------------------------------

    assert bool(constants["Status"]["is_near_constant"]) is True

    # ---------------------------------------------------------
    # DataFrame integrity test
    # ---------------------------------------------------------

    # Make sure the profiler did not modify the DataFrame.

    assert df.shape == (10, 5)

    # =========================================================
    # SUCCESS
    # =========================================================

    print("\n" + "=" * 70)
    print("STATISTICS PROFILING TEST PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()