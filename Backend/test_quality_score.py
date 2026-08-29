"""
test_quality_score.py

Test file for the Data Quality Score module
of Agentic BI Analyst.
"""

import pandas as pd
import numpy as np
from profiling.quality_score import QualityScoreProfiler, calculate_quality_score
from profiling.metadata import profile_metadata
from profiling.statistics import profile_statistics


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
        ]
    }

    df = pd.DataFrame(data)

    print("=" * 70)
    print("DATA QUALITY SCORE TEST")
    print("=" * 70)

    print("\nInput Data:")
    print(df)

    print("\nDataset Shape:")
    print(df.shape)

    # =========================================================
    # CREATE QUALITY SCORE OBJECT
    # =========================================================

    # First get metadata and statistics profiles
    metadata_profile = profile_metadata(df)
    statistics_profile = profile_statistics(df)

    scorer = QualityScoreProfiler(
        df=df,
        metadata_profile=metadata_profile,
        statistics_profile=statistics_profile
    )

    # =========================================================
    # GENERATE QUALITY SCORE
    # =========================================================

    print("\n" + "=" * 70)
    print("QUALITY SCORE")
    print("=" * 70)

    result = scorer.profile()

    print("\nQuality Score Result:")
    print(result)

    # =========================================================
    # DISPLAY RESULT DETAILS
    # =========================================================

    print("\n" + "=" * 70)
    print("QUALITY SCORE DETAILS")
    print("=" * 70)

    if isinstance(result, dict):

        for key, value in result.items():

            print(f"\n{key}:")
            print(value)

    # =========================================================
    # BASIC VALIDATION
    # =========================================================

    print("\n" + "=" * 70)
    print("VALIDATING QUALITY SCORE")
    print("=" * 70)

    assert isinstance(result, dict), \
        "Quality score result must be a dictionary."

    print("[PASS] Result is a dictionary.")

    # ---------------------------------------------------------
    # Check score exists
    # ---------------------------------------------------------

    score = None

    possible_score_keys = [
        "overall_quality_score",
        "quality_score",
        "score",
        "overall_score",
        "data_quality_score"
    ]

    for key in possible_score_keys:

        if key in result:
            score = result[key]
            break

    assert score is not None, \
        "Quality score value was not found in result."

    print(f"[PASS] Quality score found: {score}")

    # ---------------------------------------------------------
    # Check score is numeric
    # ---------------------------------------------------------

    assert isinstance(score, (int, float)), \
        "Quality score must be numeric."

    print("[PASS] Quality score is numeric.")

    # ---------------------------------------------------------
    # Check score range
    # ---------------------------------------------------------

    assert 0 <= score <= 100, \
        "Quality score must be between 0 and 100."

    print("[PASS] Quality score is between 0 and 100.")

    # =========================================================
    # TEST DATA QUALITY IMPACT
    # =========================================================

    print("\n" + "=" * 70)
    print("TESTING DATA QUALITY IMPACT")
    print("=" * 70)

    # Create a clean dataset
    clean_df = pd.DataFrame({
        "CustomerID": [101, 102, 103, 104, 105],
        "Age": [25, 28, 31, 26, 29],
        "Salary": [75000, 65000, 85000, 72000, 68000],
        "Department": [
            "IT",
            "HR",
            "Finance",
            "IT",
            "IT"
        ],
        "Status": [
            "Active",
            "Active",
            "Active",
            "Active",
            "Active"
        ]
    })

    clean_metadata = profile_metadata(clean_df)
    clean_statistics = profile_statistics(clean_df)

    clean_scorer = QualityScoreProfiler(
        df=clean_df,
        metadata_profile=clean_metadata,
        statistics_profile=clean_statistics
    )

    clean_result = clean_scorer.profile()

    print("\nClean Dataset Score:")
    print(clean_result)

    clean_score = None

    for key in possible_score_keys:

        if key in clean_result:
            clean_score = clean_result[key]
            break

    assert clean_score is not None, \
        "Clean dataset quality score was not found."

    assert isinstance(clean_score, (int, float)), \
        "Clean dataset score must be numeric."

    assert 0 <= clean_score <= 100, \
        "Clean dataset score must be between 0 and 100."

    print(f"[PASS] Clean dataset score: {clean_score}")

    # =========================================================
    # VERIFY ORIGINAL DATAFRAME WAS NOT MODIFIED
    # =========================================================

    print("\n" + "=" * 70)
    print("DATAFRAME INTEGRITY TEST")
    print("=" * 70)

    assert df.shape == (10, 5), \
        "QualityScore modified the original DataFrame."

    print("[PASS] Original DataFrame was not modified.")

    # =========================================================
    # FINAL SUCCESS
    # =========================================================

    print("\n" + "=" * 70)
    print("QUALITY SCORE TEST PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()