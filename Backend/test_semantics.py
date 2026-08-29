"""
test_semantics.py

Comprehensive test suite for SemanticProfiler.

Tests:
1. Data type detection
2. Identifier detection
3. Measure detection
4. Dimension detection
5. Boolean / flag detection
6. Datetime detection
7. Email detection
8. Phone detection
9. URL detection
10. Name detection
11. Location detection
12. Gender detection
13. Status/category detection
14. Complete classification
15. Complete profile
16. Convenience function
17. DataFrame immutability
"""

import pandas as pd

from profiling.semantic import (
    SemanticProfiler,
    profile_semantics
)


def main():

    # =========================================================
    # CREATE TEST DATA
    # =========================================================

    data = {

        # Identifier
        "CustomerID": [
            101,
            102,
            103,
            104,
            105
        ],

        # Numerical measure
        "Age": [
            25,
            28,
            31,
            26,
            29
        ],

        # Numerical measure
        "Salary": [
            75000,
            65000,
            85000,
            72000,
            68000
        ],

        # Dimension
        "Department": [
            "IT",
            "HR",
            "Finance",
            "IT",
            "IT"
        ],

        # Status/category
        "Status": [
            "Active",
            "Active",
            "Inactive",
            "Active",
            "Active"
        ],

        # Native boolean
        "IsVerified": [
            True,
            True,
            False,
            True,
            True
        ],

        # Datetime
        "JoinDate": pd.to_datetime([
            "2024-01-10",
            "2024-01-15",
            "2024-02-01",
            "2024-02-10",
            "2024-03-01"
        ]),

        # Email
        "Email": [
            "ali@example.com",
            "sara@example.com",
            "john@example.com",
            "ahmed@example.com",
            "zara@example.com"
        ],

        # Phone
        "Phone": [
            "+923001234567",
            "+923111234567",
            "+923221234567",
            "+923331234567",
            "+923441234567"
        ],

        # URL
        "Website": [
            "https://example.com",
            "https://google.com",
            "https://github.com",
            "https://openai.com",
            "https://microsoft.com"
        ],

        # Name
        "FirstName": [
            "Ali",
            "Sara",
            "John",
            "Ahmed",
            "Zara"
        ],

        # Location
        "City": [
            "Lahore",
            "Karachi",
            "Islamabad",
            "Lahore",
            "Karachi"
        ],

        # Gender
        "Gender": [
            "Male",
            "Female",
            "Male",
            "Female",
            "Female"
        ],

        # Random text
        "RandomText": [
            "Alpha",
            "Beta",
            "Gamma",
            "Delta",
            "Epsilon"
        ]
    }

    df = pd.DataFrame(data)

    # =========================================================
    # SAVE ORIGINAL DATA
    # =========================================================

    original_shape = df.shape
    original_columns = df.columns.tolist()
    original_data = df.copy(deep=True)

    # =========================================================
    # DISPLAY DATA
    # =========================================================

    print("=" * 70)
    print("DATA PROFILING - SEMANTIC TEST")
    print("=" * 70)

    print("\nInput Data:")
    print(df)

    print("\nDataset Shape:")
    print(df.shape)

    # =========================================================
    # CREATE PROFILER
    # =========================================================

    profiler = SemanticProfiler(df)

    # =========================================================
    # 1. DATA TYPE DETECTION
    # =========================================================

    print("\n" + "=" * 70)
    print("1. DATA TYPE DETECTION")
    print("=" * 70)

    data_types = profiler.detect_data_types()

    print(data_types)

    assert data_types["CustomerID"] == "integer"
    assert data_types["Age"] == "integer"
    assert data_types["Salary"] == "integer"

    assert data_types["Department"] == "string"
    assert data_types["Status"] == "string"

    assert data_types["IsVerified"] == "boolean"

    assert data_types["JoinDate"] == "datetime"

    assert data_types["Email"] == "string"

    print("Data type detection test: PASSED")

    # =========================================================
    # 2. IDENTIFIER DETECTION
    # =========================================================

    print("\n" + "=" * 70)
    print("2. IDENTIFIER DETECTION")
    print("=" * 70)

    identifiers = profiler.detect_identifiers()

    print(identifiers)

    assert identifiers["CustomerID"]["is_identifier"] is True

    assert identifiers["CustomerID"]["name_signal"] is True

    assert identifiers["CustomerID"]["high_uniqueness"] is True

    assert (
        identifiers["Department"]["is_identifier"]
        is False
    )

    print("Identifier detection test: PASSED")

    # =========================================================
    # 3. MEASURE DETECTION
    # =========================================================

    print("\n" + "=" * 70)
    print("3. MEASURE DETECTION")
    print("=" * 70)

    measures = profiler.detect_measures()

    print(measures)

    # Age is a measure
    assert measures["Age"]["is_measure"] is True

    # Salary is a measure
    assert measures["Salary"]["is_measure"] is True

    # Department is not a measure
    assert measures["Department"]["is_measure"] is False

    # Salary should have keyword signal
    assert measures["Salary"]["keyword_signal"] is True

    # CustomerID must NOT be a measure
    assert measures["CustomerID"]["is_measure"] is False

    print("Measure detection test: PASSED")

    # =========================================================
    # 4. DIMENSION DETECTION
    # =========================================================

    print("\n" + "=" * 70)
    print("4. DIMENSION DETECTION")
    print("=" * 70)

    dimensions = profiler.detect_dimensions()

    print(dimensions)

    assert dimensions["Department"]["is_dimension"] is True

    assert dimensions["Status"]["is_dimension"] is True

    assert dimensions["Gender"]["is_dimension"] is True

    assert dimensions["IsVerified"]["is_dimension"] is True

    assert dimensions["JoinDate"]["is_dimension"] is True

    assert dimensions["Salary"]["is_dimension"] is False

    print("Dimension detection test: PASSED")

    # =========================================================
    # 5. BOOLEAN FLAG DETECTION
    # =========================================================

    print("\n" + "=" * 70)
    print("5. BOOLEAN / FLAG DETECTION")
    print("=" * 70)

    booleans = profiler.detect_boolean_flags()

    print(booleans)

    assert booleans["IsVerified"]["is_boolean_flag"] is True

    assert booleans["IsVerified"]["native_boolean"] is True

    assert booleans["IsVerified"]["confidence"] == 0.99

    assert booleans["Status"]["is_boolean_flag"] is False

    print("Boolean flag detection test: PASSED")

    # =========================================================
    # 6. DATETIME DETECTION
    # =========================================================

    print("\n" + "=" * 70)
    print("6. DATETIME DETECTION")
    print("=" * 70)

    datetimes = profiler.detect_datetime_columns()

    print(datetimes)

    assert datetimes["JoinDate"]["is_datetime"] is True

    assert datetimes["JoinDate"]["native_datetime"] is True

    assert datetimes["CustomerID"]["is_datetime"] is False

    print("Datetime detection test: PASSED")

    # =========================================================
    # 7. EMAIL DETECTION
    # =========================================================

    print("\n" + "=" * 70)
    print("7. EMAIL DETECTION")
    print("=" * 70)

    emails = profiler.detect_emails()

    print(emails)

    assert emails["Email"]["is_email"] is True

    assert (
        emails["Email"]["email_match_percentage"]
        == 100.0
    )

    assert emails["Department"]["is_email"] is False

    print("Email detection test: PASSED")

    # =========================================================
    # 8. PHONE DETECTION
    # =========================================================

    print("\n" + "=" * 70)
    print("8. PHONE DETECTION")
    print("=" * 70)

    phones = profiler.detect_phones()

    print(phones)

    assert phones["Phone"]["is_phone"] is True

    assert (
        phones["Phone"]["phone_match_percentage"]
        == 100.0
    )

    assert phones["Department"]["is_phone"] is False

    print("Phone detection test: PASSED")

    # =========================================================
    # 9. URL DETECTION
    # =========================================================

    print("\n" + "=" * 70)
    print("9. URL DETECTION")
    print("=" * 70)

    urls = profiler.detect_urls()

    print(urls)

    assert urls["Website"]["is_url"] is True

    assert (
        urls["Website"]["url_match_percentage"]
        == 100.0
    )

    assert urls["Department"]["is_url"] is False

    print("URL detection test: PASSED")

    # =========================================================
    # 10. NAME DETECTION
    # =========================================================

    print("\n" + "=" * 70)
    print("10. NAME DETECTION")
    print("=" * 70)

    names = profiler.detect_names()

    print(names)

    assert names["FirstName"]["is_name"] is True

    assert names["FirstName"]["name_signal"] is True

    assert names["Department"]["is_name"] is False

    print("Name detection test: PASSED")

    # =========================================================
    # 11. LOCATION DETECTION
    # =========================================================

    print("\n" + "=" * 70)
    print("11. LOCATION DETECTION")
    print("=" * 70)

    locations = profiler.detect_locations()

    print(locations)

    assert locations["City"]["is_location"] is True

    assert locations["City"]["name_signal"] is True

    assert locations["Department"]["is_location"] is False

    print("Location detection test: PASSED")

    # =========================================================
    # 12. GENDER DETECTION
    # =========================================================

    print("\n" + "=" * 70)
    print("12. GENDER DETECTION")
    print("=" * 70)

    genders = profiler.detect_gender()

    print(genders)

    assert genders["Gender"]["is_gender"] is True

    assert (
        genders["Gender"]["gender_match_percentage"]
        == 100.0
    )

    assert genders["Department"]["is_gender"] is False

    print("Gender detection test: PASSED")

    # =========================================================
    # 13. STATUS / CATEGORY DETECTION
    # =========================================================

    print("\n" + "=" * 70)
    print("13. STATUS / CATEGORY DETECTION")
    print("=" * 70)

    statuses = profiler.detect_status_categories()

    print(statuses)

    assert (
        statuses["Status"]["is_status_or_category"]
        is True
    )

    assert statuses["Status"]["name_signal"] is True

    # Department is low-cardinality text
    assert (
        statuses["Department"]["is_status_or_category"]
        is True
    )

    print("Status/category detection test: PASSED")

    # =========================================================
    # 14. COMPLETE CLASSIFICATION
    # =========================================================

    print("\n" + "=" * 70)
    print("14. COMPLETE COLUMN CLASSIFICATION")
    print("=" * 70)

    classification = profiler.classify_columns()

    print(classification)

    # Columns must exist
    assert "CustomerID" in classification
    assert "Salary" in classification
    assert "Department" in classification
    assert "Email" in classification

    # CustomerID -> identifier
    assert (
        "identifier"
        in classification["CustomerID"]["semantic_types"]
    )

    # CustomerID should NOT be a measure
    assert (
        "measure"
        not in classification["CustomerID"]["semantic_types"]
    )

    # Salary -> measure
    assert (
        "measure"
        in classification["Salary"]["semantic_types"]
    )

    # Department -> dimension
    assert (
        "dimension"
        in classification["Department"]["semantic_types"]
    )

    # Email -> email
    assert (
        "email"
        in classification["Email"]["semantic_types"]
    )

    # Phone -> phone
    assert (
        "phone"
        in classification["Phone"]["semantic_types"]
    )

    # Website -> URL
    assert (
        "url"
        in classification["Website"]["semantic_types"]
    )

    # FirstName -> name
    assert (
        "name"
        in classification["FirstName"]["semantic_types"]
    )

    # City -> location
    assert (
        "location"
        in classification["City"]["semantic_types"]
    )

    # Gender -> gender
    assert (
        "gender"
        in classification["Gender"]["semantic_types"]
    )

    print("Complete classification test: PASSED")

    # =========================================================
    # 15. COMPLETE PROFILE
    # =========================================================

    print("\n" + "=" * 70)
    print("15. COMPLETE SEMANTIC PROFILE")
    print("=" * 70)

    profile = profiler.profile()

    print(profile)

    expected_sections = [

        "data_types",

        "identifiers",

        "measures",

        "dimensions",

        "boolean_flags",

        "datetime_columns",

        "emails",

        "phones",

        "urls",

        "names",

        "locations",

        "gender",

        "status_categories",

        "column_classification"
    ]

    for section in expected_sections:

        assert section in profile

    print("Complete semantic profile test: PASSED")

    # =========================================================
    # 16. CONVENIENCE FUNCTION
    # =========================================================

    print("\n" + "=" * 70)
    print("16. CONVENIENCE FUNCTION")
    print("=" * 70)

    convenience_profile = profile_semantics(df)

    assert isinstance(
        convenience_profile,
        dict
    )

    assert "data_types" in convenience_profile

    assert "identifiers" in convenience_profile

    assert "column_classification" in convenience_profile

    assert (
        convenience_profile[
            "data_types"
        ]["CustomerID"]
        == "integer"
    )

    assert (
        convenience_profile[
            "column_classification"
        ]["Salary"]["primary_semantic_type"]
        == "measure"
    )

    print("Convenience function test: PASSED")

    # =========================================================
    # 17. DATAFRAME IMMUTABILITY
    # =========================================================

    print("\n" + "=" * 70)
    print("17. DATAFRAME IMMUTABILITY")
    print("=" * 70)

    # Shape unchanged
    assert df.shape == original_shape

    # Columns unchanged
    assert df.columns.tolist() == original_columns

    # Data unchanged
    pd.testing.assert_frame_equal(
        df,
        original_data
    )

    print("DataFrame immutability test: PASSED")

    # =========================================================
    # FINAL SUCCESS
    # =========================================================

    print("\n" + "=" * 70)
    print("ALL SEMANTIC PROFILING TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()