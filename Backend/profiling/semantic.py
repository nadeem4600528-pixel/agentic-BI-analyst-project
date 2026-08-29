"""
semantic.py

Semantic column profiling module for Agentic BI Analyst.

Responsibilities:
- Semantic column detection
- Data-type detection
- Identifier detection
- Measure detection
- Dimension detection
- Boolean/flag detection
- Date/time semantic detection
- Text semantic detection
- Email detection
- Phone detection
- URL detection
- Name detection
- Address/location detection
- Gender detection
- Status/category detection
- Confidence scoring

This module DOES NOT modify the input DataFrame.
"""

from typing import Any, Dict
import re

import pandas as pd


class SemanticProfiler:
    """
    Performs semantic profiling on a pandas DataFrame.
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self, df: pd.DataFrame):

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "SemanticProfiler requires a pandas DataFrame."
            )

        self.df = df

    # =========================================================
    # BASIC DATA TYPE DETECTION
    # =========================================================

    def detect_data_types(self) -> Dict[str, str]:
        """
        Detect the pandas data type of every column.
        """

        result = {}

        for column in self.df.columns:

            series = self.df[column]

            if pd.api.types.is_bool_dtype(series):

                detected_type = "boolean"

            elif pd.api.types.is_integer_dtype(series):

                detected_type = "integer"

            elif pd.api.types.is_float_dtype(series):

                detected_type = "float"

            elif pd.api.types.is_datetime64_any_dtype(series):

                detected_type = "datetime"

            elif pd.api.types.is_numeric_dtype(series):

                detected_type = "numeric"

            elif isinstance(series.dtype, pd.CategoricalDtype):

                detected_type = "categorical"

            elif (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            ):

                detected_type = "string"

            else:

                detected_type = "other"

            result[column] = detected_type

        return result

    # =========================================================
    # IDENTIFIER DETECTION
    # =========================================================

    def detect_identifiers(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect columns that are likely identifiers.

        Identifier signals:
        - Strong identifier column name
        - High uniqueness
        - Suitable identifier-like data type

        High uniqueness is reported independently from the
        final identifier classification.

        This prevents columns such as Age, Salary, Email,
        Website, FirstName, etc. from automatically becoming
        identifiers simply because they happen to be unique.
        """

        result = {}

        # Semantic columns that should NOT be classified as
        # identifiers merely because they have unique values.
        semantic_name_exclusions = [
            "age",
            "salary",
            "amount",
            "price",
            "cost",
            "revenue",
            "income",
            "profit",
            "score",
            "rate",
            "percentage",
            "percent",
            "quantity",
            "qty",
            "value",
            "total",
            "balance",
            "duration",
            "hours",
            "minutes",
            "email",
            "phone",
            "mobile",
            "website",
            "url",
            "name",
            "firstname",
            "first_name",
            "lastname",
            "last_name",
            "fullname",
            "full_name",
            "randomtext",
            "date",
            "time",
            "timestamp"
        ]

        for column in self.df.columns:

            series = self.df[column]

            # -------------------------------------------------
            # Uniqueness
            # -------------------------------------------------

            non_null = series.dropna()

            if len(non_null) == 0:

                uniqueness = 0.0

            else:

                uniqueness = (
                    non_null.nunique(dropna=True)
                    / len(non_null)
                ) * 100

            # -------------------------------------------------
            # Column name
            # -------------------------------------------------

            column_name = str(column).lower().strip()

            # -------------------------------------------------
            # Strong identifier-name signal
            # -------------------------------------------------

            name_signal = bool(
                column_name == "id"
                or column_name.endswith("_id")
                or column_name.endswith("id")
                or "identifier" in column_name
                or column_name.endswith("_key")
                or column_name.endswith("key")
                or column_name.endswith("_code")
            )

            # -------------------------------------------------
            # High uniqueness
            #
            # This is a statistical property.
            # It must NOT depend on dataset size.
            # -------------------------------------------------

            high_uniqueness = bool(
                uniqueness >= 95.0
            )

            # -------------------------------------------------
            # Semantic exclusion
            # -------------------------------------------------

            semantic_exclusion = any(
                keyword in column_name
                for keyword in semantic_name_exclusions
            )

            # -------------------------------------------------
            # Data type suitability
            # -------------------------------------------------

            is_numeric = bool(
                pd.api.types.is_numeric_dtype(series)
                and not pd.api.types.is_bool_dtype(series)
            )

            is_text = bool(
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            )

            suitable_type = bool(
                is_numeric
                or is_text
            )

            # -------------------------------------------------
            # Final identifier decision
            # -------------------------------------------------

            is_identifier = bool(
                suitable_type
                and (
                    name_signal
                    or (
                        high_uniqueness
                        and not semantic_exclusion
                    )
                )
            )

            # -------------------------------------------------
            # Confidence
            # -------------------------------------------------

            if name_signal and high_uniqueness:

                confidence = 0.99

            elif name_signal:

                confidence = 0.98

            elif high_uniqueness and not semantic_exclusion:

                confidence = 0.80

            elif high_uniqueness:

                confidence = 0.20

            else:

                confidence = 0.10

            result[column] = {

                "is_identifier": is_identifier,

                "uniqueness_percentage": round(
                    uniqueness,
                    2
                ),

                "name_signal": name_signal,

                "high_uniqueness": high_uniqueness,

                "confidence": confidence
            }

        return result

    # =========================================================
    # MEASURE DETECTION
    # =========================================================

    def detect_measures(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect columns that are likely numerical measures.

        Examples:
        - Salary
        - Revenue
        - Amount
        - Age
        - Quantity
        - Score
        - Duration
        """

        result = {}

        measure_keywords = [
            "amount",
            "price",
            "cost",
            "salary",
            "revenue",
            "income",
            "profit",
            "sales",
            "quantity",
            "qty",
            "score",
            "rate",
            "percentage",
            "percent",
            "value",
            "total",
            "balance",
            "age",
            "count",
            "duration",
            "hours",
            "minutes",
            "cgpa",
            "marks",
            "rating",
            "distance",
            "weight",
            "height"
        ]

        identifiers = self.detect_identifiers()

        for column in self.df.columns:

            series = self.df[column]

            # -------------------------------------------------
            # Numeric signal
            # -------------------------------------------------

            numeric = bool(
                pd.api.types.is_numeric_dtype(series)
                and not pd.api.types.is_bool_dtype(series)
            )

            # -------------------------------------------------
            # Identifier signal
            # -------------------------------------------------

            identifier = bool(
                identifiers[column]["is_identifier"]
            )

            # -------------------------------------------------
            # Keyword signal
            # -------------------------------------------------

            name = str(column).lower()

            keyword_signal = any(
                keyword in name
                for keyword in measure_keywords
            )

            # -------------------------------------------------
            # Measure decision
            #
            # Numeric + measure keyword = measure
            #
            # Numeric + NOT identifier = measure
            #
            # Identifier columns remain non-measures unless
            # they have an explicit measure keyword.
            # -------------------------------------------------

            if numeric and keyword_signal:

                is_measure = True

            elif numeric and not identifier:

                is_measure = True

            else:

                is_measure = False

            # -------------------------------------------------
            # Confidence
            # -------------------------------------------------

            if is_measure and keyword_signal:

                confidence = 0.98

            elif is_measure:

                confidence = 0.85

            elif numeric and identifier:

                confidence = 0.20

            elif keyword_signal:

                confidence = 0.60

            else:

                confidence = 0.10

            result[column] = {

                "is_measure": bool(is_measure),

                "numeric": numeric,

                "identifier": identifier,

                "keyword_signal": keyword_signal,

                "confidence": confidence
            }

        return result

    # =========================================================
    # DIMENSION DETECTION
    # =========================================================

    def detect_dimensions(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect columns that are likely dimensions.

        Dimensions include:
        - categorical columns
        - text columns
        - boolean columns
        - datetime columns
        - descriptive text columns

        Numeric measures and identifiers are excluded.
        """

        result = {}

        identifiers = self.detect_identifiers()
        measures = self.detect_measures()

        dimension_keywords = [
            "department",
            "status",
            "category",
            "type",
            "class",
            "role",
            "gender",
            "city",
            "state",
            "province",
            "country",
            "location",
            "address",
            "name",
            "first_name",
            "last_name",
            "fullname",
            "full_name"
        ]

        for column in self.df.columns:

            series = self.df[column]

            # -------------------------------------------------
            # Basic type signals
            # -------------------------------------------------

            is_boolean = bool(
                pd.api.types.is_bool_dtype(series)
            )

            is_datetime = bool(
                pd.api.types.is_datetime64_any_dtype(series)
            )

            is_object = bool(
                pd.api.types.is_object_dtype(series)
            )

            is_string = bool(
                pd.api.types.is_string_dtype(series)
            )

            is_category = bool(
                isinstance(series.dtype, pd.CategoricalDtype)
            )

            is_text = bool(
                is_object
                or is_string
                or is_category
            )

            # -------------------------------------------------
            # Cardinality
            # -------------------------------------------------

            non_null = series.dropna()

            if len(non_null) > 0:

                unique_count = int(
                    non_null.nunique()
                )

                cardinality_percentage = (
                    unique_count / len(non_null)
                ) * 100

            else:

                unique_count = 0
                cardinality_percentage = 0.0

            # -------------------------------------------------
            # Semantic name signal
            # -------------------------------------------------

            column_name = str(column).lower()

            name_signal = any(
                keyword in column_name
                for keyword in dimension_keywords
            )

            # -------------------------------------------------
            # Low cardinality
            # -------------------------------------------------

            low_cardinality = bool(
                unique_count > 0
                and unique_count <= 20
            )

            # -------------------------------------------------
            # Existing semantic classifications
            # -------------------------------------------------

            is_identifier = bool(
                identifiers[column]["is_identifier"]
            )

            is_measure = bool(
                measures[column]["is_measure"]
            )

            # -------------------------------------------------
            # Dimension decision
            # -------------------------------------------------

            if is_boolean:

                is_dimension = True
                confidence = 0.90

            elif is_datetime:

                is_dimension = True
                confidence = 0.95

            elif is_text and name_signal:

                is_dimension = True
                confidence = 0.95

            elif is_text and low_cardinality:

                is_dimension = True
                confidence = 0.85

            elif is_text:

                # General descriptive text is also a dimension.
                #
                # Example:
                # RandomText -> dimension
                #
                is_dimension = True
                confidence = 0.80

            else:

                is_dimension = False
                confidence = 0.10

            # -------------------------------------------------
            # Numeric measures and identifiers must not become
            # dimensions.
            # -------------------------------------------------

            if (
                not is_text
                and not is_boolean
                and not is_datetime
            ):

                if is_identifier or is_measure:

                    is_dimension = False
                    confidence = 0.10

            result[column] = {

                "is_dimension": bool(is_dimension),

                "confidence": confidence,

                "boolean": is_boolean,

                "datetime": is_datetime,

                "text": is_text,

                "name_signal": name_signal,

                "unique_count": unique_count,

                "cardinality_percentage": round(
                    cardinality_percentage,
                    2
                ),

                "low_cardinality": low_cardinality,

                "identifier": is_identifier,

                "measure": is_measure
            }

        return result

    # =========================================================
    # BOOLEAN / FLAG DETECTION
    # =========================================================

    def detect_boolean_flags(
        self
    ) -> Dict[str, Dict[str, Any]]:

        """
        Detect boolean and boolean-like columns.
        """

        result = {}

        boolean_names = [
            "is_",
            "has_",
            "flag",
            "active",
            "enabled",
            "verified",
            "approved",
            "deleted"
        ]

        for column in self.df.columns:

            series = self.df[column]

            non_null = series.dropna()

            native_boolean = bool(
                pd.api.types.is_bool_dtype(series)
            )

            unique_values = set(
                str(value).strip().lower()
                for value in non_null.unique()
            )

            boolean_like_values = [
                {"true", "false"},
                {"yes", "no"},
                {"y", "n"},
                {"1", "0"}
            ]

            value_signal = any(
                unique_values.issubset(values)
                and len(unique_values) > 0
                for values in boolean_like_values
            )

            name_signal = any(
                keyword in str(column).lower()
                for keyword in boolean_names
            )

            is_flag = bool(
                native_boolean
                or value_signal
                or (
                    name_signal
                    and len(unique_values) <= 2
                )
            )

            if native_boolean:

                confidence = 0.99

            elif value_signal:

                confidence = 0.95

            elif (
                name_signal
                and len(unique_values) <= 2
            ):

                confidence = 0.85

            else:

                confidence = 0.10

            result[column] = {

                "is_boolean_flag": is_flag,

                "native_boolean": native_boolean,

                "value_signal": value_signal,

                "name_signal": name_signal,

                "unique_values": sorted(
                    list(unique_values)
                ),

                "confidence": confidence
            }

        return result

    # =========================================================
    # DATETIME DETECTION
    # =========================================================

    def detect_datetime_columns(
        self
    ) -> Dict[str, Dict[str, Any]]:

        result = {}

        for column in self.df.columns:

            series = self.df[column]

            native_datetime = bool(
                pd.api.types.is_datetime64_any_dtype(series)
            )

            name = str(column).lower()

            name_signal = any(
                keyword in name
                for keyword in [
                    "date",
                    "time",
                    "timestamp",
                    "created",
                    "updated",
                    "joined",
                    "birth"
                ]
            )

            parseable = False

            if (
                not native_datetime
                and (
                    pd.api.types.is_object_dtype(series)
                    or pd.api.types.is_string_dtype(series)
                )
            ):

                non_null = series.dropna()

                if len(non_null) > 0:

                    converted = pd.to_datetime(
                        non_null,
                        errors="coerce",
                        format="mixed"
                    )

                    valid_ratio = converted.notna().mean()

                    parseable = bool(
                        valid_ratio >= 0.80
                    )

            is_datetime = bool(
                native_datetime
                or parseable
            )

            if native_datetime:

                confidence = 0.99

            elif parseable:

                confidence = 0.95

            elif name_signal:

                confidence = 0.75

            else:

                confidence = 0.10

            result[column] = {

                "is_datetime": is_datetime,

                "native_datetime": native_datetime,

                "parseable_datetime": parseable,

                "name_signal": name_signal,

                "confidence": confidence
            }

        return result

    # =========================================================
    # EMAIL DETECTION
    # =========================================================

    def detect_emails(
        self
    ) -> Dict[str, Dict[str, Any]]:

        result = {}

        pattern = re.compile(
            r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
        )

        for column in self.df.columns:

            values = (
                self.df[column]
                .dropna()
                .astype(str)
            )

            if len(values) == 0:

                match_percentage = 0.0

            else:

                matches = values.apply(
                    lambda value:
                    bool(
                        pattern.match(
                            value.strip()
                        )
                    )
                )

                match_percentage = (
                    matches.mean() * 100
                )

            is_email = bool(
                match_percentage >= 80
            )

            result[column] = {

                "is_email": is_email,

                "email_match_percentage": round(
                    match_percentage,
                    2
                ),

                "confidence": round(
                    match_percentage / 100,
                    2
                )
            }

        return result

    # =========================================================
    # PHONE DETECTION
    # =========================================================

    def detect_phones(
        self
    ) -> Dict[str, Dict[str, Any]]:

        result = {}

        pattern = re.compile(
            r"^\+?[0-9][0-9\s\-\(\)]{6,20}$"
        )

        for column in self.df.columns:

            values = (
                self.df[column]
                .dropna()
                .astype(str)
            )

            if len(values) == 0:

                match_percentage = 0.0

            else:

                matches = values.apply(
                    lambda value:
                    bool(
                        pattern.match(
                            value.strip()
                        )
                    )
                )

                match_percentage = (
                    matches.mean() * 100
                )

            name_signal = bool(
                "phone" in str(column).lower()
                or "mobile" in str(column).lower()
                or "contact" in str(column).lower()
            )

            is_phone = bool(
                match_percentage >= 80
                or (
                    name_signal
                    and match_percentage >= 50
                )
            )

            result[column] = {

                "is_phone": is_phone,

                "phone_match_percentage": round(
                    match_percentage,
                    2
                ),

                "name_signal": name_signal,

                "confidence": round(
                    match_percentage / 100,
                    2
                )
            }

        return result

    # =========================================================
    # URL DETECTION
    # =========================================================

    def detect_urls(
        self
    ) -> Dict[str, Dict[str, Any]]:

        result = {}

        pattern = re.compile(
            r"^(https?|ftp)://[^\s]+$",
            re.IGNORECASE
        )

        for column in self.df.columns:

            values = (
                self.df[column]
                .dropna()
                .astype(str)
            )

            if len(values) == 0:

                match_percentage = 0.0

            else:

                matches = values.apply(
                    lambda value:
                    bool(
                        pattern.match(
                            value.strip()
                        )
                    )
                )

                match_percentage = (
                    matches.mean() * 100
                )

            is_url = bool(
                match_percentage >= 80
            )

            result[column] = {

                "is_url": is_url,

                "url_match_percentage": round(
                    match_percentage,
                    2
                ),

                "confidence": round(
                    match_percentage / 100,
                    2
                )
            }

        return result

    # =========================================================
    # NAME DETECTION
    # =========================================================

    def detect_names(
        self
    ) -> Dict[str, Dict[str, Any]]:

        result = {}

        name_keywords = [
            "name",
            "first_name",
            "last_name",
            "fullname",
            "full_name"
        ]

        for column in self.df.columns:

            column_name = str(column).lower()

            name_signal = any(
                keyword in column_name
                for keyword in name_keywords
            )

            is_text = bool(
                pd.api.types.is_object_dtype(
                    self.df[column]
                )
                or pd.api.types.is_string_dtype(
                    self.df[column]
                )
            )

            is_name = bool(
                name_signal
                and is_text
            )

            confidence = (
                0.95
                if is_name
                else 0.10
            )

            result[column] = {

                "is_name": is_name,

                "name_signal": name_signal,

                "text_column": is_text,

                "confidence": confidence
            }

        return result

    # =========================================================
    # LOCATION / ADDRESS DETECTION
    # =========================================================

    def detect_locations(
        self
    ) -> Dict[str, Dict[str, Any]]:

        result = {}

        location_keywords = [
            "address",
            "city",
            "state",
            "province",
            "country",
            "zip",
            "zipcode",
            "postal",
            "location",
            "street"
        ]

        for column in self.df.columns:

            name = str(column).lower()

            name_signal = any(
                keyword in name
                for keyword in location_keywords
            )

            is_text = bool(
                pd.api.types.is_object_dtype(
                    self.df[column]
                )
                or pd.api.types.is_string_dtype(
                    self.df[column]
                )
            )

            is_location = bool(
                name_signal
                and is_text
            )

            result[column] = {

                "is_location": is_location,

                "name_signal": name_signal,

                "text_column": is_text,

                "confidence": (
                    0.95
                    if is_location
                    else 0.10
                )
            }

        return result

    # =========================================================
    # GENDER DETECTION
    # =========================================================

    def detect_gender(
        self
    ) -> Dict[str, Dict[str, Any]]:

        result = {}

        gender_values = {
            "male",
            "female",
            "m",
            "f",
            "man",
            "woman",
            "non-binary",
            "nonbinary",
            "other"
        }

        for column in self.df.columns:

            values = (
                self.df[column]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
            )

            if len(values) == 0:

                match_percentage = 0.0

            else:

                matches = values.isin(
                    gender_values
                )

                match_percentage = (
                    matches.mean() * 100
                )

            is_gender = bool(
                match_percentage >= 80
            )

            result[column] = {

                "is_gender": is_gender,

                "gender_match_percentage": round(
                    match_percentage,
                    2
                ),

                "confidence": round(
                    match_percentage / 100,
                    2
                )
            }

        return result

    # =========================================================
    # STATUS / CATEGORY DETECTION
    # =========================================================

    def detect_status_categories(
        self
    ) -> Dict[str, Dict[str, Any]]:

        result = {}

        status_keywords = [
            "status",
            "state",
            "type",
            "category",
            "class",
            "level",
            "role"
        ]

        for column in self.df.columns:

            series = self.df[column]

            non_null = series.dropna()

            if len(non_null) == 0:

                unique_count = 0
                cardinality_percentage = 0.0

            else:

                unique_count = int(
                    non_null.nunique()
                )

                cardinality_percentage = (
                    unique_count
                    / len(non_null)
                ) * 100

            name_signal = any(
                keyword in str(column).lower()
                for keyword in status_keywords
            )

            low_cardinality = bool(
                unique_count <= 20
                and unique_count > 0
            )

            is_text = bool(
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
            )

            is_status_category = bool(
                name_signal
                or (
                    low_cardinality
                    and is_text
                )
            )

            if name_signal and low_cardinality:

                confidence = 0.95

            elif name_signal:

                confidence = 0.85

            elif low_cardinality and is_text:

                confidence = 0.75

            else:

                confidence = 0.10

            result[column] = {

                "is_status_or_category":
                    is_status_category,

                "unique_count":
                    unique_count,

                "cardinality_percentage":
                    round(
                        cardinality_percentage,
                        2
                    ),

                "name_signal":
                    name_signal,

                "low_cardinality":
                    low_cardinality,

                "confidence":
                    confidence
            }

        return result

    # =========================================================
    # SEMANTIC CLASSIFICATION
    # =========================================================

    def classify_columns(
        self
    ) -> Dict[str, Dict[str, Any]]:

        data_types = self.detect_data_types()

        identifiers = self.detect_identifiers()

        measures = self.detect_measures()

        dimensions = self.detect_dimensions()

        booleans = self.detect_boolean_flags()

        datetimes = self.detect_datetime_columns()

        emails = self.detect_emails()

        phones = self.detect_phones()

        urls = self.detect_urls()

        names = self.detect_names()

        locations = self.detect_locations()

        genders = self.detect_gender()

        statuses = self.detect_status_categories()

        result = {}

        for column in self.df.columns:

            semantic_types = []

            # -------------------------------------------------
            # Identifier
            # -------------------------------------------------

            if identifiers[column]["is_identifier"]:

                semantic_types.append(
                    "identifier"
                )

            # -------------------------------------------------
            # Measure
            # -------------------------------------------------

            if measures[column]["is_measure"]:

                semantic_types.append(
                    "measure"
                )

            # -------------------------------------------------
            # Dimension
            # -------------------------------------------------

            if dimensions[column]["is_dimension"]:

                semantic_types.append(
                    "dimension"
                )

            # -------------------------------------------------
            # Boolean
            # -------------------------------------------------

            if booleans[column]["is_boolean_flag"]:

                semantic_types.append(
                    "boolean_flag"
                )

            # -------------------------------------------------
            # Datetime
            # -------------------------------------------------

            if datetimes[column]["is_datetime"]:

                semantic_types.append(
                    "datetime"
                )

            # -------------------------------------------------
            # Email
            # -------------------------------------------------

            if emails[column]["is_email"]:

                semantic_types.append(
                    "email"
                )

            # -------------------------------------------------
            # Phone
            # -------------------------------------------------

            if phones[column]["is_phone"]:

                semantic_types.append(
                    "phone"
                )

            # -------------------------------------------------
            # URL
            # -------------------------------------------------

            if urls[column]["is_url"]:

                semantic_types.append(
                    "url"
                )

            # -------------------------------------------------
            # Name
            # -------------------------------------------------

            if names[column]["is_name"]:

                semantic_types.append(
                    "name"
                )

            # -------------------------------------------------
            # Location
            # -------------------------------------------------

            if locations[column]["is_location"]:

                semantic_types.append(
                    "location"
                )

            # -------------------------------------------------
            # Gender
            # -------------------------------------------------

            if genders[column]["is_gender"]:

                semantic_types.append(
                    "gender"
                )

            # -------------------------------------------------
            # Status / category
            # -------------------------------------------------

            if statuses[column][
                "is_status_or_category"
            ]:

                semantic_types.append(
                    "status_or_category"
                )

            # -------------------------------------------------
            # Unknown
            # -------------------------------------------------

            if not semantic_types:

                semantic_types.append(
                    "unknown"
                )

            result[column] = {

                "data_type":
                    data_types[column],

                "semantic_types":
                    semantic_types,

                "primary_semantic_type":
                    semantic_types[0]
            }

        return result

    # =========================================================
    # CATEGORY CONSISTENCY
    # =========================================================

    def category_consistency(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze consistency of categorical values across the dataset.

        Checks:
        - Case consistency (e.g., "Active" vs "active")
        - Whitespace consistency
        - Synonym detection (e.g., "USA", "US", "United States")
        - Abbreviation consistency
        """

        results = {}

        for column in self.df.columns:
            series = self.df[column].dropna()

            if len(series) == 0:
                results[column] = {
                    "consistency_score": 100.0,
                    "issues": {}
                }
                continue

            if not (pd.api.types.is_object_dtype(series) or
                    isinstance(series.dtype, pd.CategoricalDtype) or
                    pd.api.types.is_string_dtype(series)):
                results[column] = {
                    "consistency_score": 100.0,
                    "issues": {"note": "Not a categorical column"}
                }
                continue

            str_series = series.astype(str).str.strip()
            unique_values = str_series.unique()

            issues = {}

            # Case inconsistency
            lower_values = str_series.str.lower()
            unique_lower = lower_values.nunique()
            unique_original = str_series.nunique()

            if unique_lower < unique_original:
                # Find case variants
                case_groups = lower_values.groupby(lower_values).apply(lambda x: x.index.tolist())
                variants = []
                for val in case_groups.index:
                    original_vals = str_series[lower_values == val].unique()
                    if len(original_vals) > 1:
                        variants.append({
                            "normalized": val,
                            "variants": list(original_vals),
                            "counts": {v: int((str_series == v).sum()) for v in original_vals}
                        })
                issues["case_inconsistency"] = {
                    "variant_groups": variants,
                    "total_variants": len(variants)
                }

            # Whitespace inconsistency
            ws_variants = {}
            for val in unique_values:
                stripped = val.strip()
                if stripped != val:
                    if stripped not in ws_variants:
                        ws_variants[stripped] = []
                    ws_variants[stripped].append(val)

            if ws_variants:
                issues["whitespace_inconsistency"] = {
                    "groups": {k: v for k, v in ws_variants.items()},
                    "total_affected": sum(len(v) for v in ws_variants.values())
                }

            # Potential synonyms (using string similarity)
            if len(unique_values) <= 100:  # Limit for performance
                from difflib import SequenceMatcher
                synonym_groups = []
                checked = set()

                for i, val1 in enumerate(unique_values):
                    if val1 in checked:
                        continue
                    group = [val1]
                    for val2 in unique_values[i+1:]:
                        if val2 in checked:
                            continue
                        similarity = SequenceMatcher(None, val1.lower(), val2.lower()).ratio()
                        if similarity > 0.85:  # High similarity threshold
                            group.append(val2)
                            checked.add(val2)
                    if len(group) > 1:
                        synonym_groups.append(group)
                        checked.update(group)

                if synonym_groups:
                    issues["potential_synonyms"] = {
                        "groups": synonym_groups
                    }

            # Abbreviation detection
            abbrev_pattern = re.compile(r'^[A-Z]{2,5}$')
            abbrevs = [v for v in unique_values if abbrev_pattern.match(v)]
            if abbrevs:
                issues["abbreviations_detected"] = {
                    "count": len(abbrevs),
                    "values": abbrevs[:20]
                }

            # Calculate consistency score
            penalty = 0
            if "case_inconsistency" in issues:
                penalty += issues["case_inconsistency"]["total_variants"] * 5
            if "whitespace_inconsistency" in issues:
                penalty += issues["whitespace_inconsistency"]["total_affected"] * 3
            if "potential_synonyms" in issues:
                penalty += len(issues["potential_synonyms"]["groups"]) * 10

            consistency_score = max(0, 100 - penalty)

            results[column] = {
                "consistency_score": round(consistency_score, 2),
                "unique_values": int(unique_original),
                "issues": issues
            }

        return results

    # =========================================================
    # STANDARDIZATION DETECTUTION
    # =========================================================

    def standardization_detection(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect columns that need standardization.

        Identifies:
        - Date format inconsistencies
        - Numeric format inconsistencies (decimal separators, thousands separators)
        - Phone number format variations
        - Address format variations
        - Case format (Title Case, UPPER, lower)
        """

        results = {}

        for column in self.df.columns:
            series = self.df[column].dropna()

            if len(series) == 0:
                results[column] = {
                    "needs_standardization": False,
                    "issues": []
                }
                continue

            issues = []

            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                str_series = series.astype(str)

                # Date format detection
                date_formats = set()
                date_patterns = [
                    (r'^\d{4}-\d{2}-\d{2}$', 'YYYY-MM-DD'),
                    (r'^\d{2}/\d{2}/\d{4}$', 'MM/DD/YYYY'),
                    (r'^\d{2}-\d{2}-\d{4}$', 'DD-MM-YYYY'),
                    (r'^\d{2}\.\d{2}\.\d{4}$', 'DD.MM.YYYY'),
                    (r'^\d{4}/\d{2}/\d{2}$', 'YYYY/MM/DD'),
                    (r'^\d{1,2}/\d{1,2}/\d{2,4}$', 'M/D/YY or MM/DD/YYYY'),
                ]

                for pattern, fmt in date_patterns:
                    matches = str_series.str.match(pattern).sum()
                    if matches > 0:
                        date_formats.add(fmt)

                if len(date_formats) > 1:
                    issues.append({
                        "type": "date_format_inconsistency",
                        "formats_detected": list(date_formats),
                        "message": f"Multiple date formats detected: {', '.join(date_formats)}"
                    })

                # Numeric format (decimal/thousands separators)
                num_patterns = {
                    "dot_decimal": r'^\d+\.\d+$',
                    "comma_decimal": r'^\d+,\d+$',
                    "space_thousands": r'^\d{1,3}( \d{3})*(\.\d+)?$',
                    "comma_thousands": r'^\d{1,3}(,\d{3})*(\.\d+)?$',
                    "dot_thousands": r'^\d{1,3}(\.\d{3})*(,\d+)?$'
                }

                detected_num_formats = []
                for fmt_name, pattern in num_patterns.items():
                    if str_series.str.match(pattern).any():
                        detected_num_formats.append(fmt_name)

                if len(detected_num_formats) > 1:
                    issues.append({
                        "type": "numeric_format_inconsistency",
                        "formats_detected": detected_num_formats,
                        "message": f"Multiple numeric formats: {', '.join(detected_num_formats)}"
                    })

                # Phone format variations
                phone_patterns = {
                    "international": r'^\+\d{1,3}[\s-]?\d{3,4}[\s-]?\d{3,4}$',
                    "us_parens": r'^\(\d{3}\)\s?\d{3}-\d{4}$',
                    "us_dashes": r'^\d{3}-\d{3}-\d{4}$',
                    "us_dots": r'^\d{3}\.\d{3}\.\d{4}$',
                    "plain": r'^\d{10}$'
                }

                detected_phone_formats = []
                for fmt_name, pattern in phone_patterns.items():
                    if str_series.str.match(pattern).any():
                        detected_phone_formats.append(fmt_name)

                if len(detected_phone_formats) > 1:
                    issues.append({
                        "type": "phone_format_inconsistency",
                        "formats_detected": detected_phone_formats,
                        "message": f"Multiple phone formats: {', '.join(detected_phone_formats)}"
                    })

                # Case format
                case_types = set()
                if (str_series == str_series.str.title()).all():
                    case_types.add("Title Case")
                if (str_series == str_series.str.upper()).all():
                    case_types.add("UPPER")
                if (str_series == str_series.str.lower()).all():
                    case_types.add("lower")
                if len(case_types) > 1:
                    issues.append({
                        "type": "case_format_inconsistency",
                        "formats_detected": list(case_types),
                        "message": f"Mixed case formats: {', '.join(case_types)}"
                    })

            elif pd.api.types.is_numeric_dtype(series):
                # Check for mixed precision
                decimals = series.apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
                if decimals.nunique() > 3:
                    issues.append({
                        "type": "decimal_precision_inconsistency",
                        "unique_precisions": int(decimals.nunique()),
                        "message": "Inconsistent decimal precision"
                    })

            results[column] = {
                "needs_standardization": len(issues) > 0,
                "issues": issues
            }

        return results

    # =========================================================
    # UNIT CONSISTENCY DETECTION
    # =========================================================

    def unit_consistency(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect unit inconsistencies in numeric columns.

        Looks for:
        - Currency symbols ($, €, £, etc.)
        - Measurement units (kg, lb, m, ft, etc.)
        - Percentage vs decimal
        - Time units (seconds, minutes, hours)
        """

        results = {}

        # Common unit patterns
        unit_patterns = {
            "currency": r'[$€£¥₹]',
            "percentage": r'%',
            "weight_kg": r'\bkg\b',
            "weight_lb": r'\blb\b|\blbs\b',
            "length_m": r'\bm\b',
            "length_cm": r'\bcm\b',
            "length_ft": r'\bft\b|\'',
            "length_in": r'\bin\b|\"',
            "temperature_c": r'°C|\bC\b',
            "temperature_f": r'°F|\bF\b',
            "time_s": r'\bs\b|\bsec\b',
            "time_min": r'\bmin\b',
            "time_h": r'\bh\b|\bhr\b',
            "data_bytes": r'\bB\b|\bbytes\b',
            "data_kb": r'\bKB\b|\bKb\b',
            "data_mb": r'\bMB\b|\bMb\b',
            "data_gb": r'\bGB\b|\bGb\b',
        }

        for column in self.df.columns:
            series = self.df[column].dropna()

            if len(series) == 0:
                results[column] = {
                    "consistent": True,
                    "units_detected": [],
                    "issues": []
                }
                continue

            issues = []
            detected_units = set()

            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                str_series = series.astype(str)

                for unit_name, pattern in unit_patterns.items():
                    matches = str_series.str.contains(pattern, case=False, regex=True)
                    if matches.any():
                        detected_units.add(unit_name)
                        # Check if ALL values have this unit
                        if not matches.all():
                            issues.append({
                                "unit": unit_name,
                                "coverage": round((matches.sum() / len(series)) * 100, 2),
                                "message": f"Unit '{unit_name}' not consistent across all values"
                            })

                # Check for mixed units in same column
                if len(detected_units) > 1:
                    issues.append({
                        "type": "mixed_units",
                        "units": list(detected_units),
                        "message": f"Multiple units detected in same column: {', '.join(detected_units)}"
                    })

            elif pd.api.types.is_numeric_dtype(series):
                # For numeric columns, check column name for unit hints
                col_lower = column.lower()
                name_units = []
                unit_keywords = {
                    "currency": ["amount", "price", "cost", "salary", "revenue", "budget"],
                    "percentage": ["rate", "pct", "percent", "percentage", "ratio"],
                    "weight": ["weight", "mass"],
                    "length": ["height", "width", "length", "distance", "depth"],
                    "temperature": ["temp", "temperature"],
                    "time": ["duration", "time", "latency", "response_time"],
                    "data": ["size", "memory", "storage", "disk"]
                }

                for unit, keywords in unit_keywords.items():
                    if any(kw in col_lower for kw in keywords):
                        name_units.append(unit)

                if len(name_units) > 1:
                    issues.append({
                        "type": "ambiguous_unit_in_name",
                        "possible_units": name_units,
                        "message": f"Column name suggests multiple units: {', '.join(name_units)}"
                    })

            results[column] = {
                "consistent": len(issues) == 0,
                "units_detected": list(detected_units) if detected_units else (
                    name_units if 'name_units' in locals() and name_units else []
                ),
                "issues": issues
            }

        return results

    # =========================================================
    # PII / SENSITIVE DATA DETECTION
    # =========================================================

    def detect_pii(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect personally identifiable information (PII) and sensitive data.

        Detects:
        - Email addresses
        - Phone numbers
        - Social Security Numbers (US)
        - Credit card numbers
        - IP addresses
        - Names (first/last)
        - Addresses
        - Date of birth
        - Driver's license
        - Passport numbers
        - Bank account numbers
        """

        results = {}

        # PII patterns
        pii_patterns = {
            "ssn": {
                "pattern": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
                "description": "US Social Security Number"
            },
            "credit_card": {
                "pattern": re.compile(r'\b(?:\d{4}[\s-]?){3}\d{4}\b'),
                "description": "Credit Card Number"
            },
            "ipv4": {
                "pattern": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
                "description": "IPv4 Address"
            },
            "ipv6": {
                "pattern": re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),
                "description": "IPv6 Address"
            },
            "passport_us": {
                "pattern": re.compile(r'\b[A-Z]{1}\d{8}\b'),
                "description": "US Passport Number"
            },
            "drivers_license_ca": {
                "pattern": re.compile(r'\b[Dd]\d{7}\b'),
                "description": "California Driver's License"
            },
            "bank_account": {
                "pattern": re.compile(r'\b\d{8,17}\b'),
                "description": "Bank Account Number (generic)"
            },
            "ein": {
                "pattern": re.compile(r'\b\d{2}-\d{7}\b'),
                "description": "Employer Identification Number"
            },
            "vin": {
                "pattern": re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b'),
                "description": "Vehicle Identification Number"
            }
        }

        for column in self.df.columns:
            series = self.df[column].dropna()

            if len(series) == 0:
                results[column] = {
                    "pii_detected": False,
                    "pii_types": [],
                    "confidence": 0.0
                }
                continue

            str_series = series.astype(str)
            pii_types = []
            total_confidence = 0.0

            # Check patterns
            for pii_type, info in pii_patterns.items():
                matches = str_series.apply(lambda x: bool(info["pattern"].search(x)))
                match_pct = (matches.sum() / len(series)) * 100

                if match_pct > 50:  # Majority match
                    pii_types.append({
                        "type": pii_type,
                        "description": info["description"],
                        "match_percentage": round(match_pct, 2),
                        "confidence": round(match_pct / 100, 2)
                    })
                    total_confidence += match_pct / 100

            # Check for names (using existing name detection)
            name_result = self.detect_names().get(column, {})
            if name_result.get("is_name", False):
                pii_types.append({
                    "type": "person_name",
                    "description": "Person Name",
                    "match_percentage": 90.0,
                    "confidence": 0.9
                })
                total_confidence += 0.9

            # Check for dates of birth
            dob_keywords = ["dob", "birth", "birthdate", "date_of_birth"]
            if any(kw in column.lower() for kw in dob_keywords):
                pii_types.append({
                    "type": "date_of_birth",
                    "description": "Date of Birth",
                    "match_percentage": 95.0,
                    "confidence": 0.95
                })
                total_confidence += 0.95

            # Check for addresses (using existing location detection)
            location_result = self.detect_locations().get(column, {})
            if location_result.get("is_location", False):
                pii_types.append({
                    "type": "address",
                    "description": "Physical Address",
                    "match_percentage": 85.0,
                    "confidence": 0.85
                })
                total_confidence += 0.85

            avg_confidence = total_confidence / len(pii_types) if pii_types else 0.0

            results[column] = {
                "pii_detected": len(pii_types) > 0,
                "pii_types": pii_types,
                "confidence": round(avg_confidence, 2)
            }

        return results

    # =========================================================
    # COMPLETE PROFILE
    # =========================================================

    def profile(
        self
    ) -> Dict[str, Any]:

        return {

            "data_types":
                self.detect_data_types(),

            "identifiers":
                self.detect_identifiers(),

            "measures":
                self.detect_measures(),

            "dimensions":
                self.detect_dimensions(),

            "boolean_flags":
                self.detect_boolean_flags(),

            "datetime_columns":
                self.detect_datetime_columns(),

            "emails":
                self.detect_emails(),

            "phones":
                self.detect_phones(),

            "urls":
                self.detect_urls(),

            "names":
                self.detect_names(),

            "locations":
                self.detect_locations(),

            "gender":
                self.detect_gender(),

            "status_categories":
                self.detect_status_categories(),

            "column_classification":
                self.classify_columns(),

            "category_consistency":
                self.category_consistency(),

            "standardization_detection":
                self.standardization_detection(),

            "unit_consistency":
                self.unit_consistency(),

            "pii_detection":
                self.detect_pii()
        }


# =============================================================
# CONVENIENCE FUNCTION
# =============================================================

def profile_semantics(
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Convenience function for semantic profiling.
    """

    profiler = SemanticProfiler(df)

    return profiler.profile()