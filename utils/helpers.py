"""
Utility functions for the harmonization pipeline
"""
import re
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, List, Tuple


def extract_trial_from_filename(filename: str) -> Optional[str]:
    """
    Extract NCT trial ID from filename using regex.
    Returns first match of NCT followed by 8 digits.
    """
    match = re.search(r'(NCT\d{8})', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def decode_bytes(value: Any) -> Any:
    """
    Decode bytes to string if needed.
    Handles SAS byte strings that pandas reads.
    """
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8').strip()
        except UnicodeDecodeError:
            return value.decode('latin-1').strip()
    elif isinstance(value, str):
        return value.strip()
    return value


def decode_dataframe_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decode all byte strings in a DataFrame to regular strings.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(decode_bytes)
    return df


def to_mixed_case(value: str) -> str:
    """
    Convert string to mixed case (title case).
    Handles special cases like abbreviations and preserves lowercase for
    prepositions/articles that shouldn't be capitalized.
    """
    if not value or pd.isna(value):
        return value

    value = str(value).strip()

    # Don't modify if it looks like an identifier or code
    if re.match(r'^[A-Z0-9_-]+$', value) and len(value) <= 10:
        return value

    # Words that should remain lowercase (unless first word)
    lowercase_words = {'or', 'and', 'of', 'the', 'in', 'on', 'at', 'to', 'for',
                       'with', 'than', 'as', 'by', 'from', 'into', 'nor', 'but'}

    # First apply title case
    titled = value.title()

    # Split into words and fix prepositions (keep them lowercase unless first word)
    words = titled.split()
    result_words = []
    for i, word in enumerate(words):
        # Check if lowercase version is in our exception list
        word_lower = word.lower()
        # Handle words with parentheses like "(Other"
        clean_word = word_lower.strip('()')

        if clean_word in lowercase_words and i > 0:
            # Preserve any leading/trailing punctuation but lowercase the word
            if word.startswith('('):
                result_words.append('(' + word_lower.strip('('))
            else:
                result_words.append(word_lower)
        else:
            result_words.append(word)

    return ' '.join(result_words)


def normalize_whitespace(value: str) -> str:
    """
    Trim and normalize whitespace in a string.
    """
    if not value or pd.isna(value):
        return value
    return ' '.join(str(value).split())


def sas_date_to_iso(sas_date: float, origin: str = "1960-01-01") -> Optional[str]:
    """
    Convert SAS numeric date to ISO 8601 string (YYYY-MM-DD).
    SAS dates are days since 1960-01-01.
    """
    if pd.isna(sas_date):
        return None

    try:
        base = datetime.strptime(origin, "%Y-%m-%d")
        result = base + timedelta(days=int(sas_date))
        return result.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None


def sas_datetime_to_iso(sas_datetime: float, origin: str = "1960-01-01") -> Optional[str]:
    """
    Convert SAS numeric datetime to ISO 8601 date string.
    SAS datetimes are seconds since 1960-01-01 00:00:00.
    Returns date only, removes time component.
    """
    if pd.isna(sas_datetime):
        return None

    try:
        base = datetime.strptime(origin, "%Y-%m-%d")
        result = base + timedelta(seconds=int(sas_datetime))
        return result.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None


def is_valid_iso_date(date_str: str) -> bool:
    """
    Check if string is valid ISO 8601 date format.
    Accepts YYYY, YYYY-MM, or YYYY-MM-DD.
    """
    if not date_str or pd.isna(date_str):
        return False

    date_str = str(date_str).strip()

    # Full date
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    # Year-month
    if re.match(r'^\d{4}-\d{2}$', date_str):
        try:
            datetime.strptime(date_str, "%Y-%m")
            return True
        except ValueError:
            return False

    # Year only
    if re.match(r'^\d{4}$', date_str):
        year = int(date_str)
        return 1900 <= year <= 2100

    return False


def is_full_date(date_str: str) -> bool:
    """
    Check if date string is a full date (YYYY-MM-DD).
    """
    if not date_str or pd.isna(date_str):
        return False
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', str(date_str).strip()))


def calculate_age(birth_date: str, reference_date: str) -> Optional[int]:
    """
    Calculate age in years from birth date to reference date.
    Both must be full ISO dates (YYYY-MM-DD).
    """
    if not is_full_date(birth_date) or not is_full_date(reference_date):
        return None

    try:
        birth = datetime.strptime(birth_date, "%Y-%m-%d")
        ref = datetime.strptime(reference_date, "%Y-%m-%d")

        age = ref.year - birth.year

        # Adjust if birthday hasn't occurred yet in reference year
        if (ref.month, ref.day) < (birth.month, birth.day):
            age -= 1

        return age if age >= 0 else None
    except ValueError:
        return None


def find_column_match(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find first matching column from candidate list (case-insensitive).
    Also checks for columns ending with the candidate name.
    """
    df_cols_upper = {col.upper(): col for col in df.columns}

    for candidate in candidates:
        candidate_upper = candidate.upper()

        # Exact match (case-insensitive)
        if candidate_upper in df_cols_upper:
            return df_cols_upper[candidate_upper]

        # Suffix match (e.g., RSUBJID matches column ending in SUBJID)
        for col_upper, col_original in df_cols_upper.items():
            if col_upper.endswith(candidate_upper):
                return col_original

    return None


def validate_nct_format(trial_id: str) -> bool:
    """
    Validate that trial ID matches NCT format (NCT + 8 digits).
    """
    if not trial_id or pd.isna(trial_id):
        return False
    return bool(re.match(r'^NCT\d{8}$', str(trial_id).strip(), re.IGNORECASE))


def get_unique_values_sample(series: pd.Series, max_values: int = 5) -> str:
    """
    Get a sample of unique values from a series for QC reporting.
    """
    unique_vals = series.dropna().unique()[:max_values]
    return ", ".join(str(v) for v in unique_vals)
