"""
DM Domain Output Schema and Harmonization Rules (v2.1)
Based on DM_Harmonization_Spec_v1.4

Changes from v2:
- Added PID_A and more SUBJID synonyms for sponsor-specific naming (NCT00699374)
- Added column exclusion patterns to prevent false SUBJID matches (e.g., FORMID)
- Added heuristic SUBJID detection for non-standard column names

Changes from v1:
- Enhanced SUBJID source priority with more synonyms
- Additional country codes
- Better SITEID handling
"""

# Output schema - ordered list of variables
OUTPUT_SCHEMA = [
    "TRIAL", "SUBJID", "SEX", "RACE", "AGE", "AGEU", "AGEGP", "ETHNIC",
    "COUNTRY", "SITEID", "STUDYID", "USUBJID", "ARMCD", "ARM",
    "BRTHDTC", "RFSTDTC", "RFENDTC", "DOMAIN"
]

# Required variables (must have value or trigger QC flag)
REQUIRED_VARIABLES = ["TRIAL", "SUBJID", "SEX", "RACE"]

# Conditional required (AGE required if present/derivable, else AGEGP)
CONDITIONAL_REQUIRED = ["AGE", "AGEGP"]

# Source priority mappings - ordered list of source column candidates
# ENHANCED in v2.1: Added more SUBJID synonyms and PID variants
SOURCE_PRIORITY = {
    "TRIAL": ["TRIAL"],  # Primarily extracted from filename

    # ENHANCED: Many more SUBJID synonyms - common in various trial data formats
    # NOTE: Order matters - more specific patterns should come first
    "SUBJID": [
        "SUBJID", "RSUBJID", "RUSUBJID",  # Standard CDISC
        "SUBJ", "SUBJECT", "SUBJECTID", "SUBJECT_ID",  # Common alternatives
        "PATID", "PATIENTID", "PATIENT_ID", "PATIENT",  # Patient-based naming
        "PTNO", "PT_NO", "PTID", "PT_ID",  # Abbreviated patient
        "SUBJECTNO", "SUBJECT_NO", "SUBJNO", "SUBJ_NO",  # Subject number variants
        "SCREENID", "SCREEN_ID", "SCRNO",  # Screening IDs
        "ENROLID", "ENROLL_ID", "ENROLLID",  # Enrollment IDs
        "PARTICIPANTID", "PARTICIPANT_ID",  # Participant IDs
        # v2.1 additions for sponsor-specific naming conventions
        "PID", "PID_A", "PID_B", "PID_C",  # Pfizer-style Patient ID variants
        "PATNO", "PAT_NO", "PAT_ID",  # Patient number variants
        "SUBNO", "SUB_NO", "SUB_ID",  # Subject number short forms
        "RANDNO", "RAND_NO", "RANDOMNO",  # Randomization numbers
        "SCRNNO", "SCRN_NO", "SCREENNO",  # Screening number variants
        "SBJID", "SBJ_ID", "SBJNO",  # Alternative abbreviations
    ],

    "SEX": ["SEX", "SEXC", "GENDER", "SEXCD"],
    "RACE": ["RACE", "RACESC", "RACECD", "RACEN"],
    "AGE": ["AGE", "APTS", "AGEATSCREEN", "AGEATRAND"],
    "AGEU": ["AGEU", "AGEUNITS", "AGEUNIT", "AGEUNITU"],
    "AGEGP": ["AGEGP", "AGEGRP", "AGE_GROUP", "AGE_BAND", "AGE_CAT", "AGEGR1", "AGEGR2", "AGEGR1N"],
    "ETHNIC": ["ETHNIC", "ETHGRP", "ETHNICGRP", "ETHNICN"],
    "COUNTRY": ["COUNTRY", "CNTRY", "COUNTRYC"],

    # ENHANCED: More SITEID synonyms
    "SITEID": ["SITEID", "RSITEID", "SITE", "SITENO", "SITE_ID", "CENT", "CENTER", "CENTRENO", "INVSITE", "INVID"],

    "STUDYID": ["STUDYID", "STUDY", "PROTNO", "PROTOCOL", "PROT_ID"],
    "USUBJID": ["USUBJID", "RUSUBJID"],
    "ARMCD": ["ARMCD", "TRTARMCD", "TRTCODE", "TRTN", "TRTCDN"],
    "ARM": ["ARM", "TRTARM", "TRTLONG", "ACTTRT", "TRT", "TREATMENT", "LI_ARM"],
    "BRTHDTC": ["BRTHDTC", "BIRTHDT", "DOB", "BRTHDT"],
    "RFSTDTC": ["RFSTDTC", "RFXSTDTC", "RANDDTC", "RANDDT", "STARTDT"],
    "RFENDTC": ["RFENDTC", "RFXENDTC", "ENDDT", "LSTVISDT"],
    "DOMAIN": []  # Constant 'DM'
}

# SEX code mappings
SEX_DECODE = {
    "1": "Male", "2": "Female",
    "M": "Male", "F": "Female",
    "U": "Unknown", "UNK": "Unknown",
    "MALE": "Male", "FEMALE": "Female",
    "UNKNOWN": "Unknown"
}

# RACE normalization mappings (value -> standardized)
RACE_NORMALIZE = {
    "WHITE": "Caucasian",
    "WHITE OR CAUCASIAN": "Caucasian",
    "CAUCASIAN": "Caucasian",
    "BLACK": "Black or African American",
    "BLACK OR AFRICAN AMERICAN": "Black or African American",
    "AFRICAN AMERICAN": "Black or African American",
    "ASIAN": "Asian",
    "ORIENTAL": "Asian",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": "Native Hawaiian or Other Pacific Islander",
    "AMERICAN INDIAN OR ALASKA NATIVE": "American Indian or Alaska Native",
    "OTHER": "Other",
    "MULTIPLE": "Multiple",
}

# Common ISO country code expansions (ENHANCED in v2)
COUNTRY_CODES = {
    "US": "United States", "USA": "United States",
    "CA": "Canada", "CAN": "Canada",
    "GB": "United Kingdom", "UK": "United Kingdom", "GBR": "United Kingdom",
    "DE": "Germany", "DEU": "Germany",
    "FR": "France", "FRA": "France",
    "JP": "Japan", "JPN": "Japan",
    "AU": "Australia", "AUS": "Australia",
    "BE": "Belgium", "BEL": "Belgium",
    "FI": "Finland", "FIN": "Finland",
    "CZ": "Czech Republic", "CZE": "Czech Republic",
    "KR": "Korea, Republic of", "KOR": "Korea, Republic of",
    "IN": "India", "IND": "India",
    "SE": "Sweden", "SWE": "Sweden",
    "PL": "Poland", "POL": "Poland",
    "DK": "Denmark", "DNK": "Denmark",
    "TW": "Taiwan", "TWN": "Taiwan",
    "NL": "Netherlands", "NLD": "Netherlands",
    "BR": "Brazil", "BRA": "Brazil",
    "HU": "Hungary", "HUN": "Hungary",
    "MX": "Mexico", "MEX": "Mexico",
    "RU": "Russia", "RUS": "Russia",
    "CH": "Switzerland", "CHE": "Switzerland",
    "CN": "China", "CHN": "China",
    "IT": "Italy", "ITA": "Italy",
    "SG": "Singapore", "SGP": "Singapore",
    "ES": "Spain", "ESP": "Spain",
    "NZ": "New Zealand", "NZL": "New Zealand",
    "AT": "Austria", "AUT": "Austria",
    "ZA": "South Africa", "ZAF": "South Africa",
    "AR": "Argentina", "ARG": "Argentina",
    "CL": "Chile", "CHL": "Chile",
    "CO": "Colombia", "COL": "Colombia",
    "IL": "Israel", "ISR": "Israel",
    "NO": "Norway", "NOR": "Norway",
    "PT": "Portugal", "PRT": "Portugal",
    "IE": "Ireland", "IRL": "Ireland",
    "TH": "Thailand", "THA": "Thailand",
    "MY": "Malaysia", "MYS": "Malaysia",
    "PH": "Philippines", "PHL": "Philippines",
    "VN": "Vietnam", "VNM": "Vietnam",
    "ID": "Indonesia", "IDN": "Indonesia",
    "HK": "Hong Kong", "HKG": "Hong Kong",
}

# QC issue types
QC_ISSUES = {
    "TRIAL_MISSING_OR_INVALID": "TRIAL is missing or does not match NCT format",
    "DUPLICATE_SUBJECT": "Duplicate (TRIAL, SUBJID) combination",
    "MISSING_REQUIRED_VALUE": "Required variable has missing value",
    "MISSING_AGE_AND_AGEGP": "Both AGE and AGEGP are missing",
    "CODED_VALUE_NO_DICTIONARY": "Coded value requires dictionary but none provided",
    "DATE_INVALID": "Date value is not parseable",
    "DATE_ORDER_INVALID": "RFENDTC is before RFSTDTC",
    "AGE_INCONSISTENT_WITH_DATES": "Derived AGE differs from provided AGE by >2 years",
    "AGE_DERIVATION_SKIPPED_PARTIAL_DATE": "AGE derivation blocked due to partial dates",
    "COUNTRY_CODE_UNMAPPED": "Country code could not be expanded",
    "AGEU_UNMAPPED": "Age unit not mappable",
    # v2.1 additions for mapping failure detection
    "SUBJID_MAPPING_SUSPECT": "SUBJID mapping may be incorrect - low uniqueness or all rows affected",
    "SUBJID_NOT_FOUND": "No subject identifier column could be identified in source data",
    "COLUMN_MAPPING_HEURISTIC": "Column mapped using heuristic (uniqueness-based) rather than name matching",
}

# Columns to EXCLUDE from SUBJID matching even if they end in "ID"
# These are known non-subject identifier columns
SUBJID_EXCLUSION_PATTERNS = [
    "FORMID", "FORM_ID",  # Form identifiers
    "VISITID", "VISIT_ID",  # Visit identifiers
    "PROTID", "PROT_ID", "PROTOCOLID", "PROTOCOL_ID",  # Protocol identifiers (use for STUDYID)
    "STUDYID", "STUDY_ID",  # Study identifiers (separate output field)
    "SITEID", "SITE_ID",  # Site identifiers (separate output field)
    "ARMID", "ARM_ID",  # Arm identifiers
    "DOMAINID", "DOMAIN_ID",  # Domain identifiers
    "TESTID", "TEST_ID",  # Test identifiers
    "PARAMID", "PARAM_ID",  # Parameter identifiers
    "SEQID", "SEQ_ID",  # Sequence identifiers
    "RECORDID", "RECORD_ID",  # Record identifiers
    "ROWID", "ROW_ID",  # Row identifiers
]

# SAS date origin (days since 1960-01-01)
SAS_DATE_ORIGIN = "1960-01-01"
