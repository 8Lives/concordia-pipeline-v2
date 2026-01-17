"""
Configuration module for Concordia Pipeline v2.
"""
from config.schema import (
    OUTPUT_SCHEMA,
    REQUIRED_VARIABLES,
    CONDITIONAL_REQUIRED,
    SOURCE_PRIORITY,
    SEX_DECODE,
    RACE_NORMALIZE,
    COUNTRY_CODES,
    QC_ISSUES,
    SAS_DATE_ORIGIN
)

__all__ = [
    'OUTPUT_SCHEMA',
    'REQUIRED_VARIABLES',
    'CONDITIONAL_REQUIRED',
    'SOURCE_PRIORITY',
    'SEX_DECODE',
    'RACE_NORMALIZE',
    'COUNTRY_CODES',
    'QC_ISSUES',
    'SAS_DATE_ORIGIN'
]
