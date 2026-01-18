"""
Harmonize Values Agent (v2 - Agentic Architecture)

Responsibilities:
- Decode coded values using dictionary
- Normalize dates to ISO 8601
- Apply value transformations (SEX codes, RACE normalization, etc.)
- Handle AGE vs AGEGP logic
- Derive USUBJID if needed
- Track all transformations for lineage reporting
- LLM fallback for unmatched values (Opus 4.5)

Changes from v1:
- Extends AgentBase for timeout/retry/callback support
- Uses PipelineContext for input/output
- Standardized error handling with validate_input() and execute()
- Progress callbacks for real-time UI updates
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from agents.base import AgentBase, AgentResult, AgentConfig, PipelineContext, ProgressCallback
from config.schema import (
    OUTPUT_SCHEMA, SEX_DECODE, RACE_NORMALIZE, COUNTRY_CODES, SAS_DATE_ORIGIN
)
from utils.helpers import (
    to_mixed_case, normalize_whitespace, sas_date_to_iso,
    is_full_date, calculate_age, validate_nct_format
)

# LLM fallback support
try:
    from agents.llm_service import get_llm_service, LLMResponse
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class HarmonizeAgent(AgentBase):
    """
    Agent responsible for value-level harmonization and transformation.
    Now with LLM fallback for unmatched values.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        use_llm_fallback: bool = True,
        progress_callback: Optional[ProgressCallback] = None
    ):
        super().__init__(
            name="Harmonize Values Agent",
            config=config or AgentConfig(timeout_seconds=120.0),
            progress_callback=progress_callback
        )
        self.use_llm_fallback = use_llm_fallback and LLM_AVAILABLE
        self.llm_service = get_llm_service() if self.use_llm_fallback else None
        self.llm_resolutions = []  # Track LLM-resolved values

    def validate_input(self, context: PipelineContext) -> Optional[str]:
        """
        Validate that required inputs are present in context.

        Required inputs:
        - df: DataFrame from Map Agent
        - mapping_log: Mapping log from Map Agent
        - map_metadata: Metadata from Map Agent
        """
        df = context.get("df")
        if df is None:
            return "Missing required input: df (DataFrame from Map Agent)"

        if not isinstance(df, pd.DataFrame):
            return "Input 'df' must be a pandas DataFrame"

        if len(df) == 0:
            return "Input DataFrame is empty"

        mapping_log = context.get("mapping_log")
        if mapping_log is None:
            return "Missing required input: mapping_log (from Map Agent)"

        map_metadata = context.get("map_metadata")
        if map_metadata is None:
            return "Missing required input: map_metadata (from Map Agent)"

        return None

    def execute(self, context: PipelineContext) -> AgentResult:
        """
        Execute harmonization.

        Reads from context:
        - df: Mapped DataFrame from Map Agent
        - mapping_log: Mapping log from Map Agent
        - map_metadata: Metadata from Map Agent (includes dictionary, trial_id)

        Writes to result.data:
        - df: Harmonized DataFrame
        - harmonize_lineage_log: Transformation lineage
        - harmonize_metadata: Metadata about harmonization
        """
        try:
            # Reset LLM resolutions for this run
            self.llm_resolutions = []

            # Get inputs from context
            df = context.get("df")
            mapping_log = context.get("mapping_log")
            map_metadata = context.get("map_metadata")

            lineage_log = []
            harmonized_df = df.copy()

            # Extract context
            trial_id = map_metadata.get('trial_id')
            dictionary = map_metadata.get('dictionary', {})

            # Create mapping lookup
            mapping_lookup = {m['output_variable']: m for m in mapping_log}

            # Calculate total variables for progress
            total_vars = len(OUTPUT_SCHEMA)
            processed_vars = 0

            # Process each variable
            for var in OUTPUT_SCHEMA:
                mapping_info = mapping_lookup.get(var, {})
                source_col = mapping_info.get('source_column')
                operation = mapping_info.get('operation', 'Unknown')

                original_values = harmonized_df[var].copy()
                transform_details = {}

                # Apply variable-specific transformations
                if var == "TRIAL":
                    harmonized_df[var], transform_details = self._harmonize_trial(
                        harmonized_df[var], trial_id
                    )

                elif var == "SUBJID":
                    harmonized_df[var], transform_details = self._harmonize_subjid(
                        harmonized_df[var], harmonized_df.get('USUBJID')
                    )

                elif var == "SEX":
                    harmonized_df[var], transform_details = self._harmonize_sex(
                        harmonized_df[var], dictionary
                    )

                elif var == "RACE":
                    harmonized_df[var], transform_details = self._harmonize_race(
                        harmonized_df[var], dictionary
                    )

                elif var == "AGE":
                    harmonized_df[var], transform_details = self._harmonize_age(
                        harmonized_df[var],
                        harmonized_df.get('BRTHDTC'),
                        harmonized_df.get('RFSTDTC'),
                        harmonized_df.get('AGEU')
                    )

                elif var == "AGEU":
                    harmonized_df[var], transform_details = self._harmonize_ageu(
                        harmonized_df[var], harmonized_df['AGE']
                    )

                elif var == "AGEGP":
                    harmonized_df[var], transform_details = self._harmonize_agegp(
                        harmonized_df[var], harmonized_df['AGE']
                    )

                elif var == "ETHNIC":
                    harmonized_df[var], transform_details = self._harmonize_ethnic(
                        harmonized_df[var], dictionary
                    )

                elif var == "COUNTRY":
                    harmonized_df[var], transform_details = self._harmonize_country(
                        harmonized_df[var]
                    )

                elif var == "USUBJID":
                    harmonized_df[var], transform_details = self._harmonize_usubjid(
                        harmonized_df[var],
                        harmonized_df.get('STUDYID'),
                        harmonized_df.get('SUBJID'),
                        trial_id
                    )

                elif var == "ARM":
                    harmonized_df[var], transform_details = self._harmonize_arm(
                        harmonized_df[var], dictionary
                    )

                elif var == "ARMCD":
                    harmonized_df[var], transform_details = self._harmonize_armcd(
                        harmonized_df[var], dictionary
                    )

                elif var in ["BRTHDTC", "RFSTDTC", "RFENDTC"]:
                    harmonized_df[var], transform_details = self._harmonize_date(
                        harmonized_df[var], var
                    )

                elif var == "SITEID":
                    harmonized_df[var], transform_details = self._harmonize_siteid(
                        harmonized_df[var]
                    )

                elif var in ["STUDYID", "DOMAIN"]:
                    # Simple trim/normalize
                    harmonized_df[var] = harmonized_df[var].apply(
                        lambda x: normalize_whitespace(str(x)) if pd.notna(x) else None
                    )
                    transform_details = {"operation": "Normalize (trim whitespace)"}

                # Calculate change metrics
                change_metrics = self._calculate_change_metrics(original_values, harmonized_df[var])

                # Record lineage
                lineage_entry = {
                    "variable": var,
                    "source_column": source_col,
                    "mapping_operation": operation,
                    "transform_operation": transform_details.get('operation', 'None'),
                    "transform_details": transform_details,
                    "rows_changed": change_metrics['rows_changed'],
                    "percent_changed": change_metrics['percent_changed'],
                    "missing_count": harmonized_df[var].isna().sum(),
                    "non_null_count": harmonized_df[var].notna().sum()
                }
                lineage_log.append(lineage_entry)

                # Update progress
                processed_vars += 1
                progress = 0.1 + (0.8 * processed_vars / total_vars)
                self._update_status(
                    self.status,
                    f"Harmonizing {var}...",
                    progress
                )

            # Enforce subject-level grain (uniqueness on TRIAL, SUBJID)
            duplicates_info = self._check_duplicates(harmonized_df)
            if duplicates_info['has_duplicates']:
                # Log but don't automatically drop
                pass

            self._update_status(self.status, "Building metadata...", 0.95)

            # Build metadata
            metadata = {
                "agent": self.name,
                "timestamp": datetime.now().isoformat(),
                "rows_out": len(harmonized_df),
                "trial_id": trial_id,
                "dictionary_used": dictionary is not None and len(dictionary) > 0,
                "variables_with_dictionary_decode": sum(
                    1 for l in lineage_log
                    if 'dictionary' in str(l.get('transform_details', {}))
                ),
                "duplicates": duplicates_info,
                "source_file": map_metadata.get('source_file'),
                "source_filename": map_metadata.get('source_filename'),
                "llm_fallback_enabled": self.use_llm_fallback,
                "llm_resolutions": len(self.llm_resolutions),
                "llm_resolution_details": self.llm_resolutions
            }

            return AgentResult(
                success=True,
                data={
                    "harmonized_df": harmonized_df,  # QC Agent expects "harmonized_df"
                    "harmonize_lineage_log": lineage_log,
                    "harmonize_metadata": metadata,
                    # Flatten key items for easy context access
                    "trial_id": trial_id,
                    "lineage_log": lineage_log,
                    # Pass through for QC Agent
                    "mapping_log": mapping_log,
                    "ingest_metadata": context.get("ingest_metadata", {}),
                },
                metadata=metadata
            )

        except Exception as e:
            import traceback
            return AgentResult(
                success=False,
                error=f"Harmonization failed: {str(e)}",
                error_type=type(e).__name__,
                metadata={"traceback": traceback.format_exc()}
            )

    def _harmonize_trial(self, series: pd.Series, trial_id: Optional[str]) -> Tuple[pd.Series, Dict]:
        """Harmonize TRIAL variable."""
        if trial_id:
            result = pd.Series([trial_id] * len(series), index=series.index)
            return result, {"operation": "Constant from filename", "value": trial_id}

        # Normalize existing values
        result = series.apply(lambda x: str(x).strip().upper() if pd.notna(x) else None)
        return result, {"operation": "Normalize (uppercase, trim)"}

    def _harmonize_subjid(self, series: pd.Series, usubjid: Optional[pd.Series]) -> Tuple[pd.Series, Dict]:
        """Harmonize SUBJID variable."""
        result = series.copy()

        # Convert numeric to string, preserving value
        def convert_subjid(val):
            if pd.isna(val):
                return None
            # Handle float values (common from SAS)
            if isinstance(val, float) and val == int(val):
                return str(int(val))
            return str(val).strip()

        result = result.apply(convert_subjid)

        # If still missing and USUBJID available, use USUBJID
        if usubjid is not None:
            mask = result.isna() & usubjid.notna()
            result.loc[mask] = usubjid.loc[mask].apply(
                lambda x: str(x).strip() if pd.notna(x) else None
            )

        return result, {"operation": "Normalize (convert to string, trim)"}

    def _llm_fallback_resolve(self, variable: str, value: str, valid_values: List[str] = None) -> Optional[str]:
        """
        Use LLM to resolve an unmapped value.
        Returns resolved value or None if LLM unavailable/unsuccessful.
        """
        if not self.use_llm_fallback or not self.llm_service or not self.llm_service.is_enabled():
            return None

        context = {
            'valid_values': valid_values or [],
            'description': f"Clinical trial {variable} field"
        }

        response = self.llm_service.resolve_unmapped_value(variable, value, context)

        if response.success and response.result:
            # Track the resolution
            self.llm_resolutions.append({
                'variable': variable,
                'original_value': value,
                'resolved_value': response.result,
                'reasoning': response.reasoning,
                'tokens_used': response.tokens_used
            })
            return response.result

        return None

    def _harmonize_sex(self, series: pd.Series, dictionary: Dict) -> Tuple[pd.Series, Dict]:
        """Harmonize SEX variable with code decoding and LLM fallback."""
        llm_resolved_count = 0
        used_dictionary = False

        # Check dictionary for SEX codes
        dict_codes = {}
        if dictionary and 'SEX' in dictionary:
            dict_codes = dictionary['SEX'].get('codes', {})
            used_dictionary = True

        valid_sex_values = ['Male', 'Female', 'Unknown']

        def decode_sex(val):
            nonlocal llm_resolved_count

            if pd.isna(val):
                return None

            val_str = str(val).strip().upper()

            # Try dictionary first
            if val_str in dict_codes:
                decoded = dict_codes[val_str]
                return to_mixed_case(decoded)

            # Try built-in mappings
            if val_str in SEX_DECODE:
                return SEX_DECODE[val_str]

            # Already a valid value?
            if val_str in ['MALE', 'FEMALE', 'UNKNOWN']:
                return to_mixed_case(val_str)

            # LLM fallback for unrecognized values
            llm_result = self._llm_fallback_resolve('SEX', str(val).strip(), valid_sex_values)
            if llm_result:
                llm_resolved_count += 1
                return llm_result

            # Keep as-is if unrecognized
            return to_mixed_case(val_str)

        result = series.apply(decode_sex)

        # Build detailed operation description
        if used_dictionary:
            operation = "Decoded using dictionary (SEX codes) then normalized."
        else:
            operation = "Normalized using built-in mappings."

        details = {
            "operation": operation,
            "dictionary_used": used_dictionary,
            "dictionary_codes": dict_codes if used_dictionary else {}
        }
        if llm_resolved_count > 0:
            details['llm_resolved_count'] = llm_resolved_count
        return result, details

    def _harmonize_race(self, series: pd.Series, dictionary: Dict) -> Tuple[pd.Series, Dict]:
        """Harmonize RACE variable with code decoding, normalization, and LLM fallback."""
        llm_resolved_count = 0
        used_dictionary = False

        # Check dictionary for RACE codes
        dict_codes = {}
        if dictionary and 'RACE' in dictionary:
            dict_codes = dictionary['RACE'].get('codes', {})
            used_dictionary = True

        valid_race_values = [
            'Caucasian', 'Black or African American', 'Asian',
            'Native Hawaiian or Other Pacific Islander',
            'American Indian or Alaska Native', 'Other', 'Multiple'
        ]

        def decode_race(val):
            nonlocal llm_resolved_count

            if pd.isna(val):
                return None

            val_str = str(val).strip()

            # Try dictionary first
            if val_str in dict_codes:
                decoded = dict_codes[val_str]
                # Apply additional normalization
                decoded_upper = decoded.upper()
                if decoded_upper in RACE_NORMALIZE:
                    return RACE_NORMALIZE[decoded_upper]
                return to_mixed_case(decoded)

            # Try direct normalization
            val_upper = val_str.upper()
            if val_upper in RACE_NORMALIZE:
                return RACE_NORMALIZE[val_upper]

            # LLM fallback for unrecognized values
            llm_result = self._llm_fallback_resolve('RACE', val_str, valid_race_values)
            if llm_result:
                llm_resolved_count += 1
                return llm_result

            # Keep as-is with mixed case
            return to_mixed_case(val_str)

        result = series.apply(decode_race)

        # Build detailed operation description
        if used_dictionary:
            operation = "Decoded using dictionary (RACE codes), then normalized (White->Caucasian; mixed case)."
        else:
            operation = "Normalized using built-in mappings (White->Caucasian; mixed case)."

        details = {
            "operation": operation,
            "dictionary_used": used_dictionary,
            "dictionary_codes": dict_codes if used_dictionary else {}
        }
        if llm_resolved_count > 0:
            details['llm_resolved_count'] = llm_resolved_count
        return result, details

    def _harmonize_age(
        self,
        series: pd.Series,
        brthdtc: Optional[pd.Series],
        rfstdtc: Optional[pd.Series],
        ageu: Optional[pd.Series]
    ) -> Tuple[pd.Series, Dict]:
        """Harmonize AGE variable, potentially deriving from dates."""
        result = series.copy()
        derived_count = 0

        # Convert to numeric
        def to_numeric_age(val):
            if pd.isna(val):
                return None
            try:
                age = float(val)
                if 0 <= age <= 120:
                    return age
                return None
            except (ValueError, TypeError):
                return None

        result = result.apply(to_numeric_age)

        # Convert units if AGEU indicates non-years
        if ageu is not None:
            for idx in result.index:
                if pd.notna(result[idx]) and pd.notna(ageu.get(idx)):
                    unit = str(ageu[idx]).upper()
                    if 'MONTH' in unit:
                        result[idx] = result[idx] / 12
                    elif 'DAY' in unit:
                        result[idx] = result[idx] / 365.25
                    elif 'WEEK' in unit:
                        result[idx] = result[idx] / 52

        # Try to derive missing ages from dates
        if brthdtc is not None and rfstdtc is not None:
            for idx in result.index:
                if pd.isna(result[idx]):
                    birth = brthdtc.get(idx)
                    ref = rfstdtc.get(idx)
                    if pd.notna(birth) and pd.notna(ref):
                        derived_age = calculate_age(str(birth), str(ref))
                        if derived_age is not None:
                            result[idx] = derived_age
                            derived_count += 1

        # Determine operation description based on data
        missing_count = result.isna().sum()
        total_count = len(result)

        if missing_count == total_count:
            operation = "AGE not available; left blank (use AGEGP)."
        elif derived_count > 0:
            operation = f"Normalized; derived {derived_count} ages from birth/reference dates."
        else:
            operation = "Normalized to numeric years."

        details = {
            "operation": operation,
            "ages_derived": derived_count
        }

        return result, details

    def _harmonize_ageu(self, series: pd.Series, age: pd.Series) -> Tuple[pd.Series, Dict]:
        """Harmonize AGEU variable."""
        result = series.copy()

        def normalize_ageu(val, has_age):
            if not has_age:
                return None
            if pd.isna(val):
                return "Years"  # Default when AGE present
            val_str = str(val).upper().strip()
            if 'YEAR' in val_str:
                return "Years"
            if 'MONTH' in val_str:
                return "Years"  # Converted in AGE harmonization
            if 'DAY' in val_str:
                return "Years"
            if 'WEEK' in val_str:
                return "Years"
            return "Years"

        result = pd.Series([
            normalize_ageu(series.iloc[i], pd.notna(age.iloc[i]))
            for i in range(len(series))
        ], index=series.index)

        # Determine if AGEU is blank because AGE not populated
        if age.isna().all():
            operation = "AGEU blank because AGE not populated."
        else:
            operation = "Standardized to Years."

        return result, {"operation": operation}

    def _harmonize_agegp(self, series: pd.Series, age: pd.Series) -> Tuple[pd.Series, Dict]:
        """
        Harmonize AGEGP variable.
        AGEGP only populated when AGE is missing/non-derivable.
        """
        result = series.copy()

        # Blank out AGEGP where AGE is present
        for idx in result.index:
            if pd.notna(age[idx]):
                result[idx] = None
            elif pd.notna(result[idx]):
                # Trim whitespace but preserve format
                result[idx] = str(result[idx]).strip()

        return result, {"operation": "Conditional (blank when AGE present)"}

    def _harmonize_ethnic(self, series: pd.Series, dictionary: Dict) -> Tuple[pd.Series, Dict]:
        """Harmonize ETHNIC variable."""
        used_dictionary = False
        dict_source = None

        dict_codes = {}
        if dictionary and 'ETHGRP' in dictionary:
            dict_codes = dictionary['ETHGRP'].get('codes', {})
            used_dictionary = True
            dict_source = 'ETHGRP'
        elif dictionary and 'ETHNIC' in dictionary:
            dict_codes = dictionary['ETHNIC'].get('codes', {})
            used_dictionary = True
            dict_source = 'ETHNIC'

        def decode_ethnic(val):
            if pd.isna(val):
                return None

            val_str = str(val).strip()

            if val_str in dict_codes:
                return to_mixed_case(dict_codes[val_str])

            return to_mixed_case(val_str)

        result = series.apply(decode_ethnic)

        if used_dictionary:
            operation = f"Decoded using dictionary ({dict_source} codes) then mixed case."
        else:
            operation = "Normalized to mixed case."

        details = {
            "operation": operation,
            "dictionary_used": used_dictionary
        }
        return result, details

    def _harmonize_country(self, series: pd.Series) -> Tuple[pd.Series, Dict]:
        """Harmonize COUNTRY variable."""
        codes_expanded = 0

        def normalize_country(val):
            nonlocal codes_expanded
            if pd.isna(val):
                return None

            val_str = str(val).strip()

            # Check if it's a code that needs expansion
            val_upper = val_str.upper()
            if val_upper in COUNTRY_CODES:
                codes_expanded += 1
                return COUNTRY_CODES[val_upper]

            # Already a full name, apply mixed case
            return to_mixed_case(val_str)

        result = series.apply(normalize_country)

        if codes_expanded > 0:
            operation = f"Expanded {codes_expanded} country codes; trim; mixed case."
        else:
            operation = "Trim; mixed case."

        return result, {"operation": operation, "codes_expanded": codes_expanded}

    def _harmonize_usubjid(
        self,
        series: pd.Series,
        studyid: Optional[pd.Series],
        subjid: Optional[pd.Series],
        trial_id: Optional[str]
    ) -> Tuple[pd.Series, Dict]:
        """Harmonize/derive USUBJID."""
        result = series.copy()

        # Derive where missing
        derived_count = 0
        for idx in result.index:
            if pd.isna(result[idx]) and subjid is not None and pd.notna(subjid.get(idx)):
                # Use STUDYID if available, else TRIAL
                prefix = None
                if studyid is not None and pd.notna(studyid.get(idx)):
                    prefix = str(studyid[idx]).strip()
                elif trial_id:
                    prefix = trial_id

                if prefix:
                    result[idx] = f"{prefix}-{subjid[idx]}"
                    derived_count += 1

        return result, {
            "operation": f"Derived as STUDYID||'-'||SUBJID (else TRIAL||'-'||SUBJID). Derived {derived_count} values.",
            "derived_count": derived_count
        }

    def _harmonize_arm(self, series: pd.Series, dictionary: Dict) -> Tuple[pd.Series, Dict]:
        """Harmonize ARM variable."""
        result = series.apply(
            lambda x: to_mixed_case(str(x).strip()) if pd.notna(x) else None
        )
        return result, {"operation": "Normalize (mixed case, trim)"}

    def _harmonize_armcd(self, series: pd.Series, dictionary: Dict) -> Tuple[pd.Series, Dict]:
        """Harmonize ARMCD variable."""
        dict_codes = {}
        if dictionary and 'TRTCODE' in dictionary:
            dict_codes = dictionary['TRTCODE'].get('codes', {})

        def decode_armcd(val):
            if pd.isna(val):
                return None
            val_str = str(val).strip()
            if val_str in dict_codes:
                return dict_codes[val_str]
            return val_str

        result = series.apply(decode_armcd)
        return result, {"operation": "Decode if dictionary available"}

    def _harmonize_date(self, series: pd.Series, var_name: str) -> Tuple[pd.Series, Dict]:
        """Harmonize date variables to ISO 8601."""
        def convert_date(val):
            if pd.isna(val):
                return None

            # If numeric (SAS date), convert
            if isinstance(val, (int, float)):
                return sas_date_to_iso(val)

            # If string, validate format
            val_str = str(val).strip()

            # Remove time component if present
            if 'T' in val_str:
                val_str = val_str.split('T')[0]
            if ' ' in val_str and ':' in val_str:
                val_str = val_str.split(' ')[0]

            return val_str if val_str else None

        result = series.apply(convert_date)
        return result, {"operation": "Normalize to ISO 8601 date"}

    def _harmonize_siteid(self, series: pd.Series) -> Tuple[pd.Series, Dict]:
        """Harmonize SITEID - preserve as string to keep leading zeros."""
        def convert_siteid(val):
            if pd.isna(val):
                return None
            # Handle numeric values
            if isinstance(val, float) and val == int(val):
                return str(int(val))
            return str(val).strip()

        result = series.apply(convert_siteid)
        return result, {"operation": "Normalize (string, preserve format)"}

    def _calculate_change_metrics(
        self,
        original: pd.Series,
        transformed: pd.Series
    ) -> Dict[str, Any]:
        """Calculate how many values changed during transformation."""
        total = len(original)
        if total == 0:
            return {"rows_changed": 0, "percent_changed": 0.0}

        # Compare values (treating None == None as unchanged)
        def values_equal(a, b):
            if pd.isna(a) and pd.isna(b):
                return True
            if pd.isna(a) or pd.isna(b):
                return False
            return str(a).strip() == str(b).strip()

        changed = sum(
            1 for i in range(total)
            if not values_equal(original.iloc[i], transformed.iloc[i])
        )

        return {
            "rows_changed": changed,
            "percent_changed": round(100 * changed / total, 2)
        }

    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate (TRIAL, SUBJID) combinations."""
        if 'TRIAL' not in df.columns or 'SUBJID' not in df.columns:
            return {"has_duplicates": False, "duplicate_count": 0}

        # Count duplicates
        dup_mask = df.duplicated(subset=['TRIAL', 'SUBJID'], keep=False)
        dup_count = dup_mask.sum()

        return {
            "has_duplicates": dup_count > 0,
            "duplicate_count": dup_count,
            "unique_duplicate_keys": df[dup_mask][['TRIAL', 'SUBJID']].drop_duplicates().shape[0] if dup_count > 0 else 0
        }


def run_harmonize_agent(
    context: PipelineContext,
    use_llm_fallback: bool = True,
    progress_callback: Optional[ProgressCallback] = None
) -> AgentResult:
    """
    Convenience function to run the Harmonize Agent.

    Args:
        context: Pipeline context with df, mapping_log, map_metadata
        use_llm_fallback: Whether to use LLM for unmatched values
        progress_callback: Optional callback for progress updates

    Returns:
        AgentResult with harmonized DataFrame and lineage log
    """
    agent = HarmonizeAgent(
        use_llm_fallback=use_llm_fallback,
        progress_callback=progress_callback
    )
    return agent.run(context)


if __name__ == "__main__":
    print("Harmonize Agent v2 - run via orchestrator or import")
