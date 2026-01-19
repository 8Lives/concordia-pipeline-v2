"""
QC & Report Agent (v2 - Agentic Architecture)

Responsibilities:
- Execute all QC checks per specification
- Generate QC Report (CSV)
- Generate Harmonization Transformation Report (DOCX)
- Track issues and prepare final outputs

Changes from v1:
- Extends AgentBase for timeout/retry/callback support
- Uses PipelineContext for input/output
- Standardized error handling
"""
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime

from agents.base import AgentBase, AgentResult, AgentConfig, PipelineContext, ProgressCallback
from config.schema import OUTPUT_SCHEMA, REQUIRED_VARIABLES, QC_ISSUES
from utils.helpers import validate_nct_format, is_full_date, get_unique_values_sample


class QCAgent(AgentBase):
    """
    Agent responsible for QC checks and report generation.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        super().__init__(
            name="QC & Report Agent",
            config=config or AgentConfig(timeout_seconds=120.0),
            progress_callback=progress_callback
        )

    def validate_input(self, context: PipelineContext) -> Optional[str]:
        """Validate required inputs exist."""
        harmonized_df = context.get("harmonized_df")
        if harmonized_df is None:
            return "Missing required input: harmonized_df"

        if not isinstance(harmonized_df, pd.DataFrame):
            return "harmonized_df must be a pandas DataFrame"

        lineage_log = context.get("lineage_log")
        if lineage_log is None:
            return "Missing required input: lineage_log"

        harmonize_metadata = context.get("harmonize_metadata")
        if harmonize_metadata is None:
            return "Missing required input: harmonize_metadata"

        mapping_log = context.get("mapping_log")
        if mapping_log is None:
            return "Missing required input: mapping_log"

        ingest_metadata = context.get("ingest_metadata")
        if ingest_metadata is None:
            return "Missing required input: ingest_metadata"

        return None

    def execute(self, context: PipelineContext) -> AgentResult:
        """
        Execute QC checks and generate reports.

        Reads from context:
        - harmonized_df: Harmonized DataFrame from Harmonize Agent
        - lineage_log: Lineage log from Harmonize Agent
        - harmonize_metadata: Metadata from Harmonize Agent
        - mapping_log: Mapping log from Map Agent
        - ingest_metadata: Metadata from Ingest Agent

        Writes to context (via result.data):
        - harmonized_df: Final harmonized DataFrame (passed through)
        - qc_report: QC report DataFrame
        - transformation_summary: List of transformation details
        """
        df = context.get("harmonized_df")
        lineage_log = context.get("lineage_log")
        harmonize_metadata = context.get("harmonize_metadata")
        mapping_log = context.get("mapping_log")
        ingest_metadata = context.get("ingest_metadata")

        qc_issues = []
        trial_id = harmonize_metadata.get('trial_id', 'UNKNOWN')

        # Update progress - starting QC checks
        self._update_status(self.status, "Checking TRIAL validity...", 0.1)

        # Execute QC checks
        qc_issues.extend(self._check_trial_validity(df, trial_id))

        self._update_status(self.status, "Checking uniqueness constraints...", 0.2)
        qc_issues.extend(self._check_uniqueness(df, trial_id))

        self._update_status(self.status, "Checking required values...", 0.3)
        qc_issues.extend(self._check_required_values(df, trial_id))

        self._update_status(self.status, "Checking age completeness...", 0.4)
        qc_issues.extend(self._check_age_completeness(df, trial_id))

        self._update_status(self.status, "Checking date validity...", 0.5)
        qc_issues.extend(self._check_date_validity(df, trial_id))

        self._update_status(self.status, "Checking coded values...", 0.6)
        qc_issues.extend(self._check_coded_values(df, trial_id, harmonize_metadata))

        self._update_status(self.status, "Checking column mapping quality...", 0.65)
        qc_issues.extend(self._check_mapping_quality(df, trial_id, mapping_log))

        # Build QC report DataFrame
        self._update_status(self.status, "Building QC report...", 0.7)
        qc_report = self._build_qc_report(qc_issues)

        # Build transformation summary
        self._update_status(self.status, "Building transformation summary...", 0.8)
        transformation_summary = self._build_transformation_summary(
            lineage_log, mapping_log, ingest_metadata, harmonize_metadata
        )

        # Build metadata
        self._update_status(self.status, "Finalizing results...", 0.9)
        metadata = {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "trial_id": trial_id,
            "rows_final": len(df),
            "total_qc_issues": len(qc_issues),
            "issues_by_type": self._count_issues_by_type(qc_issues),
            "source_file": ingest_metadata.get('source_file'),
            "source_filename": ingest_metadata.get('source_filename')
        }

        return AgentResult(
            success=True,
            data={
                "harmonized_df": df,
                "qc_report": qc_report,
                "transformation_summary": transformation_summary
            },
            metadata=metadata
        )

    def _check_trial_validity(self, df: pd.DataFrame, trial_id: str) -> List[Dict]:
        """Check TRIAL validity (must match NCT format)."""
        issues = []

        if 'TRIAL' not in df.columns:
            issues.append({
                "TRIAL": trial_id,
                "issue_type": "TRIAL_MISSING_OR_INVALID",
                "variable": "TRIAL",
                "n_rows_affected": len(df),
                "example_values": "",
                "notes": "TRIAL column missing from output"
            })
            return issues

        # Check for missing or invalid values
        invalid_mask = df['TRIAL'].apply(lambda x: not validate_nct_format(x) if pd.notna(x) else True)
        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            example_vals = get_unique_values_sample(df.loc[invalid_mask, 'TRIAL'])
            issues.append({
                "TRIAL": trial_id,
                "issue_type": "TRIAL_MISSING_OR_INVALID",
                "variable": "TRIAL",
                "n_rows_affected": invalid_count,
                "example_values": example_vals,
                "notes": "TRIAL does not match ^NCT\\d{8}$ pattern"
            })

        return issues

    def _check_uniqueness(self, df: pd.DataFrame, trial_id: str) -> List[Dict]:
        """Check (TRIAL, SUBJID) uniqueness."""
        issues = []

        if 'TRIAL' not in df.columns or 'SUBJID' not in df.columns:
            return issues

        # Find duplicates
        dup_mask = df.duplicated(subset=['TRIAL', 'SUBJID'], keep=False)
        dup_count = dup_mask.sum()

        if dup_count > 0:
            dup_subjects = df.loc[dup_mask, 'SUBJID'].unique()[:5]

            # Check if ALL rows are duplicates (suggests mapping failure)
            total_rows = len(df)
            dup_ratio = dup_count / total_rows if total_rows > 0 else 0

            if dup_ratio == 1.0:
                # All rows are duplicates - likely mapping failure
                unique_subjid_count = df['SUBJID'].nunique()
                issues.append({
                    "TRIAL": trial_id,
                    "issue_type": "SUBJID_MAPPING_SUSPECT",
                    "variable": "SUBJID",
                    "n_rows_affected": dup_count,
                    "example_values": ", ".join(str(s) for s in dup_subjects),
                    "notes": f"ALL {total_rows} rows flagged as duplicates ({unique_subjid_count} unique SUBJID values). "
                             f"This strongly suggests SUBJID was mapped to wrong column. Manual review required."
                })
            else:
                issues.append({
                    "TRIAL": trial_id,
                    "issue_type": "DUPLICATE_SUBJECT",
                    "variable": "TRIAL, SUBJID",
                    "n_rows_affected": dup_count,
                    "example_values": ", ".join(str(s) for s in dup_subjects),
                    "notes": f"Duplicate (TRIAL, SUBJID) combinations found"
                })

        return issues

    def _check_required_values(self, df: pd.DataFrame, trial_id: str) -> List[Dict]:
        """Check required variables for missing values."""
        issues = []
        total_rows = len(df)

        for var in REQUIRED_VARIABLES:
            if var not in df.columns:
                issues.append({
                    "TRIAL": trial_id,
                    "issue_type": "MISSING_REQUIRED_VALUE",
                    "variable": var,
                    "n_rows_affected": len(df),
                    "example_values": "",
                    "notes": f"Required variable {var} not in output"
                })
                continue

            missing_count = df[var].isna().sum()
            if missing_count > 0:
                # Check if ALL rows are missing (suggests source column not found)
                if missing_count == total_rows:
                    if var == "SUBJID":
                        issues.append({
                            "TRIAL": trial_id,
                            "issue_type": "SUBJID_NOT_FOUND",
                            "variable": var,
                            "n_rows_affected": missing_count,
                            "example_values": "",
                            "notes": f"No subject identifier column found in source data. "
                                     f"All {total_rows} rows have missing SUBJID. Manual column mapping required."
                        })
                    else:
                        issues.append({
                            "TRIAL": trial_id,
                            "issue_type": "MISSING_REQUIRED_VALUE",
                            "variable": var,
                            "n_rows_affected": missing_count,
                            "example_values": "",
                            "notes": f"ALL {missing_count} rows missing {var} - source column may not exist in data"
                        })
                else:
                    issues.append({
                        "TRIAL": trial_id,
                        "issue_type": "MISSING_REQUIRED_VALUE",
                        "variable": var,
                        "n_rows_affected": missing_count,
                        "example_values": "",
                        "notes": f"{missing_count} rows with missing {var}"
                    })

        return issues

    def _check_age_completeness(self, df: pd.DataFrame, trial_id: str) -> List[Dict]:
        """Check AGE/AGEGP completeness (at least one must be present)."""
        issues = []

        has_age = 'AGE' in df.columns
        has_agegp = 'AGEGP' in df.columns

        if not has_age and not has_agegp:
            issues.append({
                "TRIAL": trial_id,
                "issue_type": "MISSING_AGE_AND_AGEGP",
                "variable": "AGE, AGEGP",
                "n_rows_affected": len(df),
                "example_values": "",
                "notes": "Neither AGE nor AGEGP columns present"
            })
            return issues

        # Check rows where both are missing
        both_missing_count = 0
        for idx in df.index:
            age_missing = not has_age or pd.isna(df.loc[idx, 'AGE'])
            agegp_missing = not has_agegp or pd.isna(df.loc[idx, 'AGEGP'])
            if age_missing and agegp_missing:
                both_missing_count += 1

        if both_missing_count > 0:
            issues.append({
                "TRIAL": trial_id,
                "issue_type": "MISSING_AGE_AND_AGEGP",
                "variable": "AGE, AGEGP",
                "n_rows_affected": both_missing_count,
                "example_values": "",
                "notes": f"{both_missing_count} rows missing both AGE and AGEGP"
            })

        return issues

    def _check_date_validity(self, df: pd.DataFrame, trial_id: str) -> List[Dict]:
        """Check date validity and order."""
        issues = []
        date_vars = ['BRTHDTC', 'RFSTDTC', 'RFENDTC']

        for var in date_vars:
            if var not in df.columns:
                continue

            # Check for invalid dates
            invalid_count = 0
            for idx in df.index:
                val = df.loc[idx, var]
                if pd.notna(val):
                    val_str = str(val).strip()
                    # Basic validation: should look like a date
                    if val_str and not any(c.isdigit() for c in val_str):
                        invalid_count += 1

            if invalid_count > 0:
                issues.append({
                    "TRIAL": trial_id,
                    "issue_type": "DATE_INVALID",
                    "variable": var,
                    "n_rows_affected": invalid_count,
                    "example_values": get_unique_values_sample(df[var].dropna()),
                    "notes": f"{invalid_count} rows with invalid date format"
                })

        # Check RFENDTC < RFSTDTC (date order)
        if 'RFSTDTC' in df.columns and 'RFENDTC' in df.columns:
            order_invalid_count = 0
            for idx in df.index:
                start = df.loc[idx, 'RFSTDTC']
                end = df.loc[idx, 'RFENDTC']
                if pd.notna(start) and pd.notna(end):
                    if is_full_date(str(start)) and is_full_date(str(end)):
                        if str(end) < str(start):  # String comparison works for ISO dates
                            order_invalid_count += 1

            if order_invalid_count > 0:
                issues.append({
                    "TRIAL": trial_id,
                    "issue_type": "DATE_ORDER_INVALID",
                    "variable": "RFSTDTC, RFENDTC",
                    "n_rows_affected": order_invalid_count,
                    "example_values": "",
                    "notes": f"{order_invalid_count} rows where RFENDTC < RFSTDTC"
                })

        return issues

    def _check_coded_values(self, df: pd.DataFrame, trial_id: str, metadata: Dict) -> List[Dict]:
        """Check for coded values that weren't decoded."""
        issues = []
        coded_vars = ['SEX', 'RACE', 'ETHNIC', 'ARMCD']

        dictionary_used = metadata.get('dictionary_used', False)

        for var in coded_vars:
            if var not in df.columns:
                continue

            # Check if values still look like codes (single digits/short numbers)
            still_coded_count = 0
            for idx in df.index:
                val = df.loc[idx, var]
                if pd.notna(val):
                    val_str = str(val).strip()
                    # Looks like a code: 1-2 digit number
                    if val_str.isdigit() and len(val_str) <= 2:
                        still_coded_count += 1

            if still_coded_count > 0 and not dictionary_used:
                issues.append({
                    "TRIAL": trial_id,
                    "issue_type": "CODED_VALUE_NO_DICTIONARY",
                    "variable": var,
                    "n_rows_affected": still_coded_count,
                    "example_values": get_unique_values_sample(df[var]),
                    "notes": f"{var} contains apparent codes but no dictionary was provided"
                })

        return issues

    def _check_mapping_quality(self, df: pd.DataFrame, trial_id: str, mapping_log: List[Dict]) -> List[Dict]:
        """Check for mapping quality issues (heuristic mappings, warnings)."""
        issues = []

        for mapping in mapping_log:
            var = mapping.get('output_variable')
            details = mapping.get('details', {})

            # Check for heuristic mappings
            if details.get('heuristic_used'):
                source_col = details.get('matched_candidate', 'unknown')
                note = details.get('note', '')
                original_rejected = details.get('original_match_rejected')

                issue_note = f"{var} mapped to '{source_col}' using heuristic (uniqueness-based) matching. {note}"
                if original_rejected:
                    issue_note += f" Original match '{original_rejected}' was rejected due to low uniqueness."

                issues.append({
                    "TRIAL": trial_id,
                    "issue_type": "COLUMN_MAPPING_HEURISTIC",
                    "variable": var,
                    "n_rows_affected": len(df),
                    "example_values": source_col,
                    "notes": issue_note
                })

            # Check for mapping warnings
            if details.get('warning'):
                issues.append({
                    "TRIAL": trial_id,
                    "issue_type": "SUBJID_MAPPING_SUSPECT",
                    "variable": var,
                    "n_rows_affected": len(df),
                    "example_values": details.get('matched_candidate', ''),
                    "notes": details.get('warning')
                })

        return issues

    def _build_qc_report(self, issues: List[Dict]) -> pd.DataFrame:
        """Build QC report DataFrame from issues list."""
        if not issues:
            # Return empty report with correct schema
            return pd.DataFrame(columns=[
                'TRIAL', 'issue_type', 'variable', 'n_rows_affected',
                'example_values', 'notes'
            ])

        return pd.DataFrame(issues)

    def _count_issues_by_type(self, issues: List[Dict]) -> Dict[str, int]:
        """Count issues by type for metadata."""
        counts = {}
        for issue in issues:
            issue_type = issue.get('issue_type', 'UNKNOWN')
            counts[issue_type] = counts.get(issue_type, 0) + 1
        return counts

    def _build_transformation_summary(
        self,
        lineage_log: List[Dict],
        mapping_log: List[Dict],
        ingest_metadata: Dict,
        harmonize_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """Build transformation summary for the report."""
        summary = []

        # Create lookup for mapping info
        mapping_lookup = {m['output_variable']: m for m in mapping_log}

        for lineage in lineage_log:
            var = lineage['variable']
            mapping = mapping_lookup.get(var, {})

            summary.append({
                "variable": var,
                "source_column": lineage.get('source_column'),
                "mapping_operation": mapping.get('operation', 'Unknown'),
                "transform_operation": lineage.get('transform_operation', 'None'),
                "transform_details": str(lineage.get('transform_details', {})),
                "rows_changed": lineage.get('rows_changed', 0),
                "percent_changed": lineage.get('percent_changed', 0),
                "missing_count": lineage.get('missing_count', 0),
                "non_null_count": lineage.get('non_null_count', 0)
            })

        return summary


def run_qc_agent(
    context: PipelineContext,
    config: Optional[AgentConfig] = None,
    progress_callback: Optional[ProgressCallback] = None
) -> AgentResult:
    """
    Convenience function to run the QC Agent.
    """
    agent = QCAgent(config=config, progress_callback=progress_callback)
    return agent.run(context)
