"""
Map & Standardize Agent (v2 - Agentic Architecture)

Responsibilities:
- Match source columns to output schema using priority rules
- Handle column name variants (SUBJ vs SUBJID, ETHGRP vs ETHNIC)
- Apply text normalization (trim, whitespace)
- Create column mapping log for traceability

Changes from v1:
- Extends AgentBase for timeout/retry/callback support
- Uses PipelineContext for input/output
- Enhanced SUBJID mapping with more synonyms
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from agents.base import AgentBase, AgentResult, AgentConfig, PipelineContext, ProgressCallback
from config.schema import OUTPUT_SCHEMA, SOURCE_PRIORITY
from utils.helpers import find_column_match, normalize_whitespace


class MapAgent(AgentBase):
    """
    Agent responsible for mapping source columns to the output schema.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        super().__init__(
            name="Map & Standardize Agent",
            config=config or AgentConfig(timeout_seconds=60.0),
            progress_callback=progress_callback
        )

    def validate_input(self, context: PipelineContext) -> Optional[str]:
        """Validate required inputs exist."""
        df = context.get("df")
        if df is None:
            return "Missing required input: df (from Ingest Agent)"
        return None

    def execute(self, context: PipelineContext) -> AgentResult:
        """
        Execute column mapping.

        Reads from context:
        - df: DataFrame from Ingest Agent
        - ingest_metadata: Metadata from Ingest Agent
        - trial_id: Trial ID from filename

        Writes to context (via result.data):
        - mapped_df: DataFrame with columns mapped to output schema
        - mapping_log: Log of all mappings performed
        """
        df = context.get("df")
        ingest_metadata = context.get("ingest_metadata", {})
        trial_id = context.get("trial_id") or ingest_metadata.get("trial_id_from_filename")
        dictionary = context.get("dictionary")

        mapping_log = []
        mapped_data = {}

        total_vars = len(OUTPUT_SCHEMA)

        for idx, output_var in enumerate(OUTPUT_SCHEMA):
            # Update progress
            self._update_status(
                self.status,
                f"Mapping {output_var}...",
                0.1 + (0.8 * idx / total_vars)
            )

            source_col, operation, details = self._map_variable(
                df, output_var, trial_id, ingest_metadata
            )

            # Get the values
            if source_col is not None and source_col in df.columns:
                values = df[source_col].apply(
                    lambda x: normalize_whitespace(str(x)) if pd.notna(x) else None
                )
            elif operation == "Constant":
                values = pd.Series([details.get('value')] * len(df), index=df.index)
            elif operation == "Derive":
                values = pd.Series([None] * len(df), index=df.index)
            else:
                values = pd.Series([None] * len(df), index=df.index)

            mapped_data[output_var] = values

            mapping_entry = {
                "output_variable": output_var,
                "source_column": source_col,
                "operation": operation,
                "details": details,
                "non_null_count": values.notna().sum(),
                "null_count": values.isna().sum()
            }
            mapping_log.append(mapping_entry)

        # Create mapped DataFrame
        mapped_df = pd.DataFrame(mapped_data)

        # Build metadata
        metadata = {
            "agent": self.name,
            "input_columns": list(df.columns),
            "output_columns": OUTPUT_SCHEMA,
            "columns_mapped": sum(1 for m in mapping_log if m['source_column'] is not None),
            "columns_constant": sum(1 for m in mapping_log if m['operation'] == 'Constant'),
            "columns_derived": sum(1 for m in mapping_log if m['operation'] == 'Derive'),
            "columns_unmapped": sum(1 for m in mapping_log if m['source_column'] is None and m['operation'] not in ['Constant', 'Derive']),
            "rows": len(mapped_df),
            "trial_id": trial_id,
        }

        return AgentResult(
            success=True,
            data={
                "mapped_df": mapped_df,
                "mapping_log": mapping_log,
                "trial_id": trial_id,
                "dictionary": dictionary,
                "source_filename": ingest_metadata.get('source_filename'),
                "source_file": ingest_metadata.get('source_file'),
            },
            metadata=metadata
        )

    def _map_variable(
        self,
        df: pd.DataFrame,
        output_var: str,
        trial_id: Optional[str],
        ingest_metadata: Dict[str, Any]
    ) -> Tuple[Optional[str], str, Dict[str, Any]]:
        """Determine the source column and operation for an output variable."""
        details = {}

        if output_var == "TRIAL":
            if trial_id:
                return None, "Constant", {"value": trial_id, "source": "filename"}
            source = find_column_match(df, SOURCE_PRIORITY.get("TRIAL", []))
            if source:
                return source, "Copy", {"note": "TRIAL from data"}
            source = find_column_match(df, ["STUDYID", "STUDY"])
            if source:
                return source, "Copy", {"note": "Using STUDYID as TRIAL fallback"}
            return None, "Missing", {"note": "Could not determine TRIAL"}

        elif output_var == "DOMAIN":
            return None, "Constant", {"value": "DM"}

        elif output_var == "SUBJID":
            candidates = SOURCE_PRIORITY.get("SUBJID", [])
            source = find_column_match(df, candidates)
            if source:
                return source, "Copy", {"matched_candidate": source}
            usubjid_source = find_column_match(df, ["USUBJID"])
            if usubjid_source:
                return usubjid_source, "Copy", {
                    "note": "SUBJID sourced from USUBJID",
                    "matched_candidate": usubjid_source
                }
            return None, "Missing", {"note": "No subject identifier found"}

        elif output_var == "USUBJID":
            source = find_column_match(df, ["USUBJID", "RUSUBJID"])
            if source:
                return source, "Copy", {}
            return None, "Derive", {"derivation": "STUDYID||'-'||SUBJID or TRIAL||'-'||SUBJID"}

        elif output_var == "AGE":
            source = find_column_match(df, SOURCE_PRIORITY.get("AGE", []))
            if source:
                return source, "Copy", {}
            has_brthdtc = find_column_match(df, SOURCE_PRIORITY.get("BRTHDTC", [])) is not None
            has_rfstdtc = find_column_match(df, SOURCE_PRIORITY.get("RFSTDTC", [])) is not None
            if has_brthdtc and has_rfstdtc:
                return None, "Derive", {"derivation": "From BRTHDTC and RFSTDTC"}
            return None, "Missing", {"note": "AGE not present and not derivable"}

        elif output_var == "AGEGP":
            candidates = SOURCE_PRIORITY.get("AGEGP", [])
            available_sources = []
            for candidate in candidates:
                source = find_column_match(df, [candidate])
                if source:
                    distinct_count = df[source].dropna().nunique()
                    available_sources.append((source, distinct_count))
            if available_sources:
                best_source = max(available_sources, key=lambda x: x[1])
                return best_source[0], "Copy", {
                    "granularity_selection": "Most distinct values",
                    "distinct_count": best_source[1]
                }
            return None, "Missing", {}

        else:
            candidates = SOURCE_PRIORITY.get(output_var, [])
            source = find_column_match(df, candidates)
            if source:
                return source, "Copy", {"matched_candidate": source}
            return None, "Missing", {}
