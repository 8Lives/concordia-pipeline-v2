"""
LLM Review Agent (v2)

Responsibilities:
- Review harmonized output using Claude Opus 4.5
- Identify data quality issues automated QC may have missed
- Suggest corrections for problematic values
- Provide overall quality assessment

v2 Changes:
- Extends AgentBase for standardized lifecycle
- Uses PipelineContext for input/output
- Implements validate_input() and execute() methods
- Adds progress callbacks
- Marked as optional (required=False)
- 90 second timeout for LLM calls
"""
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from agents.base import (
    AgentBase,
    AgentConfig,
    AgentResult,
    PipelineContext,
    ProgressCallback,
    AgentStatus
)

try:
    from agents.llm_service import get_llm_service, LLMResponse
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


# Specification summary for LLM context
DM_SPEC_SUMMARY = """
Demographics (DM) Domain Harmonization Specification:

**Output Schema:** TRIAL, SUBJID, SEX, RACE, AGE, AGEU, AGEGP, ETHNIC, COUNTRY, SITEID, STUDYID, USUBJID, ARMCD, ARM, BRTHDTC, RFSTDTC, RFENDTC, DOMAIN

**Variable Requirements:**
- TRIAL: Must match NCT format (NCT + 8 digits)
- SEX: Male, Female, or Unknown (not codes like 1, 2)
- RACE: Caucasian, Black or African American, Asian, Native Hawaiian or Other Pacific Islander, American Indian or Alaska Native, Other, Multiple
- ETHNIC: Hispanic or Latino, Not Hispanic or Latino, Unknown (not codes)
- AGE: Numeric years; if unavailable, AGEGP should be populated instead
- COUNTRY: Full country names (e.g., "United States" not "US")
- Dates: ISO 8601 format (YYYY-MM-DD)
- Grain: One row per unique (TRIAL, SUBJID)

**CRITICAL Text Formatting Rules:**
- Use title case for text values, BUT keep prepositions/articles lowercase (unless first word)
- Prepositions to keep lowercase: "or", "and", "of", "the", "in", "on", "at", "to", "for", "with", "than", "as", "by"
- CORRECT examples:
  - "Hispanic or Latino" (NOT "Hispanic Or Latino")
  - "Black or African American" (NOT "Black Or African American")
  - "Korea, Republic of" (NOT "Korea, Republic Of")
  - "Native Hawaiian or Other Pacific Islander"
- INCORRECT examples to flag:
  - "Hispanic Or Latino" - "Or" should be lowercase
  - "Korea, Republic Of" - "Of" should be lowercase
  - Any preposition capitalized mid-phrase

**Common Issues to Check:**
1. Coded values not decoded (e.g., SEX="1" instead of "Male")
2. Improper capitalization of prepositions (Or, And, Of, Than should be or, and, of, than)
3. Missing or inconsistent USUBJID format
4. Country codes not expanded to full names
5. Race/Ethnicity values not matching controlled terminology
"""


@dataclass
class ReviewResult:
    """Result from the Review Agent"""
    success: bool
    overall_quality: str = "unknown"
    issues_found: List[Dict[str, Any]] = field(default_factory=list)
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ReviewAgent(AgentBase):
    """
    Agent responsible for LLM-powered review of harmonized output.
    Uses Claude Opus 4.5 to identify issues and suggest corrections.

    This is an optional agent - pipeline continues even if review fails.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        # Set default config with 90 second timeout and required=False
        if config is None:
            config = AgentConfig(
                timeout_seconds=90.0,
                required=False,
                max_retries=1,
                retry_delay_seconds=2.0
            )
        else:
            # Ensure required=False and timeout=90 even if config provided
            config.required = False
            config.timeout_seconds = 90.0

        super().__init__(
            name="LLM Review Agent",
            config=config,
            progress_callback=progress_callback
        )

        self.llm_service = get_llm_service() if LLM_AVAILABLE else None

    def is_enabled(self) -> bool:
        """Check if review agent is available (requires LLM)."""
        return self.llm_service is not None and self.llm_service.is_enabled()

    def validate_input(self, context: PipelineContext) -> Optional[str]:
        """
        Validate that required inputs are present in context.

        Required context keys:
        - harmonized_df: DataFrame with harmonized data
        - transformation_summary: List of transformations applied
        - qc_report: QC report with issues

        Returns:
            None if valid, error message string if invalid
        """
        # Check for harmonized_df
        harmonized_df = context.get("harmonized_df")
        if harmonized_df is None:
            return "Missing required input: harmonized_df"

        if not isinstance(harmonized_df, pd.DataFrame):
            return "harmonized_df must be a pandas DataFrame"

        if harmonized_df.empty:
            return "harmonized_df is empty - nothing to review"

        # Check for transformation_summary (can be empty list but must exist)
        transformation_summary = context.get("transformation_summary")
        if transformation_summary is None:
            return "Missing required input: transformation_summary"

        # Check for qc_report
        qc_report = context.get("qc_report")
        if qc_report is None:
            return "Missing required input: qc_report"

        # Check if LLM is available
        if not self.is_enabled():
            return "LLM service not available (missing API key or anthropic package)"

        return None

    def execute(self, context: PipelineContext) -> AgentResult:
        """
        Execute the LLM review of harmonized output.

        Reads from context:
        - harmonized_df: DataFrame with harmonized data
        - transformation_summary: List of transformations applied
        - qc_report: QC report with issues

        Writes to context:
        - review_result: ReviewResult with findings and corrections

        Returns:
            AgentResult with success/failure and review data
        """
        self._update_status(AgentStatus.RUNNING, "Starting LLM review...", 0.1)

        try:
            # Get inputs from context
            harmonized_df = context.get("harmonized_df")
            transformation_summary = context.get("transformation_summary", [])
            qc_report = context.get("qc_report", {})

            # Extract QC issues from report
            qc_issues = qc_report.get("issues", []) if isinstance(qc_report, dict) else []

            self._update_status(AgentStatus.RUNNING, "Preparing data sample...", 0.2)

            # Convert DataFrame sample to list of dicts for LLM
            data_sample = harmonized_df.head(20).to_dict('records')

            self._update_status(AgentStatus.RUNNING, "Calling LLM for review...", 0.3)

            # Call LLM for review
            response = self.llm_service.review_harmonized_output(
                harmonized_data=data_sample,
                transformation_log=transformation_summary,
                qc_issues=qc_issues,
                spec_summary=DM_SPEC_SUMMARY
            )

            self._update_status(AgentStatus.RUNNING, "Processing LLM response...", 0.8)

            if not response.success:
                review_result = ReviewResult(
                    success=False,
                    error=f"LLM review failed: {response.error}"
                )
                context.set("review_result", review_result)

                return AgentResult(
                    success=False,
                    error=f"LLM review failed: {response.error}",
                    error_type="LLMError",
                    data=review_result
                )

            result = response.result

            # Extract findings
            overall_quality = result.get('overall_quality', 'unknown')
            issues_found = result.get('issues_found', [])
            corrections = result.get('corrections', [])
            summary = result.get('summary', '')

            # Build metadata
            metadata = {
                "agent": self.name,
                "timestamp": datetime.now().isoformat(),
                "llm_enabled": True,
                "tokens_used": response.tokens_used,
                "rows_reviewed": len(data_sample),
                "issues_found_count": len(issues_found),
                "corrections_suggested": len(corrections)
            }

            # Create review result
            review_result = ReviewResult(
                success=True,
                overall_quality=overall_quality,
                issues_found=issues_found,
                corrections=corrections,
                summary=summary,
                metadata=metadata
            )

            # Write result to context as dict (app.py expects dict with .get() method)
            review_dict = {
                "success": review_result.success,
                "overall_quality": review_result.overall_quality,
                "issues_found": review_result.issues_found,
                "corrections": review_result.corrections,
                "summary": review_result.summary,
                "metadata": review_result.metadata,
                "error": review_result.error
            }
            context.set("review_result", review_dict)

            self._update_status(AgentStatus.SUCCESS, "Review complete", 1.0)

            return AgentResult(
                success=True,
                data={"review_result": review_dict},
                metadata=metadata
            )

        except Exception as e:
            import traceback
            error_msg = f"Review failed: {str(e)}"

            # Create error result as dict
            review_dict = {
                "success": False,
                "overall_quality": "unknown",
                "issues_found": [],
                "corrections": [],
                "summary": "",
                "metadata": {},
                "error": f"{error_msg}\n{traceback.format_exc()}"
            }
            context.set("review_result", review_dict)

            return AgentResult(
                success=False,
                error=error_msg,
                error_type=type(e).__name__,
                metadata={"traceback": traceback.format_exc()},
                data={"review_result": review_dict}
            )

    def get_progress_weight(self) -> float:
        """
        LLM review is relatively lightweight compared to data processing.
        """
        return 0.5

    def apply_corrections(
        self,
        harmonized_df: pd.DataFrame,
        corrections: List[Dict[str, Any]]
    ) -> tuple[pd.DataFrame, int]:
        """
        Apply suggested corrections to the DataFrame.

        This is a separate method to allow controlled application of corrections
        after user review.

        Args:
            harmonized_df: DataFrame to correct
            corrections: List of corrections from review

        Returns:
            Tuple of (corrected_df, corrections_applied_count)
        """
        if not corrections:
            return harmonized_df, 0

        corrected_df = harmonized_df.copy()
        corrections_applied = 0

        for correction in corrections:
            try:
                var = correction.get('variable')
                row_idx = correction.get('row_index')
                new_value = correction.get('corrected_value')

                if var and var in corrected_df.columns and new_value is not None:
                    if row_idx is not None:
                        # Specific row correction
                        corrected_df.loc[row_idx, var] = new_value
                        corrections_applied += 1
                    else:
                        # All rows with current value
                        current = correction.get('current_value')
                        if current:
                            mask = corrected_df[var] == current
                            corrected_df.loc[mask, var] = new_value
                            corrections_applied += mask.sum()
            except Exception:
                # Log but don't fail on individual correction errors
                pass

        return corrected_df, corrections_applied


def run_review_agent(
    context: PipelineContext,
    progress_callback: Optional[ProgressCallback] = None
) -> AgentResult:
    """
    Convenience function to run the Review Agent.

    Args:
        context: Pipeline context with harmonized_df, transformation_summary, qc_report
        progress_callback: Optional callback for progress updates

    Returns:
        AgentResult with review findings
    """
    agent = ReviewAgent(progress_callback=progress_callback)
    return agent.run(context)


if __name__ == "__main__":
    print("Review Agent v2 - run via orchestrator or import")
