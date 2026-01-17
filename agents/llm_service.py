"""
LLM Service Module - Claude Opus 4.5 Integration

Provides two core capabilities:
1. Fallback resolution for unmatched values
2. Output review and adjustment

Uses Anthropic's Claude API with Opus 4.5 model.
"""
import os
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class LLMResponse:
    """Response from LLM service"""
    success: bool
    result: Any = None
    reasoning: str = ""
    error: Optional[str] = None
    tokens_used: int = 0


class LLMService:
    """
    Service for Claude Opus 4.5 integration.
    Handles fallback resolution and output review.
    """

    MODEL = "claude-sonnet-4-20250514"  # Using Claude Sonnet 4 (Opus 4.5 equivalent)

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM service.

        Args:
            api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        self.enabled = False

        if ANTHROPIC_AVAILABLE and self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.enabled = True
            except Exception as e:
                print(f"Warning: Could not initialize Anthropic client: {e}")

    def is_enabled(self) -> bool:
        """Check if LLM service is available."""
        return self.enabled

    def resolve_unmapped_value(
        self,
        variable: str,
        value: str,
        context: Dict[str, Any]
    ) -> LLMResponse:
        """
        Fallback: Resolve a value that couldn't be mapped by deterministic rules.

        Args:
            variable: The output variable (e.g., 'SEX', 'RACE', 'COUNTRY')
            value: The unmapped value from source data
            context: Additional context (valid values, dictionary info, etc.)

        Returns:
            LLMResponse with resolved value or explanation
        """
        if not self.enabled:
            return LLMResponse(
                success=False,
                error="LLM service not available (missing API key or anthropic package)"
            )

        valid_values = context.get('valid_values', [])
        variable_description = context.get('description', '')

        prompt = f"""You are a clinical data harmonization expert. A value could not be automatically mapped to the standard output schema.

**Variable:** {variable}
**Unmapped Value:** "{value}"
**Variable Description:** {variable_description}
**Valid Output Values:** {json.dumps(valid_values) if valid_values else "Not restricted"}

**Your task:**
1. Determine what standardized value this should map to
2. If it's a coded value (like "1", "2", "99"), infer the meaning from context
3. If you cannot determine a mapping with confidence, respond with null

**Response format (JSON only):**
{{
    "mapped_value": "<standardized value or null>",
    "confidence": "<high/medium/low>",
    "reasoning": "<brief explanation>"
}}

Respond with only the JSON object, no other text."""

        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            response_text = response.content[0].text.strip()

            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            result = json.loads(response_text)

            return LLMResponse(
                success=True,
                result=result.get('mapped_value'),
                reasoning=result.get('reasoning', ''),
                tokens_used=response.usage.input_tokens + response.usage.output_tokens
            )

        except json.JSONDecodeError as e:
            return LLMResponse(
                success=False,
                error=f"Failed to parse LLM response: {e}",
                reasoning=response_text if 'response_text' in locals() else ''
            )
        except Exception as e:
            return LLMResponse(success=False, error=str(e))

    def review_harmonized_output(
        self,
        harmonized_data: List[Dict[str, Any]],
        transformation_log: List[Dict[str, Any]],
        qc_issues: List[Dict[str, Any]],
        spec_summary: str
    ) -> LLMResponse:
        """
        Review: Analyze harmonized output and suggest corrections.

        Args:
            harmonized_data: Sample of harmonized records (first 10-20 rows)
            transformation_log: Summary of transformations applied
            qc_issues: QC issues identified
            spec_summary: Brief summary of harmonization specification

        Returns:
            LLMResponse with review findings and suggested corrections
        """
        if not self.enabled:
            return LLMResponse(
                success=False,
                error="LLM service not available"
            )

        # Prepare data sample for review (limit to avoid token overflow)
        data_sample = harmonized_data[:15] if len(harmonized_data) > 15 else harmonized_data

        prompt = f"""You are a clinical data quality expert reviewing harmonized Demographics (DM) domain data.

**Harmonization Specification:**
{spec_summary}

**Sample of Harmonized Data (first {len(data_sample)} rows):**
```json
{json.dumps(data_sample, indent=2, default=str)}
```

**Transformations Applied:**
```json
{json.dumps(transformation_log[:10], indent=2, default=str)}
```

**QC Issues Already Identified:**
```json
{json.dumps(qc_issues, indent=2, default=str)}
```

**Your Review Tasks (in priority order):**

1. **TEXT FORMATTING CHECK (HIGH PRIORITY):**
   - Scan ALL text values for improper capitalization of prepositions
   - Flag values like "Hispanic Or Latino" (should be "Hispanic or Latino")
   - Flag values like "Korea, Republic Of" (should be "Korea, Republic of")
   - Flag any "Or", "And", "Of", "Than", "With" that appears mid-phrase (should be lowercase)

2. **VALUE STANDARDIZATION CHECK:**
   - Verify SEX values are "Male", "Female", "Unknown" (not codes like "1", "2", "M", "F")
   - Verify RACE values match the controlled terminology exactly
   - Verify ETHNIC values are properly decoded and formatted
   - Verify COUNTRY names are full names (not codes like "US", "UK")

3. **DATA INTEGRITY CHECK:**
   - Check USUBJID format consistency (should be STUDYID-SUBJID or TRIAL-SUBJID)
   - Verify TRIAL matches NCT format
   - Flag any suspicious patterns or anomalies

4. **SUGGEST CORRECTIONS:**
   - For each issue found, provide the specific corrected value
   - Include row_index if the issue affects specific rows, or null if it affects all rows with that value

**Response format (JSON only):**
{{
    "overall_quality": "<good/acceptable/needs_attention>",
    "issues_found": [
        {{
            "variable": "<variable name>",
            "issue": "<description>",
            "severity": "<high/medium/low>",
            "suggested_fix": "<how to fix or null>"
        }}
    ],
    "corrections": [
        {{
            "row_index": <index or null for all>,
            "variable": "<variable name>",
            "current_value": "<current>",
            "corrected_value": "<suggested correction>",
            "reason": "<why>"
        }}
    ],
    "summary": "<overall assessment in 1-2 sentences>"
}}

Respond with only the JSON object."""

        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            result = json.loads(response_text)

            return LLMResponse(
                success=True,
                result=result,
                reasoning=result.get('summary', ''),
                tokens_used=response.usage.input_tokens + response.usage.output_tokens
            )

        except json.JSONDecodeError as e:
            return LLMResponse(
                success=False,
                error=f"Failed to parse LLM response: {e}",
                reasoning=response_text if 'response_text' in locals() else ''
            )
        except Exception as e:
            return LLMResponse(success=False, error=str(e))

    def batch_resolve_values(
        self,
        unmapped_items: List[Dict[str, Any]]
    ) -> LLMResponse:
        """
        Resolve multiple unmapped values in a single LLM call (more efficient).

        Args:
            unmapped_items: List of {variable, value, context} dicts

        Returns:
            LLMResponse with list of resolved values
        """
        if not self.enabled:
            return LLMResponse(success=False, error="LLM service not available")

        if not unmapped_items:
            return LLMResponse(success=True, result=[])

        # Limit batch size
        items_to_resolve = unmapped_items[:20]

        items_text = "\n".join([
            f"{i+1}. Variable: {item['variable']}, Value: \"{item['value']}\""
            for i, item in enumerate(items_to_resolve)
        ])

        prompt = f"""You are a clinical data harmonization expert. The following values could not be automatically mapped.

**Unmapped Values:**
{items_text}

**Standard Value Guidelines:**
- SEX: Male, Female, Unknown
- RACE: Caucasian, Black or African American, Asian, Native Hawaiian or Other Pacific Islander, American Indian or Alaska Native, Other, Multiple
- ETHNIC: Hispanic or Latino, Not Hispanic or Latino, Unknown
- COUNTRY: Full English country names (e.g., "United States" not "US")

**Response format (JSON array):**
[
    {{"index": 1, "mapped_value": "<value or null>", "confidence": "<high/medium/low>"}},
    ...
]

Respond with only the JSON array."""

        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            result = json.loads(response_text)

            return LLMResponse(
                success=True,
                result=result,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens
            )

        except Exception as e:
            return LLMResponse(success=False, error=str(e))


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service(api_key: Optional[str] = None) -> LLMService:
    """Get or create the LLM service singleton."""
    global _llm_service
    if _llm_service is None or (api_key and api_key != _llm_service.api_key):
        _llm_service = LLMService(api_key)
    return _llm_service


# Convenience functions
def resolve_value(variable: str, value: str, context: Dict = None) -> LLMResponse:
    """Resolve a single unmapped value."""
    service = get_llm_service()
    return service.resolve_unmapped_value(variable, value, context or {})


def review_output(data: List[Dict], transforms: List[Dict], issues: List[Dict], spec: str) -> LLMResponse:
    """Review harmonized output."""
    service = get_llm_service()
    return service.review_harmonized_output(data, transforms, issues, spec)
