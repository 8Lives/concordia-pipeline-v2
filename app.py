"""
Concordia Harmonization Pipeline v2 - Streamlit Web Interface

Key improvements over v1:
- Real-time progress tracking with actual stage callbacks
- Per-stage status indicators (running, success, failed, skipped)
- Graceful handling of optional agent failures
- Clear error attribution when things go wrong
- Visual execution timeline
"""
import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
from datetime import datetime
import zipfile
import io
import time
from typing import Optional
import threading

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import AgenticOrchestrator, PipelineResult, PipelineStatus
from agents.base import AgentStatus

# LLM availability will be checked inside main() to avoid Streamlit context issues
LLM_AVAILABLE = None  # Will be set in main()


def check_llm_availability() -> bool:
    """Check if LLM service is available. Must be called within Streamlit context."""
    try:
        # Try to get API key from Streamlit secrets first, then env var
        api_key = None
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass

        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            return False

        from agents.llm_service import get_llm_service
        _llm_service = get_llm_service(api_key=api_key)
        return _llm_service.is_enabled()
    except Exception as e:
        print(f"LLM check failed: {e}")
        return False


# Stage display configuration
STAGE_CONFIG = {
    "ingest": {"icon": "📥", "name": "Ingest & Convert"},
    "map": {"icon": "🗺️", "name": "Map & Standardize"},
    "harmonize": {"icon": "🔧", "name": "Harmonize Values"},
    "qc": {"icon": "✅", "name": "QC & Report"},
    "review": {"icon": "🤖", "name": "LLM Review"}
}

STATUS_ICONS = {
    AgentStatus.PENDING: "⏳",
    AgentStatus.RUNNING: "🔄",
    AgentStatus.SUCCESS: "✅",
    AgentStatus.FAILED: "❌",
    AgentStatus.TIMEOUT: "⏰",
    AgentStatus.SKIPPED: "⏭️",
    AgentStatus.RETRYING: "🔁"
}


def main():
    global LLM_AVAILABLE

    # Page config
    st.set_page_config(
        page_title="Concordia DM Harmonization v2",
        page_icon="🔬",
        layout="wide"
    )

    # Check LLM availability (must be done inside Streamlit context)
    if LLM_AVAILABLE is None:
        LLM_AVAILABLE = check_llm_availability()

    # Header
    st.title("🔬 Concordia DM Harmonization Pipeline v2")
    st.markdown("""
    Upload your clinical trial Demographics (DM) domain dataset to harmonize it according to the
    CRDSA harmonization specification. **v2 features real-time progress tracking and improved error handling.**
    """)

    st.divider()

    # LLM Status indicator
    col1, col2 = st.columns([3, 1])
    with col1:
        if LLM_AVAILABLE:
            st.success("🤖 **LLM Review Enabled** - Claude will review output and suggest corrections")
        else:
            st.info("💡 **LLM Review Disabled** - Set ANTHROPIC_API_KEY in secrets to enable AI-powered review")

    with col2:
        st.caption("Pipeline v2.0 - Agentic Architecture")

    st.divider()

    # File upload section
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 Dataset Upload")
        data_file = st.file_uploader(
            "Upload your dataset",
            type=['sas7bdat', 'xpt', 'csv', 'xlsx', 'xls'],
            help="Supported formats: SAS7BDAT, XPT (SAS Transport), CSV, Excel"
        )

        if data_file:
            st.success(f"✓ Uploaded: {data_file.name}")
            st.caption(f"Size: {data_file.size / 1024:.1f} KB")

    with col2:
        st.subheader("📖 Data Dictionary (Optional)")
        dict_file = st.file_uploader(
            "Upload data dictionary",
            type=['xlsx', 'xls', 'csv'],
            help="Optional: Provide a data dictionary to decode coded values"
        )

        if dict_file:
            st.success(f"✓ Uploaded: {dict_file.name}")

    st.divider()

    # Process button
    if data_file:
        if st.button("🚀 Run Harmonization Pipeline", type="primary", use_container_width=True):
            run_pipeline(data_file, dict_file)
    else:
        st.info("👆 Please upload a dataset to begin.")

    # Footer
    st.divider()
    st.caption("Concordia Harmonization Pipeline v2.0 | CRDSA Clinical Research Data Sharing Alliance")


def run_pipeline(data_file, dict_file):
    """Run the harmonization pipeline with real-time progress tracking."""

    # Initialize session state for progress tracking
    if 'stage_status' not in st.session_state:
        st.session_state.stage_status = {}

    # Create progress display area
    progress_container = st.container()

    with progress_container:
        st.subheader("⚡ Pipeline Execution")

        # Overall progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Stage status display
        stage_cols = st.columns(5 if LLM_AVAILABLE else 4)
        stage_placeholders = {}

        stages = ["ingest", "map", "harmonize", "qc"]
        if LLM_AVAILABLE:
            stages.append("review")

        for i, stage in enumerate(stages):
            with stage_cols[i]:
                config = STAGE_CONFIG.get(stage, {"icon": "📦", "name": stage})
                stage_placeholders[stage] = st.empty()
                stage_placeholders[stage].markdown(
                    f"⏳ **{config['name']}**\n\n*Pending*"
                )

    # Create temp directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_path = temp_path / "output"
        output_path.mkdir()

        # Save uploaded files
        data_path = temp_path / data_file.name
        with open(data_path, 'wb') as f:
            f.write(data_file.getvalue())

        dict_path = None
        if dict_file:
            dict_path = temp_path / dict_file.name
            with open(dict_path, 'wb') as f:
                f.write(dict_file.getvalue())

        # Progress callback for real-time updates
        def progress_callback(
            current_stage: int,
            total_stages: int,
            stage_name: str,
            stage_status: AgentStatus,
            message: str,
            overall_pct: float
        ):
            """Callback invoked by orchestrator for progress updates."""
            # Update progress bar
            progress_bar.progress(min(overall_pct, 1.0))

            # Update status text
            config = STAGE_CONFIG.get(stage_name, {"icon": "📦", "name": stage_name})
            status_icon = STATUS_ICONS.get(stage_status, "❓")
            status_text.markdown(
                f"{status_icon} **Stage {current_stage}/{total_stages}: {config['name']}** - {message}"
            )

            # Update stage card
            if stage_name in stage_placeholders:
                if stage_status == AgentStatus.RUNNING:
                    stage_placeholders[stage_name].markdown(
                        f"🔄 **{config['name']}**\n\n*{message}*"
                    )
                elif stage_status == AgentStatus.SUCCESS:
                    stage_placeholders[stage_name].markdown(
                        f"✅ **{config['name']}**\n\n*Complete*"
                    )
                elif stage_status == AgentStatus.FAILED:
                    stage_placeholders[stage_name].markdown(
                        f"❌ **{config['name']}**\n\n*Failed*"
                    )
                elif stage_status == AgentStatus.TIMEOUT:
                    stage_placeholders[stage_name].markdown(
                        f"⏰ **{config['name']}**\n\n*Timeout*"
                    )
                elif stage_status == AgentStatus.SKIPPED:
                    stage_placeholders[stage_name].markdown(
                        f"⏭️ **{config['name']}**\n\n*Skipped*"
                    )

        # Run the pipeline
        try:
            status_text.text("🔄 Initializing pipeline...")
            progress_bar.progress(0.02)

            # Ensure API key is available to agents via environment variable
            if LLM_AVAILABLE:
                try:
                    api_key = st.secrets.get("ANTHROPIC_API_KEY")
                    if api_key:
                        os.environ["ANTHROPIC_API_KEY"] = api_key
                except Exception:
                    pass

            orchestrator = AgenticOrchestrator(
                output_dir=str(output_path),
                use_llm=LLM_AVAILABLE,
                progress_callback=progress_callback
            )

            # Execute pipeline with real progress tracking
            result = orchestrator.run(
                str(data_path),
                str(dict_path) if dict_path else None,
                save_outputs=True,
                enable_llm_review=LLM_AVAILABLE
            )

            progress_bar.progress(1.0)

            if result.success:
                if result.status == PipelineStatus.PARTIAL_SUCCESS:
                    status_text.markdown("⚠️ **Pipeline completed with warnings** (optional stages failed)")
                else:
                    status_text.markdown("✅ **Pipeline completed successfully!**")
                display_results(result, output_path)
            else:
                status_text.markdown(f"❌ **Pipeline failed at stage: {result.error_stage}**")
                st.error(f"Error: {result.error}")

                # Show stage results for debugging
                if result.stage_results:
                    with st.expander("🔍 View Stage Details"):
                        for stage_name, stage_result in result.stage_results.items():
                            if stage_result.success:
                                st.write(f"✅ {stage_name}: Success ({stage_result.execution_time_ms}ms)")
                            else:
                                st.write(f"❌ {stage_name}: {stage_result.error}")
                                if stage_result.metadata.get('traceback'):
                                    st.code(stage_result.metadata['traceback'])

        except Exception as e:
            progress_bar.progress(1.0)
            status_text.markdown("❌ **Pipeline failed with error**")
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def display_results(result: PipelineResult, output_path: Path):
    """Display pipeline results and download options."""

    st.divider()
    st.subheader("📊 Results")

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    ingest_meta = result.run_metadata.get('stages', {}).get('ingest', {})
    harmonize_meta = result.run_metadata.get('stages', {}).get('harmonize', {})
    qc_meta = result.run_metadata.get('stages', {}).get('qc', {})
    review_meta = result.run_metadata.get('stages', {}).get('review', {})

    with col1:
        st.metric("Trial ID", ingest_meta.get('trial_id', 'Unknown'))

    with col2:
        st.metric("Rows Processed", harmonize_meta.get('rows_out', 0))

    with col3:
        st.metric("QC Issues", qc_meta.get('total_issues', 0))

    with col4:
        dict_status = "Yes" if ingest_meta.get('dictionary_loaded') else "No"
        st.metric("Dictionary Used", dict_status)

    with col5:
        if review_meta.get('success', False):
            quality = review_meta.get('overall_quality', 'N/A')
            st.metric("LLM Quality", quality.replace('_', ' ').title())
        elif review_meta.get('error'):
            st.metric("LLM Review", "Failed")
        else:
            st.metric("LLM Review", "Disabled")

    st.divider()

    # Execution timeline
    with st.expander("⏱️ Execution Timeline"):
        timeline_data = []
        for stage_name, stage_info in result.run_metadata.get('stages', {}).items():
            config = STAGE_CONFIG.get(stage_name, {"icon": "📦", "name": stage_name})
            timeline_data.append({
                "Stage": f"{config['icon']} {config['name']}",
                "Status": "✅ Success" if stage_info.get('success') else "❌ Failed",
                "Time (ms)": stage_info.get('execution_time_ms', 0),
                "Retries": stage_info.get('retries_used', 0)
            })
        if timeline_data:
            st.dataframe(pd.DataFrame(timeline_data), use_container_width=True)

    # Tabs for different views
    has_review = result.review_result is not None

    if has_review:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Harmonized Data",
            "⚠️ QC Report",
            "🤖 LLM Review",
            "🔄 Transformations",
            "📥 Downloads"
        ])
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Harmonized Data",
            "⚠️ QC Report",
            "🔄 Transformations",
            "📥 Downloads"
        ])

    with tab1:
        st.subheader("Harmonized Dataset Preview")
        if result.harmonized_df is not None:
            st.dataframe(result.harmonized_df.head(50), use_container_width=True)
            st.caption(f"Showing first 50 of {len(result.harmonized_df)} rows")

    with tab2:
        st.subheader("QC Issues")
        if result.qc_report is not None and len(result.qc_report) > 0:
            st.dataframe(result.qc_report, use_container_width=True)

            st.subheader("Issues by Type")
            issues_by_type = qc_meta.get('issues_by_type', {})
            if issues_by_type:
                for issue_type, count in issues_by_type.items():
                    st.write(f"• **{issue_type}**: {count} issue(s)")
        else:
            st.success("✅ No QC issues found!")

    # LLM Review tab
    if has_review:
        with tab3:
            st.subheader("🤖 LLM Review (Claude)")

            review = result.review_result

            quality = review.get('overall_quality', 'unknown')
            quality_colors = {
                'good': '🟢',
                'acceptable': '🟡',
                'needs_attention': '🔴',
                'unknown': '⚪'
            }
            quality_icon = quality_colors.get(quality, '⚪')
            st.markdown(f"### Overall Quality: {quality_icon} **{quality.replace('_', ' ').title()}**")

            if review.get('summary'):
                st.info(review['summary'])

            st.subheader("Issues Identified by LLM")
            issues = review.get('issues_found', [])
            if issues:
                for i, issue in enumerate(issues, 1):
                    severity = issue.get('severity', 'medium')
                    severity_badge = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(severity, '⚪')
                    with st.expander(f"{severity_badge} {issue.get('variable', 'Unknown')}: {issue.get('issue', 'No description')[:50]}..."):
                        st.write(f"**Variable:** {issue.get('variable')}")
                        st.write(f"**Issue:** {issue.get('issue')}")
                        st.write(f"**Severity:** {severity}")
                        if issue.get('suggested_fix'):
                            st.write(f"**Suggested Fix:** {issue.get('suggested_fix')}")
            else:
                st.success("✅ No additional issues identified by LLM review")

            st.subheader("Suggested Corrections")
            corrections = review.get('corrections', [])
            if corrections:
                corrections_df = pd.DataFrame(corrections)
                st.dataframe(corrections_df, use_container_width=True)
                st.caption("Note: Corrections are suggestions only. Review before applying.")
            else:
                st.info("No specific corrections suggested")

        # Transformations tab (shifted)
        with tab4:
            st.subheader("Variable Transformations")
            if result.transformation_summary:
                transform_df = pd.DataFrame(result.transformation_summary)
                st.dataframe(transform_df, use_container_width=True)

        downloads_tab = tab5
    else:
        with tab3:
            st.subheader("Variable Transformations")
            if result.transformation_summary:
                transform_df = pd.DataFrame(result.transformation_summary)
                st.dataframe(transform_df, use_container_width=True)

        downloads_tab = tab4

    with downloads_tab:
        st.subheader("Download Results")

        output_files = result.run_metadata.get('output_files', {})

        col1, col2 = st.columns(2)

        with col1:
            if 'harmonized_csv' in output_files:
                csv_path = output_files['harmonized_csv']
                if os.path.exists(csv_path):
                    with open(csv_path, 'rb') as f:
                        st.download_button(
                            "📄 Download Harmonized Dataset (CSV)",
                            f.read(),
                            file_name=Path(csv_path).name,
                            mime="text/csv",
                            use_container_width=True
                        )

            if 'qc_csv' in output_files:
                qc_path = output_files['qc_csv']
                if os.path.exists(qc_path):
                    with open(qc_path, 'rb') as f:
                        st.download_button(
                            "⚠️ Download QC Report (CSV)",
                            f.read(),
                            file_name=Path(qc_path).name,
                            mime="text/csv",
                            use_container_width=True
                        )

        with col2:
            if 'transformation_csv' in output_files:
                transform_path = output_files['transformation_csv']
                if os.path.exists(transform_path):
                    with open(transform_path, 'rb') as f:
                        st.download_button(
                            "🔄 Download Transformation Summary (CSV)",
                            f.read(),
                            file_name=Path(transform_path).name,
                            mime="text/csv",
                            use_container_width=True
                        )

            if 'report_docx' in output_files:
                docx_path = output_files['report_docx']
                if os.path.exists(docx_path):
                    with open(docx_path, 'rb') as f:
                        st.download_button(
                            "📑 Download Full Report (DOCX)",
                            f.read(),
                            file_name=Path(docx_path).name,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )

        st.divider()

        # Download all as ZIP
        st.subheader("📦 Download All Files")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_type, file_path in output_files.items():
                if os.path.exists(file_path):
                    zf.write(file_path, Path(file_path).name)

        zip_buffer.seek(0)
        trial_id = ingest_meta.get('trial_id', 'output')
        st.download_button(
            "📦 Download All Results (ZIP)",
            zip_buffer.getvalue(),
            file_name=f"{trial_id}_harmonization_results.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary"
        )


if __name__ == "__main__":
    main()
