# Concordia DM Harmonization Pipeline v2

A multi-agent pipeline for harmonizing clinical trial Demographics (DM) domain data according to the CRDSA harmonization specification.

## Features

- **Agentic Architecture**: Each pipeline stage runs as an independent agent with timeout handling, retry logic, and progress callbacks
- **Real-time Progress Tracking**: See exactly which stage is running, not just a spinning wheel
- **Graceful Degradation**: Optional stages (like LLM review) can fail without stopping the pipeline
- **LLM-Powered Review**: Optional Claude integration for intelligent data quality review
- **Enhanced Column Mapping**: 28+ synonyms for SUBJID and expanded support for common column naming conventions

## Pipeline Stages

| Stage | Agent | Description |
|-------|-------|-------------|
| 1 | **Ingest & Convert** | Parse SAS7BDAT, XPT, CSV, or Excel files |
| 2 | **Map & Standardize** | Match source columns to output schema |
| 3 | **Harmonize Values** | Decode, normalize, and standardize values |
| 4 | **QC & Report** | Run quality checks and generate reports |
| 5 | **LLM Review** | (Optional) AI-powered review with Claude |

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path to `app.py`
5. Deploy

### Enable LLM Features

Add your Anthropic API key in Streamlit Cloud:
- Go to **Settings** → **Secrets**
- Add: `ANTHROPIC_API_KEY = "sk-ant-xxxxx"`

## Input Formats

- **SAS7BDAT** - SAS data files
- **XPT** - SAS Transport files
- **CSV** - Comma-separated values
- **XLSX/XLS** - Excel spreadsheets

Optional: Upload a data dictionary (Excel/CSV) to decode coded values.

## Output Files

| File | Description |
|------|-------------|
| `{TRIAL}_DM_harmonized_{timestamp}.csv` | Harmonized dataset |
| `{TRIAL}_QC_report_{timestamp}.csv` | Quality control issues |
| `{TRIAL}_transformation_summary_{timestamp}.csv` | Transformation details |
| `{TRIAL}_Harmonization_Report_{timestamp}.docx` | Full report document |

## Architecture

```
concordia_pipeline_v2/
├── app.py                 # Streamlit UI
├── orchestrator.py        # Pipeline coordination
├── agents/
│   ├── base.py            # AgentBase with timeout/retry
│   ├── ingest_agent.py    # File parsing
│   ├── map_agent.py       # Column mapping
│   ├── harmonize_agent.py # Value harmonization
│   ├── qc_agent.py        # Quality checks
│   ├── review_agent.py    # LLM review (optional)
│   └── llm_service.py     # Claude API integration
├── config/
│   └── schema.py          # Output schema & mappings
└── utils/
    └── helpers.py         # Utility functions
```

## Key Improvements in v2

| Feature | v1 | v2 |
|---------|----|----|
| Progress Tracking | Simulated | Real callbacks per stage |
| Timeouts | None | 60-120s per agent |
| Retry Logic | None | Configurable with backoff |
| Error Handling | All-or-nothing | Per-agent isolation |
| SUBJID Mapping | 5 synonyms | 28 synonyms |

## Configuration

Agent timeouts and retry behavior can be configured in `orchestrator.py`:

```python
ingest_config = AgentConfig(
    timeout_seconds=60.0,
    max_retries=0,
    required=True
)
```

## License

CRDSA Clinical Research Data Sharing Alliance

---

Built with [Streamlit](https://streamlit.io) and [Claude](https://anthropic.com)
