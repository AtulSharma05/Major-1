"""Enterprise Streamlit interface for the Self-Validating Autonomous Data Analytics Agent.

Architectural Context:
    Presents an integrated, professional workbench for:
    1. Dataset Management: Ingestion, schema inspection, profiling, and data quality metrics.
    2. Query Workspace: Natural-language query execution with multi-step DAG planning.
    3. Results Presentation: Consolidated metrics, tabular datasets, charts, and generated SQL.
    4. Workflow Trace: Inspection of the 7-node LangGraph execution state machine.
    5. Validation Centre: 3-Layer verification audit (Execution, Numerical, and Semantic).
    6. Self-Correction History: Complete attempt history, diagnostics, and retry budget tracking.
"""

from __future__ import annotations

import importlib
import os
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

import src.agent.data.ingestion as ingestion_module
import src.ui.helpers as ui_helpers
from src.agent.codegen.generator import CodeGenerator, OllamaClient
from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import format_profile_for_prompt, profile_table
from src.agent.execution.engine import ControlledExecutionEngine
from src.agent.planning.planner import QueryPlanner
from src.agent.validation.contracts import ValidationStatus, WorkflowValidationReport
from src.agent.validation.semantic import SemanticValidator
from src.agent.workflow.graph import build_analytical_graph
from src.agent.workflow.state import AgentState

importlib.reload(ingestion_module)
importlib.reload(ui_helpers)

(
    format_correction_history_display,
    format_data_quality_metrics,
    format_numerical_validation_display,
    format_semantic_validation_display,
    format_step_validation_summary,
    get_excel_sheet_names,
    get_status_badge_html,
    get_table_preview_df,
    get_workflow_nodes_summary,
    get_workflow_execution_path,
    ingest_uploaded_file,
    prepare_chart_data,
    prepare_comparison_chart_df,
    profile_to_display_df,
    serialize_validation_report_for_json,
    validation_report_to_display_rows,
) = (
    ui_helpers.format_correction_history_display,
    ui_helpers.format_data_quality_metrics,
    ui_helpers.format_numerical_validation_display,
    ui_helpers.format_semantic_validation_display,
    ui_helpers.format_step_validation_summary,
    ui_helpers.get_excel_sheet_names,
    ui_helpers.get_status_badge_html,
    ui_helpers.get_table_preview_df,
    ui_helpers.get_workflow_nodes_summary,
    ui_helpers.get_workflow_execution_path,
    ui_helpers.ingest_uploaded_file,
    ui_helpers.prepare_chart_data,
    ui_helpers.prepare_comparison_chart_df,
    ui_helpers.profile_to_display_df,
    ui_helpers.serialize_validation_report_for_json,
    ui_helpers.validation_report_to_display_rows,
)


def render_error_boundary(section_name: str, exc: Exception) -> None:
    """Render a concise enterprise error notification with collapsible developer diagnostics."""
    st.error(
        f"Application Rendering Notice: Unable to complete rendering for {section_name}. "
        "The analytical engine finished, but the presentation component "
        "encountered an internal issue."
    )
    with st.expander("Technical Diagnostics", expanded=False):
        st.caption("Detailed Python exception traceback for debugging:")
        st.code(traceback.format_exc(), language="text")


def register_all_sample_datasets(session_db: SessionDatabase) -> list[str]:
    """Ensure all sample CSV files in data/ are registered into the session database."""
    registered: list[str] = []
    data_dir = Path("data")
    if data_dir.exists():
        for csv_file in sorted(data_dir.glob("*.csv")):
            table_name = csv_file.stem
            if not session_db.is_table_allowed(table_name):
                try:
                    session_db.register_csv(csv_file, table_name)
                except Exception:
                    continue
            registered.append(table_name)
    return registered


# ---------------------------------------------------------------------------
# Page Configuration & Professional Dark Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Autonomous Data Analytics Agent",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom enterprise dark theme CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #E2E8F0;
}

.stApp {
    background-color: #0B0F17 !important;
}

/* Sidebar Dark Styling */
[data-testid="stSidebar"] {
    background-color: #0D131F !important;
    border-right: 1px solid #1E293B !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F8FAFC !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.25rem !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown p {
    color: #CBD5E1 !important;
    font-size: 0.85rem !important;
    line-height: 1.45 !important;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #94A3B8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #E2E8F0 !important;
    font-weight: 600 !important;
    font-size: 0.825rem !important;
}

[data-testid="stSidebar"] hr {
    margin: 0.65rem 0 !important;
    border-color: #1E293B !important;
}

/* Secondary Descriptions & Global Captions Contrast */
.stCaption,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #94A3B8 !important;
    font-size: 0.825rem !important;
    font-weight: 500 !important;
}

/* Typography Hierarchy */
h1 {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    color: #F8FAFC !important;
    margin-top: 0.2rem !important;
    margin-bottom: 0.2rem !important;
}

h2, h3 {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #F1F5F9 !important;
    margin-top: 0.6rem !important;
    margin-bottom: 0.35rem !important;
}

h4 {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #E2E8F0 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.25rem !important;
}

h5 {
    font-size: 0.825rem !important;
    font-weight: 600 !important;
    color: #CBD5E1 !important;
    margin-top: 0.4rem !important;
    margin-bottom: 0.2rem !important;
}

/* Compact Metric Cards with High-Contrast Dark Styling */
[data-testid="stMetric"] {
    background-color: #131B2A !important;
    border: 1px solid #1E293B !important;
    border-radius: 6px !important;
    padding: 0.45rem 0.75rem !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.3) !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    color: #94A3B8 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
}

/* Compact Textual Status Badges (Dark Mode Tuned) */
.status-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.12rem 0.45rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    border-radius: 4px;
    border: 1px solid transparent;
    line-height: 1.25;
}

.status-verified {
    background-color: rgba(16, 185, 129, 0.15) !important;
    color: #34D399 !important;
    border-color: rgba(16, 185, 129, 0.35) !important;
}

.status-failed {
    background-color: rgba(239, 68, 68, 0.15) !important;
    color: #F87171 !important;
    border-color: rgba(239, 68, 68, 0.35) !important;
}

.status-inconclusive {
    background-color: rgba(245, 158, 11, 0.15) !important;
    color: #FBBF24 !important;
    border-color: rgba(245, 158, 11, 0.35) !important;
}

.status-unverified {
    background-color: rgba(148, 163, 184, 0.12) !important;
    color: #CBD5E1 !important;
    border-color: rgba(148, 163, 184, 0.25) !important;
}

.status-running {
    background-color: rgba(59, 130, 246, 0.15) !important;
    color: #60A5FA !important;
    border-color: rgba(59, 130, 246, 0.35) !important;
}

/* Hide Streamlit Top Header & Toolbar (Removes Top Black Bar) */
header[data-testid="stHeader"],
[data-testid="stHeader"] {
    background-color: transparent !important;
    background: transparent !important;
    color: transparent !important;
    border-bottom: none !important;
    box-shadow: none !important;
    height: 0px !important;
    min-height: 0px !important;
    padding: 0 !important;
}

[data-testid="stToolbar"] {
    display: none !important;
    visibility: hidden !important;
    height: 0px !important;
    width: 0px !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

#MainMenu, footer {
    display: none !important;
    visibility: hidden !important;
}

/* Dark DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #1E293B !important;
    border-radius: 6px !important;
    background-color: #131B2A !important;
}

/* Dark Expanders */
.streamlit-expanderHeader {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #E2E8F0 !important;
    background-color: #131B2A !important;
    border: 1px solid #1E293B !important;
    border-radius: 6px !important;
}

[data-testid="stExpander"] {
    border: 1px solid #1E293B !important;
    border-radius: 6px !important;
    background-color: #0E1524 !important;
}

/* Dark Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.5rem;
    border-bottom: 1px solid #1E293B !important;
    background-color: transparent;
    padding-bottom: 2px;
}

.stTabs [data-baseweb="tab"] {
    height: 2.35rem;
    padding: 0 0.4rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: #94A3B8 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s ease-in-out;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #F1F5F9 !important;
}

.stTabs [aria-selected="true"] {
    color: #38BDF8 !important;
    font-weight: 600 !important;
    border-bottom-color: #38BDF8 !important;
}

/* Dark Buttons & Execute CTA */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    border: none !important;
    font-size: 0.875rem !important;
    height: 2.5rem !important;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.25) !important;
    transition: all 0.15s ease-in-out !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 8px rgba(37, 99, 235, 0.35) !important;
}

.stButton > button:not([kind="primary"]) {
    background-color: #131B2A !important;
    color: #E2E8F0 !important;
    border: 1px solid #1E293B !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.825rem !important;
    transition: all 0.15s ease-in-out !important;
}
.stButton > button:not([kind="primary"]):hover {
    background-color: #1E293B !important;
    border-color: #38BDF8 !important;
    color: #F8FAFC !important;
}

/* Dark Inputs & Number Inputs */
input, textarea, [data-baseweb="select"] {
    background-color: #131B2A !important;
    color: #F8FAFC !important;
    border-color: #334155 !important;
}

div[data-testid="stNumberInput"] input {
    height: 2.5rem !important;
}

/* File Uploader Dark Styling */
[data-testid="stFileUploader"] {
    background-color: #131B2A !important;
    border: 1px dashed #334155 !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38BDF8 !important;
}

/* Tighten Layout & Spacing */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 96% !important;
}

hr {
    margin: 0.65rem 0 !important;
    border-color: #1E293B !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "db" not in st.session_state or not hasattr(st.session_state["db"], "cursor"):
    st.session_state["db"] = SessionDatabase()

db: SessionDatabase = st.session_state["db"]

if "active_table" not in st.session_state:
    st.session_state["active_table"] = None

if "current_question" not in st.session_state:
    st.session_state["current_question"] = "What is the total sales by region?"

if "workflow_execution_state" not in st.session_state:
    st.session_state["workflow_execution_state"] = None

if "ollama_url" not in st.session_state:
    st.session_state["ollama_url"] = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = "qwen2.5-coder:7b"

if "engine_mode" not in st.session_state:
    st.session_state["engine_mode"] = "Auto (Ollama LLM with Autonomous Intent Fallback)"

# Auto-register benchmark datasets on initial boot if no tables exist
if not db.get_allowed_tables():
    loaded_tables = register_all_sample_datasets(db)
    if "sample_superstore" in loaded_tables:
        st.session_state["active_table"] = "sample_superstore"
    elif loaded_tables:
        st.session_state["active_table"] = loaded_tables[0]

# ---------------------------------------------------------------------------
# Sidebar: Dataset Management & Session Status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Data Ingestion")
    st.caption("Session-isolated DuckDB storage for tabular datasets.")

    uploaded_files = st.file_uploader(
        label="Upload tabular files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Upload CSV or Excel files. Column identifiers will be sanitized automatically.",
    )

    if uploaded_files:
        for uf in uploaded_files:
            file_bytes = uf.getvalue()
            filename = uf.name
            ext = Path(filename).suffix.lower()

            sheet_name: str | int = 0
            if ext in {".xlsx", ".xls"}:
                try:
                    sheets = get_excel_sheet_names(file_bytes)
                    if len(sheets) > 1:
                        sheet_name = st.selectbox(
                            f"Sheet for {filename}",
                            options=sheets,
                            key=f"sheet_{filename}",
                        )
                except Exception as exc:
                    st.error(f"Cannot read sheets from {filename}: {exc}")
                    continue

            btn_key = f"reg_{filename}_{len(file_bytes)}"
            if st.button(f"Register {filename}", key=btn_key, use_container_width=True):
                with st.spinner(f"Ingesting {filename}..."):
                    success, table_or_msg, err_detail = ingest_uploaded_file(
                        db=db,
                        file_bytes=file_bytes,
                        filename=filename,
                        sheet_name=sheet_name,
                    )
                if success:
                    st.session_state["active_table"] = table_or_msg
                    st.session_state["workflow_execution_state"] = None
                    st.rerun()
                else:
                    st.error(f"Ingestion failed ({table_or_msg}): {err_detail}")

    st.markdown("---")
    st.markdown("### Benchmark Datasets")
    st.caption("Auto-discovered canonical verification datasets.")

    if st.button("Populate All Sample Datasets", use_container_width=True):
        loaded = register_all_sample_datasets(db)
        st.success(f"Populated {len(loaded)} dataset(s): {', '.join(loaded)}")
        if not st.session_state.get("active_table") and loaded:
            st.session_state["active_table"] = loaded[0]
        st.rerun()

    # Quick Switch Grid for all 4 canonical sample datasets
    col_bench1, col_bench2 = st.columns(2)
    if col_bench1.button("Superstore (1)", use_container_width=True):
        p = Path("data/sample_superstore.csv")
        if p.exists():
            db.register_csv(p, "sample_superstore")
            st.session_state["active_table"] = "sample_superstore"
            st.session_state["workflow_execution_state"] = None
            st.rerun()

    if col_bench2.button("Superstore 2", use_container_width=True):
        p = Path("data/sample_superstore_2.csv")
        if p.exists():
            db.register_csv(p, "sample_superstore_2")
            st.session_state["active_table"] = "sample_superstore_2"
            st.session_state["workflow_execution_state"] = None
            st.rerun()

    col_bench3, col_bench4 = st.columns(2)
    if col_bench3.button("Superstore 3", use_container_width=True):
        p = Path("data/sample_superstore_3.csv")
        if p.exists():
            db.register_csv(p, "sample_superstore_3")
            st.session_state["active_table"] = "sample_superstore_3"
            st.session_state["workflow_execution_state"] = None
            st.rerun()

    if col_bench4.button("Sample Retail", use_container_width=True):
        p = Path("data/sample_retail.csv")
        if p.exists():
            db.register_csv(p, "sample_retail")
            st.session_state["active_table"] = "sample_retail"
            st.session_state["workflow_execution_state"] = None
            st.rerun()

    st.markdown("---")
    st.markdown("### Active Dataset Selection")
    allowed_tables = db.get_allowed_tables()

    if allowed_tables:
        current_active = st.session_state.get("active_table")
        current_idx = (
            allowed_tables.index(current_active) if current_active in allowed_tables else 0
        )
        selected_table = st.selectbox(
            "Selected table",
            options=allowed_tables,
            index=current_idx,
            label_visibility="collapsed",
        )
        if selected_table != current_active:
            st.session_state["active_table"] = selected_table
            st.session_state["workflow_execution_state"] = None
            st.rerun()
    else:
        st.info("No tables currently registered in session.")

    st.markdown("---")
    st.markdown("### Execution & Model Settings")
    with st.expander("Connection & Engine Controls", expanded=False):
        c_mode = st.selectbox(
            "Engine Mode",
            options=[
                "Auto (Ollama LLM with Autonomous Intent Fallback)",
                "Autonomous Deterministic Engine (No Ollama Required)",
                "Ollama LLM Only (Strict)",
            ],
            index=0,
            key="engine_mode_select",
        )
        st.session_state["engine_mode"] = c_mode

        c_url = st.text_input(
            "Ollama Base URL",
            value=st.session_state["ollama_url"],
            help="HTTP endpoint for the Ollama local daemon (default http://localhost:11434).",
        )
        st.session_state["ollama_url"] = c_url

        c_model = st.text_input(
            "Model Name",
            value=st.session_state["ollama_model"],
            help="Target LLM model tag in local Ollama daemon.",
        )
        st.session_state["ollama_model"] = c_model

        if st.button("Test Connection", use_container_width=True):
            test_client = OllamaClient(base_url=c_url, model=c_model, timeout=5.0)
            health = test_client.check_health()
            if health.get("connected"):
                models_list = ", ".join(health.get("models", [])) or "None listed"
                st.success(
                    f"Connected (HTTP {health['status_code']}). Available models: {models_list}"
                )
            else:
                err_text = health.get("error", "Unknown error")
                code_text = health.get("status_code", "N/A")
                st.warning(
                    f"Connection unreachable (Status: {code_text}): {err_text}\n\n"
                    "Autonomous Intent Fallback is enabled to execute queries deterministically."
                )

    st.markdown("---")
    st.markdown("### Engine Status")
    st.caption(
        "**Storage:** DuckDB In-Memory Session\n\n"
        "**Execution:** Thread-Isolated Subprocess\n\n"
        "**Validation:** 3-Layer Invariant Verification\n\n"
        "**Max Retries:** 2 Retries (3 Total Attempts)"
    )

# ---------------------------------------------------------------------------
# Main Workbench Interface
# ---------------------------------------------------------------------------
active_table = st.session_state.get("active_table")

if not active_table or not db.is_table_allowed(active_table):
    st.title("Autonomous Data Analytics Agent")
    st.info(
        "No dataset is currently selected.\n\n"
        "Select an active dataset or click **Populate All Sample Datasets** "
        "in the sidebar to begin."
    )
    st.stop()

# Generate or retrieve active profile with defensive concurrency retry
try:
    profile = profile_table(db, active_table)
except Exception:
    import time

    time.sleep(0.1)
    try:
        profile = profile_table(db, active_table)
    except Exception as exc:
        st.error(f"Failed to profile active table `{active_table}`: {exc}")
        st.stop()

# Top Application Bar
top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    tag_style = (
        "background: rgba(56, 189, 248, 0.1); color: #38BDF8; font-size: 0.68rem; "
        "font-weight: 600; padding: 0.1rem 0.45rem; border-radius: 4px; "
        "border: 1px solid rgba(56, 189, 248, 0.25);"
    )
    title_html = (
        '<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.2rem;">'
        '<h1 style="margin:0;font-size:1.45rem;font-weight:700;color:#F8FAFC;">'
        "Autonomous Data Analytics Agent</h1>"
        f'<span style="{tag_style}">v2.0</span>'
        "</div>"
    )
    st.markdown(title_html, unsafe_allow_html=True)

    meta_html = (
        '<div style="display:flex;align-items:center;gap:0.5rem;color:#94A3B8;font-size:0.8rem;">'
        '<span>Active Dataset: <code style="color:#F1F5F9;background:#131B2A;padding:0.1rem 0.4rem;'
        f'border-radius:4px;border:1px solid #1E293B;">{profile.table_name}</code></span>'
        '<span style="color:#475569;">•</span>'
        f'<span>Rows: <strong style="color:#E2E8F0;">{profile.row_count:,}</strong></span>'
        '<span style="color:#475569;">•</span>'
        f'<span>Columns: <strong style="color:#E2E8F0;">{profile.column_count}</strong></span>'
        "</div>"
    )
    st.markdown(meta_html, unsafe_allow_html=True)
with top_col2:
    engine_badge = get_status_badge_html("VERIFIED", "ENGINE READY")
    badge_html = (
        '<div style="display:flex;justify-content:flex-end;align-items:center;'
        f'height:100%;padding-top:0.25rem;">{engine_badge}</div>'
    )
    st.markdown(badge_html, unsafe_allow_html=True)

# Main Tabbed Workbench
tab_workspace, tab_validation, tab_trace, tab_correction, tab_dataset = st.tabs(
    [
        "Query Workspace & Results",
        "Validation Centre",
        "Workflow Trace",
        "Self-Correction History",
        "Dataset & Profiling",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1: Query Workspace & Results
# ---------------------------------------------------------------------------
with tab_workspace:
    st.markdown("### Query Input")
    st.caption(
        "Submit analytical questions for autonomous DAG planning, SQL generation, "
        "execution, and validation."
    )

    # Benchmark Question Shortcuts
    st.markdown("**Sample Benchmark Questions:**")
    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
    if b_col1.button("Total Sales (Scalar)", use_container_width=True):
        st.session_state["current_question"] = "What is the total sales?"
    if b_col2.button("Sales by Region (Grouped)", use_container_width=True):
        st.session_state["current_question"] = "What are the total sales by region?"
    if b_col3.button("Lowest 5 Sub-Categories", use_container_width=True):
        st.session_state["current_question"] = (
            "What were the 5 sub-categories with the lowest total sales in 2017?"
        )
    if b_col4.button("Compare 2016 vs 2017", use_container_width=True):
        st.session_state["current_question"] = "Compare 2016 and 2017 total sales"
    if b_col5.button("Demo Self-Correction", use_container_width=True):
        st.session_state["current_question"] = (
            "What were the 5 sub-categories with the lowest total sales in 2017? "
            "(Demonstrate Self-Correction)"
        )

    user_query = st.text_input(
        "Analytical Question",
        value=st.session_state["current_question"],
        placeholder="e.g. What were the top 5 sub-categories by sales in 2017?",
        label_visibility="collapsed",
    )

    ctrl_col1, ctrl_col2 = st.columns([5, 1])
    with ctrl_col1:
        run_query = st.button("Execute Query", type="primary", use_container_width=True)
    with ctrl_col2:
        query_timeout = st.number_input(
            "Timeout (s)",
            min_value=1.0,
            max_value=60.0,
            value=15.0,
            step=1.0,
            label_visibility="collapsed",
        )

    # Execution Trigger
    if run_query and user_query:
        ollama_base_url = st.session_state.get("ollama_url", "http://localhost:11434")
        ollama_model_tag = st.session_state.get("ollama_model", "qwen2.5-coder:7b")
        mode = st.session_state.get(
            "engine_mode", "Auto (Ollama LLM with Autonomous Intent Fallback)"
        )

        is_strict = "Strict" in mode
        is_autonomous_only = "Autonomous Deterministic Engine" in mode

        if is_autonomous_only:
            planner = QueryPlanner(client=None, db=db)
            generator = CodeGenerator(client=None)
            generator.allow_fallback = True
        else:
            ollama_client = OllamaClient(
                base_url=ollama_base_url,
                model=ollama_model_tag,
                timeout=45.0,
            )
            planner = QueryPlanner(client=ollama_client, db=db)
            generator = CodeGenerator(client=ollama_client)
            generator.allow_fallback = not is_strict

        with st.spinner("Executing analytical workflow across compiled LangGraph nodes..."):
            try:
                workflow_graph = build_analytical_graph(
                    db=db,
                    generator=generator,
                    planner=planner,
                    semantic_validator=SemanticValidator(),
                    default_timeout_seconds=query_timeout,
                )
                initial_state: AgentState = {
                    "question": user_query,
                    "session_id": "streamlit_session",
                    "active_table": active_table,
                    "schema_profile": profile,
                    "retry_count": 0,
                    "max_retries": 2,
                    "status": "executing",
                }
                final_state = workflow_graph.invoke(initial_state)
                st.session_state["workflow_execution_state"] = final_state
            except ConnectionError as conn_err:
                st.session_state["workflow_execution_state"] = None
                st.error(
                    f"Ollama Connection Error: {conn_err}. Verify that Ollama daemon is running, "
                    "or select 'Autonomous Deterministic Engine' in the sidebar."
                )
            except Exception as exc:
                st.session_state["workflow_execution_state"] = None
                st.error(f"Analytical Workflow Error: {exc}")
                with st.expander("Technical Diagnostics", expanded=False):
                    st.code(traceback.format_exc(), language="text")

    # Results Display
    state = st.session_state.get("workflow_execution_state")
    if state:
        try:
            st.markdown("---")
            status_raw = state.get("status", "unknown").upper()
            val_report = state.get("validation_report")
            assembled = state.get("assembled_result")
            plan = state.get("plan")
            step_sql = state.get("step_sql", {})
            step_results = state.get("step_results", {})

            # Status Summary Header
            b_c1, b_c2, b_c3, b_c4 = st.columns(4)
            status_badge_html = get_status_badge_html(status_raw)
            b_c1.markdown(f"**Workflow Status**<br>{status_badge_html}", unsafe_allow_html=True)

            if isinstance(val_report, WorkflowValidationReport):
                conf_badge = get_status_badge_html(
                    "VERIFIED" if val_report.workflow_confidence.value == "HIGH" else "UNVERIFIED",
                    f"{val_report.workflow_confidence.value} CONFIDENCE",
                )
                b_c2.markdown(
                    f"**Verification Confidence**<br>{conf_badge}", unsafe_allow_html=True
                )
                verified_steps = sum(
                    1
                    for s in val_report.latest_step_results.values()
                    if s.status == ValidationStatus.VERIFIED
                )
                total_steps = len(val_report.latest_step_results)
                b_c3.metric("Verified Steps", f"{verified_steps} / {total_steps}")
                b_c4.metric("Total Attempts", val_report.total_retries + total_steps)

            # Section: Consolidated Analytical Answer
            st.markdown("### Consolidated Analytical Answer")

            if assembled:
                # Summary Metrics for Comparative or Multi-Step Queries
                metrics = assembled.summary_metrics or {}
                if "step_1_winner" in metrics and "step_2_winner" in metrics:
                    w1 = metrics.get("step_1_winner", "N/A")
                    w2 = metrics.get("step_2_winner", "N/A")
                    w_col1, w_col2, w_col3 = st.columns(3)
                    w_col1.metric("Metric 1 Leader", str(w1))
                    w_col2.metric("Metric 2 Leader", str(w2))
                    alignment = (
                        "Aligned (Same Entity)" if w1 == w2 else "Divergent (Different Entities)"
                    )
                    w_col3.metric("Objective Alignment", alignment)

                if "absolute_difference" in metrics and "step_1_value" in metrics:
                    s1_val = metrics.get("step_1_value")
                    s2_val = metrics.get("step_2_value")
                    abs_diff = metrics.get("absolute_difference")
                    pct_chg = metrics.get("percentage_change")
                    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                    m_c1.metric(
                        "Step 1 Value",
                        f"{s1_val:,.2f}" if isinstance(s1_val, (int, float)) else str(s1_val),
                    )
                    m_c2.metric(
                        "Step 2 Value",
                        f"{s2_val:,.2f}" if isinstance(s2_val, (int, float)) else str(s2_val),
                    )
                    m_c3.metric(
                        "Absolute Variance",
                        f"{abs_diff:,.2f}" if isinstance(abs_diff, (int, float)) else str(abs_diff),
                    )
                    m_c4.metric(
                        "Percentage Variance",
                        f"{pct_chg:+.2f}%" if isinstance(pct_chg, (int, float)) else "N/A",
                    )

                # Primary Output (Comparative Chart, Scalar, or Tabular)
                comp_df, comp_x, comp_y = prepare_comparison_chart_df(assembled, plan)
                if comp_df is not None and comp_x and comp_y:
                    st.markdown(f"**Annual Comparison ({comp_y} by {comp_x}):**")
                    st.bar_chart(comp_df)
                    st.markdown(f"**Comparative Summary ({comp_x} & {comp_y}):**")
                    tbl_disp = comp_df.reset_index()
                    st.dataframe(tbl_disp, use_container_width=True, hide_index=True)
                elif assembled.primary_scalar is not None:
                    ps = assembled.primary_scalar
                    ps_str = f"{ps:,.2f}" if isinstance(ps, (int, float)) else str(ps)
                    st.metric(label="Calculated Result", value=ps_str)
                elif assembled.primary_dataframe is not None:
                    df_out = assembled.primary_dataframe
                    if df_out.empty:
                        st.info(
                            "Query executed successfully but returned 0 rows matching criteria."
                        )
                    else:
                        st.dataframe(df_out, use_container_width=True)

                        # Analytical Chart (Only when grain is appropriate)
                        chart_type, x_col, y_col = prepare_chart_data(df_out)
                        if chart_type == "bar" and x_col and y_col:
                            st.markdown(f"**Visual Distribution ({y_col} by {x_col}):**")
                            chart_df = df_out[[x_col, y_col]].dropna().set_index(x_col)
                            st.bar_chart(chart_df)
                        elif chart_type == "line" and x_col and y_col:
                            st.markdown(f"**Temporal Trend ({y_col} over {x_col}):**")
                            chart_df = df_out[[x_col, y_col]].dropna().set_index(x_col)
                            st.line_chart(chart_df)
                elif assembled.assembly_notes:
                    for note in assembled.assembly_notes:
                        st.warning(note)

                # Assembly Notes
                if assembled.assembly_notes:
                    with st.expander("Consolidation Notes", expanded=False):
                        for note in assembled.assembly_notes:
                            st.markdown(f"- {note}")
            else:
                st.info("No assembled analytical output produced.")

            # Section: Generated DuckDB SQL
            st.markdown("### Generated DuckDB SQL")
            if step_sql:
                for sid, sql_code in step_sql.items():
                    with st.expander(f"SQL Statement (Step {sid})", expanded=True):
                        st.code(sql_code, language="sql")
            else:
                st.caption("No SQL statements recorded.")
        except Exception as exc:
            render_error_boundary("Analytical Results & Workspace", exc)

    else:
        empty_callout = (
            '<div style="border:1px dashed #1E293B;border-radius:8px;padding:2.25rem 1.5rem;'
            'text-align:center;background:rgba(19, 27, 42, 0.35);margin-top:0.75rem;">'
            '<div style="color:#38BDF8;font-size:1.25rem;margin-bottom:0.35rem;">✦</div>'
            '<div style="color:#F8FAFC;font-weight:600;font-size:0.95rem;margin-bottom:0.25rem;">'
            "Ready for Autonomous Query Execution</div>"
            '<div style="color:#94A3B8;font-size:0.825rem;max-width:520px;margin:0 auto;'
            'line-height:1.5;">'
            "Select a benchmark question above or type your natural-language question, then click "
            '<strong style="color:#E2E8F0;">Execute Query</strong> to generate DuckDB SQL, '
            "execute with thread isolation, and perform 3-layer verification.</div></div>"
        )
        st.markdown(empty_callout, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 2: Validation Centre
# ---------------------------------------------------------------------------
with tab_validation:
    try:
        st.markdown("### Multi-Layer Validation Centre")
        st.caption(
            "Rigorous verification across Layer 1 (Execution), Layer 2 (Numerical), "
            "and Layer 3 (Semantic AST)."
        )

        state = st.session_state.get("workflow_execution_state")
        if state and isinstance(state.get("validation_report"), WorkflowValidationReport):
            val_report: WorkflowValidationReport = state["validation_report"]

            # High-level Summary Table
            audit_rows = validation_report_to_display_rows(val_report)
            if audit_rows:
                st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### Per-Step Layer Deep-Dive")

            for step_id, step_val in val_report.latest_step_results.items():
                step_summary = format_step_validation_summary(val_report, step_id)
                if not step_summary:
                    continue

                with st.expander(
                    f"Validation Details — Step {step_id} ({step_val.status.value})", expanded=True
                ):
                    # Overview Indicators
                    ov_c1, ov_c2, ov_c3 = st.columns(3)
                    ov_c1.markdown(
                        f"**Step Status:** {get_status_badge_html(step_summary['step_status'])}",
                        unsafe_allow_html=True,
                    )
                    conf_val = step_summary["confidence"]
                    conf_cls = "VERIFIED" if conf_val == "HIGH" else "UNVERIFIED"
                    ov_c2.markdown(
                        f"**Confidence:** {get_status_badge_html(conf_cls, conf_val)}",
                        unsafe_allow_html=True,
                    )
                    ov_c3.markdown(f"**Shape:** `{step_summary['shape_description']}`")

                    # Layer 1: Execution & Schema
                    st.markdown("#### Layer 1: Physical Execution & Schema Verification")
                    l1_badge = get_status_badge_html(step_summary["execution_status"])
                    st.markdown(f"Status: {l1_badge}", unsafe_allow_html=True)
                    exec_time_str = f"{step_val.execution_result.execution_time_ms:.2f} ms"
                    st.caption(
                        f"**Schema Grounding:** `{step_summary['schema_matched']}` | "
                        f"**Dimensions:** `{step_summary['row_count']}r × "
                        f"{step_summary['column_count']}c` | "
                        f"**Latency:** `{exec_time_str}` | "
                        f"**Message:** *{step_summary['message']}*"
                    )

                    # Layer 2: Numerical Validation
                    st.markdown("#### Layer 2: Independent Numerical Calculation")
                    num_disp = format_numerical_validation_display(step_summary)
                    l2_badge = get_status_badge_html(num_disp["status"])
                    st.markdown(f"Status: {l2_badge}", unsafe_allow_html=True)

                    if num_disp["mode"] == "scalar":
                        n_c1, n_c2, n_c3 = st.columns(3)
                        n_c1.metric("DuckDB Output", str(num_disp.get("duckdb_value")))
                        n_c2.metric(
                            "Independent Pandas Calculation",
                            str(num_disp.get("independent_value")),
                        )
                        abs_d = num_disp.get("absolute_difference")
                        n_c3.metric(
                            "Absolute Difference",
                            f"{abs_d:.6g}" if isinstance(abs_d, float) else str(abs_d),
                        )
                        tol_desc = (
                            f"Absolute (`atol={num_disp['atol']}`), "
                            f"Relative (`rtol={num_disp['rtol']}`)"
                        )
                        st.caption(
                            f"**Tolerances:** {tol_desc} | "
                            f"**Relative Difference:** `{num_disp.get('relative_difference')}` | "
                            f"**Message:** *{num_disp.get('message')}*"
                        )
                    elif num_disp["mode"] == "grouped":
                        ng_c1, ng_c2, ng_c3, ng_c4 = st.columns(4)
                        ng_c1.metric("Expected Groups", num_disp.get("expected_groups", 0))
                        ng_c2.metric("Actual Groups", num_disp.get("actual_groups", 0))
                        ng_c3.metric("Matched Groups", num_disp.get("matched_groups", 0))
                        pass_str = (
                            f"{num_disp.get('passed_comparisons', 0)} / "
                            f"{num_disp.get('compared_metrics', 0)}"
                        )
                        ng_c4.metric("Passed Comparisons", pass_str)

                        if num_disp.get("comparison_rows"):
                            st.dataframe(
                                pd.DataFrame(num_disp["comparison_rows"]),
                                use_container_width=True,
                                hide_index=True,
                            )

                        if num_disp.get("failure_type"):
                            st.caption(
                                f"**Failure Type:** `{num_disp.get('failure_type')}` | "
                                f"**Reason:** `{num_disp.get('failure_reason')}` | "
                                f"**Missing Keys:** `{num_disp.get('missing_group_keys')}`"
                            )
                    else:
                        inc_msg = (
                            num_disp.get("message")
                            or num_disp.get("inconclusive_reason")
                            or "Unverified"
                        )
                        st.caption(f"Reason: *{inc_msg}*")

                    # Layer 3: Semantic AST Validation
                    st.markdown("#### Layer 3: Scope-Aware Semantic AST Coverage")
                    sem_disp = format_semantic_validation_display(step_summary)
                    l3_badge = get_status_badge_html(sem_disp["status"])
                    st.markdown(f"Status: {l3_badge}", unsafe_allow_html=True)
                    checks_ratio = f"{sem_disp['passed_count']} / {sem_disp['total_count']}"
                    sem_msg = sem_disp.get("message") or "Semantic verification complete"
                    st.caption(
                        f"**Canonical Checks Verified:** `{checks_ratio}` | "
                        f"**Message:** *{sem_msg}*"
                    )

                    if sem_disp["checks"]:
                        checks_df = pd.DataFrame(
                            [
                                {
                                    "Canonical Check": c["Check Name"],
                                    "Status": c["Status"],
                                    "Diagnostic Code": c["Diagnostic Code"],
                                    "Explanation": c["Explanation"],
                                }
                                for c in sem_disp["checks"]
                            ]
                        )
                        st.dataframe(checks_df, use_container_width=True, hide_index=True)

                        with st.expander(
                            "Detailed Intent & AST Evidence per Check", expanded=False
                        ):
                            for c in sem_disp["checks"]:
                                st.markdown(f"**{c['Check Name']}** ({c['Status']})")
                                st.caption(f"Explanation: {c['Explanation']}")
                                ev1, ev2 = st.columns(2)
                                ev1.caption("Structured Intent Evidence:")
                                ev1.json(c.get("intent_evidence") or {"info": "None"})
                                ev2.caption("SQL AST Evidence:")
                                ev2.json(c.get("ast_evidence") or {"info": "None"})

            # Export Raw Report
            st.markdown("---")
            with st.expander("Raw WorkflowValidationReport (JSON)", expanded=False):
                st.json(serialize_validation_report_for_json(val_report))
        else:
            st.info(
                "Validation report unavailable. Run a query in the Query Workspace to inspect "
                "multi-layer validation."
            )
    except Exception as exc:
        render_error_boundary("Validation Centre", exc)

# ---------------------------------------------------------------------------
# Tab 3: Workflow Trace
# ---------------------------------------------------------------------------
with tab_trace:
    try:
        st.markdown("### LangGraph Compiled State Machine Trace")
        st.caption(
            "Operational inspection of state transitions, DAG topology, and execution latencies."
        )

        state = st.session_state.get("workflow_execution_state")
        if state:
            # Traversed Execution Path Sequence
            exec_path = get_workflow_execution_path(state)
            if exec_path:
                st.markdown("#### Execution Traversal Sequence")
                path_str = "  ➔  ".join(f"`{node}`" for node in ["START"] + exec_path + ["END"])
                st.markdown(f"**Runtime Transition Path:** {path_str}")

            # State Machine Nodes Table (8 compiled nodes)
            nodes_trace = get_workflow_nodes_summary(state)
            if nodes_trace:
                st.markdown("#### State Machine Nodes")
                trace_df = pd.DataFrame(
                    [
                        {
                            "Node": n["node_id"],
                            "Component": n["name"],
                            "Architectural Role": n["role"],
                            "Execution Status": n["status"],
                            "Operational Summary": n["details"],
                        }
                        for n in nodes_trace
                    ]
                )
                st.dataframe(trace_df, use_container_width=True, hide_index=True)

            # Compiled Transition Routing Rules
            with st.expander("Compiled Graph Architecture & Transition Routing", expanded=False):
                st.markdown(
                    """
- **Initial Baseline Execution:** `START` ➔ `planner` ➔ `codegen` ➔ `executor`
- **Dynamic Routing after `executor`:**
  - `route_after_executor` evaluates execution and multi-layer validation:
    - If actionable failure & retry budget remains: ➔ `error_analyzer`
    - If successful with remaining steps: ➔ `step_advance` (iterates DAG, loops to `codegen`)
    - If terminal or all steps complete: ➔ `assembler`
- **Dynamic Routing after `error_analyzer`:**
  - `route_after_error_analyzer` evaluates failure diagnosability:
    - If actionable / correctable: ➔ `corrector` (remediates query, loops back to `executor`)
    - If non-correctable or retries exhausted: ➔ `assembler` (preserves diagnostics)
- **Consolidation & Final Verification:** `assembler` ➔ `validator` ➔ `END`
"""
                )

            # Plan DAG & Step Dependencies
            st.markdown("#### Analytical Plan & Execution DAG")
            plan = state.get("plan")
            if plan and plan.steps:
                dag_c1, dag_c2, dag_c3 = st.columns(3)
                dag_c1.metric(
                    "Execution Mode",
                    "Multi-Step Decomposition" if plan.is_multi_step else "Single-Step Direct",
                )
                dag_c2.metric("Total Steps", len(plan.steps))
                order_str = " -> ".join(f"Step {s}" for s in plan.execution_order)
                dag_c3.metric("Topological Order", order_str)

                plan_df = pd.DataFrame(
                    [
                        {
                            "Step ID": f"Step {s.step_id}",
                            "Sub-Question": s.question,
                            "Target Table": s.target_table,
                            "Dependencies": ", ".join(f"Step {d}" for d in s.depends_on)
                            if s.depends_on
                            else "None (Independent)",
                        }
                        for s in plan.steps
                    ]
                )
                st.dataframe(plan_df, use_container_width=True, hide_index=True)

            # Step Execution Latencies
            st.markdown("#### Execution Latencies & Output Grain")
            step_results = state.get("step_results", {})
            if step_results:
                exec_rows = []
                for sid, r in step_results.items():
                    grain = (
                        "Scalar (1×1)"
                        if r.is_scalar
                        else f"Tabular ({r.row_count}×{r.column_count})"
                    )
                    exec_rows.append(
                        {
                            "Step": f"Step {sid}",
                            "Status": "Success" if r.success else "Failed",
                            "Latency (ms)": f"{r.execution_time_ms:.2f}",
                            "Rows": r.row_count,
                            "Columns": r.column_count,
                            "Grain": grain,
                            "Error": r.error_message or "None",
                        }
                    )
                st.dataframe(pd.DataFrame(exec_rows), use_container_width=True, hide_index=True)
        else:
            st.info(
                "No workflow state available. Run an analytical query to inspect the "
                "state machine trace."
            )
    except Exception as exc:
        render_error_boundary("Workflow Trace", exc)

# ---------------------------------------------------------------------------
# Tab 4: Self-Correction History
# ---------------------------------------------------------------------------
with tab_correction:
    try:
        st.markdown("### Self-Correction Audit History")
        st.caption(
            "Chronological trace of initial executions, error diagnostics, and bounded "
            "correction retries."
        )

        state = st.session_state.get("workflow_execution_state")
        val_report = state.get("validation_report") if state else None

        if isinstance(val_report, WorkflowValidationReport):
            history_records = format_correction_history_display(val_report)

            # Summary KPIs
            total_steps = len(val_report.step_history)
            corrected_steps = sum(
                1 for sid, atts in val_report.step_history.items() if len(atts) > 1
            )
            total_retries = val_report.total_retries
            final_workflow_status = val_report.workflow_status.value

            kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
            kpi_c1.metric("Total Steps", total_steps)
            kpi_c2.metric("Corrected Steps", corrected_steps)
            kpi_c3.metric("Retries Used", f"{total_retries} / {total_steps * 2} max")
            kpi_c4.markdown(
                f"**Final Status**<br>{get_status_badge_html(final_workflow_status)}",
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("#### Chronological Attempt History")

            for sid, attempts in val_report.step_history.items():
                st.markdown(f"##### Step {sid} Attempt History ({len(attempts)} attempt(s))")

                for att in attempts:
                    att_num = att.attempt_number
                    att_status = att.status.value
                    badge = get_status_badge_html(att_status, f"ATTEMPT {att_num}: {att_status}")

                    with st.expander(
                        f"Step {sid} — Attempt {att_num} ({att_status})",
                        expanded=(att_num == len(attempts)),
                    ):
                        a_c1, a_c2, a_c3 = st.columns(3)
                        a_c1.markdown(f"**Status:** {badge}", unsafe_allow_html=True)
                        a_c2.metric("Confidence", att.confidence.value)
                        exec_time = (
                            att.raw_execution_result.execution_time_ms
                            if att.raw_execution_result
                            else 0.0
                        )
                        a_c3.metric("Execution Latency", f"{exec_time:.2f} ms")

                        st.markdown("**Executed SQL:**")
                        st.code(att.executed_sql or "-- No SQL recorded", language="sql")

                        # Layer Outcomes
                        st.caption(
                            f"**Layer 1 (Execution):** `{att.execution_result.status.value}` | "
                            f"**Layer 2 (Numerical):** `{att.numerical_result.status.value}` | "
                            f"**Layer 3 (Semantic):** `{att.semantic_result.status.value}`"
                        )

                        # Diagnostic Details
                        diag = att.diagnostic
                        if diag:
                            st.markdown("##### Correction Diagnostic")
                            d_c1, d_c2 = st.columns(2)
                            d_c1.caption(f"**Error Category:** `{diag.error_category.value}`")
                            is_corr = "Yes" if diag.is_correctable else "No"
                            d_c2.caption(f"**Actionable / Correctable:** `{is_corr}`")
                            st.caption(f"**Summary:** *{diag.error_summary}*")
                            st.caption(
                                f"**Remediation Guidance:** *{diag.remediation_instruction}*"
                            )

                        # Decision Note
                        rec = next(
                            (
                                r
                                for r in history_records
                                if r["step_id"] == sid and r["attempt_number"] == att_num
                            ),
                            None,
                        )
                        if rec:
                            st.info(f"**Decision Note:** {rec['decision_note']}")
        else:
            st.info(
                "No correction history available. Execute a query to observe "
                "self-correction behavior."
            )
    except Exception as exc:
        render_error_boundary("Self-Correction History", exc)

# ---------------------------------------------------------------------------
# Tab 5: Dataset & Profiling
# ---------------------------------------------------------------------------
with tab_dataset:
    try:
        st.markdown("### Active Dataset Profiling & Data Quality")
        st.caption(f"Profile metrics and schema grounding for table `{profile.table_name}`.")

        # Data Quality Overview
        dq = format_data_quality_metrics(profile)
        q_c1, q_c2, q_c3, q_c4 = st.columns(4)
        q_c1.metric("Total Records", f"{profile.row_count:,}")
        q_c2.metric("Total Attributes", profile.column_count)
        q_c3.metric("Data Completeness", f"{dq['completeness_pct']:.2f}%")
        q_c4.metric("Columns with Nulls", f"{dq['columns_with_nulls']} / {profile.column_count}")

        if dq["ambiguous_date_columns"]:
            ambig_cols = ", ".join(dq["ambiguous_date_columns"])
            st.warning(
                f"Ambiguous date format detected in column(s): `{ambig_cols}`. "
                "Formats where day and month are both <= 12 cannot be reliably determined "
                "without schema guidance."
            )

        # Column Profiling Table
        st.markdown("#### Column Inventory & Summary Statistics")
        disp_df = profile_to_display_df(profile)
        st.dataframe(disp_df, use_container_width=True, hide_index=True)

        # Table Preview (First 20 Rows)
        st.markdown("#### Table Preview (First 20 Rows)")
        try:
            preview_df = get_table_preview_df(db, active_table, limit=20)
            st.dataframe(preview_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Cannot retrieve table preview: {exc}")

        # Sanitized Column Mapping
        mapping = db.get_column_mapping(active_table)
        if mapping:
            with st.expander("Sanitized Column Identifier Mapping", expanded=False):
                map_df = pd.DataFrame(
                    [
                        {"Original Header": orig, "Sanitized Identifier": clean}
                        for orig, clean in mapping.items()
                    ]
                )
                st.dataframe(map_df, use_container_width=True, hide_index=True)

        # Token-Budgeted Schema Representation
        with st.expander("Token-Budgeted Prompt Schema Representation", expanded=False):
            st.caption(
                "Exact Markdown schema representation supplied to Ollama LLM nodes for grounding."
            )
            token_slider = st.slider(
                "Token Budget", min_value=200, max_value=3000, value=1200, step=100
            )
            prompt_schema = format_profile_for_prompt(profile, max_token_budget=token_slider)
            st.code(prompt_schema, language="markdown")

        # Manual SQL Sandbox
        with st.expander("Direct DuckDB SQL Sandbox", expanded=False):
            st.caption("Execute custom read-only DuckDB SQL directly through the timeout engine.")
            default_sql = f"SELECT * FROM {profile.table_name} LIMIT 5"
            custom_sql = st.text_area("SQL Query", value=default_sql)
            if st.button("Run SQL Sandbox", use_container_width=True):
                engine = ControlledExecutionEngine(db=db, default_timeout_seconds=10.0)
                res = engine.execute(custom_sql)
                if res.success:
                    st.caption(
                        f"Executed in {res.execution_time_ms:.2f} ms ({res.row_count} rows):"
                    )
                    st.dataframe(res.dataframe, use_container_width=True)
                else:
                    st.error(f"Execution Error ({res.error_type}): {res.error_message}")
    except Exception as exc:
        render_error_boundary("Dataset & Profiling", exc)
