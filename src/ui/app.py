"""Interactive Streamlit dashboard for Phase 2 data ingestion and profiling testing.

Enables manual verification of:
    - Multi-dataset CSV/Excel uploads into session-isolated DuckDB.
    - Table-name collision handling and deduplication.
    - Safe table preview (first 20 rows without raw SQL).
    - Comprehensive column profiling (null rates, distinct values, ranges, means, samples).
    - Token-budgeted schema prompt serialization for LLM grounding.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import format_profile_for_prompt, profile_table
from src.ui.helpers import (
    get_excel_sheet_names,
    get_table_preview_df,
    ingest_uploaded_file,
    profile_to_display_df,
)

# Page configuration
st.set_page_config(
    page_title="Self-Validating Analytics Agent — Phase 2 Data Profiler",
    page_icon="📊",
    layout="wide",
)

# Initialize SessionDatabase in Streamlit session state to preserve across reruns
if "db" not in st.session_state:
    st.session_state["db"] = SessionDatabase()

db: SessionDatabase = st.session_state["db"]

# Ensure tracking for active table
if "active_table" not in st.session_state:
    st.session_state["active_table"] = None

# --- SIDEBAR: Upload & Session Management ---
with st.sidebar:
    st.title("📂 Data Ingestion")
    st.markdown("Upload tabular datasets into the active session-isolated DuckDB catalog.")

    # Multi-file uploader
    uploaded_files = st.file_uploader(
        label="Choose CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Upload CSV or Excel spreadsheets. Column headers will be sanitized automatically.",
    )

    if uploaded_files:
        for uf in uploaded_files:
            file_bytes = uf.getvalue()
            filename = uf.name
            ext = Path(filename).suffix.lower()

            # Handle Excel sheet selection if workbook has multiple sheets
            sheet_name: str | int = 0
            if ext in {".xlsx", ".xls"}:
                try:
                    sheets = get_excel_sheet_names(file_bytes)
                    if len(sheets) > 1:
                        sheet_name = st.selectbox(
                            f"Select sheet for {filename}",
                            options=sheets,
                            key=f"sheet_{filename}",
                        )
                except Exception as exc:
                    st.error(f"Failed to read sheets from {filename}: {exc}")
                    continue

            # Upload button per file to prevent repeated auto-registration
            reg_button_key = f"reg_{filename}_{len(file_bytes)}"
            if st.button(f"Register '{filename}'", key=reg_button_key):
                with st.spinner(f"Ingesting {filename}..."):
                    success, table_or_msg, err_detail = ingest_uploaded_file(
                        db=db,
                        file_bytes=file_bytes,
                        filename=filename,
                        sheet_name=sheet_name,
                    )
                if success:
                    st.success(f"Registered table: `{table_or_msg}`")
                    st.session_state["active_table"] = table_or_msg
                else:
                    st.error(f"Error ({table_or_msg}): {err_detail}")

    st.markdown("---")
    st.subheader("Benchmark Datasets")
    st.markdown("Load pre-packaged canonical datasets for fast verification.")

    sample_superstore_path = Path("data/sample_superstore.csv")
    sample_retail_path = Path("data/sample_retail.csv")

    if st.button("Load Canonical Samples (Superstore & Retail)", use_container_width=True):
        loaded = []
        if sample_superstore_path.exists():
            t1 = db.register_csv(sample_superstore_path, "sample_superstore")
            loaded.append(t1)
        if sample_retail_path.exists():
            t2 = db.register_csv(sample_retail_path, "sample_retail")
            loaded.append(t2)

        if loaded:
            st.success(f"Loaded: {', '.join(f'`{t}`' for t in loaded)}")
            st.session_state["active_table"] = loaded[0]
            st.rerun()

    st.markdown("---")
    st.subheader("Registered Tables")
    allowed_tables = db.get_allowed_tables()

    if allowed_tables:
        st.caption(f"Total Tables in Session: **{len(allowed_tables)}**")
        selected_table = st.selectbox(
            "Active Dataset",
            options=allowed_tables,
            index=(
                allowed_tables.index(st.session_state["active_table"])
                if st.session_state["active_table"] in allowed_tables
                else 0
            ),
        )
        st.session_state["active_table"] = selected_table
    else:
        st.info("No tables currently registered in session.")


# --- MAIN AREA: Preview & Profiling ---
st.title("📊 Dataset Profiling & Session Inspector")
st.caption(
    "Phase 2 Manual Testing Interface — Self-Validating Autonomous Data Analytics Agent"
)

active_table = st.session_state.get("active_table")

if not active_table or not db.is_table_allowed(active_table):
    # Empty State
    st.info(
        "👋 Welcome! No dataset is currently selected.\n\n"
        "**To get started:**\n"
        "1. Upload a `.csv` or `.xlsx` file using the sidebar, or\n"
        "2. Click **'Load Canonical Samples'** in the sidebar to test immediately."
    )
else:
    # Selected dataset header
    try:
        profile = profile_table(db, active_table)
    except Exception as exc:
        st.error(f"Failed to profile table `{active_table}`: {exc}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Table", f"`{profile.table_name}`")
    col2.metric("Total Rows", f"{profile.row_count:,}")
    col3.metric("Total Columns", profile.column_count)

    # Tabs for structured inspection
    tab_preview, tab_profile, tab_prompt = st.tabs([
        "📋 Data Preview (First 20 Rows)",
        "🔍 Column Profiling & Statistics",
        "🤖 LLM Prompt Schema View",
    ])

    # Tab 1: Safe Data Preview
    with tab_preview:
        st.subheader("First 20 Rows")
        try:
            preview_df = get_table_preview_df(db, active_table, limit=20)
            st.dataframe(preview_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Error fetching preview: {exc}")

        st.subheader("Sanitized Column Mapping")
        mapping = db.get_column_mapping(active_table)
        if mapping:
            map_data = [
                {"Original Header": orig, "Sanitized Identifier": clean}
                for orig, clean in mapping.items()
            ]
            st.dataframe(map_data, use_container_width=True)

    # Tab 2: Column Profiling & Statistics
    with tab_profile:
        st.subheader("Column Cardinality, Nulls & Summary Statistics")
        disp_df = profile_to_display_df(profile)
        st.dataframe(
            disp_df,
            use_container_width=True,
            column_config={
                "Null %": st.column_config.TextColumn(
                    "Null Rate",
                    help="Percentage of missing values in this column",
                ),
            },
        )

    # Tab 3: Prompt Schema Export
    with tab_prompt:
        st.subheader("Token-Budgeted Schema Representation")
        st.markdown(
            "This exact Markdown schema representation is supplied to LLM nodes "
            "(Planner, Code Generator) for grounding SQL generation."
        )

        token_budget = st.slider(
            "Token Budget for Schema",
            min_value=100,
            max_value=3000,
            value=1200,
            step=100,
        )

        formatted_schema = format_profile_for_prompt(profile, max_token_budget=token_budget)

        char_count = len(formatted_schema)
        est_tokens = char_count // 4
        m1, m2 = st.columns(2)
        m1.metric("Estimated Tokens", f"~{est_tokens:,}")
        m2.metric("Character Count", f"{char_count:,}")

        st.code(formatted_schema, language="markdown")

