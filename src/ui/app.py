"""Interactive Streamlit dashboard for Phase 2 & Phase 3 manual testing.

Enables manual verification of:
    - Multi-dataset CSV/Excel uploads into session-isolated DuckDB (Phase 2).
    - Table-name collision handling and deduplication (Phase 2).
    - Safe table preview (first 20 rows without raw SQL) (Phase 2).
    - Comprehensive column profiling and token-budgeted schema serialization (Phase 2).
    - Question-to-SQL generation with Qwen2.5-Coder 7B via local Ollama (Phase 3).
    - Controlled, thread-isolated DuckDB execution with error capture & timeouts (Phase 3).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.agent.codegen.generator import CodeGenerator, OllamaClient
from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import format_profile_for_prompt, profile_table
from src.agent.execution.engine import ControlledExecutionEngine
from src.ui.helpers import (
    get_excel_sheet_names,
    get_table_preview_df,
    ingest_uploaded_file,
    profile_to_display_df,
)

# Page configuration
st.set_page_config(
    page_title="Self-Validating Analytics Agent — Testing Dashboard",
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

# Track user query state
if "current_question" not in st.session_state:
    st.session_state["current_question"] = "What is the total sales by region?"

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
                    st.rerun()
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


# --- MAIN AREA: Preview, Profiling & Analytical Execution ---
st.title("📊 Autonomous Analytics Agent — Testing Dashboard")
st.caption("Phase 2 Data Profiling & Phase 3 Single-Step Analytical Engine")

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
    tab_exec, tab_preview, tab_profile, tab_prompt = st.tabs([
        "⚡ Single-Step Query Execution (Phase 3)",
        "📋 Data Preview (First 20 Rows)",
        "🔍 Column Profiling & Statistics",
        "🤖 LLM Prompt Schema View",
    ])

    # Tab 1: Single-Step Analytical Query Execution (Phase 3)
    with tab_exec:
        st.subheader("Natural Language Question → DuckDB SQL Execution")
        st.markdown(
            "Uses **Qwen2.5-Coder 7B** via local Ollama (`localhost:11434`) grounded with the "
            "active table schema, executed deterministically via `ControlledExecutionEngine`."
        )

        # Quick sample question buttons for Superstore
        if "superstore" in active_table:
            st.caption("💡 Quick test questions for Superstore:")
            quick_cols = st.columns(4)
            if quick_cols[0].button("Total Sales (Scalar)", use_container_width=True):
                st.session_state["current_question"] = "What is the total sales?"
            if quick_cols[1].button("Sales by Region (Grouped)", use_container_width=True):
                st.session_state["current_question"] = "What are the total sales by region?"
            if quick_cols[2].button("Furniture Sales (Filtered)", use_container_width=True):
                q = "What is total sales for Furniture category?"
                st.session_state["current_question"] = q
            if quick_cols[3].button("Top 3 Sub-Categories", use_container_width=True):
                q = "What are the top 3 sub-categories by total sales?"
                st.session_state["current_question"] = q

        user_question = st.text_input(
            "Enter your analytical question:",
            value=st.session_state["current_question"],
            key="user_query_input",
        )

        col_run, col_timeout = st.columns([4, 1])
        run_btn = col_run.button(
            "🚀 Generate & Execute SQL",
            type="primary",
            use_container_width=True,
        )
        query_timeout = col_timeout.number_input(
            "Timeout (sec)",
            min_value=1.0,
            max_value=30.0,
            value=5.0,
        )

        if run_btn and user_question:
            generator = CodeGenerator(client=OllamaClient(model="qwen2.5-coder:7b", timeout=30.0))
            engine = ControlledExecutionEngine(db=db, default_timeout_seconds=query_timeout)

            # Step 1: SQL Code Generation
            with st.spinner("🤖 Grounding schema & generating SQL with Qwen2.5-Coder 7B..."):
                try:
                    generated_sql = generator.generate_sql(
                        question=user_question,
                        profile=profile,
                    )
                except ConnectionError as conn_err:
                    st.error(
                        f"❌ Could not connect to local Ollama daemon: {conn_err}\n\n"
                        "Please ensure Ollama is running (`ollama serve`)."
                    )
                    st.stop()
                except Exception as gen_err:
                    st.error(f"❌ SQL Generation Error: {gen_err}")
                    st.stop()

            st.markdown("#### Generated DuckDB SQL")
            st.code(generated_sql, language="sql")

            # Step 2: Controlled Execution
            with st.spinner("⚡ Executing query in DuckDB with AST & timeout protection..."):
                res = engine.execute(generated_sql, timeout_seconds=query_timeout)

            # Step 3: Present Execution Results
            m_stat, m_time, m_rows = st.columns(3)
            m_stat.metric("Status", "✅ Success" if res.success else "❌ Failed")
            m_time.metric("Execution Time", f"{res.execution_time_ms:.2f} ms")
            m_rows.metric("Rows Returned", res.row_count)

            if res.success:
                if res.is_scalar:
                    st.success("Calculated Scalar Result:")
                    val = res.scalar_value
                    formatted_val = f"{val:,.2f}" if isinstance(val, (int, float)) else str(val)
                    st.metric(label="Metric Value", value=formatted_val)
                elif res.is_empty():
                    st.info("Query executed successfully, but returned 0 rows.")
                else:
                    st.success("Query Results Table:")
                    st.dataframe(res.dataframe, use_container_width=True)
            else:
                st.error(f"Execution Error ({res.error_type}): {res.error_message}")

        st.markdown("---")
        with st.expander("🛠️ Manual SQL Sandbox (Test Direct SQL Execution)"):
            st.caption(
                "Write and execute custom DuckDB SQL directly through the timeout engine."
            )
            default_manual_sql = (
                f"SELECT region, SUM(sales) AS total_sales "
                f"FROM {profile.table_name} "
                "GROUP BY region ORDER BY total_sales DESC"
            )
            custom_sql = st.text_area("SQL Query", value=default_manual_sql)
            if st.button("Run Manual SQL"):
                engine = ControlledExecutionEngine(db=db, default_timeout_seconds=query_timeout)
                manual_res = engine.execute(custom_sql, timeout_seconds=query_timeout)
                if manual_res.success:
                    st.success(f"Executed in {manual_res.execution_time_ms:.2f} ms:")
                    st.dataframe(manual_res.dataframe, use_container_width=True)
                else:
                    st.error(f"Error ({manual_res.error_type}): {manual_res.error_message}")

    # Tab 2: Safe Data Preview (Phase 2)
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

    # Tab 3: Column Profiling & Statistics (Phase 2)
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

    # Tab 4: Prompt Schema Export (Phase 2)
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
