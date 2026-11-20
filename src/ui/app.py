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
from src.agent.planning.planner import QueryPlanner
from src.agent.validation.contracts import ValidationStatus, WorkflowValidationReport
from src.agent.workflow.graph import build_analytical_graph
from src.agent.workflow.state import AgentState
from src.ui.helpers import (
    format_step_validation_summary,
    get_excel_sheet_names,
    get_lifecycle_status_badge,
    get_table_preview_df,
    ingest_uploaded_file,
    profile_to_display_df,
    serialize_validation_report_for_json,
    validation_report_to_display_rows,
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
    sample_superstore_2_path = Path("data/sample_superstore_2.csv")
    sample_retail_path = Path("data/sample_retail.csv")

    if st.button("Load Canonical Samples (Superstore & Retail)", use_container_width=True):
        loaded = []
        if sample_superstore_path.exists():
            t1 = db.register_csv(sample_superstore_path, "sample_superstore")
            loaded.append(t1)
        if sample_superstore_2_path.exists():
            t2 = db.register_csv(sample_superstore_2_path, "sample_superstore_2")
            loaded.append(t2)
        if sample_retail_path.exists():
            t3 = db.register_csv(sample_retail_path, "sample_retail")
            loaded.append(t3)

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
st.caption("Phase 2 Data Profiling & Phase 4 Multi-Step Analytical Workflow (LangGraph)")

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
        "⚡ Analytical Workflow (Phase 4 LangGraph)",
        "📋 Data Preview (First 20 Rows)",
        "🔍 Column Profiling & Statistics",
        "🤖 LLM Prompt Schema View",
    ])

    # Tab 1: Multi-Step & Single-Step Analytical Query Execution (Phase 4)
    with tab_exec:
        st.subheader("Autonomous Multi-Step Query Planning & Workflow Orchestration")
        st.markdown(
            "Orchestrates the 7-node **LangGraph** analytical lifecycle: "
            "**Planner** decomposes questions into structured DAGs, "
            "**Code Generator** synthesizes SQL with **Qwen2.5-Coder 7B**, "
            "**ControlledExecutionEngine** runs queries safely, and "
            "**ResultAssembler** merges intermediate findings."
        )

        # Quick sample question buttons for Superstore
        if "superstore" in active_table:
            st.caption("💡 Quick test questions for Superstore:")
            st.markdown("**Single-Step Queries:**")
            quick_cols_s = st.columns(3)
            if quick_cols_s[0].button("Total Sales (Scalar)", use_container_width=True):
                st.session_state["current_question"] = "What is the total sales?"
            if quick_cols_s[1].button("Sales by Region (Grouped)", use_container_width=True):
                st.session_state["current_question"] = "What are the total sales by region?"
            if quick_cols_s[2].button("Furniture Sales (Filtered)", use_container_width=True):
                st.session_state["current_question"] = "What is total sales for Furniture category?"

            st.markdown("**Multi-Step Analytical Queries:**")
            quick_cols_m = st.columns(4)
            if quick_cols_m[0].button("Compare 2016 vs 2017", use_container_width=True):
                st.session_state["current_question"] = "Compare 2016 and 2017 total sales"
            if quick_cols_m[1].button("Compare Central vs East", use_container_width=True):
                st.session_state["current_question"] = "Compare Central and East region sales"
            if quick_cols_m[2].button("2016 vs 2017 by Region", use_container_width=True):
                st.session_state["current_question"] = "Compare 2016 and 2017 sales by region"
            if quick_cols_m[3].button("Top Sales & Profit Item", use_container_width=True):
                q = (
                    "Which product has the highest sales and which product has the highest "
                    "profit? They can be the same or different."
                )
                st.session_state["current_question"] = q

        user_question = st.text_input(
            "Enter your analytical question (single-step or multi-step comparison):",
            value=st.session_state["current_question"],
            key="user_query_input",
        )

        col_run, col_timeout = st.columns([4, 1])
        run_btn = col_run.button(
            "🚀 Execute Analytical Workflow",
            type="primary",
            use_container_width=True,
        )
        query_timeout = col_timeout.number_input(
            "Timeout (sec/step)",
            min_value=1.0,
            max_value=30.0,
            value=10.0,
        )

        if run_btn and user_question:
            ollama_client = OllamaClient(model="qwen2.5-coder:7b", timeout=45.0)
            planner = QueryPlanner(client=ollama_client)
            generator = CodeGenerator(client=ollama_client)

            with st.spinner("🔄 Running 7-node LangGraph analytical state machine..."):
                try:
                    workflow_app = build_analytical_graph(
                        db=db,
                        generator=generator,
                        planner=planner,
                        default_timeout_seconds=query_timeout,
                    )
                    initial_state: AgentState = {
                        "question": user_question,
                        "session_id": "streamlit_session",
                        "active_table": active_table,
                        "schema_profile": profile,
                        "retry_count": 0,
                        "max_retries": 2,
                        "status": "starting",
                    }
                    final_state = workflow_app.invoke(initial_state)
                except ConnectionError as conn_err:
                    st.error(
                        f"❌ Could not connect to local Ollama daemon: {conn_err}\n\n"
                        "Please ensure Ollama is running (`ollama serve`)."
                    )
                    st.stop()
                except Exception as wf_err:
                    st.error(f"❌ Workflow Execution Failed: {wf_err}")
                    st.stop()

            # Retrieve final state elements
            plan = final_state.get("plan")
            step_sql = final_state.get("step_sql", {})
            step_results = final_state.get("step_results", {})
            assembled = final_state.get("assembled_result")
            status = final_state.get("status", "unknown")
            val_report = final_state.get("validation_report")

            # --- Workflow Lifecycle & Validation Status Banner ---
            badge_label, badge_icon, alert_type = get_lifecycle_status_badge(status)
            if alert_type == "success":
                st.success(f"{badge_icon} **{badge_label}**")
            elif alert_type == "error":
                st.error(f"{badge_icon} **{badge_label}**")
            elif alert_type == "warning":
                st.warning(f"{badge_icon} **{badge_label}**")
            else:
                st.info(f"{badge_icon} **{badge_label}**")

            if isinstance(val_report, WorkflowValidationReport):
                s_c1, s_c2, s_c3, s_c4 = st.columns(4)
                s_c1.metric("Lifecycle Status", status.upper())
                s_c2.metric("Workflow Confidence", val_report.workflow_confidence.value)
                verified_count = sum(
                    1
                    for s in val_report.latest_step_results.values()
                    if s.execution_result.status == ValidationStatus.VERIFIED
                )
                total_step_count = len(val_report.latest_step_results)
                s_c3.metric("Layer 1 Verified", f"{verified_count} / {total_step_count}")
                s_c4.metric("Total Attempts", val_report.total_retries + total_step_count)

            # --- Section 1: Workflow Execution Summary ---
            st.markdown("### 1. Analytical Plan & Execution DAG")
            if plan:
                badge_type = (
                    "Multi-Step Decomposition"
                    if plan.is_multi_step
                    else "Single-Step Direct Execution"
                )
                p_col1, p_col2, p_col3 = st.columns(3)
                p_col1.metric("Plan Structure", badge_type)
                p_col2.metric("Total Steps", len(plan.steps))
                exec_order_str = " ➔ ".join(f"Step {s}" for s in plan.execution_order)
                p_col3.metric("Topological Order", exec_order_str)

                # Steps Table
                steps_display = [
                    {
                        "Step ID": f"Step {s.step_id}",
                        "Sub-Question": s.question,
                        "Target Table": s.target_table,
                        "Dependencies": (
                            ", ".join(f"Step {d}" for d in s.depends_on)
                            if s.depends_on
                            else "None (Independent)"
                        ),
                    }
                    for s in plan.steps
                ]
                st.dataframe(steps_display, use_container_width=True)
            else:
                st.warning("No structured plan was produced.")

            # --- Section 2: Step-by-Step Execution Traces ---
            st.markdown("### 2. Step-by-Step Execution Traces")
            if plan and plan.execution_order:
                for step_id in plan.execution_order:
                    s_obj = plan.get_step(step_id)
                    s_desc = s_obj.question if s_obj else f"Step {step_id}"
                    s_sql = step_sql.get(step_id, "-- No SQL recorded")
                    s_res = step_results.get(step_id)

                    with st.expander(f"📍 Step {step_id}: {s_desc}", expanded=True):
                        s_tbl = s_obj.target_table if s_obj else active_table
                        s_deps = s_obj.depends_on if s_obj else []
                        st.caption(f"**Target Table:** `{s_tbl}` | **Dependencies:** `{s_deps}`")
                        st.markdown("**Generated DuckDB SQL:**")
                        st.code(s_sql, language="sql")

                        if s_res:
                            r_c1, r_c2, r_c3 = st.columns(3)
                            stat_label = "✅ Success" if s_res.success else "❌ Failed"
                            r_c1.metric("Execution Status", stat_label)
                            r_c2.metric("Latency", f"{s_res.execution_time_ms:.2f} ms")
                            r_c3.metric("Rows Returned", s_res.row_count)

                            if s_res.success:
                                if s_res.is_scalar:
                                    st.markdown("**Step Output (Scalar):**")
                                    s_val = s_res.scalar_value
                                    if isinstance(s_val, (int, float)):
                                        disp_val = f"{s_val:,.2f}"
                                    else:
                                        disp_val = str(s_val)
                                    st.metric(label=f"Step {step_id} Value", value=disp_val)
                                elif s_res.is_empty():
                                    st.info("Query succeeded but returned 0 rows.")
                                else:
                                    st.markdown("**Step Output (Table Preview):**")
                                    st.dataframe(s_res.dataframe, use_container_width=True)
                            else:
                                err_text = (
                                    f"Execution Error ({s_res.error_type}): {s_res.error_message}"
                                )
                                st.error(err_text)
                        else:
                            st.warning("No execution result recorded for this step.")

                        # Phase 5 Stage 2 Validation Trace
                        step_val_info = format_step_validation_summary(val_report, step_id)
                        if step_val_info:
                            st.markdown("---")
                            st.markdown("##### 🛡️ Phase 5 Validation (Layer 1: Execution)")
                            v_c1, v_c2, v_c3 = st.columns(3)
                            v_c1.metric("Step Status", step_val_info["step_status"])
                            v_c2.metric("Confidence", step_val_info["confidence"])
                            v_c3.metric("Layer 1 Execution", step_val_info["execution_status"])

                            st.caption(
                                f"**Execution Passed:** "
                                f"`{'✅ Yes' if step_val_info['execution_passed'] else '❌ No'}` | "
                                f"**Output Shape:** `{step_val_info['shape_description']}` | "
                                f"**Schema Match:** `{step_val_info['schema_matched']}` | "
                                f"**Empty Output:** `{step_val_info['is_empty']}`"
                            )
                            if step_val_info["message"]:
                                st.caption(f"ℹ️ *{step_val_info['message']}*")

                            # Numerical and semantic layer statuses
                            l_c1, l_c2 = st.columns(2)
                            l2_note = (
                                "*(Intent unavailable)*"
                                if step_val_info["numerical_status"] == "INCONCLUSIVE"
                                else "*(Pending Stage 3)*"
                            )
                            l3_note = (
                                "*(Intent unavailable)*"
                                if step_val_info["semantic_status"] == "INCONCLUSIVE"
                                else "*(Pending Stage 4)*"
                            )
                            l_c1.caption(
                                f"**Layer 2 (Numerical):** "
                                f"`{step_val_info['numerical_status']}` {l2_note}"
                            )
                            l_c2.caption(
                                f"**Layer 3 (Semantic):** "
                                f"`{step_val_info['semantic_status']}` {l3_note}"
                            )

                            if step_val_info["diagnostic"]:
                                diag = step_val_info["diagnostic"]
                                st.error(
                                    f"**Validation Diagnostic ({diag['category']}):** "
                                    f"{diag['summary']} (Correctable: {diag['correctable']})"
                                )

            # --- Section 3: Final Assembled Analytical Answer ---
            st.markdown("### 3. Final Assembled Analytical Answer")
            if assembled:
                if status == "inconclusive":
                    msg = "⚠️ Execution completed — analytical correctness not established."
                    if assembled.assembly_notes:
                        msg += f" ({assembled.assembly_notes[0]})"
                    st.warning(msg)
                elif status == "validation_failed" or not assembled.all_steps_succeeded:
                    st.error("❌ Workflow encountered execution or validation failures.")
                elif status == "execution_verified":
                    msg = "🔵 Execution verified (Layer 1) — analytical correctness unverified."
                    if assembled.assembly_notes:
                        msg += f" ({assembled.assembly_notes[0]})"
                    st.info(msg)
                elif status == "validated":
                    msg = "✅ Workflow Completed & Fully Validated Across All Layers"
                    if assembled.assembly_notes:
                        msg += f" ({assembled.assembly_notes[0]})"
                    st.success(msg)
                elif assembled.all_steps_succeeded:
                    msg = "Execution completed"
                    if assembled.assembly_notes:
                        msg += f" ({assembled.assembly_notes[0]})"
                    st.info(msg)
                else:
                    st.error("❌ Workflow encountered partial or complete step execution errors.")

                # If independent ranking summary metrics are available
                metrics = assembled.summary_metrics
                if metrics and "step_1_winner" in metrics and "step_2_winner" in metrics:
                    st.markdown("#### Independent Objective Winners")
                    w1 = metrics.get("step_1_winner", "N/A")
                    w2 = metrics.get("step_2_winner", "N/A")
                    r_c1, r_c2, r_c3 = st.columns(3)
                    r_c1.metric("Top Sales Item", w1)
                    r_c2.metric("Top Profit Item", w2)
                    match_str = (
                        "Same Product"
                        if w1 == w2
                        else "Different Products"
                    )
                    r_c3.metric("Product Alignment", match_str)

                # If scalar comparison summary metrics are available
                if metrics and "absolute_difference" in metrics:
                    st.markdown("#### Comparative Summary Metrics")
                    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                    s1_val = metrics.get("step_1_value")
                    s2_val = metrics.get("step_2_value")
                    diff_val = metrics.get("absolute_difference")
                    pct_val = metrics.get("percentage_change")

                    fmt_s1 = f"{s1_val:,.2f}" if isinstance(s1_val, (int, float)) else str(s1_val)
                    fmt_s2 = f"{s2_val:,.2f}" if isinstance(s2_val, (int, float)) else str(s2_val)
                    if isinstance(diff_val, (int, float)):
                        fmt_diff = f"{diff_val:,.2f}"
                    else:
                        fmt_diff = str(diff_val)

                    if pct_val is None:
                        fmt_pct = "N/A (zero baseline)"
                    elif isinstance(pct_val, (int, float)):
                        fmt_pct = f"{pct_val:+.2f}%"
                    else:
                        fmt_pct = str(pct_val)

                    m_c1.metric("Step 1 Result", fmt_s1)
                    m_c2.metric("Step 2 Result", fmt_s2)
                    m_c3.metric("Absolute Variance", fmt_diff)
                    m_c4.metric("Percentage Change", fmt_pct)

                # Primary tabular or scalar display
                if assembled.primary_dataframe is not None:
                    st.markdown("#### Consolidated Result Dataset")
                    st.dataframe(assembled.primary_dataframe, use_container_width=True)
                elif assembled.primary_scalar is not None:
                    st.markdown("#### Consolidated Result Metric")
                    ps_val = assembled.primary_scalar
                    fmt_ps = f"{ps_val:,.2f}" if isinstance(ps_val, (int, float)) else str(ps_val)
                    st.metric(label="Calculated Answer", value=fmt_ps)

                # Assembly Notes
                if assembled.assembly_notes:
                    with st.expander("📝 Assembly & Consolidation Notes", expanded=False):
                        for note in assembled.assembly_notes:
                            st.markdown(f"- {note}")

            # --- Section 4: Multi-Layer Validation Audit & Traceability ---
            st.markdown("---")
            st.markdown("### 4. Multi-Layer Validation Audit & Traceability")
            if isinstance(val_report, WorkflowValidationReport):
                audit_rows = validation_report_to_display_rows(val_report)
                if audit_rows:
                    st.dataframe(audit_rows, use_container_width=True)

                st.caption(
                    f"**Workflow Status:** `{status}` | "
                    "**Validation Layers:** Layer 1 (Physical Execution & Shape/Schema) active. "
                    "Layers 2 & 3: INCONCLUSIVE when intent is missing, or UNVERIFIED."
                )

                with st.expander("🔍 View Raw WorkflowValidationReport (JSON)", expanded=False):
                    st.json(serialize_validation_report_for_json(val_report))
            else:
                st.caption(
                    f"**Workflow Status:** `{status}` | "
                    "Validation report not available for this execution."
                )

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

        ambig_cols = [
            c.name for c in profile.columns.values()
            if c.date_format and "AMBIGUOUS" in c.date_format
        ]
        if ambig_cols:
            st.warning(
                f"⚠️ Ambiguous date format detected in column(s): {', '.join(ambig_cols)}. "
                "Ambiguous date formats (e.g. DD/MM/YYYY vs MM/DD/YYYY where both are <= 12) "
                "cannot be reliably filtered without user clarification."
            )

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
