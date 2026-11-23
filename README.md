# Self-Validating Autonomous Data Analytics Agent

> **Diagnostic and Evidence-Based Decision Support**  
> B.Tech Major Project — Jaypee Institute of Information Technology (JIIT), Sector 128, Noida  
> **Team:** Parth Tiwari & Atul Sharma | **Supervisor:** Mr. Noor Mohammad

---

## 1. Project Overview

This project implements an autonomous conversational analytics agent that translates natural-language business questions into executable SQL, verifies analytical correctness using a three-layer validation system (Execution, Numerical, and Semantic Validation), performs additive diagnostic root-cause analysis (RCA), and assigns transparent categorical confidence scores with verifiable execution evidence.

Key architectural pillars:
- **Local-First LLM:** Powered by `qwen2.5-coder:7b` served via Ollama (`localhost:11434`).
- **In-Process Analytical Engine:** DuckDB for rapid, zero-overhead analytical SQL processing over tabular datasets (CSV/Excel).
- **Three-Layer Deterministic Validation:**
  1. *Execution Validation:* SQLGlot AST validation, table allowlisting, runtime execution guards, and timeout limits.
  2. *Numerical Validation:* Vectorized Pandas/NumPy independent recalculation from structured intent with strict null/duplicate handling and compound tolerance ($\text{atol}=10^{-4}, \text{rtol}=10^{-4}$).
  3. *Semantic Validation:* Requirement checklist verification, SQL AST structural checks, and output schema alignment.
- **Diagnostic RCA:** Additive variance decomposition ($\Delta M_i / \Delta M$) across categorical dimensions with strict mathematical eligibility checks.
- **Evidence-First Presentation:** Interactive Streamlit interface surfacing generated SQL, verification traces, discrepancy breakdowns, and Plotly visualizations.

---

## 2. Repository Layout

```text
major2.0/
├── docs/                   # Architectural blueprints & Master Execution Plan
├── data/                   # Analytical datasets (Superstore, Online Retail)
├── src/
│   ├── agent/              # Core analytical agent logic
│   │   ├── data/           # Dynamic table registration & profiling
│   │   ├── planning/       # Structured intent & result assembly
│   │   ├── codegen/        # Prompt templates & SQL code generation
│   │   ├── execution/      # Controlled DuckDB execution & timeout guards
│   │   ├── validation/     # Execution, Numerical & Semantic validators
│   │   ├── rca/            # Additive variance decomposition & dimension drilldown
│   │   └── workflow/       # 7-node LangGraph orchestration & state machine
│   └── ui/                 # Streamlit application, components, and visualizations
└── tests/
    ├── unit/               # Fast, 100% offline mocked unit tests
    └── integration/        # Live Ollama integration tests
```

---

## 3. Quickstart & Setup

### Prerequisites
- **Python 3.10+** (Tested on Python 3.13)
- **[Ollama](https://ollama.ai)** installed and running locally (`ollama serve`)
- Model pulled: `qwen2.5-coder:7b`

### Environment Installation
```bash
# 1. Clone the repository and enter directory
cd major2.0

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install package in editable mode with dev dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 4. How to Run the Project

### Step 1: Start the Local LLM Service (Ollama)
Make sure the Ollama daemon is active and the Qwen2.5-Coder model is downloaded:

```bash
# Start Ollama service (if not already running as a daemon)
ollama serve

# In a separate terminal, pull the required model
ollama pull qwen2.5-coder:7b

# (Optional) Verify Ollama is reachable
curl http://localhost:11434/api/tags
```

### Step 2: Launch the Interactive Streamlit Dashboard
With your virtual environment activated, start the web interface:

```bash
streamlit run src/ui/app.py
```

By default, the dashboard opens at **`http://localhost:8501`**.

---

### Step 3: Using the Dashboard

1. **Upload Dataset:**
   - In the sidebar under **"1. Dataset Ingestion & Profiling"**, upload any `.csv`, `.xlsx`, or `.xls` file, or load one of the included datasets from `data/` (e.g. `data/sample_superstore.csv`, `data/sample_retail.csv`).
   - The engine automatically sanitizes column names, detects datetime patterns, registers the table into an isolated DuckDB session, and captures a zero-copy Pandas snapshot for independent verification.

2. **Inspect Schema & Profiling:**
   - Review table dimensions, inferred data types, null counts, distinct cardinalities, and numerical stats before querying.

3. **Ask Analytical Questions:**
   - In **"2. Conversational Analytical Query"**, type natural-language business questions, for example:
     - *"What is the total sales and profit by region?"*
     - *"Calculate average order value by customer segment."*
     - *"Show top 5 categories with highest sales."*
   - Click **Run Analytical Query**.

4. **Inspect Generated SQL & Multi-Layer Validation:**
   - **Step-by-Step Plan & SQL:** View the decomposition plan and generated DuckDB SQL.
   - **Layer 1 (Execution Validation):** Checks physical execution success, syntax/table allowlists, output grain, and projected column deduplication.
   - **Layer 2 (Numerical Validation):** Pure Pandas independent recalculation directly from structured intent, validating results within strict compound tolerances ($\text{atol}=10^{-4}, \text{rtol}=10^{-4}$) while distinguishing SQL NULL from IEEE NaN.
   - **Layer 3 (Semantic Validation):** Output schema and dimension checklist alignment.
   - **Audit Table & Evidence JSON:** Review the complete multi-layer validation table and full diagnostic report under **"4. Multi-Layer Validation Audit & Traceability"**.

---

## 5. Verification & Testing

```bash
# Run code formatting and linter checks
ruff check src/ tests/

# Run the complete offline unit test suite (176 tests, ~3s)
pytest tests/unit -v

# Run end-to-end integration tests (requires Ollama running with qwen2.5-coder:7b)
pytest tests/integration -v
```

---

## 6. Documentation

For deep-dive architecture, decisions, and development guidelines, refer to:
- [Master Execution Plan](docs/MASTER_EXECUTION_PLAN.md)
- [System Architecture](docs/ARCHITECTURE.md)
- [Architecture Decisions (ADR-01 to ADR-12)](docs/ARCHITECTURE_DECISIONS.md)
- [Testing Strategy](docs/TESTING_STRATEGY.md)
- [Development Protocol](docs/DEVELOPMENT_PROTOCOL.md)

