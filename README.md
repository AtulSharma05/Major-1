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
- Python 3.10+ (Tested on Python 3.13)
- [Ollama](https://ollama.ai) running locally (`ollama serve`) with `qwen2.5-coder:7b`

### Environment Installation
```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install package with dev dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

### Running Tests
```bash
# Run offline unit tests
pytest tests/unit -v

# Run live integration tests (requires Ollama running)
pytest tests/integration -v
```

---

## 4. Documentation
For deep-dive architecture, decisions, and development guidelines, refer to:
- [Master Execution Plan](docs/MASTER_EXECUTION_PLAN.md)
- [System Architecture](docs/ARCHITECTURE.md)
- [Architecture Decisions (ADR-01 to ADR-12)](docs/ARCHITECTURE_DECISIONS.md)
- [Testing Strategy](docs/TESTING_STRATEGY.md)
- [Development Protocol](docs/DEVELOPMENT_PROTOCOL.md)

