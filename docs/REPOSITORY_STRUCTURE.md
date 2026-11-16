# Practical Repository Structure

A clean, modular Python repository designed for clarity, testability, and alignment with the **10 Project Phases** and the **Three-Layer Validation Architecture**.

```
major2.0/
├── docs/                                  # Project documentation & reference blueprints
│   ├── PROJECT_ROADMAP.md                 # 10-phase development plan
│   ├── ARCHITECTURE.md                    # Simplified architecture & validation specification
│   ├── TECH_STACK.md                      # Core dependencies & rationale
│   ├── REPOSITORY_STRUCTURE.md            # Directory structure specification
│   ├── TESTING_STRATEGY.md                # Testing & evaluation strategy
│   ├── DEVELOPMENT_PROTOCOL.md            # Mandatory feature & phase lifecycle
│   └── ARCHITECTURE_DECISIONS.md          # Key project decisions & guidelines
│
├── pyproject.toml                         # Project dependencies and tool configurations
├── README.md                              # Project overview and run instructions
│
├── data/                                  # Datasets directory (Phase 2 & 8)
│   ├── superstore.csv                     # Standard development dataset
│   ├── online_retail.csv                  # UCI Online Retail dataset
│   └── synthetic_diagnostic.csv           # Ground-truth injected diagnostic dataset
│
├── src/                                   # Application source code
│   ├── agent/                             # Core Analytics Engine & Agent Logic
│   │   ├── __init__.py
│   │   │
│   │   ├── data/                          # Phase 2: Ingestion & Profiling
│   │   │   ├── __init__.py
│   │   │   ├── ingestion.py               # CSV & Excel loading into Pandas & DuckDB
│   │   │   └── profiling.py               # Column profiling, statistics, and metadata
│   │   │
│   │   ├── analytics/                     # Phase 3: Basic Analytics Engine (Single-Step)
│   │   │   ├── __init__.py
│   │   │   ├── ollama_client.py           # Local Ollama client (Qwen2.5-Coder 7B)
│   │   │   ├── question_parser.py         # Structured JSON question interpretation & preservation
│   │   │   ├── metric_library.py          # Standard metric definitions (SUM, AVG, AOV, etc.)
│   │   │   ├── code_generator.py          # Python/Pandas & SQL code generation
│   │   │   └── executor.py                # Safe in-process code execution with timeouts
│   │   │
│   │   ├── planning/                      # Phase 4: Planning & LangGraph Agent (Multi-Step)
│   │   │   ├── __init__.py
│   │   │   ├── state.py                   # LangGraph shared agent state definition
│   │   │   ├── planner.py                 # Multi-step analysis planner & dependency tracker
│   │   │   └── graph.py                   # LangGraph 6-node workflow definition & routing
│   │   │
│   │   ├── validation/                    # Phase 5: Three-Layer Validation & Self-Correction
│   │   │   ├── __init__.py
│   │   │   ├── execution_validator.py     # Layer 1: AST parsing, runtime trap, output structure
│   │   │   ├── numerical_validator.py     # Layer 2: Independent alternate engine calculation & tolerances
│   │   │   ├── semantic_validator.py      # Layer 3: Plan/code inspection, output grain, requirement coverage
│   │   │   ├── analyzer.py                # Error Analyzer (correctable vs non-correctable classification)
│   │   │   └── correction.py              # Self-correction feedback & retry limiter (max 2 attempts)
│   │   │
│   │   ├── diagnostics/                   # Phase 6: Diagnostic Root-Cause Analysis
│   │   │   ├── __init__.py
│   │   │   ├── hypotheses.py              # Candidate factor generation & dimension exploration
│   │   │   └── contribution.py            # Variance decomposition & factor ranking
│   │   │
│   │   └── reporting/                     # Phase 7: Evidence & Provenance Store
│   │       ├── __init__.py
│   │       ├── provenance.py              # Lineage tracking & calculation store
│   │       └── confidence.py              # Rule-based confidence scoring & sufficiency check
│   │
│   └── ui/                                # Phase 7: Cleanly Separated Streamlit Interface
│       ├── __init__.py
│       ├── app.py                         # Main Streamlit application entry point
│       ├── components.py                  # UI cards, tables, 3-tier validation badges, evidence drawer
│       └── charts.py                      # Plotly visualization builders
│
└── tests/                                 # Automated tests (Phases 2 – 9)
    ├── conftest.py                        # Pytest fixtures and sample data loaders
    ├── test_profiling.py                  # Unit tests for data profiling (Phase 2)
    ├── test_analytics.py                  # Tests for single-step code generation & execution (Phase 3)
    ├── test_planning.py                   # Tests for LangGraph multi-step agent workflow (Phase 4)
    ├── test_validation.py                 # Tests for 3-layer validation & self-correction (Phase 5)
    ├── test_diagnostics.py                # Tests for RCA & contribution analysis (Phase 6)
    ├── test_reporting.py                  # Tests for evidence, confidence, and provenance (Phase 7)
    └── benchmark/                         # Phases 8 & 9: Evaluation & Ablation
        ├── benchmark_questions.json       # 50–100 benchmark analytical questions
        ├── run_benchmark.py               # Evaluation benchmark runner
        └── ablation_runner.py             # 5-configuration ablation study runner
```

---

## Phased Introduction of Modules

| Phase | Modules / Directories Introduced |
| :--- | :--- |
| **Phase 1** | Project setup: `docs/`, `pyproject.toml`, `README.md` |
| **Phase 2** | `data/`, `src/agent/data/`, `tests/test_profiling.py` |
| **Phase 3** | `src/agent/analytics/`, `tests/test_analytics.py` |
| **Phase 4** | `src/agent/planning/`, `tests/test_planning.py` |
| **Phase 5** | `src/agent/validation/`, `tests/test_validation.py` |
| **Phase 6** | `src/agent/diagnostics/`, `tests/test_diagnostics.py` |
| **Phase 7** | `src/agent/reporting/`, `src/ui/`, `tests/test_reporting.py` |
| **Phase 8** | `tests/benchmark/benchmark_questions.json`, `tests/benchmark/run_benchmark.py` |
| **Phase 9** | `tests/benchmark/ablation_runner.py` |
| **Phase 10**| Final documentation, reports, presentation slides |
