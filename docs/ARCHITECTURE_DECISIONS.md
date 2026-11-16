# Architecture Decisions & Practical Guidelines

## Key Architecture Decisions (ADRs)

### ADR-01: Modular Python Application with UI in `src/ui/`
- **Decision:** Build as a clean modular Python application with the user interface located directly in `src/ui/` (`app.py`, `components.py`, `charts.py`), while analytical logic lives in `src/agent/`.
- **Rationale:** Separates presentation concerns cleanly from core agent logic while avoiding the complexity of a separate FastAPI microservice.

### ADR-02: Structured JSON Question Interpretation Without Upfront Pydantic Overhead
- **Decision:** The LLM parses user queries into a simple structured JSON format capturing metrics, aggregations, grouping dimensions, filters, time constraints, and derived calculations. Initially, simple JSON validation is sufficient. A formal Pydantic contract framework can be introduced later if needed.
- **Rationale:** Prevents premature schema rigidity while guaranteeing that the original question and its structured interpretation are preserved throughout the workflow for downstream validation.

### ADR-03: Clear Separation Between Phase 3 and Phase 4
- **Decision:** Phase 3 is strictly confined to proving that the model can answer simple, single-step analytical questions using generated code and basic charts. Multi-step analysis planning and LangGraph stateful orchestration are introduced in Phase 4.
- **Rationale:** Avoids over-complicating early phases and ensures foundational code generation and execution work reliably before introducing multi-step graphs.

### ADR-04: Three-Tier Deterministic Validation
- **Decision:** Validation in Phase 5 comprises three distinct gates:
  1. *Execution Check:* Verifies code executed cleanly without runtime errors.
  2. *Numerical Check:* Recomputes calculations using a **genuinely separate implementation** (e.g., executing an independent Pandas/NumPy calculation when the primary was DuckDB SQL), rather than running the same code twice.
  3. *Semantic Check:* Audits output to ensure all requested metrics, filters, and grouping dimensions from the preserved structured JSON were retained.
- **Rationale:** Ensures that code which runs without errors but computes the wrong aggregate or drops a user constraint is caught and sent to self-correction.

### ADR-05: Diagnostic RCA and Dimension Exploration
- **Decision:** In the completed system, RCA is invoked for diagnostic inquiries. For questions without an explicit measurable delta, the system explores available dimensions or issues an explicit insufficient-evidence report. Results are reported as evidence-supported "contributing factors", not proven causality.
- **Rationale:** Aligns with academic rigor; business observational data supports contribution and correlation, but not definitive causal claims without controlled experiments.

### ADR-06: Calibrated Confidence and Insufficient Evidence
- **Decision:** The system computes an evidence-confidence score based on observable factors (validation status, sample size, hypothesis test results) and must output an explicit *"Insufficient evidence"* outcome when data is lacking.
- **Rationale:** Prevents false business certainty and ensures transparent, auditable decision support.

### ADR-07: Phase-by-Phase User Testing and Sign-Off Governance
- **Decision:** After each phase is implemented and automated tests pass, the Project Owner (User) performs manual host testing according to specific testing instructions. No phase proceeds without explicit user approval.
- **Rationale:** Guarantees incremental quality, verifies host reproducibility, and prevents regressions.

### ADR-08: 6-Node Self-Correction & Error Taxonomy
- **Decision:** Implement a 6-node LangGraph workflow (`Planner` -> `Code Generator` -> `Executor` -> `Validator` -> `Error Analyzer` -> `Corrector`). Only correctable errors (syntax, column name typos, type casting, formula mistakes, missing filters/groupings, numerical deltas) trigger retries (capped at `max_retries = 2`). Non-correctable failures (unsupported operations, missing columns, ambiguous questions, empty data) halt immediately. If correction fails after 2 attempts, the system honestly reports the failure, outputs any best-effort result flagged as UNVERIFIED, and severely degrades the confidence score.
- **Rationale:** Prevents endless hallucination loops and maintains intellectual honesty about unverified outputs.

### ADR-09: Primary Execution Engine & Independent Verification Pair
- **Decision:** The LLM's primary generated code will be **DuckDB SQL**. `SQLGlot` is used to parse the SQL AST, enforce read-only safety, and inspect semantic coverage. The independent numerical verification engine will be implemented using **pure vectorized Pandas/NumPy** constructed programmatically from the structured specification.
- **Rationale:** SQL is the natural target for relational queries; DuckDB offers blazing in-process speed; SQLGlot provides precise AST inspection; and Pandas provides an orthogonal mathematical verification path.

### ADR-10: Dataset Pairing for Development & Benchmarks
- **Decision:** Both `superstore.csv` (tabular business data with sales, profit, orders, dates, categories) and `online_retail.csv` (UCI transactional invoice data) will be bundled in `data/`.
- **Rationale:** Provides rich coverage for both standard retail analytics and invoice-level diagnostic RCA scenarios.

### ADR-11: Strict Local Ollama Execution Environment
- **Decision:** All automated and interactive tests strictly require a live local Ollama instance running on `localhost:11434` with the `qwen2.5-coder:7b` model pulled.
- **Rationale:** Guarantees that the exact local model and inference environment intended for project demonstration is verified at every development checkpoint.

### ADR-12: 1st-Order Dimensional Variance Decomposition
- **Decision:** Phase 6 Diagnostic RCA will implement 1st-order dimensional contribution analysis ($\Delta M_i / \Delta M$ across individual candidate dimensions like Region, Category, Segment, Product), ensuring mathematical exactness and clear interpretability.
- **Rationale:** 1st-order decomposition guarantees that segment contributions sum cleanly to 100% of the observed variance, keeping the diagnostic output mathematically watertight and easy to explain during academic evaluation.
