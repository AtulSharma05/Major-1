# Technology Stack: Practical Major Project Specification

Aligned directly with the project report (Chapter 4, Table 3) and practical B.Tech requirements.

---

## 1. Core Technology Selection

| Component | Tool / Technology | Practical Role in Major Project |
| :--- | :--- | :--- |
| **Programming Language** | **Python 3.10+** | Core language for data profiling, agent planning, execution, and validation. |
| **Interface / Frontend** | **Streamlit** | Fast, interactive web UI for file upload, question input, and rendering tables, charts, and validation cards. |
| **LLM Inference** | **Ollama** | Local LLM host running on `localhost:11434`; private, reproducible, zero API cost. |
| **Model** | **Qwen2.5-Coder 7B** | Compact open-weights code & reasoning LLM capable of fast structured JSON generation and Python/SQL code synthesis. |
| **Agent Orchestration** | **LangGraph** | Coordinates the analytical workflow: planner -> coder -> executor -> validator -> self-correction loop. |
| **Data Processing** | **Pandas & NumPy** | In-memory data manipulation, statistical calculations, and primary/independent execution. |
| **Analytical SQL Engine** | **DuckDB** | Fast in-process columnar SQL database for querying CSV/Excel files and providing independent calculation validation. |
| **Data Contracts** | **Pydantic v2** | Simple typed structures for question intent, plan steps, validation reports, and evidence records. |
| **Visualization** | **Plotly** | Clean, interactive charts (bar, line, scatter) embedded directly in Streamlit. |
| **Testing** | **Pytest** | Automated unit, integration, validation, and benchmark test suite. |
| **Isolation (Optional)** | **Docker** | Optional sandboxed container for isolating generated code execution. |

---

## 2. Minimal Dependency List

```toml
[project]
name = "self-validating-analytics-agent"
version = "0.1.0"
description = "Self-Validating Autonomous Data Analytics Agent for Diagnostic & Evidence-Based Decision Support"
requires-python = ">=3.10"
dependencies = [
    "streamlit>=1.32.0",
    "pandas>=2.0.0",
    "duckdb>=0.10.0",
    "numpy>=1.24.0",
    "plotly>=5.19.0",
    "langgraph>=0.0.30",
    "langchain-core>=0.1.30",
    "httpx>=0.27.0",
    "pydantic>=2.6.0",
    "openpyxl>=3.1.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.3.0"
]
```

---

## 3. Technology Boundary & Simplicity Rules

- **No FastAPI Backend Required:** The application runs as a cohesive, modular Streamlit Python application, eliminating unnecessary REST API serialization and microservice network complexity.
- **In-Process Database:** DuckDB runs directly in-memory within the Python process; no PostgreSQL or external database installation is required for core functionality.
- **Direct Local Ollama:** Connects to local Ollama via simple HTTP requests (`httpx`), avoiding heavy framework overhead.
