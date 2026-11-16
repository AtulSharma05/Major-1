# Testing & Evaluation Strategy

Aligned directly with the evaluation plan in the Major Project Report (Chapter 5.12, Chapter 8) and finalized Three-Layer Validation Architecture.

---

## 1. Testing Hierarchy

```
       [Host Testing / Streamlit UI Demo]
                      │
       [Ablation Study: 5 Configurations]
                      │
   [Benchmark Suite: 50–100 Questions (DABStep/Retail)]
                      │
[Three-Layer Validation & Self-Correction Tests]
 ├── Layer 1: Execution Validation Tests
 ├── Layer 2: Numerical Validation Tests
 └── Layer 3: Semantic Validation Tests
                      │
      [Unit Tests (Pandas, DuckDB, Ollama)]
```

---

## 2. Test Types

### 2.1 Unit & Component Tests
- **Data Ingestion & Profiling:** Validates parsing of CSV/Excel files, schema inference, missing value detection, and statistical distribution computations.
- **Analytics & Execution:** Verifies that generated Python/SQL code runs safely, obeys timeouts, and catches runtime errors gracefully.
- **Planning Logic:** Verifies that LangGraph transitions properly manage agent state, step completion, and stopping conditions.

### 2.2 Three-Layer Validation Tests

#### 2.2.1 Layer 1: Execution Validation Tests
- **Syntax Parsing:** Verifies `ast.parse()` traps invalid Python and `sqlglot.parse_one()` traps invalid SQL.
- **Runtime Traps:** Injects missing variables, undefined imports, and non-existent column references to verify clean exception handling.
- **Resource Constraints:** Tests timeout enforcement on long-running loops or queries.
- **Output Structure:** Tests detection of empty, malformed, or non-tabular outputs.

#### 2.2.2 Layer 2: Numerical Validation Tests
- **Independent Re-computation:** Asserts that an aggregate computed via Pandas matches an independent re-computation via DuckDB SQL within tolerance (`atol=1e-4`).
- **Metric Library Accuracy:** Validates formulas for `SUM`, `AVG`, `COUNT`, `DISTINCT COUNT`, and `AOV = SUM(sales)/COUNT(DISTINCT order_id)`.
- **Discrepancy Detection:** Injects intentional calculation errors (e.g. AOV calculated as `AVG(sales)`) to verify that the numerical validator marks the status as `FAIL`.
- **Unsupported Metric Handling:** Tests that novel/unsupported metrics are marked as partially validated (`INCONCLUSIVE`) and never falsely marked `PASS`.

#### 2.2.3 Layer 3: Semantic Validation Tests
- **Plan & Code Inspection:** Uses `SQLGlot` to parse generated SQL and assert that all requested `WHERE` filters, `GROUP BY` dimensions, and aggregations are present in the query AST.
- **Output Inspection:** Verifies that final DataFrame columns match requested metrics and dimensions, and sorting order is preserved.
- **Omission Detection:** Injects plans that omit a user filter (e.g., omitting `year = 2017`) and asserts that semantic validation fails even if numerical execution succeeded.

#### 2.2.4 Combined Decision & Self-Correction Tests
- Verifies the full validation decision matrix: only marks `PASS` when all three layers pass.
- Tests that failures trigger the 6-node LangGraph self-correction loop, that diagnostic feedback is delivered to the LLM, and that retries halt strictly at `max_retries = 2`.
- Verifies that when retries fail, the result is reported with an UNVERIFIED flag and a penalized confidence score.

---

## 3. Benchmark Evaluation (50–100 Questions)
As defined in the project report (Table 4, Chapter 5.12), the agent is evaluated against 50–100 business analytics questions across:
1. **Simple single-step questions:** Total revenue, average order value.
2. **Multi-step questions:** Top 5 products by margin compared to last month.
3. **Diagnostic questions:** "Why did revenue decrease in Q3?"
4. **Insufficient-evidence questions:** Questions where the dataset lacks necessary explanatory variables.

**Evaluation Metrics:**
- Analytical accuracy (%)
- Calculation error rate (%)
- Validation accuracy (%)
- Self-correction success rate (%)
- Unsupported claim rate (%)
- Evidence completeness (%)
- Average response time (s)

---

## 4. Ablation Study (5 Configurations)
Quantifies the exact contribution of each system component as defined in Table 5 of the project report:

| Config | Configuration Name | Code Execution | Validation & Correction | Hypothesis RCA | Provenance & Confidence |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **C1** | LLM-only analytics | No | No | No | No |
| **C2** | LLM + code execution | Yes | No | No | No |
| **C3** | LLM + validation | Yes | Yes | No | No |
| **C4** | LLM + validation + RCA | Yes | Yes | Yes | No |
| **C5** | Complete proposed system | Yes | Yes | Yes | Yes |

---

## 5. Manual Host Testing (Project Owner / Faculty Demo)

For every implemented phase, the project owner performs manual validation:
1. Start local Ollama: `ollama run qwen2.5-coder:7b`.
2. Launch Streamlit UI: `streamlit run src/ui/app.py`.
3. Upload target dataset (`superstore.csv` or `online_retail.csv`).
4. Ask a sample question and observe:
   - Data profiling summary.
   - Plan decomposition in LangGraph.
   - Three validation badges: `Execution: PASS`, `Numerical: PASS`, `Semantic: PASS`.
   - Ranked contributing factors (for diagnostic queries).
   - Evidence records and calibrated confidence score.
