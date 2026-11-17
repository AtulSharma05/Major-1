"""Ollama client integration and schema-grounded SQL code generation."""

from __future__ import annotations

import re

import httpx

from src.agent.codegen.prompts import SYSTEM_PROMPT, build_sql_prompt
from src.agent.data.profiling import TableProfile
from src.agent.planning.intent import QuerySpecification


def extract_sql_from_response(raw_text: str) -> str:
    """Extract clean SQL string from LLM output, stripping markdown code fences."""
    text = raw_text.strip()

    # Match ```sql ... ``` or ``` ... ```
    sql_block_pattern = r"```(?:sql)?\s*([\s\S]*?)\s*```"
    match = re.search(sql_block_pattern, text, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip()
    else:
        # If no code block, strip leading commentary and look for SELECT or WITH
        extracted = text

    # Remove trailing semicolon for uniform DuckDB execution
    extracted = re.sub(r";\s*$", "", extracted).strip()
    return extracted


class OllamaClient:
    """HTTP client for communicating with local Ollama daemon."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:7b",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Call Ollama /api/generate endpoint with temperature 0.0 for deterministic output."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Ensure 'ollama serve' is running."
            ) from exc
        except httpx.HTTPStatusError as exc:
            err_code = exc.response.status_code
            raise RuntimeError(f"Ollama API returned HTTP {err_code}: {exc}") from exc


class CodeGenerator:
    """Orchestrates prompt formatting and SQL synthesis."""

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def generate_sql(
        self,
        question: str,
        profile: TableProfile,
        intent: QuerySpecification | None = None,
        error_feedback: str | None = None,
    ) -> str:
        """Generate executable DuckDB SQL for an analytical question."""
        prompt = build_sql_prompt(
            question=question,
            profile=profile,
            intent=intent,
            error_feedback=error_feedback,
        )
        raw_response = self.client.generate(prompt=prompt)
        return extract_sql_from_response(raw_response)
