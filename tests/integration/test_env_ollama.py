"""Live integration tests for local Ollama service.

Verifies connectivity to Ollama on http://localhost:11434 and checks
for available LLM models (specifically qwen2.5-coder:7b).
"""

import pytest


def test_ollama_service_connectivity():
    """Check if local Ollama daemon is reachable on localhost:11434."""
    try:
        import httpx
    except ImportError:
        pytest.skip("httpx is not installed in the current environment")

    url = "http://localhost:11434/api/tags"
    try:
        response = httpx.get(url, timeout=5.0)
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        pytest.skip(f"Ollama server is not running on localhost:11434: {exc}")

    assert response.status_code == 200, f"Ollama returned HTTP status {response.status_code}"
    data = response.json()
    assert "models" in data, "Malformed response from Ollama: 'models' key missing"

    available_models = [m.get("name", "") for m in data.get("models", [])]
    # Check if target model or any qwen model is pulled
    qwen_present = any("qwen2.5-coder" in name for name in available_models)

    # We record this check: if qwen is missing, warn or assert based on installation state
    if not qwen_present:
        pytest.skip(
            f"Ollama is reachable, but 'qwen2.5-coder:7b' is not yet downloaded. "
            f"Installed models: {available_models}. Run 'ollama pull qwen2.5-coder:7b'."
        )
