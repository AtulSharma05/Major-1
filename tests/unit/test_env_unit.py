"""Offline environment and packaging unit tests.

These tests run 100% offline without any external services or Ollama.
"""

import importlib
import sys


def test_python_version_minimum():
    """Verify that Python runtime is at least version 3.10."""
    assert sys.version_info >= (3, 10), (
        f"Python 3.10+ is required, but running under {sys.version}"
    )


def test_core_package_structure():
    """Verify that the project packages are structured and importable."""
    packages = ["src", "src.agent", "src.ui"]
    for pkg in packages:
        module = importlib.import_module(pkg)
        assert module is not None, f"Failed to import package: {pkg}"


def test_repo_root_fixture(repo_root, data_dir):
    """Verify that test fixtures locate project root and data directory."""
    assert repo_root.exists(), "Repo root path does not exist"
    assert (repo_root / "pyproject.toml").exists(), "pyproject.toml not found in repo root"
    assert data_dir.exists(), "data/ directory not found in repo root"

