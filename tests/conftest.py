"""Shared fixtures for all test modules."""

import os
import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent real LLM calls in tests
os.environ.setdefault("GEMINI_API_KEY", "test-key-no-llm")
os.environ.setdefault("GEMINI_MODEL", "gemini-2.0-flash")


@pytest.fixture(scope="session")
def cleaned_data_dir() -> Path:
    """Return path to cleaned Parquet files, ensuring they exist."""
    d = PROJECT_ROOT / "data_cleaned"
    assert d.is_dir(), f"{d} does not exist — run `python src/prepare_data.py` first"
    return d


@pytest.fixture(scope="session")
def raw_data_dir() -> Path:
    d = PROJECT_ROOT / "data" / "fmcg-sales-copilot-ai-engineer-mid-4to6"
    assert d.is_dir(), f"{d} does not exist"
    return d


@pytest.fixture(scope="session")
def reconciliation_report_path() -> Path:
    p = PROJECT_ROOT / "reconciliation_report.json"
    assert p.is_file(), f"{p} does not exist — run `python src/prepare_data.py` first"
    return p