"""Test setup.

The environment variables are set *before* `gridwise` is imported, so the whole
test session runs against a throwaway database and trace log instead of the
developer's local files.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_TMP = Path(tempfile.mkdtemp(prefix="gridwise-tests-"))
os.environ["GRIDWISE_DB"] = str(_TMP / "test.duckdb")
os.environ["GRIDWISE_TRACES"] = str(_TMP / "traces.jsonl")
os.environ["GRIDWISE_DRAFTS"] = str(_TMP / "drafts.sqlite")
# Force the deterministic stand-in, whatever the developer's shell has configured.
for _name in ("ANTHROPIC_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_KEY",
              "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "AZURE_OPENAI_API_VERSION", "GRIDWISE_PROVIDER"):
    os.environ.pop(_name, None)
sys.path.insert(0, str(ROOT / "tests"))

import pytest  # noqa: E402

from gridwise.data import build_database  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def database() -> Path:
    """One small database for the whole session."""
    return build_database(days=120, seed=11)


@pytest.fixture()
def trace_file() -> Path:
    return Path(os.environ["GRIDWISE_TRACES"])
