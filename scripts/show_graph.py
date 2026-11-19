"""Print the LangGraph pipeline as a Mermaid diagram (paste into mermaid.live)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridwise.llm import RuleBasedLLM  # noqa: E402
from gridwise.graph import GridWiseGraph  # noqa: E402

print(GridWiseGraph(RuleBasedLLM()).mermaid())
