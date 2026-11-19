"""Public entry point for running the agent.

The pipeline itself is the LangGraph state machine in `graph.py`. This module
keeps a small, stable surface (`Agent.run`, `ask`) for the API, the scripts and
the tests, and owns one thing the graph does not: a fresh `Tracer` per run.
"""

from __future__ import annotations

from .config import settings
from .data import validate_site
from .graph import GridWiseGraph
from .llm import LLM, get_llm
from .observability import Tracer
from .schemas import AgentAnswer


class Agent:
    def __init__(self, llm: LLM | None = None, max_steps: int | None = None, max_repairs: int | None = None):
        self.llm = llm or get_llm()
        self.graph = GridWiseGraph(self.llm, max_steps=max_steps or settings.max_steps, max_repairs=max_repairs)

    def run(self, question: str, site_id: str = "NL-UTR-01") -> tuple[AgentAnswer, Tracer]:
        validate_site(site_id)  # the stand-in router writes it into SQL
        tracer = Tracer()
        answer = self.graph.invoke(question, site_id, tracer)
        return answer, tracer


def ask(question: str, site_id: str = "NL-UTR-01") -> tuple[AgentAnswer, Tracer]:
    return Agent().run(question, site_id=site_id)
