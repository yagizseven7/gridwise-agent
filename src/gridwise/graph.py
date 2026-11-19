"""The agent pipeline as a LangGraph state machine.

    START
      |
    input_guard ──(injection)──────────────────────────► refuse ──┐
      |  PII redacted from the question                              |
    router ──(provider content filter)─────────────────► refuse ───┤
      |  router agent: picks tools -> validated Plan                 |
    act ──(no usable evidence)─────────────────────────► no_evidence ┤
      |  tools run; values flow between steps; passages screened     |
    synthesize ◄────────────┐  answer writer                          |
      |──(writer failed)────┼──────────────────────────► withhold ───┤
      |                     │                                        |
    verify ──(not grounded, │ repairs left: feedback)                |
      |  verification agent: claims + numbers                        |
      ├──(not grounded, no repairs left)──────────────► withhold ───┤
      |                                                              |
    finalize ◄───────────────────────────────────────────────────────┘
      |  quality gates, PII scrub on output, citations, confidence
     END

Each node is a plain function of the state. The model, the tracer and the
verification agent are injected; nothing is global, so every node can be unit
tested and the whole graph can run against a fake model.

Model calls can fail (rate limits, timeouts, an Azure content filter). No such
failure crashes a run: the router falls back to document search, a failed writer
leads to the evidence being returned as-is, and a failed verifier counts as "not
verified" (fail closed). Each case leaves an `llm_error:<role>:<kind>` flag.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from . import guardrails, pii
from .config import settings
from .llm import LLM, LLMError, extract_json
from .observability import Tracer
from .schemas import AgentAnswer, Evidence, Plan, Verification
from .tools import TOOLS, get_tool
from .verifier import VerificationAgent

PLACEHOLDER_PREFIX = "$"

REFUSAL = (
    "This request looks like an attempt to override the assistant's instructions, so it "
    "was not processed. Please rephrase it as a question about demand, forecasts, costs "
    "or the client's operating rules."
)
NO_EVIDENCE = "I could not gather any evidence for that question, so I am not going to guess."


class AgentState(TypedDict, total=False):
    question: str            # redacted before any model sees it
    site_id: str
    flags: list[str]
    blocked: bool
    plan: Plan
    evidence: list[Evidence]
    draft: dict[str, Any]
    feedback: str
    attempts: int
    writer_failed: bool
    verification: Verification
    answer: AgentAnswer


def _tracer(config: RunnableConfig) -> Tracer:
    return config["configurable"]["tracer"]


def _add_flags(state: AgentState, *new: str) -> list[str]:
    return sorted(set(state.get("flags", [])) | {f for f in new if f})


class GridWiseGraph:
    def __init__(self, llm: LLM, max_steps: int | None = None, max_repairs: int | None = None):
        self.llm = llm
        self.verifier = VerificationAgent(llm)
        self.max_steps = max_steps or settings.max_steps
        self.max_repairs = settings.max_repairs if max_repairs is None else max_repairs
        self.app = self._build()

    # --- nodes -------------------------------------------------------------
    def input_guard(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        with _tracer(config).step("guardrail", "input_guard") as sink:
            question, pii_kinds = pii.redact(state["question"])
            verdict = guardrails.detect_prompt_injection(question)
            flags: list[str] = []
            if pii_kinds:
                flags.append("pii_redacted_input")
            if verdict.blocked:
                flags.append("prompt_injection_blocked")
            elif verdict.suspicious:
                flags.append("possible_prompt_injection")
            sink["detail"].update({
                "question": question, "pii_kinds": pii_kinds,
                "injection_score": verdict.score, "injection_rules": verdict.rules,
            })
            sink["ok"] = not verdict.blocked
        return {"question": question, "blocked": verdict.blocked, "flags": _add_flags(state, *flags)}

    def router(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        question, site_id = state["question"], state["site_id"]
        with _tracer(config).step("router", "router_agent", site_id=site_id) as sink:
            try:
                response = self.llm.route(question, site_id)
            except LLMError as exc:
                sink["ok"] = False
                sink["detail"]["llm_error"] = {"kind": exc.kind, "detail": exc.detail}
                flag = f"llm_error:router:{exc.kind}"
                if exc.kind == "content_filter":
                    # The provider's own safety filter (e.g. Azure Prompt Shields)
                    # judged the input an attack: treat it exactly like ours did.
                    return {"blocked": True, "flags": _add_flags(state, flag, "provider_content_filter")}
                plan = self._fallback_plan(question, "Router unavailable; falling back to document search.")
                sink["detail"]["route"] = [s.tool for s in plan.steps]
                return {"plan": plan, "flags": _add_flags(state, flag)}
            Tracer.record_llm(sink, response)
            try:
                plan = Plan.model_validate(extract_json(response.text))
            except (ValidationError, ValueError) as exc:
                # A malformed plan is not a crash: fall back to the safest useful
                # action (document search) and record why.
                sink["detail"]["plan_error"] = str(exc)[:300]
                plan = self._fallback_plan(question, "Router output failed validation; falling back to document search.")
            sink["detail"]["objective"] = plan.objective
            sink["detail"]["route"] = [s.tool for s in plan.steps]
        return {"plan": plan}

    @staticmethod
    def _fallback_plan(question: str, why: str) -> Plan:
        return Plan(objective=question[:200],
                    steps=[{"tool": "search_playbook", "args": {"query": question, "k": 3}, "why": why}])

    @staticmethod
    def resolve_args(args: dict[str, Any], blackboard: dict[str, float]) -> dict[str, Any]:
        """Fill values a later step needs from what earlier steps produced.

        `None` means "look me up by parameter name"; "$peak_kw" means "look me up
        under that key". This is how `estimate_imbalance_cost` gets the peak that
        `forecast_demand` or the SQL lookup computed, without the model ever
        handling the number.
        """
        resolved: dict[str, Any] = {}
        for key, value in args.items():
            if value is None and key in blackboard:
                resolved[key] = blackboard[key]
            elif isinstance(value, str) and value.startswith(PLACEHOLDER_PREFIX):
                resolved[key] = blackboard.get(value[1:], value)
            else:
                resolved[key] = value
        return resolved

    def act(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        tracer = _tracer(config)
        evidence: list[Evidence] = []
        blackboard: dict[str, float] = {}
        flags: list[str] = []

        for call in state["plan"].steps[: self.max_steps]:
            if call.tool not in TOOLS:
                continue
            args = self.resolve_args(call.args, blackboard)
            with tracer.step("tool", call.tool, args=args, why=call.why) as sink:
                try:
                    result = get_tool(call.tool)(**args)
                except Exception as exc:  # noqa: BLE001
                    # A failing tool degrades the answer and says so; it does not end the run.
                    reason = f"{type(exc).__name__}: {exc}"[:300]
                    sink["detail"]["handled_error"] = reason
                    sink["ok"] = False
                    flags.append(f"tool_failed:{call.tool}")
                    evidence.append(Evidence(
                        tool=call.tool, ok=False, values={}, citations=[], raw={"error": reason},
                        summary=f"Tool {call.tool} failed: {reason}",
                        brief=f"I could not complete the {call.tool.replace('_', ' ')} step, so part of the question is unanswered.",
                    ))
                    continue

                quarantined = result.get("raw", {}).get("quarantined") or []
                if quarantined:
                    flags.append("retrieved_content_quarantined")
                    sink["detail"]["quarantined"] = quarantined
                sink["detail"]["summary"] = result["summary"][:400]
                sink["detail"]["values"] = result.get("values", {})
                sink["detail"]["citations"] = result.get("citations", [])
                blackboard.update(result.get("values", {}))
                evidence.append(Evidence(
                    tool=call.tool,
                    summary=result["summary"],
                    brief=result.get("brief", result["summary"]),
                    values=result.get("values", {}),
                    citations=result.get("citations", []),
                    raw=result.get("raw", {}),
                ))
        return {"evidence": evidence, "flags": _add_flags(state, *flags)}

    def synthesize(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        attempt = state.get("attempts", 0) + 1
        payload = [e.model_dump(exclude={"raw"}) for e in state["evidence"]]
        with _tracer(config).step("synthesis", f"answer_writer_attempt_{attempt}") as sink:
            try:
                response = self.llm.synthesize(state["question"], payload, state.get("feedback", ""))
            except LLMError as exc:
                sink["ok"] = False
                sink["detail"]["llm_error"] = {"kind": exc.kind, "detail": exc.detail}
                flags = [f"llm_error:synthesis:{exc.kind}"]
                if exc.kind == "content_filter":
                    flags.append("provider_content_filter")
                return {"writer_failed": True, "attempts": attempt, "flags": _add_flags(state, *flags)}
            Tracer.record_llm(sink, response)
            try:
                draft = extract_json(response.text)
            except ValueError:
                draft = {"answer": response.text.strip(), "confidence": "low", "flags": ["unparsed_model_output"]}
            sink["detail"]["answer_chars"] = len(str(draft.get("answer", "")))
        return {"draft": draft, "attempts": attempt}

    def verify(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        attempt = state.get("attempts", 1)
        answer = str(state["draft"].get("answer", ""))
        llm_flags: list[str] = []
        with _tracer(config).step("verification", f"verification_agent_attempt_{attempt}") as sink:
            try:
                verdict, response = self.verifier.verify(state["question"], answer, state["evidence"])
                Tracer.record_llm(sink, response)
            except LLMError as exc:
                # Fail closed: an answer nobody could check is not a checked answer.
                sink["detail"]["llm_error"] = {"kind": exc.kind, "detail": exc.detail}
                llm_flags.append(f"llm_error:verification:{exc.kind}")
                verdict = Verification(grounded=False, unsupported_claims=[f"verifier unavailable ({exc.kind})"],
                                       verifier=exc.provider)
            sink["ok"] = verdict.grounded
            sink["detail"].update({
                "grounded": verdict.grounded,
                "ungrounded_numbers": verdict.ungrounded_numbers,
                "unsupported_claims": verdict.unsupported_claims,
                "claims_checked": len(verdict.claims),
            })
        flags = []
        if verdict.ungrounded_numbers:
            flags.append("ungrounded_numbers_detected")
        if verdict.unsupported_claims:
            flags.append("unsupported_claims_detected")
        return {"verification": verdict, "feedback": verdict.feedback(),
                "flags": _add_flags(state, *flags, *llm_flags)}

    def withhold(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        with _tracer(config).step("guardrail", "withhold_unverified_answer"):
            usable = [e for e in state["evidence"] if e.ok]
            reason = (
                "The answer-writing model was unavailable, so here is the evidence as gathered"
                if state.get("writer_failed")
                else "I could not verify every statement in my draft answer against the tool results, "
                     "so I am withholding it"
            )
            text = f"{reason}. Evidence gathered: " + " ".join(e.brief or e.summary for e in usable)
        return {"draft": {"answer": text, "confidence": "low", "flags": []},
                "flags": _add_flags(state, "answer_withheld")}

    def refuse(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        return {"draft": {"answer": REFUSAL, "confidence": "low", "flags": []}, "evidence": []}

    def no_evidence(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        return {"draft": {"answer": NO_EVIDENCE, "confidence": "low", "flags": []},
                "flags": _add_flags(state, "no_evidence")}

    def finalize(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        evidence = state.get("evidence", [])
        draft = state["draft"]
        payload = [e.model_dump(exclude={"raw"}) for e in evidence]
        flags = set(state.get("flags", [])) | set(draft.get("flags", [])) | set(guardrails.quality_flags(payload))

        text, out_pii = pii.redact(str(draft.get("answer", "")).strip())
        if out_pii:
            flags.add("pii_redacted_output")

        confidence = draft.get("confidence", "medium")
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        hard_stops = {"prompt_injection_blocked", "provider_content_filter", "answer_withheld", "no_evidence"}
        if flags & hard_stops or any(f.startswith("tool_failed:") for f in flags):
            confidence = "low"
        elif flags and confidence == "high":
            confidence = "medium"

        answer = AgentAnswer(
            answer=text,
            evidence=evidence,
            citations=sorted({c for e in evidence for c in e.citations}),
            confidence=confidence,
            flags=sorted(flags),
            verification=state.get("verification"),
        )
        with _tracer(config).step("answer", "final_answer") as sink:
            sink["detail"].update({
                "answer": answer.answer, "citations": answer.citations,
                "confidence": answer.confidence, "flags": answer.flags,
            })
        return {"answer": answer, "flags": answer.flags}

    # --- edges ---------------------------------------------------------------
    @staticmethod
    def after_guard(state: AgentState) -> Literal["refuse", "router"]:
        return "refuse" if state.get("blocked") else "router"

    @staticmethod
    def after_router(state: AgentState) -> Literal["refuse", "act"]:
        return "refuse" if state.get("blocked") else "act"

    @staticmethod
    def after_synthesize(state: AgentState) -> Literal["verify", "withhold"]:
        return "withhold" if state.get("writer_failed") else "verify"

    @staticmethod
    def after_act(state: AgentState) -> Literal["synthesize", "no_evidence"]:
        # Usable = the tool succeeded *and* found something (numbers or a citation).
        usable = [e for e in state.get("evidence", []) if e.ok and (e.values or e.citations)]
        return "synthesize" if usable else "no_evidence"

    def after_verify(self, state: AgentState) -> Literal["finalize", "synthesize", "withhold"]:
        if state["verification"].grounded:
            return "finalize"
        if state.get("attempts", 1) <= self.max_repairs:
            return "synthesize"  # repair: the writer gets the verifier's feedback
        return "withhold"

    def _build(self):
        graph = StateGraph(AgentState)
        for name in ("input_guard", "router", "act", "synthesize", "verify",
                     "withhold", "refuse", "no_evidence", "finalize"):
            graph.add_node(name, getattr(self, name))

        graph.add_edge(START, "input_guard")
        graph.add_conditional_edges("input_guard", self.after_guard, ["refuse", "router"])
        graph.add_conditional_edges("router", self.after_router, ["refuse", "act"])
        graph.add_conditional_edges("act", self.after_act, ["synthesize", "no_evidence"])
        graph.add_conditional_edges("synthesize", self.after_synthesize, ["verify", "withhold"])
        graph.add_conditional_edges("verify", self.after_verify, ["finalize", "synthesize", "withhold"])
        for terminal in ("withhold", "refuse", "no_evidence"):
            graph.add_edge(terminal, "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    # --- entry point -------------------------------------------------------------
    def invoke(self, question: str, site_id: str, tracer: Tracer) -> AgentAnswer:
        state = self.app.invoke(
            {"question": question, "site_id": site_id, "flags": [], "attempts": 0},
            config={"configurable": {"tracer": tracer}, "recursion_limit": 30},
        )
        return state["answer"]

    def mermaid(self) -> str:
        return self.app.get_graph().draw_mermaid()
