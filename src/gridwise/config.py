"""Central configuration.

Everything that might change between laptop / staging / production lives here and
is read from environment variables. No secrets or magic numbers are hard-coded in
the agent logic itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- storage -----------------------------------------------------------
    db_path: Path = field(default_factory=lambda: Path(os.environ.get("GRIDWISE_DB", ROOT / "data" / "gridwise.duckdb")))
    knowledge_dir: Path = field(default_factory=lambda: Path(os.environ.get("GRIDWISE_KNOWLEDGE", ROOT / "knowledge")))
    trace_path: Path = field(default_factory=lambda: Path(os.environ.get("GRIDWISE_TRACES", ROOT / "data" / "traces.jsonl")))
    drafts_path: Path = field(default_factory=lambda: Path(os.environ.get("GRIDWISE_DRAFTS", ROOT / "data" / "drafts.sqlite")))
    evals_path: Path = field(default_factory=lambda: Path(os.environ.get("GRIDWISE_EVALS", ROOT / "evals" / "golden.jsonl")))

    # --- model provider ----------------------------------------------------
    # GRIDWISE_PROVIDER = auto | azure_openai | anthropic | offline
    # "auto" picks Azure OpenAI when an endpoint and deployment are configured,
    # else Anthropic when ANTHROPIC_API_KEY is set, else the deterministic
    # offline stand-ins (so the demo and the test suite run with no secrets).
    provider: str = field(default_factory=lambda: os.environ.get("GRIDWISE_PROVIDER", "auto").strip().lower())
    max_tokens: int = field(default_factory=lambda: _env_int("GRIDWISE_MAX_TOKENS", 1200))
    request_timeout_s: float = field(default_factory=lambda: _env_float("GRIDWISE_TIMEOUT", 60.0))
    max_retries: int = field(default_factory=lambda: _env_int("GRIDWISE_MAX_RETRIES", 2))
    # "none" (default) sends no temperature and uses the model's own default. Many
    # current models reject anything else: Claude 5-series via LangChain, and the
    # o-series / gpt-5 reasoning models on Azure. Set e.g. 0 for models that accept it.
    temperature: str = field(default_factory=lambda: os.environ.get("GRIDWISE_TEMPERATURE", "none").strip().lower())

    # Anthropic
    model: str = field(default_factory=lambda: os.environ.get("GRIDWISE_MODEL", "claude-sonnet-5-5"))
    api_key: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"))
    anthropic_base_url: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_BASE_URL") or None)

    # Azure OpenAI
    # https://<resource>.openai.azure.com  (or https://<resource>.services.ai.azure.com)
    azure_endpoint: str | None = field(default_factory=lambda: (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").rstrip("/") or None)
    azure_deployment: str | None = field(default_factory=lambda: os.environ.get("AZURE_OPENAI_DEPLOYMENT") or None)
    azure_api_key: str | None = field(default_factory=lambda: os.environ.get("AZURE_OPENAI_API_KEY") or None)
    # Empty -> the v1 GA API (<endpoint>/openai/v1/, no api-version). Set a dated
    # version (e.g. 2024-10-21) to use the classic deployment-scoped API instead.
    azure_api_version: str | None = field(default_factory=lambda: os.environ.get("AZURE_OPENAI_API_VERSION") or None)
    # Keyless auth with Microsoft Entra ID (managed identity, az login, ...).
    azure_use_entra_id: bool = field(default_factory=lambda: _env_bool("AZURE_OPENAI_USE_ENTRA_ID", False))
    azure_entra_scope: str | None = field(default_factory=lambda: os.environ.get("AZURE_OPENAI_ENTRA_SCOPE") or None)
    # Optional: an embedding deployment (e.g. text-embedding-3-small) for retrieval.
    azure_embedding_deployment: str | None = field(default_factory=lambda: os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or None)

    # Illustrative unit prices in EUR per 1M tokens. Override per model/contract.
    price_in_per_mtok: float = field(default_factory=lambda: _env_float("GRIDWISE_PRICE_IN", 3.0))
    price_out_per_mtok: float = field(default_factory=lambda: _env_float("GRIDWISE_PRICE_OUT", 15.0))

    # --- agent behaviour ---------------------------------------------------
    max_steps: int = field(default_factory=lambda: _env_int("GRIDWISE_MAX_STEPS", 5))
    max_repairs: int = field(default_factory=lambda: _env_int("GRIDWISE_MAX_REPAIRS", 1))

    # --- guardrails --------------------------------------------------------
    # Prompt-injection score at or above which a question is refused and a
    # retrieved passage is quarantined.
    injection_block_score: float = field(default_factory=lambda: _env_float("GRIDWISE_INJECTION_BLOCK", 1.0))
    # Share of a claim's content words that must appear in one evidence unit for
    # the offline verifier to count the claim as supported.
    claim_support_threshold: float = field(default_factory=lambda: _env_float("GRIDWISE_CLAIM_SUPPORT", 0.6))
    # Minimum cosine similarity for a retrieved chunk to count as a match. Unset:
    # 0.05 for the local TF-IDF vectors, 0.30 for neural (Azure) embeddings, whose
    # similarities between unrelated texts are much higher.
    retrieval_min_score_override: float | None = field(
        default_factory=lambda: _env_float("GRIDWISE_RETRIEVAL_MIN", -1.0) if os.environ.get("GRIDWISE_RETRIEVAL_MIN") else None)

    # --- monitoring --------------------------------------------------------
    # Forecast quality gate: above this MAPE the answer is flagged for review.
    mape_alert_threshold: float = field(default_factory=lambda: _env_float("GRIDWISE_MAPE_ALERT", 0.12))
    # Log (redacted) prompts and completions in the trace. Turn off if your data
    # policy forbids storing prompt text at all.
    log_prompts: bool = field(default_factory=lambda: _env_bool("GRIDWISE_LOG_PROMPTS", True))
    log_max_chars: int = field(default_factory=lambda: _env_int("GRIDWISE_LOG_MAX_CHARS", 4000))

    @property
    def resolved_provider(self) -> str:
        if self.provider in {"azure_openai", "anthropic", "offline"}:
            return self.provider
        if self.provider not in {"", "auto"}:
            raise ValueError(f"GRIDWISE_PROVIDER must be auto, azure_openai, anthropic or offline, not {self.provider!r}")
        if self.azure_endpoint and self.azure_deployment:
            return "azure_openai"
        if self.api_key:
            return "anthropic"
        return "offline"

    @property
    def offline(self) -> bool:
        """True when no provider is configured -> deterministic stand-in agents."""
        return self.resolved_provider == "offline"

    @property
    def model_label(self) -> str:
        provider = self.resolved_provider
        if provider == "azure_openai":
            return f"azure_openai:{self.azure_deployment}"
        if provider == "anthropic":
            return f"anthropic:{self.model}"
        return "rule_based_stub"

    @property
    def uses_azure_embeddings(self) -> bool:
        return bool(self.azure_endpoint and self.azure_embedding_deployment)

    @property
    def retrieval_min_score(self) -> float:
        if self.retrieval_min_score_override is not None:
            return self.retrieval_min_score_override
        return 0.30 if self.uses_azure_embeddings else 0.05


settings = Settings()
