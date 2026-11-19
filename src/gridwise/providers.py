"""Model providers: build the LangChain chat model and embeddings from settings.

    GRIDWISE_PROVIDER=azure_openai   Azure OpenAI (chat + optional embeddings)
    GRIDWISE_PROVIDER=anthropic      Claude via the Anthropic API
    GRIDWISE_PROVIDER=offline        deterministic stand-ins, no network
    GRIDWISE_PROVIDER=auto           (default) the first one that is configured

Azure OpenAI supports two API styles, both built on LangChain's OpenAI classes:

* **v1 GA API** (default): base URL `<endpoint>/openai/v1/`, the deployment name
  passed as the model, no `api-version`. This is the style Microsoft recommends
  for new work.
* **Classic API**: set `AZURE_OPENAI_API_VERSION` (e.g. `2024-10-21`) and requests go
  to `<endpoint>/openai/deployments/<deployment>/...?api-version=...` through
  `AzureChatOpenAI` / `AzureOpenAIEmbeddings`.

Authentication is either `AZURE_OPENAI_API_KEY` or keyless Microsoft Entra ID
(`AZURE_OPENAI_USE_ENTRA_ID=true`), which uses `DefaultAzureCredential`: a managed
identity in Azure, `az login` on a laptop. Keyless is what you want in production.

Everything here is imported lazily, so the offline path needs none of these SDKs.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings, settings

V1_ENTRA_SCOPE = "https://ai.azure.com/.default"
CLASSIC_ENTRA_SCOPE = "https://cognitiveservices.azure.com/.default"


class ProviderConfigError(ValueError):
    """The selected provider is missing configuration. Raised at start-up."""


def _temperature(cfg: Settings) -> dict[str, Any]:
    if cfg.temperature in {"", "none", "default"}:
        return {}
    try:
        return {"temperature": float(cfg.temperature)}
    except ValueError as exc:
        raise ProviderConfigError(f"GRIDWISE_TEMPERATURE must be a number or 'none', not {cfg.temperature!r}") from exc


def entra_token_provider(cfg: Settings) -> Callable[[], str]:
    """A callable that returns a fresh Entra ID bearer token for Azure OpenAI."""
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ProviderConfigError("AZURE_OPENAI_USE_ENTRA_ID=true needs `pip install azure-identity`.") from exc
    scope = cfg.azure_entra_scope or (CLASSIC_ENTRA_SCOPE if cfg.azure_api_version else V1_ENTRA_SCOPE)
    return get_bearer_token_provider(DefaultAzureCredential(), scope)


def _azure_auth(cfg: Settings, token_provider: Callable[[], str] | None) -> Callable[[], str] | str:
    if token_provider is not None:
        return token_provider
    if cfg.azure_use_entra_id:
        return entra_token_provider(cfg)
    if cfg.azure_api_key:
        return cfg.azure_api_key
    raise ProviderConfigError("Azure OpenAI needs AZURE_OPENAI_API_KEY or AZURE_OPENAI_USE_ENTRA_ID=true.")


def _require_azure(cfg: Settings, deployment: str | None, what: str) -> None:
    missing = [name for name, value in (("AZURE_OPENAI_ENDPOINT", cfg.azure_endpoint), (what, deployment)) if not value]
    if missing:
        raise ProviderConfigError(f"Azure OpenAI is selected but {', '.join(missing)} is not set.")


def azure_chat_model(
    cfg: Settings = settings,
    *,
    token_provider: Callable[[], str] | None = None,
    http_client: Any = None,
) -> BaseChatModel:
    _require_azure(cfg, cfg.azure_deployment, "AZURE_OPENAI_DEPLOYMENT")
    auth = _azure_auth(cfg, token_provider)
    common: dict[str, Any] = {
        "max_tokens": cfg.max_tokens,
        "timeout": cfg.request_timeout_s,
        "max_retries": cfg.max_retries,
        **_temperature(cfg),
    }
    if http_client is not None:
        common["http_client"] = http_client

    if cfg.azure_api_version:
        from langchain_openai import AzureChatOpenAI

        if callable(auth):
            common["azure_ad_token_provider"] = auth
        else:
            common["api_key"] = auth
        return AzureChatOpenAI(
            azure_endpoint=cfg.azure_endpoint,
            azure_deployment=cfg.azure_deployment,
            api_version=cfg.azure_api_version,
            **common,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=f"{cfg.azure_endpoint}/openai/v1/",
        model=cfg.azure_deployment,  # in the v1 API the deployment name is the model
        api_key=auth,
        **common,
    )


def azure_embeddings(
    cfg: Settings = settings,
    *,
    token_provider: Callable[[], str] | None = None,
    http_client: Any = None,
) -> Embeddings:
    _require_azure(cfg, cfg.azure_embedding_deployment, "AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    auth = _azure_auth(cfg, token_provider)
    # Token-length checks need tiktoken encodings from the internet and do not
    # know Azure deployment names; our chunks are far below the input limit.
    common: dict[str, Any] = {"check_embedding_ctx_length": False, "max_retries": cfg.max_retries,
                              "timeout": cfg.request_timeout_s}
    if http_client is not None:
        common["http_client"] = http_client

    if cfg.azure_api_version:
        from langchain_openai import AzureOpenAIEmbeddings

        if callable(auth):
            common["azure_ad_token_provider"] = auth
        else:
            common["api_key"] = auth
        return AzureOpenAIEmbeddings(
            azure_endpoint=cfg.azure_endpoint,
            azure_deployment=cfg.azure_embedding_deployment,
            api_version=cfg.azure_api_version,
            **common,
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        base_url=f"{cfg.azure_endpoint}/openai/v1/",
        model=cfg.azure_embedding_deployment,
        api_key=auth,
        **common,
    )


def anthropic_chat_model(cfg: Settings = settings) -> BaseChatModel:
    if not cfg.api_key:
        raise ProviderConfigError("Anthropic is selected but ANTHROPIC_API_KEY is not set.")
    from langchain_anthropic import ChatAnthropic

    kwargs: dict[str, Any] = {}
    if cfg.anthropic_base_url:
        kwargs["base_url"] = cfg.anthropic_base_url
    return ChatAnthropic(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        timeout=cfg.request_timeout_s,
        api_key=cfg.api_key,
        max_retries=cfg.max_retries,
        **_temperature(cfg),
        **kwargs,
    )


def chat_model(cfg: Settings = settings) -> BaseChatModel | None:
    """The configured chat model, or None for the offline stand-ins."""
    provider = cfg.resolved_provider
    if provider == "azure_openai":
        return azure_chat_model(cfg)
    if provider == "anthropic":
        return anthropic_chat_model(cfg)
    return None
