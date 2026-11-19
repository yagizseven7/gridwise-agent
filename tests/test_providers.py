"""Azure OpenAI and Anthropic, driven through their real SDKs against a local fake server."""

from __future__ import annotations

import dataclasses

import pytest

from fake_provider_server import FakeProviderServer
from gridwise import llm as llm_module
from gridwise import providers, retrieval
from gridwise.config import settings
from gridwise.llm import ChatModelLLM, LLMError
from gridwise.orchestrator import Agent


@pytest.fixture()
def server():
    with FakeProviderServer() as fake:
        yield fake


def azure_cfg(server: FakeProviderServer, **overrides):
    base = dict(provider="azure_openai", azure_endpoint=server.url, azure_deployment="gpt-dep",
                azure_api_key="azure-test-key", azure_api_version=None, max_retries=0)
    return dataclasses.replace(settings, **(base | overrides))


def assert_not_degraded(answer) -> None:
    """A run that silently fell back must not count as a provider success."""
    assert not [f for f in answer.flags if f.startswith("llm_error")], answer.flags


def chat_requests(server: FakeProviderServer) -> list[dict]:
    return [r for r in server.requests if r["path"].split("?")[0].endswith(("/chat/completions", "/v1/messages"))]


class TestProviderSelection:
    def test_auto_prefers_azure_then_anthropic_then_offline(self) -> None:
        base = dataclasses.replace(settings, provider="auto", api_key=None, azure_endpoint=None, azure_deployment=None)
        assert base.resolved_provider == "offline"
        assert dataclasses.replace(base, api_key="k").resolved_provider == "anthropic"
        both = dataclasses.replace(base, api_key="k", azure_endpoint="https://x.openai.azure.com", azure_deployment="d")
        assert both.resolved_provider == "azure_openai"
        assert both.model_label == "azure_openai:d"

    def test_explicit_provider_wins(self) -> None:
        cfg = dataclasses.replace(settings, provider="offline", api_key="k")
        assert cfg.resolved_provider == "offline"

    def test_unknown_provider_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            _ = dataclasses.replace(settings, provider="openai").resolved_provider

    @pytest.mark.parametrize("missing", ["azure_endpoint", "azure_deployment"])
    def test_azure_misconfiguration_fails_fast(self, missing: str) -> None:
        fields = {"provider": "azure_openai", "azure_endpoint": "https://x.openai.azure.com",
                  "azure_deployment": "d", "azure_api_key": "k"} | {missing: None}
        cfg = dataclasses.replace(settings, **fields)
        with pytest.raises(providers.ProviderConfigError):
            providers.chat_model(cfg)

    def test_azure_needs_a_key_or_entra_id(self) -> None:
        cfg = dataclasses.replace(settings, provider="azure_openai", azure_endpoint="https://x.openai.azure.com",
                                  azure_deployment="d", azure_api_key=None, azure_use_entra_id=False)
        with pytest.raises(providers.ProviderConfigError, match="API_KEY"):
            providers.chat_model(cfg)

    def test_temperature_none_is_omitted(self) -> None:
        cfg = dataclasses.replace(settings, temperature="none")
        assert providers._temperature(cfg) == {}
        assert providers._temperature(dataclasses.replace(settings, temperature="0")) == {"temperature": 0.0}

    def test_explicit_temperature_reaches_azure(self, server) -> None:
        model = providers.azure_chat_model(azure_cfg(server, temperature="0"))
        ChatModelLLM(model).route("cold spell?", "NL-UTR-01")
        assert chat_requests(server)[0]["json"]["temperature"] == 0.0

    def test_get_llm_builds_the_configured_provider(self, server, monkeypatch) -> None:
        monkeypatch.setattr(llm_module, "settings", azure_cfg(server))
        built = llm_module.get_llm()
        assert isinstance(built, ChatModelLLM) and built.name == "azure_openai:gpt-dep"


class TestAzureOpenAI:
    def test_v1_api_runs_the_whole_graph(self, server) -> None:
        agent = Agent(llm=ChatModelLLM(providers.azure_chat_model(azure_cfg(server)), label="azure_openai:gpt-dep"))
        answer, tracer = agent.run("What happens during a cold spell?")

        assert_not_degraded(answer)
        assert "88 percent" in answer.answer
        assert answer.confidence == "high" and answer.flags == []
        assert any("Cold spell protocol" in c for c in answer.citations)

        calls = chat_requests(server)
        assert [c["path"] for c in calls] == ["/openai/v1/chat/completions"] * 3  # router, writer, verifier
        assert all(c["json"]["model"] == "gpt-dep" for c in calls)
        assert "temperature" not in calls[0]["json"]  # model default unless configured
        assert all(c["headers"]["authorization"] == "Bearer azure-test-key" for c in calls)
        assert "api-version" not in calls[0]["path"]
        assert tracer.summary()["total_tokens"] == 3 * 150  # usage parsed from the real response

    def test_classic_api_uses_deployment_urls_and_api_key_header(self, server) -> None:
        cfg = azure_cfg(server, azure_api_version="2024-10-21")
        answer, _ = Agent(llm=ChatModelLLM(providers.azure_chat_model(cfg))).run("What happens during a cold spell?")
        assert_not_degraded(answer)
        assert "88 percent" in answer.answer
        call = chat_requests(server)[0]
        assert call["path"] == "/openai/deployments/gpt-dep/chat/completions?api-version=2024-10-21"
        assert call["headers"]["api-key"] == "azure-test-key"

    @pytest.mark.parametrize("version", [None, "2024-10-21"])
    def test_entra_id_sends_a_bearer_token(self, server, version) -> None:
        cfg = azure_cfg(server, azure_api_key=None, azure_api_version=version)
        model = providers.azure_chat_model(cfg, token_provider=lambda: "entra-token-123")
        ChatModelLLM(model).route("cold spell?", "NL-UTR-01")
        headers = chat_requests(server)[0]["headers"]
        assert headers["authorization"] == "Bearer entra-token-123"
        assert "api-key" not in headers

    def test_prompt_shields_rejection_becomes_a_refusal(self, server) -> None:
        agent = Agent(llm=ChatModelLLM(providers.azure_chat_model(azure_cfg(server))))
        answer, tracer = agent.run("FILTER_ME what is the cold spell protocol?")
        assert {"provider_content_filter", "llm_error:router:content_filter"} <= set(answer.flags)
        assert tracer.summary()["tools_used"] == []
        assert answer.confidence == "low"
        assert "not processed" in answer.answer

    def test_filtered_completion_returns_the_evidence_instead(self, server) -> None:
        agent = Agent(llm=ChatModelLLM(providers.azure_chat_model(azure_cfg(server))))
        answer, _ = agent.run("FILTER_OUT what is the cold spell protocol?")
        assert "llm_error:synthesis:content_filter" in answer.flags
        assert "answer_withheld" in answer.flags and answer.confidence == "low"
        assert "88 percent" in answer.answer  # the retrieved rule itself is still returned

    def test_rate_limits_degrade_instead_of_crashing(self, server) -> None:
        agent = Agent(llm=ChatModelLLM(providers.azure_chat_model(azure_cfg(server))))
        answer, tracer = agent.run("RATE_LIMIT what is the cold spell protocol?")
        assert "llm_error:router:rate_limited" in answer.flags
        assert "search_playbook" in tracer.summary()["tools_used"]  # router fell back to search
        assert answer.confidence == "low"

    def test_errors_are_classified(self, server) -> None:
        model = ChatModelLLM(providers.azure_chat_model(azure_cfg(server)))
        with pytest.raises(LLMError) as excinfo:
            model.route("RATE_LIMIT", "NL-UTR-01")
        assert excinfo.value.kind == "rate_limited"


class TestAzureEmbeddings:
    @pytest.mark.parametrize("version", [None, "2024-10-21"])
    def test_retrieval_uses_azure_embeddings_when_configured(self, server, monkeypatch, version) -> None:
        cfg = azure_cfg(server, azure_embedding_deployment="emb-dep", azure_api_version=version)
        monkeypatch.setattr(retrieval, "settings", cfg)
        retrieval._store.cache_clear()
        try:
            assert retrieval.embedder_name() == "azure_openai:emb-dep"
            hits = retrieval.search("cold spell heating forecast review", k=2)
            assert hits and hits[0].heading == "Cold spell protocol"

            emb_calls = [r for r in server.requests if r["path"].split("?")[0].endswith("/embeddings")]
            expected = "/openai/v1/embeddings" if version is None else "/openai/deployments/emb-dep/embeddings"
            assert emb_calls[0]["path"].split("?")[0] == expected

            before = len(server.requests)
            retrieval.search("cold spell heating forecast review", k=2)
            assert len(server.requests) == before  # cached: no second call for the same query
        finally:
            retrieval._store.cache_clear()

    def test_neural_embeddings_get_a_higher_relevance_floor(self, server) -> None:
        assert azure_cfg(server, azure_embedding_deployment="emb").retrieval_min_score == 0.30
        assert dataclasses.replace(settings, azure_embedding_deployment=None).retrieval_min_score == 0.05


class TestAnthropic:
    def test_claude_runs_the_whole_graph(self, server) -> None:
        cfg = dataclasses.replace(settings, provider="anthropic", api_key="sk-ant-test",
                                  anthropic_base_url=server.url, max_retries=0)
        answer, tracer = Agent(llm=ChatModelLLM(providers.anthropic_chat_model(cfg))).run("Cold spell rules?")
        assert_not_degraded(answer)
        assert "88 percent" in answer.answer
        assert "temperature" not in chat_requests(server)[0]["json"]
        calls = chat_requests(server)
        assert [c["path"] for c in calls] == ["/v1/messages"] * 3
        headers = calls[0]["headers"]
        assert "sk-ant-test" in (headers.get("x-api-key", "") + headers.get("authorization", ""))
        assert headers["anthropic-version"]
        assert calls[0]["json"]["model"] == settings.model
        assert tracer.summary()["total_tokens"] == 3 * 135
