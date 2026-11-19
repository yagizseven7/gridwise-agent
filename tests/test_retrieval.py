from __future__ import annotations

from gridwise import retrieval, summarize


def test_every_section_becomes_a_chunk_with_source_metadata() -> None:
    docs = retrieval.all_documents()
    assert len(docs) >= 12
    assert all(d.metadata["source"].endswith(".md") and d.metadata["heading"] for d in docs)


def test_embeddings_follow_the_langchain_interface() -> None:
    vector = retrieval.embedder().embed_query("peak shaving trigger")
    assert isinstance(vector, list) and abs(sum(v * v for v in vector) - 1.0) < 1e-6


def test_stemming_matches_inflected_words() -> None:
    hits = retrieval.search("who signs it off", k=1)
    assert hits and hits[0].heading == "Escalation and human sign-off"


def test_queries_with_no_known_words_return_nothing() -> None:
    assert retrieval.search("capital of France", k=3) == []


def test_summaries_are_verbatim_and_cited() -> None:
    sentences = summarize.summarize(source="tariff", max_sentences=4)
    docs = {d.metadata["heading"]: d.page_content for d in retrieval.all_documents()}
    assert len(sentences) == 4
    for s in sentences:
        assert s.citation.startswith("tariffs_and_imbalance.md#")
        assert s.text in docs[s.citation.split("#", 1)[1]]
