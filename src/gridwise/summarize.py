"""Extractive summarisation.

The summary is made of sentences copied verbatim from the documents, each one
tagged with the section it came from. That is a deliberate trade-off: an
extractive summary is less fluent than one a model writes, but every sentence in
it is grounded by construction, so it can never introduce a figure the client's
documents do not contain.

Scoring: a sentence ranks high when it is close to the centre of the selected
text (it is representative) and, if a topic was given, close to the topic (it is
relevant). Both use the same embeddings as retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from . import retrieval

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

DOCUMENT_ALIASES = {
    "tariff": "tariffs_and_imbalance.md",
    "handbook": "tariffs_and_imbalance.md",
    "imbalance": "tariffs_and_imbalance.md",
    "playbook": "site_operations_playbook.md",
    "operations": "site_operations_playbook.md",
    "model card": "model_card_demand_forecaster.md",
    "forecaster": "model_card_demand_forecaster.md",
}


@dataclass(frozen=True)
class SummarySentence:
    text: str
    citation: str
    order: int


def resolve_source(name: str | None) -> str | None:
    if not name:
        return None
    lowered = name.lower()
    names = {d.metadata["source"] for d in retrieval.all_documents()}
    for source in names:
        if lowered in source.lower() or source.lower().startswith(lowered):
            return source
    for alias, source in DOCUMENT_ALIASES.items():
        if alias in lowered:
            return source
    return None


def summarize(topic: str = "", source: str | None = None, max_sentences: int = 4) -> list[SummarySentence]:
    """Pick the most representative sentences from one document or from the top hits."""
    docs = retrieval.all_documents()
    resolved = resolve_source(source)
    if resolved:
        chunks = [(d.page_content, f"{d.metadata['source']}#{d.metadata['heading']}") for d in docs
                  if d.metadata["source"] == resolved]
    else:
        chunks = [(p.text, p.citation) for p in retrieval.search(topic, k=3)]
    if not chunks:
        return []

    sentences: list[SummarySentence] = []
    for text, citation in chunks:
        for piece in _SENTENCE.split(text):
            piece = piece.strip()
            if len(piece) > 20:
                sentences.append(SummarySentence(piece, citation, len(sentences)))

    embed = retrieval.embedder()
    vectors = np.array(embed.embed_documents([s.text for s in sentences]))
    centroid = vectors.mean(axis=0)
    scores = vectors @ centroid
    if topic:
        scores = 0.5 * scores + 0.5 * (vectors @ np.array(embed.embed_query(topic)))

    # One sentence per section first, so a document summary covers its sections.
    ranked = list(np.argsort(scores)[::-1])
    chosen: list[int] = []
    seen_sections: set[str] = set()
    for idx in ranked:
        if sentences[idx].citation not in seen_sections:
            chosen.append(int(idx))
            seen_sections.add(sentences[idx].citation)
        if len(chosen) >= max_sentences:
            break
    for idx in ranked:
        if len(chosen) >= max_sentences:
            break
        if int(idx) not in chosen:
            chosen.append(int(idx))

    return [sentences[i] for i in sorted(chosen)]
