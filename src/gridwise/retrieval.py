"""Retrieval (the R in RAG), built on LangChain primitives.

Pipeline:

    knowledge/*.md
        -> MarkdownHeaderTextSplitter        one chunk per '##' section, headings kept as metadata
        -> RecursiveCharacterTextSplitter    long sections split further, with overlap
        -> LocalTfidfEmbeddings              each chunk embedded as a vector
        -> InMemoryVectorStore               cosine-similarity search
        -> Passage(source, heading, text)    every hit carries a citation "file.md#Heading"

About the embeddings: with `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` set, chunks and
queries are embedded by an Azure OpenAI embedding model (e.g.
text-embedding-3-small), behind a cache so each text is embedded once. Without it,
`LocalTfidfEmbeddings` implements the same LangChain `Embeddings` interface with
TF-IDF vectors (word unigrams + bigrams, a light stemmer): lexical, not neural,
but it needs no API key or download and keeps the tests deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .config import settings

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

_TOKEN = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """A deliberately tiny suffix stripper: signs/signed/sign-off -> sign."""
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def content_tokens(text: str) -> list[str]:
    """Lower-cased, stop-word-free, stemmed tokens. Shared with the verifier."""
    return [
        stem(t) for t in _TOKEN.findall((text or "").lower())
        if t not in ENGLISH_STOP_WORDS and not t.isdigit() and len(t) > 1
    ]


class LocalTfidfEmbeddings(Embeddings):
    """LangChain `Embeddings` backed by a TF-IDF vocabulary fitted on the corpus."""

    def __init__(self, corpus: list[str]):
        self.vectorizer = TfidfVectorizer(
            analyzer=lambda text: self._analyze(text), sublinear_tf=True
        )
        self.vectorizer.fit(corpus)

    @staticmethod
    def _analyze(text: str) -> list[str]:
        tokens = content_tokens(text)
        return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(texts).toarray()
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class CachedEmbeddings(Embeddings):
    """Memoise a remote embedder: documents and repeated queries are embedded once."""

    def __init__(self, inner: Embeddings, max_entries: int = 5000):
        self.inner = inner
        self.max_entries = max_entries
        self._cache: dict[str, list[float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        fresh = dict(zip(missing, self.inner.embed_documents(missing))) if missing else {}
        for text, vector in fresh.items():
            if len(self._cache) < self.max_entries:
                self._cache[text] = vector
        return [self._cache[t] if t in self._cache else fresh[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if text not in self._cache:
            vector = self.inner.embed_query(text)
            if len(self._cache) < self.max_entries:
                self._cache[text] = vector
            return vector
        return self._cache[text]


def make_embeddings(corpus: list[str]) -> Embeddings:
    """Azure OpenAI embeddings when configured, otherwise local TF-IDF vectors."""
    if settings.uses_azure_embeddings:
        from .providers import azure_embeddings  # lazy: offline needs no SDK

        return CachedEmbeddings(azure_embeddings(settings))
    return LocalTfidfEmbeddings(corpus)


def embedder_name() -> str:
    if settings.uses_azure_embeddings:
        return f"azure_openai:{settings.azure_embedding_deployment}"
    return "local_tfidf"


@dataclass(frozen=True)
class Passage:
    source: str
    heading: str
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.heading}"


def load_documents(knowledge_dir: Path) -> list[Document]:
    """Split every markdown file into section chunks with source metadata."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=True
    )
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    documents: list[Document] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        sections = header_splitter.split_text(path.read_text(encoding="utf-8"))
        for section in size_splitter.split_documents(sections):
            heading = section.metadata.get("section") or section.metadata.get("title") or path.stem
            text = " ".join(section.page_content.split())
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "heading": heading, "title": section.metadata.get("title", "")},
                )
            )
    if not documents:
        raise FileNotFoundError(f"No knowledge documents found in {knowledge_dir}")
    return documents


def build_store(documents: list[Document], embeddings: Embeddings | None = None) -> InMemoryVectorStore:
    """Embed the chunks into a vector store. The heading is embedded with the text."""
    for_embedding = [
        Document(page_content=f"{d.metadata['heading']}. {d.page_content}", metadata=d.metadata | {"text": d.page_content})
        for d in documents
    ]
    embeddings = embeddings or make_embeddings([d.page_content for d in for_embedding])
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(for_embedding)
    return store


@lru_cache(maxsize=4)
def _store(knowledge_dir: str) -> tuple[InMemoryVectorStore, tuple[Document, ...]]:
    documents = load_documents(Path(knowledge_dir))
    return build_store(documents), tuple(documents)


def all_documents(knowledge_dir: Path | None = None) -> list[Document]:
    return list(_store(str(knowledge_dir or settings.knowledge_dir))[1])


def search(query: str, k: int = 3, knowledge_dir: Path | None = None) -> list[Passage]:
    store, _ = _store(str(knowledge_dir or settings.knowledge_dir))
    vector = store.embeddings.embed_query(query)
    if not any(vector):  # no known word in the query: nothing can match
        return []
    hits = store.similarity_search_with_score_by_vector(vector, k=k)
    return [
        Passage(
            source=doc.metadata["source"],
            heading=doc.metadata["heading"],
            text=doc.metadata["text"],
            score=round(float(score), 4),
        )
        for doc, score in hits
        if score >= settings.retrieval_min_score
    ]


def embedder(knowledge_dir: Path | None = None) -> Embeddings:
    store, _ = _store(str(knowledge_dir or settings.knowledge_dir))
    return store.embeddings
