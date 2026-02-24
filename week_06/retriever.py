"""
Retrieval for RAG baselines: DuckDuckGo or Wikipedia (choose via backend).
"""
from __future__ import annotations

from typing import Any

RETRIEVER_BACKENDS = ("duckduckgo", "wikipedia")


def _retrieve_duckduckgo(query: str, k: int = 10) -> list[dict[str, Any]]:
    """Retrieve top-k snippets from DuckDuckGo. Returns list of dicts with 'text' (and 'title')."""
    try:
        from ddgs import DDGS
    except ImportError:
        raise ImportError("Install ddgs: pip install ddgs (replaces duckduckgo-search)")
    docs = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=k):
            title = r.get("title") or ""
            body = r.get("body") or ""
            text = f"{title}: {body}".strip() if title else body
            if text:
                docs.append({"text": text, "title": title})
    return docs


def _retrieve_wikipedia(query: str, k: int = 10, sentences_per_page: int = 5) -> list[dict[str, Any]]:
    """Search Wikipedia and return top-k page summaries. Uses MediaWiki API via wikipedia package."""
    try:
        import wikipedia
    except ImportError:
        raise ImportError("Install wikipedia: pip install wikipedia")
    docs = []
    try:
        titles = wikipedia.search(query, results=k)
    except Exception:
        return docs
    for title in titles:
        if len(docs) >= k:
            break
        try:
            summary = wikipedia.summary(title, sentences=sentences_per_page, auto_suggest=False)
            if summary and summary.strip():
                docs.append({"text": summary.strip(), "title": title})
        except wikipedia.exceptions.DisambiguationError as e:
            # use first disambiguation option
            if e.options:
                try:
                    summary = wikipedia.summary(e.options[0], sentences=sentences_per_page, auto_suggest=False)
                    if summary and summary.strip():
                        docs.append({"text": summary.strip(), "title": e.options[0]})
                except Exception:
                    pass
        except Exception:
            pass
    return docs


class Retriever:
    """Retriever with backend: 'duckduckgo' or 'wikipedia'."""

    def __init__(self, backend: str = "wikipedia", top_k: int = 10, **kwargs: Any):
        backend = backend.lower().strip()
        if backend not in RETRIEVER_BACKENDS:
            raise ValueError(f"backend must be one of {RETRIEVER_BACKENDS}, got {backend!r}")
        self.backend = backend
        self.top_k = top_k
        self._kwargs = kwargs

    def retrieve(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        """Return list of dicts with 'text' (and 'title') for the query."""
        k = k if k is not None else self.top_k
        if self.backend == "duckduckgo":
            return _retrieve_duckduckgo(query, k=k)
        if self.backend == "wikipedia":
            return _retrieve_wikipedia(query, k=k, **self._kwargs)
        raise ValueError(f"Unknown backend: {self.backend}")


def format_context(docs: list[dict[str, Any]], max_chars: int = 8000) -> str:
    """Concatenate doc texts up to max_chars. Docs have 'text' or 'content'."""
    parts = []
    total = 0
    for d in docs:
        t = d.get("text") or d.get("content") or str(d)
        if isinstance(t, dict):
            t = t.get("text", str(t))
        if total + len(t) + 1 > max_chars:
            remain = max_chars - total
            if remain > 0:
                parts.append(t[:remain])
            break
        parts.append(t)
        total += len(t) + 1
    return "\n\n".join(parts)
