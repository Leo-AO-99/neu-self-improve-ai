"""
FAISS + sentence-transformers retriever for RAG baselines.
Corpus: JSONL with "text" or "content" per line; index saved/loaded from index_path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


def _load_corpus_texts(corpus_path: Path) -> list[str]:
    texts = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text") or obj.get("content") or obj.get("passage") or ""
            if isinstance(t, list):
                t = " ".join(str(x) for x in t)
            texts.append(t)
    return texts


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


class Retriever:
    """Dense retriever over a JSONL corpus using sentence-transformers + FAISS."""

    def __init__(
        self,
        corpus_path: str,
        index_path: str | Path | None = None,
        top_k: int = 10,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        if not HAS_FAISS or not HAS_ST:
            raise ImportError("Need faiss-cpu (or faiss-gpu) and sentence-transformers for retriever.")
        self.corpus_path = Path(corpus_path)
        self.index_path = Path(index_path) if index_path else None
        self.top_k = top_k
        self.model_name = model_name
        self._encoder = None
        self._index = None
        self._corpus_texts: list[str] = []

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def load_index(self, corpus_path: Path | None = None) -> None:
        corpus_path = corpus_path or self.corpus_path
        corpus_path = Path(corpus_path)
        if self.index_path and self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            # Load texts from same dir as index, name corpus.jsonl or from corpus_path
            texts_path = self.index_path.parent / "corpus_texts.json"
            if texts_path.exists():
                with open(texts_path, encoding="utf-8") as f:
                    self._corpus_texts = json.load(f)
            else:
                self._corpus_texts = _load_corpus_texts(corpus_path)
        else:
            self._corpus_texts = _load_corpus_texts(corpus_path)
            encoder = self._get_encoder()
            embs = encoder.encode(self._corpus_texts, show_progress_bar=True)
            embs = np.array(embs, dtype=np.float32)
            faiss.normalize_L2(embs)
            d = embs.shape[1]
            self._index = faiss.IndexFlatIP(d)
            self._index.add(embs)
            if self.index_path:
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self._index, str(self.index_path))
                with open(self.index_path.parent / "corpus_texts.json", "w", encoding="utf-8") as f:
                    json.dump(self._corpus_texts, f, ensure_ascii=False)
        return None

    def retrieve(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        if self._index is None:
            self.load_index()
        k = k or self.top_k
        encoder = self._get_encoder()
        q_emb = encoder.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)
        scores, indices = self._index.search(q_emb, min(k, len(self._corpus_texts)))
        out = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._corpus_texts):
                continue
            out.append({
                "text": self._corpus_texts[idx],
                "score": float(scores[0][i]) if scores.size else 0.0,
                "idx": int(idx),
            })
        return out
