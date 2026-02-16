"""
Load evaluation data from SQLite (mcts_rag.db) or JSON/JSONL.
"""
from __future__ import annotations

import json
from pathlib import Path

# Dataset -> SQLite table name in mcts_rag.db
DATASET_TABLE = {
    "cwebqa": "cwebqa",
    "cwqa": "cwebqa",
    "complexwebqa": "cwebqa",
    "gpqa": "gpqa_test",
    "fmt": "fmt_test_with_evidence",
    "foolmetwice": "fmt_test_with_evidence",
    "scienceqa": "scienceqa_test",
    "science": "science_test",
    "gdpa": "gdpa_test_with_evidence",
}

# Tables that have an "evidence" column (for RAG context from DB)
TABLES_WITH_EVIDENCE = {"fmt_test_with_evidence", "gpqa_test_with_evidence", "gdpa_test_with_evidence"}


def load_from_sqlite(db_path: str | Path, table: str) -> list[dict]:
    """Load from mcts_rag.db. Columns: id, problem, solution [, evidence]."""
    import sqlite3
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(f"SELECT * FROM {table} ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        # Normalize to question / answer for run_cot and run_standard_rag
        d["question"] = d.pop("problem", "")
        d["answer"] = d.pop("solution", "")
        if "evidence" not in d:
            d["evidence"] = None
        out.append(d)
    return out


def _normalize_item(item: dict) -> dict:
    """Ensure question/answer keys for downstream (CoT/RAG); keep problem/solution if present."""
    out = dict(item)
    if "question" not in out and "problem" in out:
        out["question"] = out["problem"]
    if "answer" not in out and "solution" in out:
        out["answer"] = out["solution"]
    if "evidence" not in out:
        out["evidence"] = None
    return out


def load_from_file(data_path: str | Path) -> list[dict]:
    """Load from JSON or JSONL file. Normalizes problem->question, solution->answer for baseline scripts."""
    data_path = Path(data_path)
    data = []
    with open(data_path, encoding="utf-8") as f:
        if data_path.suffix.lower() == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        else:
            raw = json.load(f)
            if isinstance(raw, list):
                data = raw
            else:
                data = raw.get("data", raw.get("questions", raw.get("examples", [raw])))
    return [_normalize_item(d) for d in data]


def load_data(
    db_path: str | Path | None = None,
    table: str | None = None,
    data_path: str | Path | None = None,
    dataset: str | None = None,
) -> list[dict]:
    """
    Load evaluation data.
    - If db_path is set: load from SQLite; table = table or inferred from dataset.
    - Else: load from data_path (JSON/JSONL).
    """
    if db_path:
        t = table or (DATASET_TABLE.get(dataset.lower()) if dataset else None)
        if not t:
            raise ValueError("When using db_path, set --table or --dataset to choose table.")
        return load_from_sqlite(db_path, t)
    if data_path:
        return load_from_file(data_path)
    raise ValueError("Provide either db_path or data_path.")
