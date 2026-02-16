"""
Standard RAG baseline for MCTS-RAG.
Retrieve top-k documents once per question, then generate answer with context.
Prompt: Context + Question -> "The answer is: <ANSWER>."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from model import get_model
from retriever import Retriever, format_context
from data_loader import load_data as load_data_impl
from evaluator import (
    extract_answer_from_model_completion,
    check_answers_equiv,
)
from run_common import DEFAULT_DATASETS, normalize_dataset_arg, print_accuracy_summary

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = SCRIPT_DIR / "data" / "mcts_rag.db"

# Standard RAG prompt: context + question -> answer (no fewshot needed for simplicity; paper uses same format)
RAG_PROMPT_TEMPLATE = """Use the following context to answer the question. Think step by step, then give your final answer in the format "The answer is: <ANSWER>."

Context:
{context}

Question: {question}

Answer:"""


def get_question_and_gold(item: dict, dataset: str) -> tuple[str, str | None]:
    """Extract question and gold answer (supports question/problem, answer/solution)."""
    q = (
        item.get("question")
        or item.get("problem")
        or item.get("query")
        or item.get("claim")
        or item.get("input")
        or ""
    )
    if isinstance(q, dict):
        q = q.get("text", str(q))
    gold = item.get("answer") or item.get("solution") or item.get("answers")
    if isinstance(gold, list) and gold:
        gold = gold[0]
    if gold is None:
        gold = item.get("label") or item.get("target")
    return (q if isinstance(q, str) else str(q)), (gold if gold is None else str(gold))


def run_standard_rag(
    model_name_or_path: str,
    dataset: str,
    output_path: str | Path | None = None,
    top_k: int = 10,
    max_new_tokens: int = 1024,
    max_context_chars: int = 8000,
    backend: str = "vllm",
    limit: int | None = None,
    *,
    db_path: str | Path | None = None,
    table: str | None = None,
    data_path: str | Path | None = None,
    corpus_path: str | Path | None = None,
    index_path: str | Path | None = None,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = 4096,
) -> list[dict]:
    """
    If corpus_path is None, no retrieval is performed (context is empty) — useful for debugging.
    When using DB tables with evidence (e.g. fmt_test_with_evidence), evidence can be used as context if no corpus_path.
    """
    dataset_key = dataset.upper()
    data = load_data_impl(db_path=db_path, table=table, data_path=data_path, dataset=dataset_key)
    if limit:
        data = data[:limit]
    model = get_model(
        model_name_or_path,
        backend=backend,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )
    model.load()
    retriever = None
    if corpus_path:
        retriever = Retriever(
            corpus_path=str(corpus_path),
            index_path=str(index_path) if index_path else None,
            top_k=top_k,
        )
        retriever.load_index(Path(corpus_path))
    results = []
    for i, item in enumerate(tqdm(data, desc="Standard RAG")):
        question, gold = get_question_and_gold(item, dataset_key)
        if retriever:
            docs = retriever.retrieve(question, k=top_k)
            context = format_context(docs, max_chars=max_context_chars)
        elif item.get("evidence"):
            context = item["evidence"]
        else:
            context = "(No retrieval: corpus not provided.)"
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        raw_completion = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_tokens=["</s>", "\n\n\n"],
        )
        pred = extract_answer_from_model_completion(raw_completion or "", dataset_key)
        correct = None
        if gold is not None:
            correct = check_answers_equiv(pred, gold, dataset_key)
        results.append({
            "idx": i,
            "question": question,
            "gold": gold,
            "pred": pred or "",
            "correct": correct,
        })
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(results)} results to {output_path}")
    n_correct = sum(1 for r in results if r.get("correct") is True)
    n_total = sum(1 for r in results if r.get("correct") is not None)
    if n_total:
        print(f"Accuracy ({dataset_key}): {n_correct}/{n_total} = {100.0 * n_correct / n_total:.2f}%")
    return results


def main():
    p = argparse.ArgumentParser(description="Standard RAG baseline for MCTS-RAG")
    p.add_argument("--model_ckpt", "-m", type=str, required=True, help="Model path (local dir)")
    p.add_argument(
        "--dataset", "-d", type=str, nargs="*", default=None,
        help="Dataset(s): cwebqa, gpqa, fmt. Default: run all three (CWEBQA, FMT, GPQA).",
    )
    p.add_argument("--db_path", type=str, default=str(DEFAULT_DB_PATH), help="SQLite DB path")
    p.add_argument("--output_path", "-o", type=str, default=None, help="Output JSON (default: outputs/rag_<dataset>.json)")
    p.add_argument("--corpus_path", type=str, default=None, help="Corpus JSONL for FAISS retrieval (optional)")
    p.add_argument("--top_k", "-k", type=int, default=10, help="Number of passages to retrieve (default: 10)")
    p.add_argument("--limit", "-n", type=int, default=None, help="Max samples to run")
    args = p.parse_args()
    datasets = normalize_dataset_arg(args.dataset)
    results_by_dataset = {}
    for dataset_key in datasets:
        out = args.output_path or str(SCRIPT_DIR / "outputs" / f"rag_{dataset_key}.json")
        if len(datasets) > 1 and args.output_path:
            out = str(Path(args.output_path).parent / f"rag_{dataset_key}.json")
        res = run_standard_rag(
            model_name_or_path=args.model_ckpt,
            dataset=dataset_key,
            output_path=out,
            db_path=args.db_path if not args.data_path else None,
            data_path=args.data_path,
            corpus_path=args.corpus_path,
            top_k=args.top_k,
            limit=args.limit,
        )
        results_by_dataset[dataset_key] = res
    if len(results_by_dataset) > 1:
        print_accuracy_summary(results_by_dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
