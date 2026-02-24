"""
MCTS-RAG: Run MCTS with RAG (A1–A5: direct answer, OST step, subquestions, RAG, rephrased question).
Based on MCTS-RAG (https://github.com/yale-nlp/MCTS-RAG).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from model import get_model
from retriever import Retriever
from data_loader import load_data as load_data_impl
from evaluator import check_answers_equiv
from run_common import (
    DEFAULT_DATASETS,
    normalize_dataset_arg,
    print_accuracy_summary,
)
from mcts_rag import Generator, run_mcts_search, select_final_answer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = SCRIPT_DIR / "data" / "mcts_rag.db"
DEFAULT_PROMPTS_DIR = SCRIPT_DIR / "prompts"


def get_question_and_gold(item: dict, dataset: str) -> tuple[str, str | None]:
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


def run_mcts_rag(
    model_name_or_path: str,
    dataset: str,
    output_path: str | Path | None = None,
    num_rollouts: int = 4,
    num_votes: int = 8,
    top_k: int = 10,
    max_new_tokens: int = 512,
    max_depth: int = 5,
    exploration_weight: float = 2.0,
    backend: str = "vllm",
    limit: int | None = None,
    *,
    db_path: str | Path | None = None,
    table: str | None = None,
    data_path: str | Path | None = None,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = 4096,
    retriever_backend: str = "wikipedia",
    disable_a1: bool = False,
    disable_a4: bool = False,
    disable_a5: bool = False,
    disable_rag: bool = False,
    prompts_dir: str | Path | None = None,
) -> list[dict]:
    dataset_key = dataset.upper()
    prompts_dir = Path(prompts_dir or DEFAULT_PROMPTS_DIR).resolve()
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
    retriever = Retriever(backend=retriever_backend, top_k=top_k)
    generator = Generator(
        model=model,
        retriever=retriever,
        dataset=dataset_key,
        prompts_dir=prompts_dir,
        num_votes=num_votes,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        disable_a1=disable_a1,
        disable_a4=disable_a4,
        disable_a5=disable_a5,
        disable_rag=disable_rag,
    )
    results = []
    for i, item in enumerate(tqdm(data, desc="MCTS-RAG")):
        question, gold = get_question_and_gold(item, dataset_key)
        trajectory_rewards, _searcher, _root = run_mcts_search(
            question=question,
            generator=generator,
            num_rollouts=num_rollouts,
            exploration_weight=exploration_weight,
            max_depth=max_depth,
            disable_a1=disable_a1,
            disable_a5=disable_a5,
            disable_rag=disable_rag,
        )
        pred = select_final_answer(trajectory_rewards, dataset_key, check_equiv=None)
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
    p = argparse.ArgumentParser(description="MCTS-RAG: MCTS + RAG (A1–A5)")
    p.add_argument("--model_ckpt", "-m", type=str, required=True, help="Model path (local dir)")
    p.add_argument(
        "--dataset", "-d", type=str, nargs="*", default=None,
        help="Dataset(s): cwebqa, gpqa, fmt. Default: run all three.",
    )
    p.add_argument("--db_path", type=str, default=str(DEFAULT_DB_PATH), help="SQLite DB path")
    p.add_argument("--data_path", type=str, default=None, help="JSON/JSONL file (overrides db_path)")
    p.add_argument("--output_path", "-o", type=str, default=None, help="Output JSON")
    p.add_argument(
        "--retriever", "-r", type=str, default="wikipedia", choices=["duckduckgo", "wikipedia"],
        help="Retriever backend",
    )
    p.add_argument("--num_rollouts", type=int, default=4, help="MCTS rollouts per question")
    p.add_argument("--num_votes", type=int, default=8, help="Samples per action (for clustering)")
    p.add_argument("--top_k", "-k", type=int, default=10, help="Retrieval top-k")
    p.add_argument("--max_depth", type=int, default=5, help="Max MCTS depth")
    p.add_argument("--exploration_weight", type=float, default=2.0, help="UCT exploration weight")
    p.add_argument("--limit", "-n", type=int, default=None, help="Max samples")
    p.add_argument("--disable_a1", action="store_true", help="Disable A1 (direct answer)")
    p.add_argument("--disable_a4", action="store_true", help="Disable A4 (retrieval reasoning)")
    p.add_argument("--disable_a5", action="store_true", help="Disable A5 (rephrased question)")
    p.add_argument("--disable_rag", action="store_true", help="Disable RAG (A4 retrieve / rag step)")
    p.add_argument("--prompts_dir", type=str, default=str(DEFAULT_PROMPTS_DIR), help="Prompts root (default: week_06/prompts)")
    args = p.parse_args()
    datasets = normalize_dataset_arg(args.dataset)
    results_by_dataset = {}
    for dataset_key in datasets:
        out = args.output_path or str(SCRIPT_DIR / "outputs" / f"mcts_rag_{dataset_key}.json")
        if len(datasets) > 1 and args.output_path:
            out = str(Path(args.output_path).parent / f"mcts_rag_{dataset_key}.json")
        res = run_mcts_rag(
            model_name_or_path=args.model_ckpt,
            dataset=dataset_key,
            output_path=out,
            num_rollouts=args.num_rollouts,
            num_votes=args.num_votes,
            top_k=args.top_k,
            max_depth=args.max_depth,
            exploration_weight=args.exploration_weight,
            db_path=args.db_path if not args.data_path else None,
            data_path=args.data_path,
            retriever_backend=args.retriever,
            limit=args.limit,
            disable_a1=args.disable_a1,
            disable_a4=args.disable_a4,
            disable_a5=args.disable_a5,
            disable_rag=args.disable_rag,
            prompts_dir=args.prompts_dir,
        )
        results_by_dataset[dataset_key] = res
    if len(results_by_dataset) > 1:
        print_accuracy_summary(results_by_dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
