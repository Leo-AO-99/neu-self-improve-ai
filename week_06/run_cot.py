"""
CoT (Chain-of-Thought) baseline for MCTS-RAG.
Uses fewshot CoT prompts: step-by-step reasoning, final line "The answer is: <ANSWER>."
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from model import get_model
from data_loader import load_data as load_data_impl
from evaluator import (
    extract_answer_from_model_completion,
    check_answers_equiv,
)
from run_common import (
    DEFAULT_DATASETS,
    normalize_dataset_arg,
    print_accuracy_summary,
    load_fewshot_cot_config,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_DB_PATH = SCRIPT_DIR / "data" / "mcts_rag.db"


def _is_llama_model(model_name_or_path: str) -> bool:
    """Heuristic: treat as Llama if path/name contains 'llama' (e.g. meta-llama/Llama-3.2-8B)."""
    return "llama" in (model_name_or_path or "").lower()


def _build_llama_cot_prompt(question: str, examples: str, dataset_key: str) -> str:
    """Build a single-question prompt for Llama to avoid repeating '### Instruction' / '### Response'.
    Uses plain 'Question:' / 'Answer:' and one short example so the model does not copy template."""
    # One compact example (no ### blocks) so model answers the real question only
    if examples.strip():
        # Use first example line/s only; avoid long fewshot that triggers template copying
        first_ex = examples.split("\n\n")[0].strip() if "\n\n" in examples else examples[:400]
        example_block = f"Example:\n{first_ex}\n\n"
    else:
        example_block = ""
    return (
        "Answer the following question step by step. "
        "You must answer only this question. Do not write another question or repeat the question. "
        "End your reply with exactly one line: The answer is: <ANSWER>.\n\n"
        f"{example_block}"
        f"Question: {question}\n\n"
        "Answer:"
    )


def get_question_and_gold(item: dict, dataset: str) -> tuple[str, str | None]:
    """Extract question text and gold answer from item (supports question/problem, answer/solution)."""
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


def run_cot(
    model_name_or_path: str,
    prompts_dir: str | Path,
    dataset: str,
    output_path: str | Path | None = None,
    max_new_tokens: int = 1024,
    backend: str = "vllm",
    limit: int | None = None,
    *,
    db_path: str | Path | None = None,
    table: str | None = None,
    data_path: str | Path | None = None,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = 4096,
    show_prompt: bool = False,
) -> list[dict]:
    dataset_key = dataset.upper()
    prompts_dir = Path(prompts_dir).resolve()
    template, stop_tokens, answer_marker, examples = load_fewshot_cot_config(prompts_dir, dataset_key)
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
    use_llama_cot_prompt = _is_llama_model(model_name_or_path)
    results = []
    for i, item in enumerate(tqdm(data, desc="CoT")):
        question, gold = get_question_and_gold(item, dataset)
        if use_llama_cot_prompt:
            # Llama: use simple Question/Answer prompt so model doesn't copy "### Instruction" and invent new Qs
            # Chat wrap + <|eot_id|> stop are applied in model.generate() for any Llama model
            prompt = _build_llama_cot_prompt(question, examples, dataset_key)
        else:
            prompt = template.format(instruction=question, examples=examples)
        if show_prompt and i == 0:
            print("============= 第 1 条 prompt (送进模型的内容) =============\n")
            print(prompt)
            print("\n============= 以上 =============\n")
        raw_completion = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
        )
        completion = raw_completion or ""
        # If model repeated "### Instruction" / "### Response", use only the first reply for answer extraction
        parts = re.split(r"\n\n#+\s*[Ii]nstruction", completion, maxsplit=1)
        if len(parts) > 1:
            completion = parts[0].strip()
        pred = extract_answer_from_model_completion(completion, dataset_key)
        results.append({
            "idx": i,
            "question": question,
            "gold": gold,
            "pred": pred or "",
            "correct": check_answers_equiv(pred, gold, dataset_key) if gold is not None else None,
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
    p = argparse.ArgumentParser(description="CoT baseline for MCTS-RAG")
    p.add_argument("--model_ckpt", "-m", type=str, required=True, help="Model path (local dir)")
    p.add_argument(
        "--dataset", "-d", type=str, nargs="*", default=None,
        help="Dataset(s): cwebqa, gpqa, fmt. Default: run all three (CWEBQA, FMT, GPQA).",
    )
    p.add_argument("--db_path", type=str, default=str(DEFAULT_DB_PATH), help="SQLite DB path")
    p.add_argument("--data_path", type=str, default=None, help="JSON/JSONL file (overrides db_path when set)")
    p.add_argument("--output_path", "-o", type=str, default=None, help="Output JSON (default: outputs/cot_<dataset>.json)")
    p.add_argument("--limit", "-n", type=int, default=None, help="Max samples to run")
    p.add_argument("--prompts_dir", type=str, default=str(DEFAULT_PROMPTS_DIR), help="Prompts root (default: week_06/prompts)")
    args = p.parse_args()
    datasets = normalize_dataset_arg(args.dataset)
    results_by_dataset = {}
    for dataset_key in datasets:
        out = args.output_path or str(SCRIPT_DIR / "outputs" / f"cot_{dataset_key}.json")
        if len(datasets) > 1 and args.output_path:
            out = str(Path(args.output_path).parent / f"cot_{dataset_key}.json")
        res = run_cot(
            model_name_or_path=args.model_ckpt,
            prompts_dir=args.prompts_dir,
            dataset=dataset_key,
            output_path=out,
            db_path=args.db_path if not args.data_path else None,
            data_path=args.data_path,
            limit=args.limit,
        )
        results_by_dataset[dataset_key] = res
    if len(results_by_dataset) > 1:
        print_accuracy_summary(results_by_dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
