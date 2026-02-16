"""
CoT (Chain-of-Thought) baseline for MCTS-RAG.
Uses fewshot CoT prompts: step-by-step reasoning, final line "The answer is: <ANSWER>."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from model import get_model
from data_loader import load_data as load_data_impl
from evaluator import (
    extract_answer_from_model_completion,
    check_answers_equiv,
)
from run_common import DEFAULT_DATASETS, normalize_dataset_arg, print_accuracy_summary

# 默认路径：与当前脚本同目录
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_DB_PATH = SCRIPT_DIR / "data" / "mcts_rag.db"

# Dataset name -> folder name in prompts/
DATASET_PROMPT_DIR = {
    "cwebqa": "CWEBQA",
    "cwqa": "CWEBQA",
    "complexwebqa": "CWEBQA",
    "gpqa": "GPQA",
    "fmt": "FMT",
    "foolmetwice": "FMT",
    "scienceqa": "ScienceQA",
}


def load_prompt_config(prompts_dir: Path, dataset: str) -> tuple[str, list[str], str, str]:
    """Load prompt_template, stop_tokens, answer_marker, examples from fewshot_cot."""
    d = prompts_dir / DATASET_PROMPT_DIR.get(dataset.lower(), dataset.upper())
    config_path = d / "fewshot_cot" / "fewshot_cot_config.json"
    examples_path = d / "fewshot_cot" / "fewshot_cot_prompt.txt"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    template = config["prompt_template"]
    stop_tokens = config.get("stop_tokens") or ["</s>", "\n\n\n"]
    answer_marker = config.get("answer_marker") or "answer is"
    examples = ""
    if examples_path.exists():
        examples = examples_path.read_text().strip()
    return template, stop_tokens, answer_marker, examples


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
    template, stop_tokens, answer_marker, examples = load_prompt_config(prompts_dir, dataset_key)
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
    results = []
    for i, item in enumerate(tqdm(data, desc="CoT")):
        question, gold = get_question_and_gold(item, dataset)
        # 使用 prompts/<dataset>/fewshot_cot 的 template + examples 拼出完整 prompt
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
        pred = extract_answer_from_model_completion(raw_completion or "", dataset_key)
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
    p.add_argument("--output_path", "-o", type=str, default=None, help="Output JSON (default: outputs/cot_<dataset>.json)")
    p.add_argument("--limit", "-n", type=int, default=None, help="Max samples to run")
    args = p.parse_args()
    datasets = normalize_dataset_arg(args.dataset)
    results_by_dataset = {}
    for dataset_key in datasets:
        out = args.output_path or str(SCRIPT_DIR / "outputs" / f"cot_{dataset_key}.json")
        if len(datasets) > 1 and args.output_path:
            out = str(Path(args.output_path).parent / f"cot_{dataset_key}.json")
        res = run_cot(
            model_name_or_path=args.model_ckpt,
            prompts_dir=DEFAULT_PROMPTS_DIR,
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
