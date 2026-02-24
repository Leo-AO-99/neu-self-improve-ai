"""Shared defaults and helpers for run_* scripts."""
from __future__ import annotations

import json
from pathlib import Path

# Default: run all three paper datasets (CWEBQA, FMT, GPQA)
DEFAULT_DATASETS = ["CWEBQA", "FMT", "GPQA"]

DATASET_CHOICES = ["cwebqa", "CWEBQA", "gpqa", "GPQA", "fmt", "FMT", "scienceqa", "ScienceQA"]

# Dataset name -> folder name under prompts/
DATASET_PROMPT_DIR = {
    "cwebqa": "CWEBQA", "cwqa": "CWEBQA", "complexwebqa": "CWEBQA",
    "gpqa": "GPQA",
    "fmt": "FMT", "foolmetwice": "FMT",
    "scienceqa": "ScienceQA",
}

# Aliases for default dataset names
_DATASET_ALIAS = {
    "cwebqa": "CWEBQA", "cwqa": "CWEBQA", "complexwebqa": "CWEBQA",
    "gpqa": "GPQA",
    "fmt": "FMT", "foolmetwice": "FMT",
}


def normalize_dataset_arg(datasets: list[str] | None) -> list[str]:
    """If None or empty, return DEFAULT_DATASETS; else return list of canonical names."""
    if not datasets:
        return list(DEFAULT_DATASETS)
    out = []
    for d in datasets:
        key = d.lower().strip()
        out.append(_DATASET_ALIAS.get(key, d.upper()))
    return out


def _prompts_dir_for_dataset(prompts_dir: Path, dataset: str) -> Path:
    key = dataset.lower().strip()
    folder = DATASET_PROMPT_DIR.get(key, dataset.upper())
    return prompts_dir / folder


def load_fewshot_cot_config(
    prompts_dir: Path,
    dataset: str,
) -> tuple[str, list[str], str, str]:
    """Load prompt_template, stop_tokens, answer_marker, examples from prompts/<dataset>/fewshot_cot/."""
    d = _prompts_dir_for_dataset(prompts_dir, dataset)
    config_path = d / "fewshot_cot" / "fewshot_cot_config.json"
    examples_path = d / "fewshot_cot" / "fewshot_cot_prompt.txt"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    template = config["prompt_template"]
    stop_tokens = list(config.get("stop_tokens") or ["</s>", "\n\n\n"])
    answer_marker = config.get("answer_marker") or "answer is"
    examples = examples_path.read_text(encoding="utf-8").strip() if examples_path.exists() else ""
    return template, stop_tokens, answer_marker, examples


def load_decompose_config(
    prompts_dir: Path,
    dataset: str,
) -> tuple[str, str, int]:
    """Load decompose_prompt, decompose_prompt_rephrased, and question index from prompts/<dataset>/decompose/."""
    d = _prompts_dir_for_dataset(prompts_dir, dataset)
    template_path = d / "decompose" / "decompose_template.json"
    prompt_path = d / "decompose" / "decompose_prompt.txt"
    prompt_rephrased_path = d / "decompose" / "decompose_prompt_rephrased.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Decompose template not found: {template_path}")
    with open(template_path, encoding="utf-8") as f:
        template = json.load(f)
    question_index = int(template.get("index", 1))
    prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else ""
    prompt_rephrased = prompt_rephrased_path.read_text(encoding="utf-8").strip() if prompt_rephrased_path.exists() else prompt
    return prompt, prompt_rephrased, question_index


def load_fewshot_ost_config(
    prompts_dir: Path,
    dataset: str,
) -> dict:
    """Load fewshot_ost config (prompt_template, stop_tokens, answer_marker) from prompts/<dataset>/fewshot_ost/."""
    d = _prompts_dir_for_dataset(prompts_dir, dataset)
    config_path = d / "fewshot_ost" / "fewshot_ost_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Fewshot OST config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return config


def load_rephrasing_prompt(prompts_dir: Path, dataset: str) -> str:
    """Load rephrasing prompt template from prompts/<dataset>/rephrasing_prompt_template.txt."""
    d = _prompts_dir_for_dataset(prompts_dir, dataset)
    path = d / "rephrasing_prompt_template.txt"
    if not path.exists():
        raise FileNotFoundError(f"Rephrasing prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_fewshot_rag_config(
    prompts_dir: Path,
    dataset: str,
) -> tuple[str, list[str], str]:
    """
    Load RAG answer prompt: (answer_template, stop_tokens, examples).
    Uses prompts/<dataset>/fewshot_rag/. If config has "answer_template" use it
    (placeholders: {context}, {instruction}, {examples}); else use default template.
    """
    d = _prompts_dir_for_dataset(prompts_dir, dataset)
    config_path = d / "fewshot_rag" / "fewshot_rag_config.json"
    examples_path = d / "fewshot_rag" / "fewshot_rag_prompt.txt"
    examples = examples_path.read_text(encoding="utf-8").strip() if examples_path.exists() else ""
    stop_tokens = ["</s>", "\n\n\n"]
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        stop_tokens = list(config.get("stop_tokens") or stop_tokens)
        template = config.get("answer_template")
        if template:
            return template, stop_tokens, examples
    # Default: context + question, optionally prepend examples
    if examples:
        return (
            "Use the following context to answer the question. Think step by step, then give your final answer in the format \"The answer is: <ANSWER>.\"\n\n{examples}\n\nContext:\n{context}\n\nQuestion: {instruction}\n\nAnswer:",
            stop_tokens,
            examples,
        )
    return (
        "Use the following context to answer the question. Think step by step, then give your final answer in the format \"The answer is: <ANSWER>.\"\n\nContext:\n{context}\n\nQuestion: {instruction}\n\nAnswer:",
        stop_tokens,
        examples,
    )


def print_accuracy_summary(results_by_dataset: dict[str, list[dict]]) -> None:
    """Print accuracy for each dataset and a one-line summary."""
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    for name, results in results_by_dataset.items():
        n_correct = sum(1 for r in results if r.get("correct") is True)
        n_total = sum(1 for r in results if r.get("correct") is not None)
        if n_total:
            pct = 100.0 * n_correct / n_total
            print(f"  {name}: {n_correct}/{n_total} = {pct:.2f}%")
    print("=" * 60)
