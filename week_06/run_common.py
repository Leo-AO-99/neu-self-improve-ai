"""Shared defaults and helpers for run_* scripts."""
from __future__ import annotations

# Default: run all three paper datasets (CWEBQA, FMT, GPQA)
DEFAULT_DATASETS = ["CWEBQA", "FMT", "GPQA"]

DATASET_CHOICES = ["cwebqa", "CWEBQA", "gpqa", "GPQA", "fmt", "FMT", "scienceqa", "ScienceQA"]


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
