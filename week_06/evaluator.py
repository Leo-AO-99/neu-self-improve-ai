from __future__ import annotations

import re
import unicodedata
from typing import Callable

try:
    from fuzzywuzzy import fuzz
    HAS_FUZZ = True
except ImportError:
    HAS_FUZZ = False


ANSWER_MARKER = "answer is"


def isolate_answer(text: str, answer_marker: str = ANSWER_MARKER) -> str | None:
    """Take last segment after 'answer is' (official Evaluator.isolate_answer)."""
    if text is None:
        return None
    if not isinstance(text, str):
        return None
    text = text.lower()
    split_ans = text.split(answer_marker.lower())
    if len(split_ans) > 1:
        ans = split_ans[-1].replace(":", "").strip()
        extract_ans_temp = ans.split(".\n")[0].strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
            extract_ans = extract_ans_temp[:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip().strip("\n")
        return extract_ans
    return None


def _format_gpqa(answer: str) -> str:
    return answer.lower()


def extract_answer_gpqa(completion: str) -> str | None:
    """GPQAEvaluator: isolate_answer then take part before ')', first segment by \\n\\n or '. '."""
    if completion is None:
        return None
    answer = isolate_answer(completion)
    if answer is None:
        return None
    if ")" in answer:
        answer = answer.split(")")[0]
    if "\n\n" in answer:
        answer = answer.split("\n\n")[0]
    if ". " in answer:
        answer = answer.split(". ")[0]
    return _format_gpqa(answer)


def check_equiv_gpqa(answer_a: str, answer_b: str) -> bool:
    """GPQAEvaluator: format both, then == or fuzz >= 90."""
    if answer_a is None or answer_b is None:
        return False
    fa, fb = _format_gpqa(answer_a), _format_gpqa(answer_b)
    if fa == fb:
        return True
    if HAS_FUZZ and fuzz.token_sort_ratio(fa, fb) >= 90:
        return True
    return False


def _format_fmt(answer: str) -> str:
    a = answer.lower()
    if a in ("a", "supports", "support"):
        return "yes"
    if a in ("b", "refute", "refutes"):
        return "no"
    return a


def extract_answer_fmt(completion: str) -> str | None:
    """FMTEvaluator: same extraction as GPQA, then _format_fmt in check."""
    if completion is None:
        return None
    answer = isolate_answer(completion)
    if answer is None:
        return None
    if ")" in answer:
        answer = answer.split(")")[0]
    if "\n\n" in answer:
        answer = answer.split("\n\n")[0]
    if ". " in answer:
        answer = answer.split(". ")[0]
    return answer.lower()


def check_equiv_fmt(answer_a: str, answer_b: str) -> bool:
    """FMTEvaluator: format both (supports->yes, refutes->no), == or fuzz >= 90."""
    if answer_a is None or answer_b is None:
        return False
    fa, fb = _format_fmt(answer_a), _format_fmt(answer_b)
    if fa == fb:
        return True
    if HAS_FUZZ and fuzz.token_sort_ratio(fa, fb) >= 90:
        return True
    return False


def _format_cwebqa(answer: str) -> str:
    proc = unicodedata.normalize("NFKD", answer).encode("ascii", "ignore").decode("utf-8")
    proc = re.sub(r"\W", " ", proc).lower().strip()
    for prefix in ("the ", "a ", "an "):
        if proc.startswith(prefix):
            proc = proc[len(prefix) :]
    return proc


def extract_answer_cwebqa(completion: str) -> str | None:
    """CWEBQAEvaluator: isolate_answer; gold is raw solution."""
    if completion is None:
        return None
    return isolate_answer(completion)


def check_equiv_cwebqa(answer_a: str, answer_b: str) -> bool:
    """CWEBQAEvaluator: _format both, == or fuzz >= 45 or one in the other."""
    if answer_a is None or answer_b is None or answer_a == "" or answer_b == "":
        return False
    if answer_a == "Unknown" or answer_b == "Unknown":
        return False
    fa, fb = _format_cwebqa(answer_a), _format_cwebqa(answer_b)
    if fa == fb:
        return True
    if HAS_FUZZ and fuzz.token_sort_ratio(fa, fb) >= 45:
        return True
    if fa in fb or fb in fa:
        return True
    return False


def _format_scienceqa(answer: str) -> str:
    return answer.lower()


def extract_answer_scienceqa(completion: str) -> str | None:
    """ScienceQAEvaluator: isolate_answer then lower."""
    if completion is None:
        return None
    answer = isolate_answer(completion)
    if answer is None:
        return None
    return _format_scienceqa(answer)


def check_equiv_scienceqa(answer_a: str, answer_b: str) -> bool:
    """ScienceQAEvaluator: format both, == or fuzz >= 90."""
    if answer_a is None or answer_b is None:
        return False
    fa, fb = _format_scienceqa(answer_a), _format_scienceqa(answer_b)
    if fa == fb:
        return True
    if HAS_FUZZ and fuzz.token_sort_ratio(fa, fb) >= 90:
        return True
    return False


# Dataset -> (extract_fn, check_equiv_fn); gold is already stored as solution string
EXTRACTORS: dict[str, Callable[[str], str | None]] = {
    "GPQA": extract_answer_gpqa,
    "FMT": extract_answer_fmt,
    "CWEBQA": extract_answer_cwebqa,
    "ScienceQA": extract_answer_scienceqa,
}

CHECK_EQUIV: dict[str, Callable[[str, str], bool]] = {
    "GPQA": check_equiv_gpqa,
    "FMT": check_equiv_fmt,
    "CWEBQA": check_equiv_cwebqa,
    "ScienceQA": check_equiv_scienceqa,
}


def extract_answer_from_model_completion(completion: str, dataset: str) -> str | None:
    """Extract model answer using dataset-specific logic (official Evaluator)."""
    fn = EXTRACTORS.get(dataset.upper())
    if fn is None:
        return isolate_answer(completion) if completion else None
    return fn(completion)


def format_gold_solution(solution: str, dataset: str) -> str:
    """Format gold solution for comparison (official extract_answer_from_gold_solution)."""
    if solution is None or (isinstance(solution, str) and not solution.strip()):
        return ""
    s = str(solution).strip()
    if dataset.upper() == "FMT":
        return _format_fmt(s)
    if dataset.upper() == "GPQA":
        return _format_gpqa(s)
    if dataset.upper() == "ScienceQA":
        return _format_scienceqa(s)
    if dataset.upper() == "CWEBQA":
        return s  # gold returned as-is, comparison uses _format in check
    return s.lower()


def check_answers_equiv(pred: str | None, gold: str | None, dataset: str) -> bool:
    """Official check_answers_equiv per dataset (formats both inside each fn)."""
    if pred is None or gold is None:
        return False
    fn = CHECK_EQUIV.get(dataset.upper())
    if fn is None:
        return pred.strip().lower() == str(gold).strip().lower()
    return fn(pred, str(gold))
