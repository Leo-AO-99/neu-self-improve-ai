"""
MCTS-RAG helpers: node types, solution_trace formatting, terminal checks.
Simplified from official mcts_utils.py.
"""
from __future__ import annotations

import re
from enum import Enum, unique
from typing import Dict, Tuple

# Question index in prompts (e.g. "Question 1", "Question 1.1")
QUESTION_INDEX = 1


@unique
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"
    REPHRASED_USER_QUESTION = "REPHRASED_USER_QUESTION"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    SUBQUESTION = "SUBQUESTION"
    RE_SUBANSWER = "RE_SUBANSWER"
    OST_STEP = "OST_STEP"


def reach_terminal_subquestion(subquestion: str, user_question: str) -> bool:
    if not subquestion:
        return False
    if "Now we can answer" in subquestion:
        return True
    # If subquestion restates the original question, we're at the end
    uq_lower = (user_question or "").lower()
    sq_lower = subquestion.lower()
    if uq_lower and uq_lower in sq_lower:
        return True
    return False


def reach_terminal_ost_step(ost_step: str) -> bool:
    return ost_step is not None and "answer is" in ost_step.lower()


def concat_subqs_and_subas(
    solution_trace: Dict[int, Dict],
    question_index: int,
) -> Tuple[str, int]:
    """Return (concatenated "Question i.j: ... Answer i.j: ..." string, next_subquestion_id)."""
    parts = []
    for sid in sorted(solution_trace.keys()):
        if sid == 0:
            continue
        step = solution_trace.get(sid, {})
        if "subquestion" not in step or "subanswer" not in step:
            continue
        subq = step["subquestion"]
        suba = step["subanswer"].get("text", "") if isinstance(step["subanswer"], dict) else str(step["subanswer"])
        parts.append(f"Question {question_index}.{sid}: {subq}")
        parts.append(f"Answer {question_index}.{sid}: {suba}")
    next_id = max((k for k in solution_trace.keys() if k > 0), default=0) + 1
    return "\n".join(parts) + "\n" if parts else "", next_id


def concat_ost_steps(solution_trace: Dict[int, Dict]) -> Tuple[str, int]:
    """Return (concatenated Step 1... Step k, next_ost_step_id). Uses last block (0 or last subq id)."""
    last_key = max(solution_trace.keys()) if solution_trace else 0
    rec = solution_trace.get(last_key, {})
    ost = rec.get("ost_step") or {}
    if not ost:
        return "", 1
    lines = [ost[k] for k in sorted(ost.keys())]
    next_id = max(ost.keys(), default=0) + 1
    return "\n".join(lines) + "\n", next_id


def concat_subqs_subas_as_ost_steps(solution_trace: Dict[int, Dict]) -> Tuple[str, int]:
    """Format subq/subanswer as Step 1, Step 2, ...; return (string, 1)."""
    parts = []
    step_id = 1
    for sid in sorted(solution_trace.keys()):
        if sid == 0:
            continue
        step = solution_trace.get(sid, {})
        if "subanswer" not in step:
            return "", 1
        suba = step["subanswer"]
        text = suba.get("text", "") if isinstance(suba, dict) else str(suba)
        # Take part before "The answer is"
        match = re.search(r"(.+?\.)\s*The answer is", text, re.DOTALL | re.IGNORECASE)
        step_text = match.group(1).strip() if match else text.strip()
        parts.append(f"Step {step_id}: {step_text}")
        step_id += 1
    return "\n".join(parts) + "\n" if parts else "", 1


def make_hint(
    solution_trace: Dict[int, Dict],
    node_type: Node_Type,
    new_subq: str | None = None,
    new_suba: str | None = None,
    new_ost_step: str | None = None,
) -> str:
    if node_type in (Node_Type.SUBQUESTION, Node_Type.RE_SUBANSWER):
        hint_parts = []
        for sid in sorted(solution_trace.keys()):
            if sid == 0:
                continue
            step = solution_trace.get(sid, {})
            if "subquestion" in step and "subanswer" in step:
                suba = step["subanswer"]
                text = suba.get("text", "") if isinstance(suba, dict) else str(suba)
                hint_parts.append(f"Hint {sid}: {step['subquestion']} {text}")
        if new_subq is not None and new_suba is not None:
            hint_parts.append(f"Hint {len(solution_trace)}: {new_subq} {new_suba}")
        return "\n".join(hint_parts)
    if node_type == Node_Type.OST_STEP:
        last_key = max(solution_trace.keys()) if solution_trace else 0
        rec = solution_trace.get(last_key, {})
        ost = rec.get("ost_step") or {}
        hint = " ".join(ost.get(k, "") for k in sorted(ost.keys()))
        if new_ost_step:
            hint = (hint + " " + new_ost_step).strip()
        return hint.strip()
    return ""


def make_response_prefix(
    solution_trace: Dict[int, Dict],
    node_type: Node_Type,
    new_subq: str | None = None,
    new_suba: str | None = None,
    new_ost_step: str | None = None,
) -> str:
    marker = "The answer is"
    if node_type in (Node_Type.SUBQUESTION, Node_Type.RE_SUBANSWER):
        parts = []
        for sid in sorted(solution_trace.keys()):
            if sid == 0:
                continue
            step = solution_trace.get(sid, {})
            if "subanswer" in step:
                suba = step["subanswer"]
                text = suba.get("text", "") if isinstance(suba, dict) else str(suba)
                parts.append(text.split(marker)[0].strip())
        if new_subq is not None and new_suba is not None:
            parts.append(new_suba.split(marker)[0].strip())
        return " ".join(p for p in parts if p)
    if node_type == Node_Type.OST_STEP:
        last_key = max(solution_trace.keys()) if solution_trace else 0
        rec = solution_trace.get(last_key, {})
        ost = rec.get("ost_step") or {}
        prefix = " ".join(ost.get(k, "") for k in sorted(ost.keys()))
        if new_ost_step:
            prefix = (prefix + " " + new_ost_step).strip()
        return prefix.strip()
    return ""
