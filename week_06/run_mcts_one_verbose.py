"""
Run MCTS-RAG once for a single fixed question with real model + retriever.
Prints every RAG retrieval (query + retrieved docs) and every model call (prompt preview + completions).
All parameters are hardcoded.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------- Hardcoded config ----------
QUESTION = 'What is the government type where "Le Mali" is the national anthem?'
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DATASET = "CWEBQA"
NUM_ROLLOUTS = 2
NUM_VOTES = 2
TOP_K = 5
MAX_NEW_TOKENS = 256
RETRIEVER_BACKEND = "wikipedia"
SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"
# --------------------------------------


def _truncate(s: str, max_len: int = 400) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


class LoggingRetriever:
    """Wraps Retriever and prints query + retrieved docs for every retrieve()."""

    def __init__(self, retriever, action_labels: dict | None = None):
        self._r = retriever
        self._action_labels = action_labels or {}

    def retrieve(self, query: str, k=None):
        k = k if k is not None else self._r.top_k
        label = self._action_labels.get("retriever", "RAG")
        print("\n" + "=" * 60)
        print(f"[{label}] RETRIEVE")
        print(f"  Query: {query[:200]}{'...' if len(query) > 200 else ''}")
        print(f"  top_k: {k}")
        docs = self._r.retrieve(query, k=k)
        print(f"  Retrieved {len(docs)} doc(s):")
        for i, d in enumerate(docs):
            title = d.get("title", "")
            text = (d.get("text") or d.get("content") or str(d))[:300]
            print(f"    [{i+1}] {title}")
            print(f"        {_truncate(text, 280)}")
        print("=" * 60 + "\n")
        return docs


class LoggingModel:
    """Wraps model and prints prompt (truncated) + full output(s) for every generate()."""

    def __init__(self, model, action_labels: dict | None = None):
        self._m = model
        self._action_labels = action_labels or {}

    def generate(self, prompt: str, max_new_tokens: int = 512, stop_tokens=None, do_sample: bool = False, temperature: float = 0.0, top_p: float = 1.0, n: int = 1):
        label = self._action_labels.get("model", "MODEL")
        print("\n" + "-" * 60)
        print(f"[{label}] GENERATE")
        print(f"  Prompt (first 500 chars):\n  {_truncate(prompt, 500)}")
        print(f"  max_new_tokens={max_new_tokens}, n={n}, temperature={temperature}")
        out = self._m.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            n=n,
        )
        if isinstance(out, str):
            print(f"  Output:\n  {out}")
        else:
            for i, o in enumerate(out):
                print(f"  Output [{i+1}/{len(out)}]:\n  {o}")
        print("-" * 60 + "\n")
        return out


def main():
    from model import get_model
    from retriever import Retriever
    from mcts_rag import Generator, Reasoning_MCTS_Node, select_final_answer
    from mcts_utils import Node_Type
    from mcts_backbone import MCTS_Searcher

    print("Loading model and retriever (hardcoded)...")
    raw_model = get_model(MODEL_PATH, backend="vllm", max_model_len=4096)
    raw_model.load()
    raw_retriever = Retriever(backend=RETRIEVER_BACKEND, top_k=TOP_K)
    action_labels: dict[str, str] = {}
    model = LoggingModel(raw_model, action_labels=action_labels)
    retriever = LoggingRetriever(raw_retriever, action_labels=action_labels)

    print("Loading prompts and building Generator (A1–A5)...")
    generator = Generator(
        model=model,
        retriever=retriever,
        dataset=DATASET,
        prompts_dir=PROMPTS_DIR,
        num_votes=NUM_VOTES,
        max_new_tokens=MAX_NEW_TOKENS,
        top_k=TOP_K,
    )

    print("\n" + "#" * 60)
    print("QUESTION:", QUESTION)
    print("#" * 60)

    trajectory_rewards = []
    root = Reasoning_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        generator=generator,
        user_question=QUESTION,
        max_depth_allowed=5,
    )

    searcher = MCTS_Searcher(
        exploration_weight=2.0,
        num_rollouts=NUM_ROLLOUTS,
        weight_scheduler="const",
        discount=1.0,
    )

    for r in range(NUM_ROLLOUTS):
        print(f"\n>>> Rollout {r + 1} / {NUM_ROLLOUTS}")
        leaf = searcher.do_rollout(root, r)
        if leaf is not None:
            ans = getattr(leaf, "answer", None)
            rew = leaf.calculate_reward()
            trajectory_rewards.append((ans, rew))
            ntype = getattr(leaf, "node_type", None)
            print(f"    Leaf: {ntype}  answer={ans!r}  reward={rew:.3f}")

    final = select_final_answer(trajectory_rewards, DATASET, check_equiv=None)
    print("\n" + "#" * 60)
    print("FINAL SELECTED ANSWER:", final)
    print("#" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
