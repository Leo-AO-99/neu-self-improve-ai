"""
MCTS-RAG: MCTS + RAG reasoning nodes and generator.
Adapted from https://github.com/yale-nlp/MCTS-RAG (run_src/MCTS_for_reasoning_with_rag.py).
Actions: A1 Direct Answer, A2 OST step, A3 Subquestions, A4 RAG (retrieve/rag_step), A5 Rephrased question.
Uses prompts from prompts/<dataset>/fewshot_cot, fewshot_rag, decompose, fewshot_ost, rephrasing.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

from mcts_backbone import MCTS_Node, MCTS_Searcher
from mcts_utils import (
    Node_Type,
    concat_ost_steps,
    concat_subqs_and_subas,
    concat_subqs_subas_as_ost_steps,
    reach_terminal_ost_step,
    reach_terminal_subquestion,
)
from retriever import Retriever, format_context
from run_common import (
    load_decompose_config,
    load_fewshot_cot_config,
    load_fewshot_ost_config,
    load_fewshot_rag_config,
    load_rephrasing_prompt,
)


class Generator:
    """Generates children for MCTS: A1–A5. Loads prompts from prompts_dir."""

    def __init__(
        self,
        model: Any,
        retriever: Retriever,
        dataset: str,
        prompts_dir: str | Path,
        num_votes: int = 8,
        max_new_tokens: int = 512,
        stop_tokens: list[str] | None = None,
        max_context_chars: int = 8000,
        top_k: int = 10,
        disable_a1: bool = False,
        disable_a4: bool = False,
        disable_a5: bool = False,
        disable_rag: bool = False,
        num_subquestions: int = 3,
        num_ost_steps: int = 2,
    ):
        self.model = model
        self.retriever = retriever
        self.dataset = dataset.upper()
        self.prompts_dir = Path(prompts_dir).resolve()
        self.num_votes = max(1, num_votes)
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self.top_k = top_k
        self.disable_a1 = disable_a1
        self.disable_a4 = disable_a4
        self.disable_a5 = disable_a5
        self.disable_rag = disable_rag
        self.num_subquestions = num_subquestions
        self.num_ost_steps = num_ost_steps
        from evaluator import find_most_confident_answer

        self._find_confident = find_most_confident_answer

        cot_tpl, cot_stop, _am, cot_ex = load_fewshot_cot_config(self.prompts_dir, self.dataset)
        rag_tpl, rag_stop, rag_ex = load_fewshot_rag_config(self.prompts_dir, self.dataset)
        self._cot_template = cot_tpl
        self._cot_examples = cot_ex
        self._rag_template = rag_tpl
        self._rag_examples = rag_ex
        self._stop_tokens = stop_tokens or cot_stop
        if rag_stop != cot_stop:
            self._stop_tokens = self._stop_tokens or rag_stop

        decomp, decomp_rephr, self.question_index = load_decompose_config(self.prompts_dir, self.dataset)
        self._decompose_prompt = decomp
        self._decompose_prompt_rephrased = decomp_rephr
        self._fewshot_ost_config = load_fewshot_ost_config(self.prompts_dir, self.dataset)
        self._rephrasing_prompt = load_rephrasing_prompt(self.prompts_dir, self.dataset)

    def _generate_n(self, prompt: str, n: int, stop: list[str] | None = None) -> list[str]:
        out = self.model.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            stop_tokens=stop or self._stop_tokens,
            do_sample=True,
            temperature=0.7,
            n=n,
        )
        if isinstance(out, str):
            return [out]
        return list(out)

    def _cot_prompt(self, instruction: str, examples: str = "") -> str:
        return self._cot_template.format(instruction=instruction, examples=examples or self._cot_examples)

    def generate_direct_answer(self, question: str, hint: str | None = None) -> tuple[str | None, str, float]:
        """A1: Direct answer (no retrieval)."""
        if hint:
            instruction = question + "\n\nHint: " + hint
        else:
            instruction = question
        prompt = self._cot_prompt(instruction)
        completions = self._generate_n(prompt, self.num_votes)
        ans, comp, _idx, conf = self._find_confident(completions, self.dataset)
        return ans, comp or (completions[0] if completions else ""), max(conf, 0.01)

    def generate_direct_answers(
        self, user_question: str, paraphrased: bool, hint: str | None = None
    ) -> tuple[list[str], list[float]]:
        """A1: Multiple direct answers; returns (answer_list, value_list)."""
        ans, _comp, conf = self.generate_direct_answer(user_question, hint=hint)
        return ([ans] if ans is not None else [], [conf])

    def generate_rag_answer(self, question: str) -> tuple[str | None, str, float]:
        """A4: Retrieve then answer (root-level RAG)."""
        docs = self.retriever.retrieve(question, k=self.top_k)
        context = format_context(docs, max_chars=self.max_context_chars) if docs else "(No results.)"
        prompt = self._rag_template.format(
            context=context,
            instruction=question,
            examples=self._rag_examples,
        )
        completions = self._generate_n(prompt, self.num_votes)
        ans, comp, _idx, conf = self._find_confident(completions, self.dataset)
        return ans, comp or (completions[0] if completions else ""), max(conf, 0.01)

    def generate_ost_step(
        self,
        user_question: str,
        solution_trace: dict,
        paraphrased: bool,
        parent_is_subquestion: bool,
    ) -> tuple[list[str], list[float]]:
        """A2: One-step thought. Returns (ost_step_list, value_list)."""
        if parent_is_subquestion:
            existing_ost, next_id = concat_subqs_subas_as_ost_steps(solution_trace)
        else:
            existing_ost, next_id = concat_ost_steps(solution_trace)
        tpl = self._fewshot_ost_config.get("prompt_template", "{instruction}\n\n")
        prompt = tpl.format(examples="", instruction=user_question) + existing_ost + "\n"
        prompt += f"The text you generate must start with the string Step {next_id}:\n"
        stop = [f"Step {next_id + 1}", "\n\n\n"] + list(self._fewshot_ost_config.get("stop_tokens") or [])
        completions = self._generate_n(prompt, self.num_ost_steps, stop=stop)
        ost_list = [c.strip().strip("\n") for c in completions if c.strip().startswith(f"Step {next_id}")]
        if not ost_list:
            ost_list = [f"Step {next_id}: {c.strip().strip(chr(10))}" for c in completions]
        from mcts_utils import make_response_prefix

        values = []
        for ost in ost_list:
            prefix = make_response_prefix(solution_trace, Node_Type.OST_STEP, new_ost_step=ost)
            score_prompt = "Question: " + user_question + "\nAnswer: " + prefix
            preds = self._generate_n(score_prompt, self.num_votes)
            _a, _c, _i, conf = self._find_confident(preds, self.dataset)
            values.append(max(conf, 0.01))
        if not values:
            values = [0.01] * len(ost_list)
        return ost_list, values

    def generate_rag_step(
        self,
        user_question: str,
        solution_trace: dict,
        paraphrased: bool,
        parent_is_subquestion: bool,
    ) -> tuple[list[str], list[float]]:
        """A4: RAG step (retrieve then next step)."""
        if parent_is_subquestion:
            existing_ost, next_id = concat_subqs_subas_as_ost_steps(solution_trace)
        else:
            existing_ost, next_id = concat_ost_steps(solution_trace)
            if next_id == 1:
                return self.generate_ost_step(
                    user_question=user_question,
                    solution_trace=solution_trace,
                    paraphrased=paraphrased,
                    parent_is_subquestion=parent_is_subquestion,
                )
        query = f"{user_question}\n\n{existing_ost}"
        docs = self.retriever.retrieve(query, k=self.top_k)
        context = format_context(docs, max_chars=self.max_context_chars) if docs else ""
        tpl = self._fewshot_ost_config.get("prompt_template", "{instruction}\n\n")
        prompt = tpl.format(examples="", instruction=user_question) + existing_ost + "\n"
        prompt += f"### Relevant Context:\n{context}\n\nThe text you generate must start with Step {next_id}:"
        stop = ["\n\n\n", f"Step {next_id + 1}", str(next_id + 1)]
        completions = self._generate_n(prompt, self.num_ost_steps, stop=stop)
        ost_list = [
            (c.strip().strip("\n") + "\n\n### Relevant Context: " + context + "\n")
            for c in completions
            if c.strip()
        ]
        if not ost_list:
            ost_list = [f"Step {next_id}: " + c.strip() for c in completions]
        from mcts_utils import make_response_prefix

        values = []
        for ost in ost_list:
            prefix = make_response_prefix(solution_trace, Node_Type.OST_STEP, new_ost_step=ost)
            score_prompt = "Question: " + user_question + "\nAnswer: " + prefix
            preds = self._generate_n(score_prompt, self.num_votes)
            _a, _c, _i, conf = self._find_confident(preds, self.dataset)
            values.append(max(conf, 0.01))
        if not values:
            values = [0.01] * len(ost_list)
        return ost_list, values

    def generate_subquestions(
        self,
        user_question: str,
        solution_trace: dict,
        paraphrased: bool,
    ) -> tuple[list[str], list[str], list[float], list[Any]]:
        """A3: Decompose into subquestions and answer each. Returns (subq_list, suba_list, value_list, potential_answers)."""
        decomp = self._decompose_prompt_rephrased if paraphrased else self._decompose_prompt
        existing, next_id = concat_subqs_and_subas(solution_trace, self.question_index)
        prompt = decomp + "\n\n" + f"Question {self.question_index}: {user_question}\n" + existing + "\n"
        prompt += f"The text you generate must start with the string of subquestion index Question {self.question_index}.{next_id}:."
        stop = ["Answer", "\n", "The answer", f"Answer {self.question_index}.{next_id}", f"Answer {self.question_index}.{next_id}:", f"Answer {self.question_index}.{next_id}: "]
        raw_subqs = self._generate_n(prompt, max(1, self.num_subquestions), stop=stop)
        subq_list = list({s.strip() for s in raw_subqs if s.strip().startswith(f"Question {self.question_index}.{next_id}:")})
        if not subq_list:
            subq_list = list({s.strip() for s in raw_subqs if s.strip()})
        if not subq_list:
            return [], [], [], []

        suba_list = []
        value_list = []
        for subq in subq_list:
            io = decomp + "\n\n" + f"Question {self.question_index}: {user_question}\n" + existing
            io += f"Question {self.question_index}.{next_id}: " + subq + "\n"
            io += f"Please use one complete sentence to answer the question: {self.question_index}.{next_id}."
            comps = self._generate_n(io, self.num_votes, stop=["\n\n\n", f"Question {self.question_index}.{next_id + 1}"])
            ans, comp, _i, conf = self._find_confident(comps, self.dataset)
            suba_list.append(comp or (comps[0] if comps else ""))
            value_list.append(max(conf, 0.01))
        return subq_list, suba_list, value_list, [None] * len(subq_list)

    def generate_re_subanswers(
        self, user_question: str, solution_trace: dict, paraphrased: bool
    ) -> tuple[list[str], list[float], list[Any]]:
        """Re-answer last subquestion (CoT). Returns (re_suba_list, value_list, potential_answers)."""
        if not solution_trace:
            return [], [], []
        last_id = max(k for k in solution_trace if k > 0)
        last_subq = solution_trace[last_id]["subquestion"]
        question = f"{user_question} {last_subq}" if not paraphrased else f"{user_question} Question: {last_subq}"
        prompt = self._cot_prompt(question)
        comps = self._generate_n(prompt, self.num_votes)
        ans, comp, _i, conf = self._find_confident(comps, self.dataset)
        return [comp or ""], [max(conf, 0.01)], [None]

    def generate_rag_and_re_subanswers(
        self, user_question: str, solution_trace: dict, paraphrased: bool
    ) -> tuple[list[list[str]], list[float], list[Any]]:
        """Re-answer last subquestion with RAG. Returns (re_suba_list with context, value_list, potential_answers)."""
        if not solution_trace:
            return [], [], []
        last_id = max(k for k in solution_trace if k > 0)
        last_subq = solution_trace[last_id]["subquestion"]
        question = f"{user_question}\n\n{last_subq}" if not paraphrased else f"{user_question} Question: {last_subq}"
        docs = self.retriever.retrieve(question, k=self.top_k)
        context = format_context(docs, max_chars=self.max_context_chars) if docs else ""
        q2 = f"{user_question} {last_subq}\n\n### Relevant Context:\n{context}." if not paraphrased else f"{user_question} Question: {last_subq}"
        prompt = self._cot_prompt(q2)
        comps = self._generate_n(prompt, self.num_votes)
        ans, comp, _i, conf = self._find_confident(comps, self.dataset)
        wrapped = [f"{(comp or '').strip().strip(chr(10))}\n\n### Relevant Context: {context}\n"]
        return wrapped, [max(conf, 0.01)], [None]

    def generate_rephrased_user_question(self, user_question: str) -> tuple[list[str], list[Any]]:
        """A5: Rephrase question. Returns (rephrased_list, potential_answers)."""
        prompt = self._rephrasing_prompt + "\n\nRephrase Original Question: " + user_question + "\n"
        prompt += "Rephrased question you generate should start with Given a list of conditions, please answer the question. Condition 1:, and it should be one line"
        comps = self._generate_n(prompt, 1)
        out = comps[0] if comps else ""
        if "Condition 1:" in out:
            out = "Given a list of conditions, please answer the question: " + user_question + " Condition 1:" + out.split("Condition 1:")[-1]
        else:
            out = "Given a list of conditions, please answer the question. Condition 1: " + out
        return [out], [None]

    def generate_user_question_retrieve(self, user_question: str) -> tuple[list[str], list[Any]]:
        """A4: Retrieve for user question and format as context. Returns (retrieved_message_list, potential_answers)."""
        docs = self.retriever.retrieve(user_question, k=self.top_k)
        context = format_context(docs, max_chars=self.max_context_chars) if docs else ""
        msg = f"Given additional informations, please answer the question.\n### Relevant Context: {context}\nUser Question: {user_question}."
        return [msg], [None]

    def get_root_children(self, question: str) -> list[tuple[str, str | None, str, float]]:
        """Legacy: root children as (action_name, answer, completion, reward) for A1/A4 only."""
        children: list[tuple[str, str | None, str, float]] = []
        if not self.disable_a1:
            ans, comp, reward = self.generate_direct_answer(question)
            children.append(("A1", ans, comp, reward))
        if not self.disable_a4 and not self.disable_rag:
            ans, comp, reward = self.generate_rag_answer(question)
            children.append(("A4", ans, comp, reward))
        return children


class Reasoning_MCTS_Node(MCTS_Node):
    """MCTS node with node_type and solution_trace; find_children dispatches A1–A5 by type."""

    def __init__(
        self,
        parent: Reasoning_MCTS_Node | None,
        depth: int,
        node_type: Node_Type,
        *,
        generator: Generator | None = None,
        user_question: str | None = None,
        rephrased_user_question: str | None = None,
        direct_answer: str | None = None,
        node_value: float | None = None,
        subquestion: str | None = None,
        subanswer: str | None = None,
        is_new_subquestion: bool = False,
        re_subanswer: str | None = None,
        ost_step: str | None = None,
        max_depth_allowed: int = 5,
        disable_a1: bool = False,
        disable_a5: bool = False,
        disable_rag: bool = False,
    ):
        super().__init__()
        self.parent = parent
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.direct_answer = direct_answer
        self.subquestion = subquestion
        self.subanswer = subanswer
        self.is_new_subquestion = is_new_subquestion
        self.re_subanswer = re_subanswer
        self.ost_step = ost_step
        self._children_list: list[Reasoning_MCTS_Node] | None = None

        if parent is None:
            self.generator = generator
            self.user_question = user_question or ""
            self.max_depth_allowed = max_depth_allowed
            self.disable_a1 = disable_a1
            self.disable_a5 = disable_a5
            self.disable_rag = disable_rag
            self.paraphrased = False
            self.subquestion_counter = 0
            self.ost_step_counter = 0
            self.solution_trace = {0: {"user_question": user_question or "", "ost_step": {}}}
        else:
            self.generator = parent.generator
            self.user_question = parent.user_question
            self.max_depth_allowed = parent.max_depth_allowed
            self.disable_a1 = parent.disable_a1
            self.disable_a5 = parent.disable_a5
            self.disable_rag = parent.disable_rag
            self.paraphrased = parent.paraphrased
            if node_type == Node_Type.REPHRASED_USER_QUESTION and rephrased_user_question is not None:
                self.user_question = rephrased_user_question
                self.paraphrased = True
            self.subquestion_counter = parent.subquestion_counter + 1 if (node_type == Node_Type.SUBQUESTION and is_new_subquestion) else parent.subquestion_counter
            self.ost_step_counter = parent.ost_step_counter + 1 if node_type == Node_Type.OST_STEP else parent.ost_step_counter
            self.solution_trace = copy.deepcopy(parent.solution_trace)

            if node_type == Node_Type.REPHRASED_USER_QUESTION and rephrased_user_question is not None:
                self.solution_trace[0]["user_question"] = rephrased_user_question
            elif node_type == Node_Type.DIRECT_ANSWER and direct_answer is not None and node_value is not None:
                if self.subquestion_counter not in self.solution_trace:
                    self.solution_trace[self.subquestion_counter] = {"subquestion": "", "subanswer": {}, "ost_step": {}}
                self.solution_trace[self.subquestion_counter]["direct_answer"] = {"text": direct_answer, "value": node_value}
            elif node_type == Node_Type.SUBQUESTION and subquestion is not None and subanswer is not None and node_value is not None:
                self.solution_trace[self.subquestion_counter] = {
                    "subquestion": subquestion,
                    "subanswer": {"text": subanswer, "value": node_value},
                    "ost_step": {},
                }
            elif node_type == Node_Type.RE_SUBANSWER and re_subanswer is not None and node_value is not None:
                self.solution_trace[self.subquestion_counter]["subanswer"] = {"text": re_subanswer, "value": node_value}
            elif node_type == Node_Type.OST_STEP and ost_step is not None:
                if self.subquestion_counter not in self.solution_trace:
                    self.solution_trace[self.subquestion_counter] = {"user_question": "", "ost_step": {}}
                if "ost_step" not in self.solution_trace[self.subquestion_counter]:
                    self.solution_trace[self.subquestion_counter]["ost_step"] = {}
                self.solution_trace[self.subquestion_counter]["ost_step"][self.ost_step_counter] = ost_step

    @property
    def answer(self) -> str | None:
        """Extract final answer from this node for aggregation."""
        if self.node_type == Node_Type.DIRECT_ANSWER and self.direct_answer:
            from evaluator import isolate_answer
            return isolate_answer(self.direct_answer) or self.direct_answer.strip()
        if self.node_type == Node_Type.SUBQUESTION and self.subquestion and reach_terminal_subquestion(self.subquestion, self.user_question) and self.subanswer:
            from evaluator import isolate_answer
            return isolate_answer(self.subanswer) or (self.subanswer.strip() if isinstance(self.subanswer, str) else None)
        if self.node_type == Node_Type.OST_STEP and self.ost_step and reach_terminal_ost_step(self.ost_step):
            from evaluator import isolate_answer
            return isolate_answer(self.ost_step)
        return None

    def is_valid_leaf_node(self) -> bool:
        return (
            (self.node_type == Node_Type.SUBQUESTION and self.subquestion and reach_terminal_subquestion(self.subquestion, self.user_question))
            or self.node_type == Node_Type.DIRECT_ANSWER
        )

    def is_terminal(self) -> bool:
        if self.depth >= self.max_depth_allowed or self.is_valid_leaf_node():
            return True
        if self._children_list is not None and len(self._children_list) == 0:
            return True
        return False

    def calculate_reward(self) -> float:
        if self.is_valid_leaf_node() and self.node_value is not None:
            return max(self.node_value, 0.01)
        return 0.01

    def skip_backprop(self) -> bool:
        return self.node_type in (Node_Type.USER_QUESTION, Node_Type.REPHRASED_USER_QUESTION)

    def find_children(self, rollout_id: int) -> list[MCTS_Node]:
        if self._children_list is not None:
            return self._children_list
        if self.is_terminal() or self.generator is None:
            self._children_list = []
            return self._children_list

        gen = self.generator
        children: list[Reasoning_MCTS_Node] = []

        def add_direct_answers():
            hint = None
            if self.node_type not in (Node_Type.USER_QUESTION, Node_Type.REPHRASED_USER_QUESTION):
                from mcts_utils import make_hint
                hint = make_hint(self.solution_trace, self.node_type)
            da_list, val_list = gen.generate_direct_answers(self.user_question, self.paraphrased, hint=hint)
            for da, val in zip(da_list, val_list):
                if val is None or val <= 0:
                    val = 0.01
                children.append(
                    Reasoning_MCTS_Node(
                        parent=self, depth=self.depth + 1, node_type=Node_Type.DIRECT_ANSWER,
                        direct_answer=da, node_value=val,
                        max_depth_allowed=self.max_depth_allowed,
                        disable_a1=gen.disable_a1, disable_a5=gen.disable_a5, disable_rag=gen.disable_rag,
                    )
                )

        def add_ost_steps(parent_is_sq: bool = False):
            ost_list, val_list = gen.generate_ost_step(
                self.user_question, self.solution_trace, self.paraphrased, parent_is_subquestion=parent_is_sq
            )
            for ost, val in zip(ost_list, val_list or [0.01] * len(ost_list)):
                children.append(
                    Reasoning_MCTS_Node(
                        parent=self, depth=self.depth + 1, node_type=Node_Type.OST_STEP,
                        ost_step=ost, node_value=val,
                        max_depth_allowed=self.max_depth_allowed,
                        disable_a1=gen.disable_a1, disable_a5=gen.disable_a5, disable_rag=gen.disable_rag,
                    )
                )

        def add_rag_steps(parent_is_sq: bool = False):
            ost_list, val_list = gen.generate_rag_step(
                self.user_question, self.solution_trace, self.paraphrased, parent_is_subquestion=parent_is_sq
            )
            for ost, val in zip(ost_list, val_list or [0.01] * len(ost_list)):
                children.append(
                    Reasoning_MCTS_Node(
                        parent=self, depth=self.depth + 1, node_type=Node_Type.OST_STEP,
                        ost_step=ost, node_value=val,
                        max_depth_allowed=self.max_depth_allowed,
                        disable_a1=gen.disable_a1, disable_a5=gen.disable_a5, disable_rag=gen.disable_rag,
                    )
                )

        def add_subquestions():
            subq_list, suba_list, val_list, _pot = gen.generate_subquestions(
                self.user_question, self.solution_trace, self.paraphrased
            )
            for sq, sa, val in zip(subq_list, suba_list, val_list or [0.01] * len(subq_list)):
                if val is None or val <= 0:
                    val = 0.01
                children.append(
                    Reasoning_MCTS_Node(
                        parent=self, depth=self.depth + 1, node_type=Node_Type.SUBQUESTION,
                        subquestion=sq, subanswer=sa, node_value=val, is_new_subquestion=True,
                        max_depth_allowed=self.max_depth_allowed,
                        disable_a1=gen.disable_a1, disable_a5=gen.disable_a5, disable_rag=gen.disable_rag,
                    )
                )

        def add_re_subanswers():
            re_list, val_list, _pot = gen.generate_re_subanswers(
                self.user_question, self.solution_trace, self.paraphrased
            )
            for re_sa, val in zip(re_list, val_list or [0.01]):
                if val is None or val <= 0:
                    val = 0.01
                children.append(
                    Reasoning_MCTS_Node(
                        parent=self, depth=self.depth + 1, node_type=Node_Type.RE_SUBANSWER,
                        re_subanswer=re_sa, node_value=val,
                        max_depth_allowed=self.max_depth_allowed,
                        disable_a1=gen.disable_a1, disable_a5=gen.disable_a5, disable_rag=gen.disable_rag,
                    )
                )

        def add_rag_re_subanswers():
            re_list, val_list, _pot = gen.generate_rag_and_re_subanswers(
                self.user_question, self.solution_trace, self.paraphrased
            )
            for re_sa, val in zip(re_list, val_list or [0.01]):
                if val is None or val <= 0:
                    val = 0.01
                text = re_sa[0] if isinstance(re_sa, list) else re_sa
                children.append(
                    Reasoning_MCTS_Node(
                        parent=self, depth=self.depth + 1, node_type=Node_Type.RE_SUBANSWER,
                        re_subanswer=text, node_value=val,
                        max_depth_allowed=self.max_depth_allowed,
                        disable_a1=gen.disable_a1, disable_a5=gen.disable_a5, disable_rag=gen.disable_rag,
                    )
                )

        def add_rephrased():
            rephrased_list, _pot = gen.generate_rephrased_user_question(self.user_question)
            for rq in rephrased_list:
                children.append(
                    Reasoning_MCTS_Node(
                        parent=self, depth=self.depth + 1, node_type=Node_Type.REPHRASED_USER_QUESTION,
                        rephrased_user_question=rq, node_value=0.01,
                        max_depth_allowed=self.max_depth_allowed,
                        disable_a1=gen.disable_a1, disable_a5=gen.disable_a5, disable_rag=gen.disable_rag,
                    )
                )

        def add_question_retrieve():
            retrieved_list, _pot = gen.generate_user_question_retrieve(self.user_question)
            for rq in retrieved_list:
                children.append(
                    Reasoning_MCTS_Node(
                        parent=self, depth=self.depth + 1, node_type=Node_Type.REPHRASED_USER_QUESTION,
                        rephrased_user_question=rq, node_value=0.01,
                        max_depth_allowed=self.max_depth_allowed,
                        disable_a1=gen.disable_a1, disable_a5=gen.disable_a5, disable_rag=gen.disable_rag,
                    )
                )

        if self.node_type == Node_Type.USER_QUESTION:
            if not gen.disable_a1:
                add_ost_steps(parent_is_sq=False)
            if not gen.disable_rag:
                add_rag_steps(parent_is_sq=False)
                add_question_retrieve()
            add_direct_answers()
            add_subquestions()
            if not gen.disable_a5:
                add_rephrased()
        elif self.node_type == Node_Type.REPHRASED_USER_QUESTION:
            if not gen.disable_a1:
                add_ost_steps(parent_is_sq=False)
            if not gen.disable_rag:
                add_rag_steps(parent_is_sq=False)
            add_direct_answers()
            add_subquestions()
        elif self.node_type == Node_Type.SUBQUESTION:
            if not gen.disable_a1:
                add_ost_steps(parent_is_sq=True)
            add_re_subanswers()
            if not gen.disable_rag:
                add_rag_re_subanswers()
            add_direct_answers()
            add_subquestions()
        elif self.node_type == Node_Type.RE_SUBANSWER:
            if not gen.disable_a1:
                add_ost_steps(parent_is_sq=True)
            if not gen.disable_rag:
                add_rag_steps(parent_is_sq=True)
            add_direct_answers()
            add_subquestions()
        elif self.node_type == Node_Type.OST_STEP:
            if not gen.disable_rag:
                add_rag_steps(parent_is_sq=False)
            if not gen.disable_a1:
                add_ost_steps(parent_is_sq=False)
            add_direct_answers()
        else:
            self._children_list = []
            return self._children_list

        self._children_list = children
        return self._children_list


def run_mcts_search(
    question: str,
    generator: Generator,
    num_rollouts: int = 4,
    exploration_weight: float = 2.0,
    max_depth: int = 5,
    disable_a1: bool = False,
    disable_a5: bool = False,
    disable_rag: bool = False,
) -> tuple[list[tuple[str | None, float]], MCTS_Searcher, Reasoning_MCTS_Node]:
    """
    Run MCTS from root (USER_QUESTION node); return list of (answer, trajectory_reward) for each rollout leaf,
    the searcher, and the root node.
    """
    root = Reasoning_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        generator=generator,
        user_question=question,
        max_depth_allowed=max_depth,
        disable_a1=disable_a1,
        disable_a5=disable_a5,
        disable_rag=disable_rag,
    )
    searcher = MCTS_Searcher(
        exploration_weight=exploration_weight,
        num_rollouts=num_rollouts,
        weight_scheduler="const",
        discount=1.0,
    )
    trajectory_rewards: list[tuple[str | None, float]] = []
    for r in range(num_rollouts):
        leaf = searcher.do_rollout(root, r)
        if leaf is not None and isinstance(leaf, Reasoning_MCTS_Node):
            trajectory_rewards.append((leaf.answer, leaf.calculate_reward()))
    return trajectory_rewards, searcher, root


def select_final_answer(
    trajectory_rewards: list[tuple[str | None, float]],
    dataset: str,
    check_equiv: Callable[[str | None, str | None], bool] | None = None,
) -> str | None:
    """
    Score(a) = sum(reward of trajectories with answer a) / sum(all rewards).
    Return argmax Score(a). Uses check_equiv to group equivalent answers.
    """
    from evaluator import _equiv_for_dataset
    equiv = check_equiv if check_equiv is not None else _equiv_for_dataset(dataset)
    total = sum(r for _, r in trajectory_rewards)
    if total <= 0:
        return trajectory_rewards[0][0] if trajectory_rewards else None
    answer_scores: dict[str, float] = {}
    for ans, reward in trajectory_rewards:
        key = None
        for existing in answer_scores:
            if equiv(ans, existing):
                key = existing
                break
        if key is None:
            key = ans if ans is not None else ""
        answer_scores[key] = answer_scores.get(key, 0.0) + reward
    best = max(answer_scores.items(), key=lambda x: x[1])
    return best[0] if best[0] else None
