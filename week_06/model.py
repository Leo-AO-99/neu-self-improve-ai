"""
LLM wrapper for MCTS-RAG baselines (CoT and Standard RAG).
Supports vLLM (default) and HuggingFace transformers.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

HF_TOKEN: str | None = None  # e.g. "hf_xxxxxxxxxxxx"

if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN

# Reduce vLLM log noise so accuracy is visible at end
if "VLLM_LOGGING_LEVEL" not in os.environ:
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

# Llama 3.x chat format: applied in generate() so all baselines (CoT, RAG, etc.) get it automatically
LLAMA_EOT = "<|eot_id|>"


def is_llama(model_name_or_path: str) -> bool:
    """True if model looks like Llama (e.g. meta-llama/Llama-3.2-8B). Used to wrap prompt in chat format."""
    return "llama" in (model_name_or_path or "").lower()


def wrap_prompt_for_llama_chat(prompt: str) -> str:
    """Wrap prompt in Llama 3.x user/assistant format. Used by generate() when is_llama()."""
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}{LLAMA_EOT}<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def extract_answer(text: str, answer_marker: str = "answer is") -> str | None:
    """Extract answer after 'The answer is: ...' or similar."""
    if not text or not answer_marker:
        return None

    def clean(ans: str) -> str:
        ans = ans.split("\n")[0].strip().rstrip(".")
        ans = re.sub(r"^[Oo]ption\s+", "", ans)
        if len(ans) > 1 and ans.endswith(")"):
            ans = ans.rstrip(")").strip()
        return ans

    # Primary: "The answer is: X." or "answer is: X"
    pattern = rf"(?:[Tt]he\s+)?{re.escape(answer_marker)}\s*:\s*(.+?)(?:\.|$|\n)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if m:
        ans = clean(m.group(1).strip())
        return ans if ans else None
    # Fallback: "correct answer is X", "answer: X", "choose (A)"
    for fallback in [
        r"[Cc]orrect\s+answer\s+is\s*[:\s]+(\w+)",
        r"answer\s*:\s*(\w+)",
        r"choose\s*[\(\s]*([A-Da-d])[\)\s.]*",
        r"(?:is|are)\s+([A-Da-d])[\)\s.]*$",
    ]:
        m = re.search(fallback, text)
        if m:
            ans = m.group(1).strip()
            return clean(ans) if ans else None
    return None


class BaseModel:
    """Base class for generation and answer parsing."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        stop_tokens: list[str] | None = None,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        n: int = 1,
    ) -> str | list[str]:
        """Generate text. If n > 1, returns list of n samples; else single string."""
        raise NotImplementedError

    def answer(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        stop_tokens: list[str] | None = None,
        answer_marker: str = "answer is",
    ) -> str | None:
        raw = self.generate(prompt, max_new_tokens=max_new_tokens, stop_tokens=stop_tokens)
        return extract_answer(raw, answer_marker=answer_marker)


class VLLMModel(BaseModel):
    """vLLM backend for faster inference."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = 4096,
    ):
        super().__init__(model_name_or_path, device, torch_dtype)
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self._llm = None

    def load(self) -> None:
        from vllm import LLM

        kwargs = dict(
            model=self.model_name_or_path,
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        self._llm = LLM(**kwargs)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        stop_tokens: list[str] | None = None,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        n: int = 1,
    ) -> str | list[str]:
        from vllm import SamplingParams

        if self._llm is None:
            self.load()
        if is_llama(self.model_name_or_path):
            prompt = wrap_prompt_for_llama_chat(prompt)
            stop_tokens = list(stop_tokens) if stop_tokens else []
            if LLAMA_EOT not in stop_tokens:
                stop_tokens = stop_tokens + [LLAMA_EOT]
        stop = list(stop_tokens) if stop_tokens else []
        n = max(1, int(n))
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            stop=stop if stop else None,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
            n=n,
        )
        outputs = self._llm.generate([prompt], sampling_params)
        req_out = outputs[0].outputs
        def clean_one(text: str) -> str:
            if stop_tokens:
                for s in stop_tokens:
                    if s in text:
                        text = text.split(s)[0]
            return text.strip()
        if n == 1:
            return clean_one(req_out[0].text)
        return [clean_one(o.text) for o in req_out]


class HFModel(BaseModel):
    """HuggingFace Transformers model."""

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        stop_tokens: list[str] | None = None,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        n: int = 1,
    ) -> str | list[str]:
        if self.model is None or self.tokenizer is None:
            self.load()
        if is_llama(self.model_name_or_path):
            prompt = wrap_prompt_for_llama_chat(prompt)
            stop_tokens = list(stop_tokens) if stop_tokens else []
            if LLAMA_EOT not in stop_tokens:
                stop_tokens = stop_tokens + [LLAMA_EOT]
        n = max(1, int(n))
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)
        stop_ids = []
        if stop_tokens:
            for t in stop_tokens:
                ids = self.tokenizer.encode(t, add_special_tokens=False)
                if ids:
                    stop_ids.append(ids[0])
        need_sample = do_sample or n > 1
        gen = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=need_sample,
            temperature=temperature if need_sample else 1.0,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id if not stop_ids else None,
            num_return_sequences=n,
        )
        seq_len = inputs["input_ids"].shape[1]
        def clean_one(seq):
            out = self.tokenizer.decode(seq[seq_len:], skip_special_tokens=True)
            if stop_tokens:
                for s in stop_tokens:
                    if s in out:
                        out = out.split(s)[0]
            return out.strip()
        if n == 1:
            return clean_one(gen[0])
        return [clean_one(gen[i]) for i in range(n)]


def get_model(
    model_name_or_path: str,
    backend: str = "vllm",
    device: str | None = None,
    torch_dtype: torch.dtype | None = None,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = 4096,
) -> BaseModel:
    if backend == "vllm":
        return VLLMModel(
            model_name_or_path=model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )
    if backend == "transformers":
        return HFModel(
            model_name_or_path=model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
        )
    raise ValueError(f"Unknown backend: {backend}")