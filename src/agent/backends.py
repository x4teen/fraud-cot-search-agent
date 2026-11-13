"""Text-continuation backends for the agent loop.

A backend turns (system, user) into a prompt prefix and continues many
prefixes at once, stopping each one at the first stop string. Batching is what
makes the multi-turn agent affordable on one GPU: every active session advances
one step per batched call.

  hf     Hugging Face Transformers, greedy, left-padded batches, stop strings
  vllm   vLLM offline engine, greedy (optional, faster on A100)
  mock   deterministic scripted model for tests and offline demos
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re

log = logging.getLogger(__name__)


def _cut(text: str, stops: list[str]) -> str:
    """Keep text up to and including the earliest stop string."""
    best = None
    for s in stops:
        i = text.find(s)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, s)
    return text if best is None else text[: best[0] + len(best[1])]


class HFBackend:
    name = "hf"

    def __init__(self, model: str, batch_size: int = 16, four_bit: bool | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id, self.batch_size = model, batch_size
        if four_bit is None:
            four_bit = os.getenv("HF_4BIT") == "1"
        kw = {"dtype": torch.bfloat16, "device_map": "auto"}
        if four_bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            kw["dtype"] = torch.float16
        log.info("Loading %s (4-bit=%s) ...", model, four_bit)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.mdl = AutoModelForCausalLM.from_pretrained(model, **kw).eval()

    def format(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:  # Qwen3: built-in thinking off, our own <think> instruction applies
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prefixes: list[str], stop: list[str], max_new_tokens: int) -> list[str]:
        import torch
        from transformers import GenerationConfig
        gen = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, stop_strings=stop,
                               pad_token_id=self.tok.pad_token_id,
                               eos_token_id=self.mdl.generation_config.eos_token_id)
        outs = []
        for i in range(0, len(prefixes), self.batch_size):
            chunk = prefixes[i:i + self.batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.mdl.device)
            with torch.inference_mode():
                ids = self.mdl.generate(**enc, generation_config=gen, tokenizer=self.tok)
            new = ids[:, enc["input_ids"].shape[1]:]
            outs += [_cut(t, stop) for t in self.tok.batch_decode(new, skip_special_tokens=True)]
        return outs


class VLLMBackend:
    name = "vllm"

    def __init__(self, model: str, **_):
        from vllm import LLM
        self.model_id = model
        self.llm = LLM(model=model, dtype="bfloat16", seed=0, max_model_len=16384)
        self.tok = self.llm.get_tokenizer()

    format = HFBackend.format

    def generate(self, prefixes, stop, max_new_tokens):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop=stop,
                            include_stop_str_in_output=True)
        return [o.outputs[0].text for o in self.llm.generate(prefixes, sp, use_tqdm=False)]


class MockBackend:
    """Scripted, deterministic stand-in for a model.

    In agent mode it issues one search per tool family it has not used yet (up
    to two), then answers; the score is a hash of the case packet, nudged by the
    order condition so the order-bias code has something to measure.
    """
    name = "mock"
    model_id = "mock"

    def format(self, system: str, user: str) -> str:
        return f"<sys>{system}</sys>\n<user>{user}</user>\n<assistant>"

    def generate(self, prefixes, stop, max_new_tokens):
        return [self._one(p) for p in prefixes]

    @staticmethod
    def _one(prefix: str) -> str:
        user = prefix.split("<user>", 1)[1]
        packet = user.split("Assess whether", 1)[0]
        h = int(hashlib.md5(packet.encode()).hexdigest(), 16) % 81
        n_search = prefix.count("\n<information>")
        agent = "<search>TOOL: QUERY</search>" in prefix
        perf_first = "(1) performance" in prefix
        if agent and n_search < 2:
            q = ["history: receivables", "benchmark: soft assets"][n_search]
            if perf_first:
                q = ["history: sales", "benchmark: ROA"][n_search]
            return f"<think>Check {q.split(': ')[1]}.</think>\n<search>{q}</search>"
        score = h + (-5 if perf_first else 5 if "(1) earnings quality" in prefix else 0)
        score = max(0, min(100, score + 10))
        ans = {"risk_score": score, "red_flag": score >= 50, "warning_type": "none",
               "rationale": "mock output"}
        return f"<think>mock reasoning</think>\n<answer>{json.dumps(ans)}</answer>"


class ClaudeBackend:
    """Marker for the Claude path (the loop calls the Messages API itself)."""
    name = "claude"

    def __init__(self, model: str, **_):
        self.model_id = model


def make_backend(kind: str, model: str, **kw):
    if kind == "hf":
        return HFBackend(model, **kw)
    if kind == "vllm":
        return VLLMBackend(model, **kw)
    if kind == "mock":
        return MockBackend()
    if kind == "claude":
        return ClaudeBackend(model)
    raise ValueError(kind)


SEARCH_RE = re.compile(r"<search>\s*([A-Za-z_]+)\s*:\s*(.+?)\s*</search>", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
