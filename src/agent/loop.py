"""Order-controlled search agent (CP3 core method).

Adapted from the Search-R1 rollout (Jin et al., 2025, Algorithm 1) without RL
training: the frozen model interleaves <reasoning> steps with <search> calls,
retrieved text is appended inside <information> (conditioned on, not
generated), and the episode ends at <answer> or when the search budget B is
spent. The reasoning-order condition c is placed in the reasoning instruction (and
the first reasoning block is pre-filled), so
one loop serves every c.

Open models run through a text backend in batched lock-step (all active
sessions advance one step per call). Claude runs the same tools through native
API tool use, because current Claude models do not write out their reasoning.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..llm_baseline import parse as parse_cp2
from .backends import ANSWER_RE, SEARCH_RE
from .prompts import SYSTEM, build_prompt, prefill, split
from .tools import Case, ToolBox, item_focus

log = logging.getLogger(__name__)

STOPS = ["</search>", "</answer>"]
RETHINK = "\nMy previous output was not in the required format. Let me rethink.\n"


def parse_answer(text: str) -> dict | None:
    """JSON inside the last <answer>...</answer>; else the CP2 fallback parser."""
    for m in reversed(ANSWER_RE.findall(text or "")):
        try:
            out = json.loads(m.strip())
            out["risk_score"] = float(out["risk_score"])
            return out
        except (ValueError, KeyError, TypeError):
            continue
    return parse_cp2(text)


@dataclass
class Session:
    case: Case
    condition: str
    prompt: str
    prefix: str
    tools: ToolBox
    static_ids: list[str] = field(default_factory=list)
    out: str = ""
    searches: list[dict] = field(default_factory=list)
    retries: int = 0
    malformed: int = 0
    forced: bool = False
    done: bool = False
    steps: int = 0
    t0: float = field(default_factory=time.time)


class SearchAgent:
    def __init__(self, backend, budget: int = 4, top_k: int = 3, max_new_tokens: int = 1024,
                 max_retries: int = 1, verbose: bool = False):
        self.backend, self.B, self.k = backend, budget, top_k
        self.max_new, self.max_retries, self.verbose = max_new_tokens, max_retries, verbose

    # ---------------------------------------------------------------- open models
    def _new_session(self, case: Case, condition: str, tools: ToolBox) -> Session:
        _, knowledge = split(condition)
        ctx, ids = tools.static_context() if knowledge == "static" else (None, [])
        prompt = build_prompt(case.packet, condition, self.B, ctx)
        return Session(case, condition, prompt, self.backend.format(SYSTEM, prompt), tools, ids,
                       out=prefill(condition))

    def _step(self, s: Session, y: str) -> None:
        """Apply one model continuation `y` to session `s` (Algorithm 1 body)."""
        _, knowledge = split(s.condition)
        s.out += y
        s.steps += 1
        if "</answer>" in y:
            s.done = True
            return
        m = list(SEARCH_RE.finditer(y))
        if m and knowledge == "agent":
            if len(s.searches) >= self.B:  # over budget -> force the answer
                s.out += ("\n<information>Search budget exhausted. Give your final answer "
                          "now.</information>\n<answer>")
                s.forced = True
                return
            tool, query = m[-1].group(1), m[-1].group(2)
            text, ids, item = s.tools.run(tool, query)
            if tool.strip().lower() not in s.tools.TOOLS:
                s.malformed += 1
            s.searches.append({"turn": len(s.searches) + 1, "tool": tool.lower(), "query": query,
                               "item": item, "focus": item_focus(item), "ids": ids})
            s.out += f"\n<information>{text}</information>\n"
            if self.verbose:
                log.info("  %s [%s] search %d  %s: %s -> %s", s.case.case_id, s.condition,
                         len(s.searches), tool, query, ids)
            return
        # malformed: neither a valid search nor an answer
        s.malformed += 1
        if s.retries < self.max_retries and not s.forced:
            s.retries += 1
            s.out += RETHINK
        elif not s.forced:
            s.out += "\n<answer>"
            s.forced = True
        else:
            s.done = True  # forced answer still failed; give up

    def run(self, cases: list[Case], condition: str, toolboxes: dict[str, ToolBox]) -> list[dict]:
        sessions = [self._new_session(c, condition, toolboxes[c.case_id]) for c in cases]
        max_steps = self.B + self.max_retries + 3
        for step in range(max_steps):
            active = [s for s in sessions if not s.done]
            if not active:
                break
            outs = self.backend.generate([s.prefix + s.out for s in active], STOPS, self.max_new)
            for s, y in zip(active, outs):
                self._step(s, y)
            # sessions that hit the forced-answer path get one more call; any
            # session still open at the last step is closed
        for s in sessions:
            s.done = True
        return [self._record(s) for s in sessions]

    # ---------------------------------------------------------------- Claude (native tools)
    def run_claude(self, cases: list[Case], condition: str, toolboxes: dict[str, ToolBox],
                   model: str, effort_direct: str = "low", effort_cot: str = "high",
                   max_tokens: int = 16000, workers: int = 4) -> list[dict]:
        def one(case: Case) -> dict:
            return self._claude_session(case, condition, toolboxes[case.case_id], model,
                                        effort_direct, effort_cot, max_tokens)
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(one, cases))

    def _claude_session(self, case, condition, tools, model, eff_d, eff_c, max_tokens) -> dict:
        import anthropic
        client = anthropic.Anthropic()
        order, knowledge = split(condition)
        ctx, ids = tools.static_context() if knowledge == "static" else (None, [])
        prompt = build_prompt(case.packet, condition, self.B, ctx, native_tools=True)
        s = Session(case, condition, prompt, "", tools, ids)
        effort = eff_d if order == "direct" else eff_c
        tool_defs = [
            {"name": "history", "description": "The same company's values of one item for years t-3 to t.",
             "input_schema": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}},
            {"name": "benchmark", "description": "Percentile of the company's year-t value of one ratio among reference firm-years.",
             "input_schema": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}},
            {"name": "cards", "description": "Forensic-accounting technique cards relevant to a described pattern.",
             "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        ]
        msgs = [{"role": "user", "content": prompt}]
        usage = {"input_tokens": 0, "output_tokens": 0}
        stop_reason, thinking = None, []
        for _ in range(self.B + 2):
            kw = dict(model=model, max_tokens=max_tokens, system=SYSTEM, messages=msgs)
            extra = {"output_config": {"effort": effort}}
            if order != "direct":
                extra["thinking"] = {"type": "adaptive", "display": "summarized"}
            if knowledge == "agent":
                kw["tools"] = tool_defs
                if len(s.searches) >= self.B:
                    kw["tool_choice"] = {"type": "none"}
            try:
                r = client.messages.create(**kw, **extra)
            except TypeError:
                r = client.messages.create(**kw, extra_body=extra)
            usage["input_tokens"] += r.usage.input_tokens
            usage["output_tokens"] += r.usage.output_tokens
            stop_reason = r.stop_reason
            thinking += [getattr(b, "thinking", "") or "" for b in r.content if b.type == "thinking"]
            s.out += "".join(b.text for b in r.content if b.type == "text")
            calls = [b for b in r.content if b.type == "tool_use"]
            if r.stop_reason != "tool_use" or not calls:
                break
            msgs.append({"role": "assistant",
                         "content": [b.model_dump(exclude_none=True) for b in r.content]})
            results = []
            for b in calls:
                q = b.input.get("item") or b.input.get("query") or ""
                text, ids_, item = tools.run(b.name, q)
                s.searches.append({"turn": len(s.searches) + 1, "tool": b.name, "query": q,
                                   "item": item, "focus": item_focus(item), "ids": ids_})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": text})
            msgs.append({"role": "user", "content": results})
        rec = self._record(s)
        rec.update({**usage, "stop_reason": stop_reason, "effort": effort,
                    "thinking_summary": "\n".join(thinking) or None})
        return rec

    # ---------------------------------------------------------------- logging
    def _record(self, s: Session) -> dict:
        order, knowledge = split(s.condition)
        parsed = parse_answer(s.out)
        c = s.case
        return {"case_id": c.case_id, "gvkey": c.gvkey, "fyear": c.fyear, "label": c.label,
                "backend": self.backend.name, "model": self.backend.model_id,
                "condition": s.condition, "order": order, "knowledge": knowledge,
                "prompt_sha1": hashlib.sha1(s.prompt.encode()).hexdigest()[:10],
                "risk_score": parsed["risk_score"] if parsed else None,
                "red_flag": parsed.get("red_flag") if parsed else None,
                "warning_type": parsed.get("warning_type") if parsed else None,
                "rationale": parsed.get("rationale") if parsed else None,
                "n_search": len(s.searches), "searches": s.searches, "static_ids": s.static_ids,
                "malformed": s.malformed, "forced_answer": s.forced, "steps": s.steps,
                "latency_s": round(time.time() - s.t0, 2), "raw_output": s.out}
