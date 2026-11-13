"""Zero-shot LLM baseline on the same out-of-sample test firm-years.

Each sampled test firm-year becomes an anonymised, numbers-only case packet:
  - no gvkey, no calendar year (years shown as t-1 and t)
  - all dollar amounts multiplied by a random per-case constant, so absolute
    size cannot be used to look a firm up (ratios are unaffected)
The model returns JSON: risk_score (0-100), red_flag, warning_type, rationale.

Prompt conditions c:
  direct  - answer immediately with the JSON
  cot     - reason step by step through the numbers first, then answer
(The narrative-first vs. numbers-first CoT conditions need the 8-K / MD&A
text pipeline and are added for the full study.)

Backends:
  claude  - Anthropic Messages API (needs ANTHROPIC_API_KEY). Current models
            (e.g. claude-sonnet-5-5) reject temperature/top_p/top_k and always
            run adaptive thinking. They also refuse requests to write out
            reasoning, so for Claude: Direct = answer only at --effort (low);
            CoT = analysis-order instruction, answer only, at --cot-effort
            (high), with Anthropic's thinking summaries logged.
  hf      - open model in-process via Hugging Face Transformers, greedy decoding
  mock    - deterministic fake model, for testing the pipeline offline

Usage:
  python -m src.llm_baseline --backend claude --model claude-sonnet-5-5 --n 20
  python -m src.llm_baseline --backend hf --model Qwen/Qwen3-8B --n 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
log = logging.getLogger("llm_baseline")

LABELS = {
    "act": "Current assets", "che": "Cash and short-term investments",
    "rect": "Receivables", "invt": "Inventory", "at": "Total assets",
    "ppegt": "PP&E (gross)", "ivao": "Other investments and advances",
    "ivst": "Short-term investments", "lct": "Current liabilities",
    "ap": "Accounts payable", "dlc": "Debt in current liabilities",
    "txp": "Income taxes payable", "dltt": "Long-term debt", "lt": "Total liabilities",
    "ceq": "Common equity", "pstk": "Preferred stock", "re": "Retained earnings",
    "sale": "Sales", "cogs": "Cost of goods sold", "dp": "Depreciation and amortization",
    "xint": "Interest expense", "txt": "Income taxes", "ib": "Income before extraordinary items",
    "ni": "Net income", "dltis": "Long-term debt issued", "sstk": "Stock issued",
    "csho": "Shares outstanding (millions)", "prcc_f": "Share price, fiscal year end",
}
DOLLAR_VARS = [v for v in data.RAW_VARS if v not in ("csho", "prcc_f")]
RATIO_LABELS = {
    "dch_wc": "Change in working capital accruals", "ch_rsst": "Change in RSST accruals",
    "dch_rec": "Change in receivables / avg assets", "dch_inv": "Change in inventory / avg assets",
    "soft_assets": "Soft assets / total assets", "ch_cs": "Change in cash sales",
    "ch_cm": "Change in cash margin", "ch_roa": "Change in ROA",
    "issue": "Securities issued this year (1 = yes)", "bm": "Book-to-market",
    "dpi": "Depreciation index", "reoa": "Retained earnings / total assets",
    "EBIT": "EBIT / total assets", "ch_fcf": "Change in free cash flow",
}

SYSTEM = ("You are a forensic accountant screening annual financial statements "
          "for signs of material misstatement (accounting fraud).")
INSTRUCTIONS = {
    "direct": "Do not explain your reasoning. Reply with the JSON object only.",
    "cot": ("First reason step by step through the numbers: revenue and margin "
            "trends, accruals vs. cash flow, receivables and inventory build-up, "
            "soft assets, and financing activity. Write this analysis inside "
            "<reasoning>...</reasoning>. Then output the JSON object."),
}
# Current Claude models refuse requests to write out step-by-step reasoning
# (refusal category "reasoning_extraction", Anthropic commercial terms). For
# Claude the CoT condition therefore instructs the ORDER of analysis, raises the
# thinking effort, and asks for the answer only; Anthropic's own thinking
# summaries (thinking.display = "summarized") are logged for auditing.
CLAUDE_INSTRUCTIONS = {
    "direct": INSTRUCTIONS["direct"],
    "cot": ("Before deciding, work through the numbers in this order: revenue and "
            "margin trends, accruals vs. cash flow, receivables and inventory "
            "build-up, soft assets, and financing activity. Reply with the JSON "
            "object only."),
}
SCHEMA = ('{"risk_score": <integer 0-100, probability-like risk of material '
          'misstatement>, "red_flag": <true|false>, "warning_type": "<revenue '
          'recognition | expense manipulation | asset overstatement | liability '
          'understatement | cash-flow classification | none>", "rationale": '
          '"<one sentence>"}')


def sample_cases(test: pd.DataFrame, prior: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Balanced sample of test firm-years that have a prior-year record."""
    m = test.merge(prior, on=["gvkey", "fyear_prev"], suffixes=("", "_prev"), how="inner")
    per = n // 2
    s = pd.concat([m[m.misstate == 1].sample(per, random_state=seed),
                   m[m.misstate == 0].sample(n - per, random_state=seed)])
    return s.sample(frac=1, random_state=seed).reset_index(drop=True)


def build_packet(row: pd.Series, rng: np.random.Generator) -> str:
    scale = float(rng.uniform(0.5, 2.0))  # per-case anonymisation constant
    lines = ["Selected financial data (amounts in scaled currency units):",
             f"{'Item':<36}{'Year t-1':>14}{'Year t':>14}"]
    for v in LABELS:  # balance sheet, income statement, financing order
        cur, prev = row[v], row[f"{v}_prev"]
        if v in DOLLAR_VARS:
            cur, prev = cur * scale, prev * scale
        fmt = lambda x: "n/a" if pd.isna(x) else f"{x:,.2f}"
        lines.append(f"{LABELS[v]:<36}{fmt(prev):>14}{fmt(cur):>14}")
    lines += ["", "Derived ratios for year t:"]
    for v in data.RATIO_VARS:
        x = row[v]
        lines.append(f"  {RATIO_LABELS[v]:<40}{'n/a' if pd.isna(x) else f'{x:.4f}'}")
    return "\n".join(lines)


def make_prompt(packet: str, condition: str, backend: str = "hf") -> str:
    instr = (CLAUDE_INSTRUCTIONS if backend == "claude" else INSTRUCTIONS)[condition]
    return (f"{packet}\n\nAssess whether this company's year-t financial statements "
            f"are likely to contain a material misstatement.\n{instr}\n"
            f"JSON format:\n{SCHEMA}")


# ---------------------------------------------------------------- backends
_CLIENT = {}


def call_claude(model: str, prompt: str, max_tokens: int, effort: str | None,
                summarize_thinking: bool = False) -> dict:
    """Anthropic Messages API, current-generation request shape.

    No sampling parameters are sent (removed for current models). Adaptive
    thinking is always on; `effort` sets its depth. With summarize_thinking,
    the API returns Anthropic's summaries of the thinking (thinking.display =
    "summarized"), which are logged for auditing.
    """
    import anthropic
    client = _CLIENT.setdefault("claude", anthropic.Anthropic())
    kw = dict(model=model, max_tokens=max_tokens, system=SYSTEM,
              messages=[{"role": "user", "content": prompt}])
    extra = {}
    if effort:
        extra["output_config"] = {"effort": effort}
    if summarize_thinking:
        extra["thinking"] = {"type": "adaptive", "display": "summarized"}
    try:
        r = client.messages.create(**kw, **extra)
    except TypeError:  # older SDK without these typed parameters
        r = client.messages.create(**kw, extra_body=extra) if extra else client.messages.create(**kw)
    details = getattr(r, "stop_details", None)
    if details is not None and hasattr(details, "model_dump"):
        details = details.model_dump()
    return {"text": "".join(b.text for b in r.content if b.type == "text"),
            "thinking_blocks": sum(b.type == "thinking" for b in r.content),
            "thinking_summary": "\n".join(getattr(b, "thinking", "") or ""
                                           for b in r.content if b.type == "thinking") or None,
            "stop_reason": r.stop_reason, "stop_details": details,
            "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}


def call_hf(model: str, prompt: str, max_tokens: int) -> dict:
    """Open model in-process with Hugging Face Transformers, greedy decoding.

    Set HF_4BIT=1 to load in 4-bit (needed for an 8B model on a 16 GB T4).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    if model not in _CLIENT:
        log.info("Loading %s with transformers ...", model)
        kw = {"dtype": torch.bfloat16, "device_map": "auto"}
        if os.getenv("HF_4BIT") == "1":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            kw["dtype"] = torch.float16
        tok = AutoTokenizer.from_pretrained(model)
        mdl = AutoModelForCausalLM.from_pretrained(model, **kw).eval()
        # Greedy decoding: a clean config with no sampling fields at all
        gen = GenerationConfig(do_sample=False, max_new_tokens=max_tokens,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id,
                               eos_token_id=mdl.generation_config.eos_token_id)
        _CLIENT[model] = (tok, mdl, gen)
    tok, mdl, gen = _CLIENT[model]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    try:  # Qwen3: disable built-in thinking so only our CoT instruction applies
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(mdl.device)
    with torch.inference_mode():
        out = mdl.generate(**ids, generation_config=gen)
    new = out[0, ids["input_ids"].shape[1]:]
    return {"text": tok.decode(new, skip_special_tokens=True),
            "input_tokens": int(ids["input_ids"].shape[1]), "output_tokens": int(new.shape[0])}


def call_mock(prompt: str) -> dict:
    h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
    score = h % 101
    return {"text": json.dumps({"risk_score": score, "red_flag": score >= 50,
                                "warning_type": "none", "rationale": "mock output"})}


def parse(text: str) -> dict | None:
    """Extract the answer JSON: drop reasoning blocks, then try JSON objects from last to first."""
    body = re.sub(r"<(think|reasoning)>.*?</\1>", "", text or "", flags=re.S)
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", body, flags=re.S))):
        try:
            out = json.loads(m.group(0))
            out["risk_score"] = float(out["risk_score"])
            return out
        except (ValueError, KeyError, TypeError):
            continue
    return None


def main(backend: str, model: str, n: int, conditions: list[str], seed: int,
         max_tokens: int, effort: str | None, cot_effort: str | None) -> None:
    RESULTS.mkdir(exist_ok=True)
    df = data.load()
    _, test = data.split(df)
    test = test.assign(fyear_prev=test.fyear - 1)
    prior = df[data.RAW_VARS + ["gvkey", "fyear"]].rename(columns={"fyear": "fyear_prev"})
    cases = sample_cases(test, prior, n, seed)
    log.info("Sampled %d test cases (%d fraud / %d non-fraud)", len(cases),
             int(cases.misstate.sum()), int((cases.misstate == 0).sum()))
    if backend == "claude":
        log.info("Claude request: model=%s, effort direct=%s / cot=%s, no sampling parameters",
                 model, effort, cot_effort)

    tag = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")
    out_path = RESULTS / f"llm_{backend}_{tag}.jsonl"
    rng = np.random.default_rng(seed)
    packets = [build_packet(r, rng) for _, r in cases.iterrows()]

    n_fail = 0
    with out_path.open("w") as f:
        for i, row in cases.iterrows():
            case_id = f"case_{i:03d}"
            for c in conditions:
                prompt = make_prompt(packets[i], c, backend)
                t0 = time.time()
                if backend == "claude":
                    res = call_claude(model, prompt, max_tokens,
                                      cot_effort if c == "cot" else effort,
                                      summarize_thinking=(c == "cot"))
                elif backend == "hf":
                    res = call_hf(model, prompt, max_tokens)
                else:
                    res = call_mock(prompt)
                parsed = parse(res["text"])
                n_fail += parsed is None
                if res.get("stop_reason") == "refusal":
                    log.warning("%s [%s] refusal: %s", case_id, c, res.get("stop_details"))
                rec = {"case_id": case_id, "gvkey": int(row.gvkey), "fyear": int(row.fyear),
                       "label": int(row.misstate), "backend": backend, "model": model,
                       "condition": c,
                       "effort": (cot_effort if c == "cot" else effort) if backend == "claude" else None,
                       "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest()[:10],
                       "risk_score": parsed["risk_score"] if parsed else None,
                       "red_flag": parsed.get("red_flag") if parsed else None,
                       "warning_type": parsed.get("warning_type") if parsed else None,
                       "rationale": parsed.get("rationale") if parsed else None,
                       "latency_s": round(time.time() - t0, 2),
                       **{k: v for k, v in res.items() if k != "text"},
                       "raw_output": res["text"]}
                f.write(json.dumps(rec) + "\n")
                log.info("%s [%-6s] label=%d -> risk=%s flag=%s (%.1fs)", case_id, c,
                         rec["label"], rec["risk_score"], rec["red_flag"], rec["latency_s"])
    (RESULTS / "example_prompt.txt").write_text(make_prompt(packets[0], conditions[-1], backend))
    log.info("Wrote %s (%d parse failures)", out_path.name, n_fail)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "anthropic", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["claude", "hf", "mock"], default="mock")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--n", type=int, default=20, help="number of test cases (balanced)")
    ap.add_argument("--conditions", nargs="+", default=["direct", "cot"],
                    choices=list(INSTRUCTIONS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="Claude: limit on thinking + answer; HF: max new tokens")
    ap.add_argument("--effort", default="low",
                    choices=["low", "medium", "high", "xhigh", "max", "none"],
                    help="Claude thinking depth for Direct ('none' = omit, use model default)")
    ap.add_argument("--cot-effort", default="high",
                    choices=["low", "medium", "high", "xhigh", "max", "none"],
                    help="Claude thinking depth for CoT")
    a = ap.parse_args()
    a.effort = None if a.effort == "none" else a.effort
    a.cot_effort = None if a.cot_effort == "none" else a.cot_effort
    main(**vars(a))
