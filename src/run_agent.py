"""Run the CP3 experiment grid: order x knowledge conditions on the same cases.

Cases are balanced samples (half later-AAER, half not) from the out-of-sample
test window 2003-2008 (`--split test`) or, for fitting the red-flag threshold
only, from the validation window 2000-2001 (`--split dev`). Every condition
sees the same anonymised packet for a case (same scaling constant).

Usage:
  python -m src.run_agent --backend mock --n 20                          # offline check
  python -m src.run_agent --backend hf --model Qwen/Qwen3-8B --n 60
  python -m src.run_agent --backend claude --model claude-sonnet-5-5 --n 30
  python -m src.run_agent --backend hf --model Qwen/Qwen3-8B --n 4 \\
          --conditions perf-agent qual-agent --verbose --run-tag demo     # demo batch
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .agent.backends import make_backend
from .agent.loop import SearchAgent
from .agent.prompts import CONDITIONS
from .agent.tools import Benchmark, CardRetriever, Case, ToolBox
from .llm_baseline import DOLLAR_VARS, LABELS, RATIO_LABELS, sample_cases

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SAMPLES = data.DATA_DIR / "samples"
log = logging.getLogger("run_agent")


def build_packet(row: pd.Series, scale: float) -> str:
    """Same layout as the CP2 packet, with the scale constant passed in."""
    fmt = lambda x: "n/a" if pd.isna(x) else f"{x:,.2f}"
    lines = ["Selected financial data (amounts in scaled currency units):",
             f"{'Item':<36}{'Year t-1':>14}{'Year t':>14}"]
    for v in LABELS:
        cur, prev = row[v], row[f"{v}_prev"]
        if v in DOLLAR_VARS:
            cur, prev = cur * scale, prev * scale
        lines.append(f"{LABELS[v]:<36}{fmt(prev):>14}{fmt(cur):>14}")
    lines += ["", "Derived ratios for year t:"]
    for v in data.RATIO_VARS:
        x = row[v]
        lines.append(f"  {RATIO_LABELS[v]:<40}{'n/a' if pd.isna(x) else f'{x:.4f}'}")
    return "\n".join(lines)


def load_cases(df: pd.DataFrame, split: str, n: int, seed: int) -> list[Case]:
    _, ev = data.split(df) if split == "test" else data.validation_split(df)
    ev = ev.assign(fyear_prev=ev.fyear - 1)
    prior = df[data.RAW_VARS + ["gvkey", "fyear"]].rename(columns={"fyear": "fyear_prev"})
    sampled = sample_cases(ev, prior, n, seed)
    rng = np.random.default_rng(seed)
    cases = []
    for i, row in sampled.iterrows():
        row = pd.to_numeric(row, errors="coerce")
        scale = float(rng.uniform(0.5, 2.0))
        cases.append(Case(f"{split}_{i:03d}", int(row.gvkey), int(row.fyear),
                          int(row.misstate), scale, build_packet(row, scale), row))
    return cases


def main(backend: str, model: str, n: int, split: str, conditions: list[str], seed: int,
         budget: int, top_k: int, max_new_tokens: int, batch_size: int, encoder: str,
         verbose: bool, effort: str, cot_effort: str, run_tag: str = "") -> Path:
    RESULTS.mkdir(exist_ok=True)
    df = data.load()
    cases = load_cases(df, split, n, seed)
    log.info("%s split: %d cases (%d later-AAER / %d not)", split, len(cases),
             sum(c.label for c in cases), sum(1 - c.label for c in cases))

    bench, cards = Benchmark(df), CardRetriever(encoder=encoder)
    boxes = {c.case_id: ToolBox(c, df, bench, cards, top_k=top_k) for c in cases}

    # a small public sample of anonymised packets for the repo (no gvkey)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    if split == "test" and not (SAMPLES / "case_packets_sample.jsonl").exists():
        with (SAMPLES / "case_packets_sample.jsonl").open("w") as f:
            for c in cases[:4]:
                f.write(json.dumps({"case_id": c.case_id, "label": c.label, "packet": c.packet}) + "\n")

    kw = {"batch_size": batch_size} if backend == "hf" else {}
    be = make_backend(backend, model, **kw)
    agent = SearchAgent(be, budget=budget, top_k=top_k, max_new_tokens=max_new_tokens,
                        verbose=verbose)
    tag = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")
    suffix = f"_{run_tag}" if run_tag else ""  # e.g. "demo": kept out of evaluation
    out = RESULTS / f"agent_{backend}_{tag}_{split}{suffix}.jsonl"
    meta = {"backend": backend, "model": model, "split": split, "n": len(cases), "seed": seed,
            "budget": budget, "top_k": top_k, "retriever": cards.kind,
            "conditions": conditions, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (RESULTS / f"{out.stem}.meta.json").write_text(json.dumps(meta, indent=2))

    with out.open("w") as f:
        for cond in conditions:
            t0 = time.time()
            recs = (agent.run_claude(cases, cond, boxes, model, effort, cot_effort)
                    if backend == "claude" else agent.run(cases, cond, boxes))
            for r in recs:
                r["split"] = split
                f.write(json.dumps(r, default=str) + "\n")
            ok = [r for r in recs if r["risk_score"] is not None]
            log.info("[%-11s] %d/%d parsed | mean risk %.1f | mean searches %.2f | %.0fs",
                     cond, len(ok), len(recs),
                     np.mean([r["risk_score"] for r in ok]) if ok else float("nan"),
                     np.mean([r["n_search"] for r in recs]), time.time() - t0)
            if verbose:
                for r in recs:
                    log.info("  %s label=%d -> risk=%s flag=%s searches=%d", r["case_id"],
                             r["label"], r["risk_score"], r["red_flag"], r["n_search"])
    log.info("Wrote %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "anthropic", "transformers", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "vllm", "claude", "mock"], default="mock")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--n", type=int, default=60, help="balanced number of cases")
    ap.add_argument("--split", choices=["test", "dev"], default="test")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--budget", type=int, default=4, help="max searches B")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=1024, help="per generation step")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--encoder", choices=["auto", "e5", "tfidf"], default="auto")
    ap.add_argument("--effort", default="low", help="Claude effort, direct")
    ap.add_argument("--cot-effort", default="high", help="Claude effort, CoT conditions")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--run-tag", default="", help="suffix for the output file, e.g. demo")
    main(**vars(ap.parse_args()))
