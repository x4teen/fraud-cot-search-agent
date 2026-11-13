"""Sanity check of one results file before trusting (or extending) a run.

Prints the score distribution, how often the two reasoning orders gave the
same score, sample agent queries, forced/malformed rates and per-condition
run time. Reads only; writes nothing.

Usage:  python -m src.check_run results/agent_hf_Qwen-Qwen3-8B_dev.jsonl
"""
from __future__ import annotations

import argparse

import pandas as pd


def main(path: str, n_queries: int = 3) -> None:
    r = pd.read_json(path, lines=True)
    n = r.case_id.nunique()
    print(f"== {path}: {n} cases, {len(r)} sessions, parse rate "
          f"{r.risk_score.notna().mean():.0%}")

    print("\n== score distribution per condition")
    print(r.groupby("condition").risk_score.agg(["min", "median", "max", "nunique"]))

    w = r.pivot_table(index="case_id", columns="condition", values="risk_score")
    same = {k: int((w[f"perf-{k}"] == w[f"qual-{k}"]).sum())
            for k in ("none", "static", "agent")
            if {f"perf-{k}", f"qual-{k}"} <= set(w.columns)}
    print(f"\n== perf vs qual identical on how many cases (of {n}):", same)

    agent = r[r.knowledge == "agent"]
    if len(agent):
        print(f"\n== agent queries, first {n_queries} qual-agent sessions")
        for s in r[r.condition == "qual-agent"].searches.head(n_queries):
            print([f'{q["tool"]}: {q["query"]}' for q in s])
        print("\n== agent sessions: searches / forced answer / malformed steps (means)")
        print(agent.groupby("condition")[["n_search", "forced_answer", "malformed"]].mean().round(2))

    print("\n== seconds per condition (slowest session)")
    print(r.groupby("condition").latency_s.max().round(0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--n-queries", type=int, default=3)
    main(**vars(ap.parse_args()))
