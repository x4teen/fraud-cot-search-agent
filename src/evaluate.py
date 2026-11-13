"""Evaluate all baselines on the same sampled test cases.

Reads results/llm_*.jsonl and results/classical_test_scores.csv, aligns them on
(gvkey, fyear), and reports ranking and threshold metrics per model x condition,
plus the paired score difference between CoT and Direct.

Usage:  python -m src.evaluate [--tau 50]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .metrics import ranking_metrics, threshold_metrics

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main(tau: float) -> None:
    files = sorted(RESULTS.glob("llm_*.jsonl"))
    if not files:
        raise SystemExit("No LLM results found. Run src.llm_baseline first.")
    llm = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    n_bad = llm.risk_score.isna().sum()
    if "stop_reason" in llm.columns:
        ref = llm[llm.stop_reason == "refusal"].groupby(["model", "condition"]).size()
        if len(ref):
            print("Refusals by model x condition:\n" + ref.to_string())
    llm = llm.dropna(subset=["risk_score"])

    cpath = RESULTS / "classical_test_scores.csv"
    if not cpath.exists():
        raise SystemExit("No classical scores found. Run src.train_classical first.")
    classical = pd.read_csv(cpath)
    cases = llm[["gvkey", "fyear", "label"]].drop_duplicates()
    sub = cases.merge(classical, on=["gvkey", "fyear"], how="left")

    rows = []
    for col in [c for c in classical.columns if c.startswith("B")]:
        r = ranking_metrics(sub.label, sub[col])
        # classical probabilities are calibrated to ~0.7% prevalence, so use the
        # median score on this balanced subset as the threshold
        r.update(threshold_metrics(sub.label, sub[col], sub[col].median()))
        rows.append({"model": col, "condition": "-", **r})
    for (model, cond), g in llm.groupby(["model", "condition"]):
        r = ranking_metrics(g.label, g.risk_score)
        r.update(threshold_metrics(g.label, g.risk_score, tau))
        rows.append({"model": model, "condition": cond, **r})

    table = pd.DataFrame(rows)
    cols = ["model", "condition", "n", "n_fraud", "AUC", "AUPRC",
            "precision", "recall", "F1", "FPR", "score_sep"]
    print(f"\n=== Evaluation on {len(cases)} sampled out-of-sample test cases "
          f"({int(cases.label.sum())} fraud) ===")
    print(f"LLM red-flag threshold tau = {tau}; parse failures dropped: {n_bad}\n")
    print(table[cols].round(3).to_string(index=False))

    # Paired effect of the prompting condition (preview of the H1 analysis)
    wide = llm.pivot_table(index=["model", "case_id"], columns="condition",
                           values="risk_score")
    if {"direct", "cot"} <= set(wide.columns):
        d = (wide["cot"] - wide["direct"]).dropna()
        print("\nPaired delta = risk(CoT) - risk(Direct), by model:")
        print(d.groupby(level="model").agg(["mean", "median", "std", "count"])
              .round(2).to_string())

    table.to_csv(RESULTS / "evaluation_summary.csv", index=False)
    print(f"\nSaved {RESULTS / 'evaluation_summary.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=50.0)
    main(**vars(ap.parse_args()))
