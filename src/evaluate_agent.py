"""CP3 evaluation: baselines vs. the search agent on the same test cases.

Reads results/agent_*_test.jsonl (and *_dev.jsonl if --tau-from-dev) plus the
classical scores, and writes
  results/tables/comparison.{csv,md}   detection metrics per model x condition
  results/tables/order_bias.csv        H1: Delta = perf - qual, |Delta|, reversals
  results/tables/search_gain.csv       H3: G = AUPRC(agent) - AUPRC(static)
  results/tables/query_focus.csv       what the agent looked up, by order
  results/figures/comparison.png, order_delta.png

Uncertainty: 95% percentile bootstrap over cases (each case is one firm-year
of a distinct firm, so case resampling is company-clustered), paired across
conditions for differences.

Usage:  python -m src.evaluate_agent [--tau 50 | --tau-from-dev] [--boot 1000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import threshold_metrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES, FIGS = RESULTS / "tables", RESULTS / "figures"
ORDER = ["B0_random", "B1_logreg_ratios", "B2_xgboost_raw28", "direct-none", "perf-none",
         "qual-none", "perf-static", "qual-static", "perf-agent", "qual-agent"]


def short(model: str) -> str:
    return model.split("/")[-1]


def load(split: str) -> pd.DataFrame:
    files = sorted(RESULTS.glob(f"agent_*_{split}.jsonl"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    df["model"] = df.model.map(short)
    return df


def boot_ci(fn, idx_groups: list[np.ndarray], n: int, B: int, seed: int = 0):
    """Bootstrap a statistic fn(sample_indices) over case indices 0..n-1."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        s = rng.integers(0, n, n)
        try:
            v = fn(s)
        except ValueError:  # resample with one class only
            continue
        if np.isfinite(v):
            vals.append(v)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def fit_tau(dev: pd.DataFrame) -> dict:
    """Per (model, condition) threshold maximising F1 on the dev split."""
    taus = {}
    for (m, c), g in dev.dropna(subset=["risk_score"]).groupby(["model", "condition"]):
        cands = np.unique(g.risk_score)
        f1 = [threshold_metrics(g.label, g.risk_score, t)["F1"] for t in cands]
        taus[(m, c)] = float(cands[int(np.argmax(f1))])
    return taus


def detection_table(llm: pd.DataFrame, classical: pd.DataFrame, taus: dict, tau0: float,
                    B: int) -> pd.DataFrame:
    rows = []
    cases = llm[["case_id", "gvkey", "fyear", "label"]].drop_duplicates("case_id")
    sub = cases.merge(classical, on=["gvkey", "fyear"], how="left")
    for col in [c for c in classical.columns if c.startswith("B")]:
        y, s = sub.label.to_numpy(), sub[col].to_numpy()
        rows.append(_metrics("classical", col, y, s, float(np.median(s)), B, n_search=np.nan))
    for (m, c), g in llm.groupby(["model", "condition"]):
        g = g.sort_values("case_id")
        ok = g.dropna(subset=["risk_score"])
        r = _metrics(m, c, ok.label.to_numpy(), ok.risk_score.to_numpy(),
                     taus.get((m, c), tau0), B, n_search=g.n_search.mean())
        r["parse_rate"] = len(ok) / len(g)
        rows.append(r)
    t = pd.DataFrame(rows)
    t["_o"] = t.condition.map({k: i for i, k in enumerate(ORDER)})
    return t.sort_values(["model", "_o"]).drop(columns="_o")


def _metrics(model, cond, y, s, tau, B, n_search) -> dict:
    n = len(y)
    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    lo, hi = boot_ci(lambda i: average_precision_score(y[i], s[i]), [], n, B)
    alo, ahi = boot_ci(lambda i: roc_auc_score(y[i], s[i]), [], n, B, seed=1)
    th = threshold_metrics(y, s, tau)
    return {"model": model, "condition": cond, "n": n, "AUC": auc, "AUC_lo": alo, "AUC_hi": ahi,
            "AUPRC": ap, "AUPRC_lo": lo, "AUPRC_hi": hi, "tau": tau, "F1": th["F1"],
            "precision": th["precision"], "recall": th["recall"], "FPR": th["FPR"],
            "score_sep": th["score_sep"], "mean_searches": n_search}


def order_bias(llm: pd.DataFrame, taus: dict, tau0: float, B: int) -> pd.DataFrame:
    from scipy.stats import wilcoxon
    rows = []
    for (m, k), g in llm[llm.order.isin(["perf", "qual"])].groupby(["model", "knowledge"]):
        w = g.pivot_table(index=["case_id", "label"], columns="order", values="risk_score").dropna()
        if not {"perf", "qual"} <= set(w.columns) or len(w) < 3:
            continue
        d = (w.perf - w.qual).to_numpy()
        lab = w.index.get_level_values("label").to_numpy()
        tp, tq = taus.get((m, f"perf-{k}"), tau0), taus.get((m, f"qual-{k}"), tau0)
        rev = ((w.perf >= tp) != (w.qual >= tq)).mean()
        n = len(d)
        try:
            p = wilcoxon(d, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = np.nan
        rows.append({"model": m, "knowledge": k, "n_pairs": n,
                     "mean_delta": d.mean(), "delta_lo": boot_ci(lambda i: d[i].mean(), [], n, B)[0],
                     "delta_hi": boot_ci(lambda i: d[i].mean(), [], n, B)[1],
                     "mean_delta_fraud": d[lab == 1].mean() if (lab == 1).any() else np.nan,
                     "mean_abs_delta": np.abs(d).mean(),
                     "abs_lo": boot_ci(lambda i: np.abs(d[i]).mean(), [], n, B)[0],
                     "abs_hi": boot_ci(lambda i: np.abs(d[i]).mean(), [], n, B)[1],
                     "reversal_rate": rev, "wilcoxon_p": p})
    return pd.DataFrame(rows)


def search_gain(llm: pd.DataFrame, B: int) -> pd.DataFrame:
    """Paired bootstrap of AUPRC(agent) - AUPRC(static), and agent vs none."""
    rows = []
    for (m, o), g in llm[llm.order.isin(["perf", "qual"])].groupby(["model", "order"]):
        w = g.pivot_table(index=["case_id", "label"], columns="knowledge", values="risk_score").dropna()
        y = w.index.get_level_values("label").to_numpy()
        for base in ("static", "none"):
            if not {"agent", base} <= set(w.columns):
                continue
            a, b = w["agent"].to_numpy(), w[base].to_numpy()
            fn = lambda i: average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i])
            lo, hi = boot_ci(fn, [], len(y), B, seed=2)
            rows.append({"model": m, "order": o, "comparison": f"agent - {base}", "n": len(y),
                         "G_AUPRC": fn(np.arange(len(y))), "G_lo": lo, "G_hi": hi,
                         "dAUC": roc_auc_score(y, a) - roc_auc_score(y, b)})
    return pd.DataFrame(rows)


def query_focus(llm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in llm[llm.knowledge == "agent"].iterrows():
        for s in r.searches or []:
            rows.append({"model": r.model, "order": r.order, "turn": s["turn"],
                         "tool": s["tool"], "focus": s["focus"]})
    if not rows:
        return pd.DataFrame()
    q = pd.DataFrame(rows)
    out = (q.groupby(["model", "order"]).agg(
        n_queries=("focus", "size"),
        share_quality=("focus", lambda f: (f == "quality").mean()),
        share_performance=("focus", lambda f: (f == "performance").mean()),
        share_cards=("tool", lambda t: (t == "cards").mean())).reset_index())
    first = q[q.turn == 1].groupby(["model", "order"]).focus.apply(
        lambda f: (f == "quality").mean()).rename("first_query_quality")
    return out.merge(first.reset_index(), on=["model", "order"], how="left")


def figures(table: pd.DataFrame, llm: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    t = table.copy()
    t["label"] = np.where(t.model == "classical", t.condition, t.model + " · " + t.condition)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(t) + 1))
    colors = ["#888" if m == "classical" else ("C2" if "agent" in c else "C0")
              for m, c in zip(t.model, t.condition)]
    yy = np.arange(len(t))[::-1]
    ax.barh(yy, t.AUPRC, color=colors)
    ax.errorbar(t.AUPRC, yy, xerr=[t.AUPRC - t.AUPRC_lo, t.AUPRC_hi - t.AUPRC],
                fmt="none", ecolor="k", lw=0.8)
    prev = llm.drop_duplicates("case_id").label.mean()
    ax.axvline(prev, ls="--", color="k", lw=0.8)
    ax.text(prev, len(t) - 0.4, " random", fontsize=7)
    ax.set_yticks(yy, t.label, fontsize=8)
    ax.set_xlabel("AUPRC on balanced test cases (95% bootstrap CI)")
    ax.set_title("Baselines vs. search agent")
    plt.tight_layout(); plt.savefig(FIGS / "comparison.png", dpi=150); plt.close(fig)

    pairs = []
    for (m, k), g in llm[llm.order.isin(["perf", "qual"])].groupby(["model", "knowledge"]):
        w = g.pivot_table(index="case_id", columns="order", values="risk_score").dropna()
        if {"perf", "qual"} <= set(w.columns):
            pairs.append((f"{m}\n{k}", (w.perf - w.qual).to_numpy()))
    if pairs:
        fig, ax = plt.subplots(figsize=(1.4 * len(pairs) + 2, 3.4))
        ax.boxplot([p[1] for p in pairs], showmeans=True)
        ax.set_xticks(range(1, len(pairs) + 1), [p[0] for p in pairs], fontsize=8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Δ = risk(perf-first) − risk(quality-first)")
        ax.set_title("Reasoning-order effect by knowledge setting")
        plt.tight_layout(); plt.savefig(FIGS / "order_delta.png", dpi=150); plt.close(fig)


def to_markdown(t: pd.DataFrame) -> str:
    cols = ["model", "condition", "n", "AUC", "AUPRC", "F1", "FPR", "mean_searches"]
    d = t[cols].copy()
    d["AUC"] = [f"{a:.3f} [{l:.2f}, {h:.2f}]" for a, l, h in zip(t.AUC, t.AUC_lo, t.AUC_hi)]
    d["AUPRC"] = [f"{a:.3f} [{l:.2f}, {h:.2f}]" for a, l, h in zip(t.AUPRC, t.AUPRC_lo, t.AUPRC_hi)]
    for c in ("F1", "FPR", "mean_searches"):
        d[c] = d[c].map(lambda x: "–" if pd.isna(x) else f"{x:.2f}")
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.values)


def main(tau: float, tau_from_dev: bool, boot: int) -> None:
    llm = load("test")
    if llm.empty:
        raise SystemExit("No results/agent_*_test.jsonl found. Run src.run_agent first.")
    taus = fit_tau(load("dev")) if tau_from_dev else {}
    classical = pd.read_csv(RESULTS / "classical_test_scores.csv")
    rng = np.random.default_rng(0)
    classical["B0_random"] = rng.random(len(classical))
    TABLES.mkdir(parents=True, exist_ok=True)

    table = detection_table(llm, classical, taus, tau, boot)
    ob, sg, qf = order_bias(llm, taus, tau, boot), search_gain(llm, boot), query_focus(llm)
    table.to_csv(TABLES / "comparison.csv", index=False)
    (TABLES / "comparison.md").write_text(to_markdown(table) + "\n")
    for name, d in (("order_bias", ob), ("search_gain", sg), ("query_focus", qf)):
        d.to_csv(TABLES / f"{name}.csv", index=False)
    if taus:
        (TABLES / "tau_from_dev.json").write_text(json.dumps({f"{m}|{c}": t for (m, c), t in taus.items()}, indent=2))
    figures(table, llm)

    pd.set_option("display.width", 200)
    n = llm.case_id.nunique()
    print(f"\n=== Detection on {n} balanced out-of-sample test cases "
          f"({int(llm.drop_duplicates('case_id').label.sum())} later-AAER) ===")
    print(f"red-flag threshold: {'fit on dev split' if taus else f'tau = {tau}'}\n")
    print(table[["model", "condition", "n", "AUC", "AUPRC", "AUPRC_lo", "AUPRC_hi", "F1", "FPR",
                 "mean_searches"]].round(3).to_string(index=False))
    if not ob.empty:
        print("\n=== H1 order effect: Delta = risk(perf-first) - risk(quality-first) ===")
        print(ob.round(3).to_string(index=False))
    if not sg.empty:
        print("\n=== H3 knowledge gain (paired bootstrap) ===")
        print(sg.round(3).to_string(index=False))
    if not qf.empty:
        print("\n=== What the agent looked up ===")
        print(qf.round(3).to_string(index=False))
    print(f"\nTables in {TABLES}, figures in {FIGS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=50.0)
    ap.add_argument("--tau-from-dev", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    main(**vars(ap.parse_args()))
