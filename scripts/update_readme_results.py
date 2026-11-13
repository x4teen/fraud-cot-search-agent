"""Paste the latest evaluation tables and figures into README.md (between the RESULTS markers)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"


def fmt(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda x: "–" if pd.isna(x) else f"{x:.3f}")
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values)


def main() -> None:
    parts = ["**Detection** (95% bootstrap CIs in brackets; classical models thresholded at the "
             "median score, LLMs at the τ fit on the dev split):\n",
             (T / "comparison.md").read_text(),
             "\n![Comparison](results/figures/comparison.png)\n"]
    ob = T / "order_bias.csv"
    if ob.exists() and ob.stat().st_size > 5:
        d = pd.read_csv(ob)[["model", "knowledge", "n_pairs", "mean_delta", "delta_lo", "delta_hi",
                             "mean_abs_delta", "reversal_rate", "wilcoxon_p"]]
        parts += ["\n**Reasoning-order effect** Δ = risk(perf-first) − risk(quality-first):\n", fmt(d),
                  "\n![Order effect](results/figures/order_delta.png)\n"]
    sg = T / "search_gain.csv"
    if sg.exists() and sg.stat().st_size > 5:
        parts += ["\n**Knowledge gain** (paired bootstrap):\n", fmt(pd.read_csv(sg))]
    qf = T / "query_focus.csv"
    if qf.exists() and qf.stat().st_size > 5:
        parts += ["\n**What the agent looked up**, by reasoning order:\n", fmt(pd.read_csv(qf))]
    block = "\n".join(parts)
    readme = (ROOT / "README.md").read_text()
    a, b = readme.index("<!-- RESULTS:START -->"), readme.index("<!-- RESULTS:END -->")
    readme = readme[: a + len("<!-- RESULTS:START -->")] + "\n" + block + "\n" + readme[b:]
    (ROOT / "README.md").write_text(readme)
    print("README results section updated.")


if __name__ == "__main__":
    main()
