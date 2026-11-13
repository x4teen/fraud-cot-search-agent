"""Pretty-print agent sessions from a results file (for the demo video).

Usage:  python -m src.show_trace results/agent_hf_Qwen-Qwen3-8B_test_demo.jsonl [--case test_000] [--condition qual-agent]
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap

COLORS = {"think": "\033[90m", "search": "\033[94m", "information": "\033[33m", "answer": "\033[92m"}
END = "\033[0m"


def show(rec: dict, width: int = 110) -> None:
    print(f"\n{'=' * width}\n{rec['case_id']}  condition={rec['condition']}  true label="
          f"{'later-AAER' if rec['label'] else 'no AAER'}  searches={rec['n_search']}  "
          f"risk={rec['risk_score']}  flag={rec['red_flag']}\n{'=' * width}")
    for tag, body in re.findall(r"<(think|search|information|answer)>(.*?)(?:</\1>|$)",
                                rec["raw_output"], flags=re.S):
        col = COLORS[tag]
        wrapped = textwrap.fill(" ".join(body.split()), width - 4, subsequent_indent="    ")
        print(f"{col}[{tag}] {wrapped}{END}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--case")
    ap.add_argument("--condition")
    ap.add_argument("--max", type=int, default=2)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.path)]
    recs = [r for r in recs if (not a.case or r["case_id"] == a.case)
            and (not a.condition or r["condition"] == a.condition)]
    for r in recs[: a.max]:
        show(r)
