"""Prompt conditions for the CP3 experiment.

Condition name = <order>-<knowledge>:
  order c      direct | perf | qual
               perf = performance-first CoT (growth, margins, ROA: the headline
                      story management leads with), then earnings quality
               qual = quality-first CoT (accruals, cash, receivables, inventory,
                      soft assets), then performance
               The two orders use the same words; only the order of the two
               analysis blocks differs.
  knowledge k  none | static | agent
               static = a fixed retrieval bundle inserted into the prompt
               agent  = the model issues its own <search> calls (budget B)

`direct-none` is the CP2 Direct baseline.
"""
from __future__ import annotations

from ..llm_baseline import SCHEMA, SYSTEM

ORDERS = ("direct", "perf", "qual")
KNOWLEDGE = ("none", "static", "agent")
CONDITIONS = ["direct-none", "perf-none", "qual-none", "perf-static", "qual-static",
              "perf-agent", "qual-agent"]

PERF_BLOCK = ("performance: sales growth, gross margin, ROA and EBIT, and how they "
              "compare with prior years")
QUAL_BLOCK = ("earnings quality: accruals versus cash flow, receivables and inventory "
              "relative to sales, soft assets, and free cash flow")
FIN_BLOCK = "financing: debt, securities issuance and valuation"

ORDER_TEXT = {
    "perf": f"(1) {PERF_BLOCK}; (2) {QUAL_BLOCK}; (3) {FIN_BLOCK}",
    "qual": f"(1) {QUAL_BLOCK}; (2) {PERF_BLOCK}; (3) {FIN_BLOCK}",
}

TOOL_HELP = """You can look things up before answering. To search, write exactly one line
<search>TOOL: QUERY</search>
and stop. The result will appear inside <information>...</information>. Tools:
  history: ITEM     the same company's values for years t-3 to t (e.g. history: receivables)
  benchmark: ITEM   where the company's year-t ratio sits among all firm-years in a
                    reference period (e.g. benchmark: soft assets)
  cards: TEXT       forensic-accounting techniques relevant to a pattern
                    (e.g. cards: receivables growing faster than sales)
You may search at most {budget} times. Search only when it helps your judgment."""


def split(condition: str) -> tuple[str, str]:
    order, knowledge = condition.split("-")
    assert order in ORDERS and knowledge in KNOWLEDGE, condition
    return order, knowledge


def build_prompt(packet: str, condition: str, budget: int = 4,
                 static_context: str | None = None, native_tools: bool = False) -> str:
    """User prompt. `native_tools` = Claude (tools passed through the API)."""
    order, knowledge = split(condition)
    parts = [packet]
    if knowledge == "static":
        parts.append("Reference material (retrieved for this case):\n" + (static_context or ""))
    parts.append("Assess whether this company's year-t financial statements are likely "
                 "to contain a material misstatement.")
    if knowledge == "agent":
        parts.append("You have lookup tools; you may use them up to "
                     f"{budget} times." if native_tools else TOOL_HELP.format(budget=budget))
    if order == "direct":
        parts.append("Do not explain your reasoning. Reply with the JSON object only, "
                     "inside <answer>...</answer>.")
    elif native_tools:
        # Claude does not write out its reasoning; the order is an instruction only
        parts.append(f"Before deciding, work through the evidence in this order: "
                     f"{ORDER_TEXT[order]}. Reply with the JSON object only, inside "
                     f"<answer>...</answer>.")
    else:
        parts.append(f"Reason inside <think>...</think>, working through the evidence in "
                     f"this order: {ORDER_TEXT[order]}. Then give the JSON object inside "
                     f"<answer>...</answer>.")
    parts.append(f"JSON format:\n{SCHEMA}")
    return "\n\n".join(parts)


__all__ = ["SYSTEM", "CONDITIONS", "ORDERS", "KNOWLEDGE", "build_prompt", "split"]
