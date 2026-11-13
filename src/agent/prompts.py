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

TOOL_HELP = """You can look things up while you reason. To search, write one search line and stop.
The first word is the tool name (history, benchmark or cards), then a colon, then what you
want to look up. Examples of valid searches:
<search>history: receivables</search>
<search>benchmark: soft assets</search>
<search>cards: receivables growing faster than sales</search>
The tools:
- history: the same company's values of one item for years t-3 to t
- benchmark: the percentile of the company's year-t value of one ratio among all
  firm-years in a reference period
- cards: forensic-accounting techniques relevant to a pattern you describe
The result appears inside <information>...</information>; then continue reasoning.
You may search at most {budget} times. Search only when it would change your judgment,
and never repeat a search you already made."""


# The model's output is pre-filled with the opening of its reasoning, so every CoT
# session really starts with the block its order prescribes (instructions alone were
# ignored by Qwen3-8B on the dev split: it answered without reasoning).
PREFILL = {"perf": "<reasoning>\n(1) Performance:", "qual": "<reasoning>\n(1) Earnings quality:"}


def prefill(condition: str) -> str:
    return PREFILL.get(split(condition)[0], "")


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
        parts.append(f"Reason step by step inside <reasoning>...</reasoning>, working through "
                     f"the evidence in this order: {ORDER_TEXT[order]}. Then give the JSON "
                     f"object inside <answer>...</answer>.")
    parts.append(f"JSON format:\n{SCHEMA}")
    return "\n\n".join(parts)


__all__ = ["SYSTEM", "CONDITIONS", "ORDERS", "KNOWLEDGE", "build_prompt", "prefill", "split"]
