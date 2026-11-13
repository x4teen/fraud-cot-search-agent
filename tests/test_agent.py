"""Unit tests for the CP3 search agent. Run: pytest -q"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import data
from src.agent.backends import SEARCH_RE, MockBackend
from src.agent.loop import SearchAgent, parse_answer
from src.agent.prompts import ORDER_TEXT, build_prompt
from src.agent.tools import Benchmark, CardRetriever, Case, ToolBox, resolve_item
from src.run_agent import build_packet


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Synthetic Bao-format panel: 30 firms x 1991-2008."""
    rng = np.random.default_rng(0)
    rows = []
    for g in range(30):
        for y in range(1991, 2009):
            r = {v: abs(rng.normal(100, 30)) for v in data.RAW_VARS}
            r.update({v: rng.normal(0, 0.1) for v in data.RATIO_VARS})
            r.update(gvkey=1000 + g, fyear=y, misstate=int(rng.random() < 0.1), p_aaer=np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def case(frame) -> Case:
    row = frame[(frame.gvkey == 1003) & (frame.fyear == 2005)].iloc[0].copy()
    prev = frame[(frame.gvkey == 1003) & (frame.fyear == 2004)].iloc[0]
    for v in data.RAW_VARS:
        row[f"{v}_prev"] = prev[v]
    return Case("test_000", 1003, 2005, int(row.misstate), 1.7, build_packet(row, 1.7), row)


@pytest.fixture(scope="module")
def box(frame, case) -> ToolBox:
    return ToolBox(case, frame, Benchmark(frame), CardRetriever(encoder="tfidf"))


class AlwaysSearch(MockBackend):
    """Never answers on its own: exercises the budget cutoff."""

    @staticmethod
    def _one(prefix: str) -> str:
        if prefix.endswith("<answer>"):
            return '{"risk_score": 42, "red_flag": false}</answer>'
        return "<think>look again</think>\n<search>history: receivables</search>"


class Garbage(MockBackend):
    """Never follows the format: exercises rethink, then forced answer."""

    @staticmethod
    def _one(prefix: str) -> str:
        if prefix.endswith("<answer>"):
            return '{"risk_score": 7, "red_flag": false}</answer>'
        return "I am not sure."


# ---------------------------------------------------------------- parser
def test_parse_answer_prefers_last_answer_block():
    text = ('<think>{"risk_score": 99}</think><answer>{"risk_score": 10}</answer>'
            '<answer>{"risk_score": 20, "red_flag": false}</answer>')
    assert parse_answer(text)["risk_score"] == 20.0


def test_parse_answer_falls_back_and_fails_cleanly():
    assert parse_answer('Result: {"risk_score": "33"}')["risk_score"] == 33.0
    assert parse_answer("no json here") is None


def test_search_regex():
    m = SEARCH_RE.search("<think>x</think>\n<search> benchmark : soft assets </search>")
    assert m.group(1) == "benchmark" and m.group(2) == "soft assets"


def test_resolve_item_aliases():
    assert resolve_item("Receivables") == "rect"
    assert resolve_item("soft assets") == "soft_assets"
    assert resolve_item("free cash flow") == "ch_fcf"
    assert resolve_item("zzzz") is None


# ---------------------------------------------------------------- tools
def test_history_has_no_future_years(box, case):
    assert box.hist.index.max() == case.fyear
    text, ids = box.history("sales")
    assert ids == ["H:sale"] and "t-3" in text and "t+1" not in text


def test_history_uses_case_scaling(box, case, frame):
    raw = frame[(frame.gvkey == case.gvkey) & (frame.fyear == case.fyear)].sale.iloc[0]
    text, _ = box.history("sales")
    assert f"t: {raw * case.scale:,.3f}" in text


def test_benchmark_uses_training_window_only(box):
    assert box.bench.years == data.TRAIN_YEARS
    text, ids = box.benchmark("soft assets")
    assert ids == ["B:soft_assets"] and "1991-2001" in text and "percentile" in text


def test_card_retrieval_is_deterministic(box):
    a = box.cards.search("receivables growing faster than sales", 3)
    b = box.cards.search("receivables growing faster than sales", 3)
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert a[0]["id"] in {"C01", "C02"}


# ---------------------------------------------------------------- prompts
def test_order_conditions_differ_only_in_order(case):
    p, q = build_prompt(case.packet, "perf-agent"), build_prompt(case.packet, "qual-agent")
    assert p != q
    assert sorted(p.split()) == sorted(q.split())  # same words, different order
    assert ORDER_TEXT["perf"].index("performance") < ORDER_TEXT["perf"].index("earnings quality")


# ---------------------------------------------------------------- loop
def test_budget_cutoff_forces_answer(case, box):
    agent = SearchAgent(AlwaysSearch(), budget=4)
    (rec,) = agent.run([case], "qual-agent", {case.case_id: box})
    assert rec["n_search"] == 4 and rec["forced_answer"] and rec["risk_score"] == 42.0
    assert [s["turn"] for s in rec["searches"]] == [1, 2, 3, 4]
    assert all(s["ids"] == ["H:rect"] for s in rec["searches"])


def test_malformed_output_rethinks_then_forces(case, box):
    agent = SearchAgent(Garbage(), budget=4, max_retries=1)
    (rec,) = agent.run([case], "perf-none", {case.case_id: box})
    assert rec["malformed"] == 2 and rec["forced_answer"] and rec["risk_score"] == 7.0
    assert "Let me rethink" in rec["raw_output"]


def test_static_condition_logs_ids_and_no_searches(case, box):
    (rec,) = SearchAgent(MockBackend()).run([case], "qual-static", {case.case_id: box})
    assert rec["n_search"] == 0 and len(rec["static_ids"]) > 10
    assert any(i.startswith("C") for i in rec["static_ids"])


def test_record_schema(case, box):
    (rec,) = SearchAgent(MockBackend()).run([case], "perf-agent", {case.case_id: box})
    for key in ("case_id", "label", "condition", "order", "knowledge", "prompt_sha1",
                "risk_score", "searches", "raw_output"):
        assert key in rec
    json.dumps(rec, default=str)
