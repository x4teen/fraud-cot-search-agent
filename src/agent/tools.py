"""Retrieval sources for the search agent.

Three leakage-safe sources, all built only from information available at the
end of fiscal year t (the year under review):

  history    the same firm's earlier years (t-1 ... t-3), scaled with the same
             per-case anonymisation constant as the case packet
  benchmark  where one of the firm's year-t ratios sits in the cross-sectional
             distribution of the training window (1991-2001, unlabelled stats)
  cards      short forensic-accounting technique cards (original paraphrases),
             retrieved by dense (E5) or, as a fallback, TF-IDF similarity

Each tool returns (text, ids) so every retrieved item is logged by ID.
"""
from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import data
from ..llm_baseline import DOLLAR_VARS, LABELS, RATIO_LABELS

log = logging.getLogger(__name__)
CARDS_FILE = data.DATA_DIR / "cards" / "technique_cards.jsonl"

# Items analysts look at first when they lead with "performance" vs. "quality".
# Used to tag each agent query, so query focus can be compared across orders.
PERFORMANCE_ITEMS = {"sale", "cogs", "ib", "ni", "ch_roa", "EBIT", "ch_cm", "bm", "prcc_f", "csho"}
QUALITY_ITEMS = {"rect", "invt", "dch_wc", "ch_rsst", "dch_rec", "dch_inv", "soft_assets",
                 "ch_cs", "ch_fcf", "dpi", "ppegt", "dp", "ivao", "ap", "txp", "txt"}

# synonyms -> variable code (lower-case keys)
_ALIASES = {**{v.lower(): k for k, v in LABELS.items()},
            **{v.lower(): k for k, v in RATIO_LABELS.items()},
            **{k.lower(): k for k in list(LABELS) + list(RATIO_LABELS)},
            "revenue": "sale", "sales": "sale", "receivable": "rect", "receivables": "rect",
            "accounts receivable": "rect", "inventories": "invt", "inventory": "invt",
            "cogs": "cogs", "cost of sales": "cogs", "net income": "ni", "earnings": "ib",
            "depreciation": "dp", "debt": "dltt", "long-term debt": "dltt", "cash": "che",
            "payables": "ap", "accounts payable": "ap", "equity": "ceq", "pp&e": "ppegt",
            "ppe": "ppegt", "roa": "ch_roa", "accruals": "ch_rsst", "rsst accruals": "ch_rsst",
            "working capital accruals": "dch_wc", "free cash flow": "ch_fcf", "fcf": "ch_fcf",
            "cash sales": "ch_cs", "cash margin": "ch_cm", "soft assets": "soft_assets",
            "book to market": "bm", "book-to-market": "bm", "depreciation index": "dpi",
            "issuance": "issue", "ebit": "EBIT", "taxes": "txt", "interest": "xint"}


def resolve_item(query: str) -> str | None:
    """Map a free-text item name to a variable code (exact, alias, then fuzzy)."""
    q = query.strip().lower().strip(".:\"' ")
    if q in _ALIASES:
        return _ALIASES[q]
    hit = difflib.get_close_matches(q, list(_ALIASES), n=1, cutoff=0.75)
    return _ALIASES[hit[0]] if hit else None


def item_focus(code: str | None) -> str:
    if code in PERFORMANCE_ITEMS:
        return "performance"
    if code in QUALITY_ITEMS:
        return "quality"
    return "other"


# ------------------------------------------------------------------ cards
class CardRetriever:
    """Top-k retrieval over technique cards. E5 if available, else TF-IDF.

    Both are deterministic: same query -> same ranked IDs.
    """

    def __init__(self, path: Path = CARDS_FILE, encoder: str = "auto",
                 model_name: str = "intfloat/e5-base-v2"):
        self.cards = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        self.ids = [c["id"] for c in self.cards]
        docs = [f"{c['title']}. {c['text']}" for c in self.cards]
        self.kind = None
        if encoder in ("auto", "e5"):
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.emb = self.model.encode([f"passage: {d}" for d in docs],
                                             normalize_embeddings=True)
                self.kind = f"e5:{model_name}"
            except Exception as e:  # no package / no download -> fallback
                if encoder == "e5":
                    raise
                log.warning("E5 unavailable (%s); using TF-IDF retriever", type(e).__name__)
        if self.kind is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(docs)
            self.emb = self.vec.transform(docs)
            self.kind = "tfidf"
        log.info("CardRetriever: %d cards, encoder=%s", len(self.cards), self.kind)

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self.kind == "tfidf":
            sims = (self.vec.transform([query]) @ self.emb.T).toarray()[0]
        else:
            q = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
            sims = self.emb @ q
        order = sorted(range(len(sims)), key=lambda i: (-sims[i], self.ids[i]))[:k]
        return [self.cards[i] for i in order]


# ------------------------------------------------------------------ benchmark
class Benchmark:
    """Percentiles of each variable over training-window firm-years (no labels)."""

    def __init__(self, df: pd.DataFrame, years: tuple[int, int] = data.TRAIN_YEARS):
        ref = df[df.fyear.between(*years)].copy()
        # size-free versions of dollar items so firms of any scale compare
        ref = _with_scaled_items(ref)
        self.ref = {c: np.sort(ref[c].dropna().to_numpy()) for c in ref.columns
                    if c in RATIO_LABELS or c.endswith("_at") or c.endswith("_growth")}
        self.years = years

    def lookup(self, code: str, value: float) -> str:
        arr = self.ref.get(code)
        if arr is None or len(arr) == 0 or pd.isna(value):
            return f"No benchmark available for '{code}'."
        pct = 100.0 * np.searchsorted(arr, value, side="right") / len(arr)
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        p = int(round(pct))
        suf = "th" if 10 <= p % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(p % 10, "th")
        return (f"{code} = {value:.4f} is at the {p}{suf} percentile of "
                f"{len(arr):,} firm-years in {self.years[0]}-{self.years[1]} "
                f"(25th {q25:.4f}, median {q50:.4f}, 75th {q75:.4f}).")


def _with_scaled_items(frame: pd.DataFrame) -> pd.DataFrame:
    """Add scale-free variables: item / total assets and one-year growth."""
    out = frame.copy()
    for v in ("rect", "invt", "che", "ppegt", "ivao", "ap", "lt", "dltt", "sale", "ni"):
        out[f"{v}_at"] = out[v] / out["at"].where(out["at"] > 0)
    if "sale_prev" in out:
        out["sale_growth"] = out["sale"] / out["sale_prev"].where(out["sale_prev"] > 0) - 1
    return out.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------ per-case toolbox
@dataclass
class Case:
    case_id: str
    gvkey: int
    fyear: int
    label: int
    scale: float
    packet: str
    row: pd.Series = field(repr=False)


class ToolBox:
    """All tools for one case. Holds only pre-period data for that firm."""

    TOOLS = ("history", "benchmark", "cards")

    def __init__(self, case: Case, df: pd.DataFrame, bench: Benchmark,
                 cards: CardRetriever, top_k: int = 3, history_years: int = 3):
        self.case, self.bench, self.cards, self.top_k = case, bench, cards, top_k
        firm = df[(df.gvkey == case.gvkey)
                  & df.fyear.between(case.fyear - history_years, case.fyear)]
        # leakage guard: nothing after year t
        assert (firm.fyear <= case.fyear).all()
        self.hist = firm.set_index("fyear").sort_index()

    # each method returns (text, ids)
    def history(self, query: str) -> tuple[str, list[str]]:
        code = resolve_item(query)
        if code is None or code not in self.hist.columns:
            return (f"Unknown item '{query}'. Try e.g. sales, receivables, inventory, "
                    f"net income, soft assets, accruals.", [])
        t = self.case.fyear
        parts = []
        for y, val in self.hist[code].items():
            if code in DOLLAR_VARS and pd.notna(val):
                val = val * self.case.scale
            rel = "t" if y == t else f"t-{t - y}"
            parts.append(f"{rel}: {'n/a' if pd.isna(val) else f'{val:,.3f}'}")
        name = LABELS.get(code) or RATIO_LABELS.get(code, code)
        return f"{name} by year — " + "; ".join(parts), [f"H:{code}"]

    def benchmark(self, query: str) -> tuple[str, list[str]]:
        code = resolve_item(query)
        if code is None:
            return f"Unknown item '{query}'.", []
        row = _with_scaled_items(self.case.row.to_frame().T.assign(
            sale_prev=self.case.row.get("sale_prev"))).iloc[0]
        key = code if code in RATIO_LABELS else f"{code}_at"
        if key not in self.bench.ref:
            return f"No benchmark for '{query}'. Benchmarks exist for ratios and item/total assets.", []
        return self.bench.lookup(key, float(row[key])), [f"B:{key}"]

    def cards_search(self, query: str) -> tuple[str, list[str]]:
        hits = self.cards.search(query, self.top_k)
        txt = "\n".join(f"[{c['id']}] {c['title']}: {c['text']}" for c in hits)
        return txt, [c["id"] for c in hits]

    def run(self, tool: str, query: str) -> tuple[str, list[str], str | None]:
        """Dispatch. Returns (text, ids, resolved item code or None)."""
        tool = tool.strip().lower()
        if tool == "history":
            t, ids = self.history(query)
            return t, ids, resolve_item(query)
        if tool == "benchmark":
            t, ids = self.benchmark(query)
            return t, ids, resolve_item(query)
        if tool == "cards":
            t, ids = self.cards_search(query)
            return t, ids, None
        return f"Unknown tool '{tool}'. Use history, benchmark or cards.", [], None

    def static_context(self, n_cards: int = 6) -> tuple[str, list[str]]:
        """Static RAG: a fixed bundle chosen without the model.

        History of the key items, benchmarks for all 14 ratios, and the top cards
        for a query built from the firm's most extreme ratios.
        """
        lines, ids = ["Firm history (scaled units):"], []
        for code in ("sale", "rect", "invt", "ni"):
            t, i = self.history(code); lines.append("  " + t); ids += i
        lines.append("Benchmarks against 1991-2001 firm-years:")
        pcts = {}
        for code in RATIO_LABELS:
            if code == "issue":  # binary; shown in the packet already
                continue
            t, i = self.benchmark(code)
            lines.append("  " + t); ids += i
            v = self.case.row.get(code)
            arr = self.bench.ref.get(code)
            if arr is not None and pd.notna(v):
                pcts[code] = abs(np.searchsorted(arr, v) / len(arr) - 0.5)
        extreme = sorted(pcts, key=lambda c: (-pcts[c], c))[:3]
        q = " ".join(RATIO_LABELS[c] for c in extreme)
        hits = self.cards.search(q, n_cards)
        lines.append("Relevant techniques:")
        lines += [f"  [{c['id']}] {c['title']}: {c['text']}" for c in hits]
        ids += [c["id"] for c in hits]
        return "\n".join(lines), ids
