"""Dataset loading and train/test split for the IDAI-780 baseline.

Source: Bao, Ke, Li, Yu & Zhang (2020), "Detecting Accounting Fraud in Publicly
Traded U.S. Firms Using a Machine Learning Approach", Journal of Accounting
Research 58(1). Public data: github.com/JarFraud/FraudDetection.
Labels (misstate = 1) come from SEC Accounting and Auditing Enforcement
Releases (AAERs). Firms are identified only by Compustat gvkey (no names).
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA_URL = ("https://raw.githubusercontent.com/JarFraud/FraudDetection/"
            "master/data_FraudDetection_JAR2020.csv")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_FILE = DATA_DIR / "data_FraudDetection_JAR2020.csv"

# 28 raw accounting variables (Bao et al. 2020, Table 1)
RAW_VARS = ["act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis",
            "dltt", "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni",
            "ppegt", "pstk", "re", "rect", "sale", "sstk", "txp", "txt",
            "xint", "prcc_f"]
# 14 financial ratios (Dechow et al. 2011 F-score family)
RATIO_VARS = ["dch_wc", "ch_rsst", "dch_rec", "dch_inv", "soft_assets",
              "ch_cs", "ch_cm", "ch_roa", "issue", "bm", "dpi", "reoa",
              "EBIT", "ch_fcf"]

# Bao et al. out-of-sample design: train on 1991-2001, 2-year gap, test 2003-2008
TRAIN_YEARS = (1991, 2001)
TEST_YEARS = (2003, 2008)
# Validation for model selection mirrors the test design inside the training
# window: fit on 1991-1997, 2-year gap (1998-1999), validate on 2000-2001
SUBTRAIN_YEARS = (1991, 1997)
VAL_YEARS = (2000, 2001)


def download(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists() and not force:
        log.info("Dataset already present: %s", DATA_FILE.name)
        return DATA_FILE
    log.info("Downloading %s", DATA_URL)
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    log.info("Saved %.1f MB", DATA_FILE.stat().st_size / 1e6)
    return DATA_FILE


def load() -> pd.DataFrame:
    df = pd.read_csv(download())
    df[RAW_VARS + RATIO_VARS] = (df[RAW_VARS + RATIO_VARS]
                                 .replace([np.inf, -np.inf], np.nan))
    log.info("Loaded %d firm-years, %d firms, %d misstatements (%.2f%%)",
             len(df), df.gvkey.nunique(), int(df.misstate.sum()),
             100 * df.misstate.mean())
    return df


def temporal_split(df: pd.DataFrame, fit_years: tuple[int, int],
                   eval_years: tuple[int, int], names=("train", "test")
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split with Bao et al.'s serial-fraud correction.

    A fraud often spans several years. If the same AAER case (p_aaer) appears in
    both the fitting and evaluation windows, its fitting firm-years are relabelled
    0 so the model cannot learn a firm's identity and 'recognise' it later.
    """
    fit = df[df.fyear.between(*fit_years)].copy()
    ev = df[df.fyear.between(*eval_years)].copy()

    eval_cases = set(ev.loc[ev.misstate == 1, "p_aaer"].dropna())
    serial = (fit.misstate == 1) & fit.p_aaer.isin(eval_cases)
    fit.loc[serial, "misstate"] = 0
    log.info("Serial-fraud correction (%s vs %s): relabelled %d firm-years",
             names[0], names[1], int(serial.sum()))

    for name, part in zip(names, (fit, ev)):
        log.info("%-8s %d-%d: %6d firm-years, %4d frauds (%.2f%%)", name,
                 part.fyear.min(), part.fyear.max(), len(part),
                 int(part.misstate.sum()), 100 * part.misstate.mean())
    return fit, ev


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Final split: train 1991-2001, test 2003-2008."""
    return temporal_split(df, TRAIN_YEARS, TEST_YEARS, ("train", "test"))


def validation_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model-selection split inside the training window: 1991-1997 vs 2000-2001."""
    return temporal_split(df, SUBTRAIN_YEARS, VAL_YEARS, ("subtrain", "val"))


def preprocess(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    """Median-impute and winsorize at the 1st/99th percentile.

    All statistics are fit on the training set only (no test leakage).
    """
    med = train[cols].median()
    lo, hi = train[cols].quantile(0.01), train[cols].quantile(0.99)

    def _apply(x: pd.DataFrame) -> np.ndarray:
        return x[cols].fillna(med).clip(lo, hi, axis=1).to_numpy(dtype=float)

    return _apply(train), _apply(test)
