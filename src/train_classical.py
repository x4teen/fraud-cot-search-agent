"""Classical baselines (lecture: 'simple solutions' + 'prior work').

  B0  Random scores                  - sanity-check floor (AUC ~ 0.50)
  B1  Logistic regression, 14 ratios - Dechow et al. (2011) F-score style
  B2  XGBoost, 28 raw variables      - gradient-boosted trees on Bao's inputs
                                       (stand-in for Bao et al.'s RUSBoost,
                                       published test AUC ~ 0.71)

Protocol (no test data is touched during model selection):
  1. Validation: fit each hyperparameter setting on 1991-1997 and score
     2000-2001 (2-year gap, same design as the test split); pick by val AUC.
  2. Test: refit the selected setting on 1991-2001, score 2003-2008 once.

Training snapshot (CP3): every run writes
  results/logs/train_classical.log           full training log
  results/training_curves/*.csv              per-round loss / AUC / AUPRC
  results/checkpoints/B2_xgboost_raw28/      XGBoost checkpoint every 50 rounds
  results/*.joblib                           final fitted models
  results/figures/training_curves.png        loss and validation curves

Usage:  python -m src.train_classical [--quick]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from xgboost.callback import TrainingCheckPoint

from . import data
from .metrics import ranking_metrics

# Small, pre-declared grids (kept small on purpose: few frauds per year)
LR_GRID = [{"C": c} for c in (0.01, 0.1, 1.0)]
XGB_GRID = [{"max_depth": d, "n_estimators": n} for d in (3, 4, 6) for n in (200, 400)]

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
log = logging.getLogger("train_classical")


def make_model(name: str, params: dict, pos_w: float, seed: int, callbacks=None):
    if name == "B1_logreg_ratios":
        return make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=2000, **params))
    return XGBClassifier(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=pos_w, eval_metric=["logloss", "auc", "aucpr"],
                         random_state=seed, n_jobs=-1, callbacks=callbacks, **params)


SPECS = {"B1_logreg_ratios": (data.RATIO_VARS, LR_GRID),
         "B2_xgboost_raw28": (data.RAW_VARS, XGB_GRID)}
CURVES = RESULTS / "training_curves"
CKPT = RESULTS / "checkpoints"


def xgb_curve(model, names=("train", "val")) -> pd.DataFrame:
    """Per-round metrics recorded by XGBoost for each eval set."""
    ev = model.evals_result()
    out = {}
    for key, name in zip(sorted(ev), names):
        for metric, vals in ev[key].items():
            out[f"{name}_{metric}"] = vals
    df = pd.DataFrame(out)
    df.index.name = "round"
    return df.reset_index().assign(round=lambda d: d["round"] + 1)


def plot_curves(paths: dict[str, Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    final = pd.read_csv(paths["B2_final"])
    b2v = pd.read_csv(paths["B2_val"])
    ax[0].plot(b2v["round"], b2v["train_logloss"], label="fit 1991-97")
    ax[0].plot(b2v["round"], b2v["val_logloss"], label="val 2000-01")
    ax[0].set(title="B2 XGBoost: log loss", xlabel="boosting round"); ax[0].legend()
    ax[1].plot(b2v["round"], b2v["val_auc"], label="val AUC")
    ax[1].plot(b2v["round"], b2v["val_aucpr"] / b2v["val_aucpr"].max() * b2v["val_auc"].max(),
               ls="--", label="val AUPRC (rescaled)")
    ax[1].set(title="B2 XGBoost: validation ranking", xlabel="boosting round"); ax[1].legend()
    ax[2].plot(final["round"], final["train_logloss"], color="C2")
    ax[2].set(title="B2 final refit 1991-2001: training loss", xlabel="boosting round")
    plt.tight_layout()
    (RESULTS / "figures").mkdir(exist_ok=True)
    plt.savefig(RESULTS / "figures" / "training_curves.png", dpi=150)
    plt.close(fig)


def _subsample(part: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    neg = part[part.misstate == 0]
    return pd.concat([part[part.misstate == 1], neg.sample(min(n, len(neg)), random_state=seed)])


def main(quick: bool = False, seed: int = 42) -> None:
    for d in (RESULTS, CURVES, CKPT, RESULTS / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(RESULTS / "logs" / "train_classical.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)
    specs = SPECS
    df = data.load()

    # ---------------- 1. model selection on the validation split
    subtrain, val = data.validation_split(df)
    if quick:
        subtrain = _subsample(subtrain, 10000, seed)
    y_st, y_va = subtrain.misstate.to_numpy(), val.misstate.to_numpy()
    pos_w = (y_st == 0).sum() / max(1, (y_st == 1).sum())
    val_rows, best = [], {}
    curve_paths = {}
    for name, (cols, grid) in specs.items():
        X_st, X_va = data.preprocess(subtrain, val, cols)
        for params in grid:
            m = make_model(name, params, pos_w, seed).fit(X_st, y_st)
            r = ranking_metrics(y_va, m.predict_proba(X_va)[:, 1])
            val_rows.append({"model": name, **params, "val_AUC": r["AUC"], "val_AUPRC": r["AUPRC"]})
            log.info("[val] %s %s -> AUC %.4f  AUPRC %.4f", name, params, r["AUC"], r["AUPRC"])
        rows = [v for v in val_rows if v["model"] == name]
        top = max(rows, key=lambda v: v["val_AUC"])
        best[name] = {k: top[k] for k in grid[0]}
        log.info("[val] selected %s: %s (val AUC %.4f)", name, best[name], top["val_AUC"])
        # training snapshot: per-round curves for the selected setting
        if name.startswith("B2"):
            m = make_model(name, best[name], pos_w, seed)
            m.fit(X_st, y_st, eval_set=[(X_st, y_st), (X_va, y_va)], verbose=False)
            curve_paths["B2_val"] = CURVES / "B2_xgboost_fit_vs_val.csv"
            xgb_curve(m, ("train", "val")).to_csv(curve_paths["B2_val"], index=False)
        if name.startswith("B2"):
            c = pd.read_csv(curve_paths["B2_val"])
            for _, r in c.iloc[::max(1, len(c) // 8)].iterrows():
                log.info("[curve] %s round %4d | %s", name, int(r["round"]),
                         "  ".join(f"{k} {v:.4f}" for k, v in r.items() if k != "round"))
    pd.DataFrame(val_rows).to_csv(RESULTS / "classical_validation.csv", index=False)

    # ---------------- 2. refit on full training window, evaluate test once
    train, test = data.split(df)
    if quick:  # small batch for a fast live demo; test set is never subsampled
        train = _subsample(train, 15000, seed)
        log.info("--quick: training on %d firm-years", len(train))
    y_tr, y_te = train.misstate.to_numpy(), test.misstate.to_numpy()
    pos_w = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())

    rng = np.random.default_rng(seed)
    results = {"B0_random": ranking_metrics(y_te, rng.random(len(y_te)))}
    scores = pd.DataFrame({"gvkey": test.gvkey.values, "fyear": test.fyear.values,
                           "misstate": y_te})

    for name, (cols, _) in specs.items():
        X_tr, X_te = data.preprocess(train, test, cols)
        cb = None
        if name.startswith("B2"):
            (CKPT / name).mkdir(parents=True, exist_ok=True)
            cb = [TrainingCheckPoint(directory=str(CKPT / name), name="round", interval=50)]
        model = make_model(name, best[name], pos_w, seed, callbacks=cb)
        t0 = time.time()
        log.info("[%s] training on %d x %d with %s ...", name, *X_tr.shape, best[name])
        if name.startswith("B2"):  # final refit: training loss only, test never seen
            model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr)], verbose=False)
            curve_paths["B2_final"] = CURVES / f"{name}_final_train.csv"
            xgb_curve(model, ("train",)).to_csv(curve_paths["B2_final"], index=False)
            model.save_model(CKPT / name / "final.json")
        else:
            model.fit(X_tr, y_tr)
        log.info("[%s] trained in %.1fs; scoring %d test firm-years",
                 name, time.time() - t0, len(X_te))
        s = model.predict_proba(X_te)[:, 1]
        scores[name] = s
        results[name] = {**ranking_metrics(y_te, s), "params": best[name]}
        joblib.dump(model, RESULTS / f"{name}.joblib")

    scores.to_csv(RESULTS / "classical_test_scores.csv", index=False)
    if "B2_val" in curve_paths:
        plot_curves(curve_paths)
    (RESULTS / "classical_metrics.json").write_text(json.dumps(results, indent=2))

    vbest = pd.DataFrame(val_rows).groupby("model").val_AUC.max()
    table = pd.DataFrame(results).T[["AUC", "AUPRC", "precision@top1%",
                                     "recall@top1%", "NDCG@top1%"]].astype(float)
    table.insert(0, "val_AUC", vbest.reindex(table.index))
    print("\n=== Validation 2000-2001 (selection) and test 2003-2008 (out of sample) ===")
    print(f"test n = {len(y_te):,} firm-years, {int(y_te.sum())} frauds, "
          f"prevalence = {y_te.mean():.4f} (= AUPRC of a random ranker)\n")
    print(table.round(4).to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="subsample training set")
    main(**vars(ap.parse_args()))
