#!/usr/bin/env python
"""Aggregate GeneICL benchmark results with the paper's metric protocol.

  - mean-of-folds (mof) for AUROC / balanced_accuracy / AUPRC / Pearson / Spearman
  - pooled out-of-fold (pooled-OOF) for R2 ONLY

Why R2 is the exception: R2 = 1 - SS_res/SS_tot has SS_tot (the target variance) in the DENOMINATOR,
computed per fold over ~1/5 of the samples. A fold that happens to draw low-variance targets gives a tiny
SS_tot and an exploding negative R2 — a split artifact, not a model property — that then dominates the
fold average. Pooling every fold's held-out (out-of-fold) prediction into one vector and scoring R2 once
uses the full-dataset variance as a stable, well-estimated denominator, and matches R2's meaning (fraction
of the dataset's variance explained by cross-validated predictions). Additive metrics are pooling-invariant;
AUROC/Pearson etc. are read mean-of-folds (pooling depresses AUROC via cross-fold calibration drift).

Reads the benchmark.py outputs in <dir>: <tag>_{classification,regression}.csv plus the per-sample
companions <tag>_regression.oof.csv (pooled-OOF R2) and <tag>_classification.clfoof.npz. Multi-seed models
pass their per-seed tags; each cell is the mean over seeds x common tasks.

  python pooled_oof_table.py dir=results 'models=[["GeneICL (segmented)",geneicl_trm_s0,geneicl_trm_s1,geneicl_trm_s2],["GeneICL (segmented, ens k=32)",geneicl_trm_supp_thresh_k32_s0,geneicl_trm_supp_thresh_k32_s1,geneicl_trm_supp_thresh_k32_s2]]'
  (keep the whole models=[...] override on ONE line: Hydra's override grammar rejects newlines)

Defaults: CONFIG in this file.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B                                            # reuse the exact clf_metrics


def mof(D, tag, tt, metric):
    f = D / f"{tag}_{tt}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if metric not in d.columns:
        return None
    d = d.dropna(subset=[metric])
    return d.groupby("task_id")[metric].mean() if len(d) else None


def pooled_clf(D, tag, metric):
    f = D / f"{tag}_classification.clfoof.npz"
    if not f.exists():
        return None
    by = {}
    for r in np.load(f, allow_pickle=True)["data"]:
        r = r if isinstance(r, dict) else r.item()
        by.setdefault(r["task_id"], []).append((np.asarray(r["y_true"], int), np.asarray(r["proba"], float)))
    out = {}
    for tid, folds in by.items():
        K = int(max(max(y.max() + 1 for y, _ in folds), max(P.shape[1] for _, P in folds)))
        ys, Ps = [], []
        for y, P in folds:
            if P.shape[1] < K:
                P = np.concatenate([P, np.zeros((len(P), K - P.shape[1]))], 1)
            Ps.append(P / np.clip(P.sum(1, keepdims=True), 1e-12, None)); ys.append(y)
        m = B.clf_metrics(np.concatenate(ys), np.concatenate(Ps, 0), K)
        if np.isfinite(m.get(metric, np.nan)):
            out[tid] = m[metric]
    return pd.Series(out) if out else None


def pooled_reg_r2(D, tag):
    f = D / f"{tag}_regression.oof.csv"
    if not f.exists():
        return None
    o = pd.read_csv(f); out = {}
    for tid, s in o.groupby("task_id"):
        if s.y_true.std() > 0:
            out[tid] = r2_score(s.y_true.values, s.pred.values)      # pool all folds' OOF preds, score once
    return pd.Series(out) if out else None


def val(getter, tags, keep):
    vals = []
    for t in tags:
        s = getter(t)
        if s is None:
            continue
        use = sorted(set(keep) & set(s.index)) if keep else sorted(s.index)
        if use:
            vals.append(s.loc[use].values)
    return float(np.concatenate(vals).mean()) if vals else float("nan")


CLF_HEADERS = {"auroc": "AUROC", "balanced_accuracy": "BalAcc", "auprc": "AUPRC", "f1_macro": "F1mac",
               "f1_weighted": "F1wtd", "accuracy": "Acc", "mcc": "MCC", "nll": "NLL", "brier": "Brier", "ece": "ECE"}
REG_DISPLAY = [("pearson", "Pearson", "reg"), ("spearman", "Spearman", "reg"), ("r2", "R2(pool)", "reg")]


CONFIG = {                        # defaults; override on the CLI as key=value (Hydra)
    "dir": "results",             # benchmark.py output directory
    # REQUIRED list of [display name, tag1, tag2, ...] (one entry per model; tags = per-seed runs), e.g.
    # 'models=[["GeneICL",geneicl_trm_s0,geneicl_trm_s1,geneicl_trm_s2],[CatBoost,catboost]]'
    "models": "???",
    # any of: auroc balanced_accuracy auprc f1_macro f1_weighted accuracy mcc nll brier ece (all mean-of-folds)
    "clf_metrics": ["auroc", "balanced_accuracy", "auprc"],
}
ConfigStore.instance().store(name="pooled_oof_table", node=CONFIG)


@hydra.main(version_base="1.3", config_path=None, config_name="pooled_oof_table")
def main(cfg: DictConfig) -> None:
    D = Path(cfg.dir)
    models = [(str(m[0]), [str(t) for t in m[1:]]) for m in cfg.models]
    assert all(tags for _, tags in models), "each models entry needs [display name, tag1, ...]"
    unknown = set(cfg.clf_metrics) - set(CLF_HEADERS)
    assert not unknown, f"unknown clf_metrics {sorted(unknown)}; choose from {list(CLF_HEADERS)}"
    DISPLAY = [(m, CLF_HEADERS[m], "clf") for m in cfg.clf_metrics] + REG_DISPLAY

    def common(tt, metric):
        idx = [set(mof(D, t, tt, metric).index) for _, tags in models for t in tags
               if mof(D, t, tt, metric) is not None]
        return set.intersection(*idx) if idx else set()
    CCLF, CREG = common("classification", "auroc"), common("regression", "pearson")

    print("\nmean-of-folds for every metric except R2, which is pooled-OOF (NLL/Brier/ECE: lower is better)")
    print(f"CLF n={len(CCLF)} tasks | REG n={len(CREG)} tasks\n")
    hdr = f"{'Model':<40} " + " ".join(f"{h:>9}" for _, h, _ in DISPLAY)
    print(hdr); print("-" * len(hdr))
    for name, tags in models:
        row = {}
        for key, _, kind in DISPLAY:
            if kind == "clf":
                row[key] = val(lambda t, k=key: mof(D, t, "classification", k), tags, CCLF)
            elif key == "r2":
                row[key] = val(lambda t: pooled_reg_r2(D, t), tags, CREG)
            else:
                row[key] = val(lambda t, k=key: mof(D, t, "regression", k), tags, CREG)
        print(f"{name:<40} " + " ".join(f"{row[k]:>9.4f}" for k, _, _ in DISPLAY))


if __name__ == "__main__":
    main()
