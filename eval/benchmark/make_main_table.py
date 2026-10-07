#!/usr/bin/env python
"""Aggregate the benchmark result files into the main results table (CSV).

Rows, their result files and parameter counts come from make_main_table.yaml. Protocol (as in
pooled_oof_table.py): per task the mean over CV folds of AUROC / AUPRC / balanced accuracy / Pearson /
Spearman, then the mean over tasks (multi-seed rows: over seeds x tasks); R2 is computed once per task from
the pooled out-of-fold predictions (<file>.oof.csv), then averaged over tasks. Every row is scored on the
same tasks: those of the reference row in the current eval suite (index.json). A row missing any of them, or
with different CV folds on one (a run in progress, an older task definition), is left empty with a warning.

    python make_main_table.py                        # -> results/main_table.csv + results/main_table_extra.csv
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import hydra
import torch
from omegaconf import DictConfig
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
INDEX = HERE.parent / "data" / "benchmark" / "index.json"
MEAN_METRICS = [("auroc", "classification"), ("auprc", "classification"), ("balanced_accuracy", "classification"),
                ("pearson", "regression"), ("spearman", "regression")]
COLS = [m for m, _ in MEAN_METRICS] + ["r2"]
# supplementary table (main_table_extra.csv): the other logged metrics + per-fold cost, same rows and tasks
EXTRA = {"classification": ["f1_macro", "f1_weighted", "accuracy", "mcc", "nll", "brier", "ece"],
         "regression": ["mape"]}
COST = ["train_seconds", "peak_ram_mb", "peak_vram_mb"]


def _read(path, tt):
    d = pd.read_csv(HERE / path)
    return d[d.task_type == tt] if "task_type" in d.columns else d


def load_row(row):
    """{tt: [per-file DataFrame]} for a row, or None when any listed file is missing / no files listed."""
    out = {}
    for tt, key in (("classification", "clf"), ("regression", "reg")):
        files = list(row.get(key) or [])
        if not files or not all((HERE / f).exists() for f in files):
            return None
        out[tt] = [_read(f, tt) for f in files]
    return out


def pooled_r2(path, tasks):
    """Per-task pooled-OOF R2 for one regression result file, or None without its .oof.csv companion."""
    f = HERE / path.replace(".csv", ".oof.csv")
    if not f.exists():
        return None
    o = pd.read_csv(f)
    o = o[o.task_id.isin(tasks)]
    return pd.Series({t: r2_score(s.y_true, s.pred) for t, s in o.groupby("task_id") if s.y_true.std() > 0})


def count_params(paths, exclude=()):
    """Mean parameter count over checkpoints, skipping state-dict entries under the `exclude` prefixes
    (weights stored in the checkpoint but never used by the model)."""
    counts = []
    for p in paths:
        sd = torch.load(HERE / p, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        counts.append(sum(v.numel() for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()
                          and not any(k.startswith(e) for e in exclude)))
    return float(np.mean(counts))


def fmt_params(n):
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "--"
    for unit, s in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= unit:
            return f"{n / unit:.3g}{s}"
    return f"{n:.0f}"


@hydra.main(version_base="1.3", config_path=".", config_name="make_main_table")
def main(cfg: DictConfig) -> None:
    index = {t["task_id"]: t["task_type"] for t in json.load(open(INDEX))["tasks"]}
    rows = [dict(r) for r in cfg.rows]
    data = {r["name"]: load_row(r) for r in rows}
    have = [n for n, d in data.items() if d is not None]
    assert cfg.reference_row in have, f"reference row {cfg.reference_row} has no results"

    # task set = the current-suite tasks of the reference row. Every other row must cover each of them with
    # identical CV folds (same n_train/n_test per fold); a row that does not (a run still in progress, or one
    # scored on an older task definition) is reported as incomplete (empty) with a warning.
    tasks, incomplete = {}, {}
    for tt in ("classification", "regression"):
        ref = data[cfg.reference_row][tt][0]
        ref = ref[ref.task_id.map(index) == tt].set_index(["task_id", "fold"])[["n_train", "n_test"]]
        tasks[tt] = sorted(ref.index.get_level_values(0).unique())
        for n in have:
            for d in data[n][tt]:
                j = ref.join(d.set_index(["task_id", "fold"])[["n_train", "n_test"]], rsuffix="_o", how="left")
                bad = j[(j.n_train != j.n_train_o) | j.n_train_o.isna() | (j.n_test != j.n_test_o)]
                if len(bad):
                    incomplete.setdefault(n, set()).update(bad.index.get_level_values(0))
    for n, ts in incomplete.items():
        print(f"WARNING: {n} is missing or has different folds on {len(ts)} task(s) "
              f"(e.g. {sorted(ts)[:3]}) -> reported empty", flush=True)
    have = [n for n in have if n not in incomplete]
    assert cfg.reference_row in have, f"reference row {cfg.reference_row} is itself incomplete"

    # a row with several result files (seeds) is scored per seed -- each seed's mean over tasks -- and
    # reported as the mean over seeds with the between-seed SD (sample SD, ddof=1) alongside
    scores, sds = {}, {}
    for n in have:
        s, sd = {}, {}
        for m, tt in MEAN_METRICS:
            per_seed = [d[d.task_id.isin(tasks[tt])].groupby("task_id")[m].mean().mean() for d in data[n][tt]]
            s[m], sd[m] = float(np.mean(per_seed)), (float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None)
        r2 = [pooled_r2(f, tasks["regression"]) for f in next(r for r in rows if r["name"] == n)["reg"]]
        if all(x is not None for x in r2):
            per_seed = [x.mean() for x in r2]
            s["r2"], sd["r2"] = float(np.mean(per_seed)), (float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None)
        else:
            s["r2"], sd["r2"] = None, None
        scores[n], sds[n] = s, sd
    params = {r["name"]: (count_params(r["params_ckpt"], tuple(r.get("params_exclude") or ()))
                          if r.get("params_ckpt") else r.get("params")) for r in rows}

    names = [r["name"] for r in rows]
    n_clf, n_reg = len(tasks["classification"]), len(tasks["regression"])

    # one row per model, <metric> = mean (over seeds for multi-seed
    # rows), <metric>_sd = between-seed SD (empty for single-run rows), empty metrics = not available
    recs, section = [], None
    for r in rows:
        section = r.get("section", section)
        n, s, sd = r["name"], scores.get(r["name"], {}), sds.get(r["name"], {})
        rec = {"model": n, "section": section, "params": params[n],
               "n_seeds": len(data[n]["classification"]) if n in have else 0}
        for m in COLS:
            rec[m], rec[f"{m}_sd"] = s.get(m), sd.get(m)
        recs.append(rec)
    csv = HERE / cfg.out_csv
    pd.DataFrame(recs).assign(n_clf_tasks=n_clf, n_reg_tasks=n_reg).to_csv(csv, index=False)
    # supplementary: mean over folds -> tasks -> seeds (NaN where a model did not log the column, e.g. no GPU)
    extra = []
    for r in rows:
        rec = {"model": r["name"]}
        for tt, pre in (("classification", "clf"), ("regression", "reg")):
            for m in EXTRA[tt] + COST:
                rec[f"{pre}_{m}"] = (float(np.mean([d[d.task_id.isin(tasks[tt])].groupby("task_id")[m].mean().mean()
                                                    if m in d else np.nan for d in data[r["name"]][tt]]))
                                     if r["name"] in have else None)
        extra.append(rec)
    pd.DataFrame(extra).to_csv(HERE / cfg.out_extra_csv, index=False)
    print(f"wrote {csv} + {Path(cfg.out_extra_csv).name} | {n_clf} clf + {n_reg} reg tasks | "
          f"rows with results: {len(have)}/{len(rows)}")
    for n in names:
        s = scores.get(n)
        print(f"  {n:<24} {fmt_params(params[n]):>7}  " + ("  ".join(
            (f"{m}={s[m]:.3f}" + (f"({sds[n][m]:.3f})" if sds[n].get(m) is not None else ""))
            if s[m] is not None else f"{m}=--" for m in COLS) if s else "no results"))


if __name__ == "__main__":
    main()
