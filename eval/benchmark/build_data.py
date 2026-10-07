#!/usr/bin/env python
"""Freeze the GenePFN eval datasets into eval/data/benchmark/ (the benchmark; run once).

Provenance only: reuses the task registry + data loaders of the upstream curation code (GenePFN eval/, not part
of this repo; point GENEPFN_EVAL at it) and reimplements no label logic. Produces a self-contained eval/data/benchmark/ that the benchmark reads with no
scratch/parent dependency:

    reference_genes.txt               20021 gene symbols (the frozen column order)
    datasets/<dataset>.npz            X (f16, n x 20021 log1p-CPM, ref-aligned) + obs_names   [deduped]
    tasks/<task_id>.npz               sample_ids, y, task_type, [group], dataset
    index.json                        task list + dataset row counts + n_genes

All 20021 reference genes are stored. ~2 GB f16 for the full suite.

Hydra CLI (build_data.yaml):
    python eval/benchmark/build_data.py                 # all tasks
    python eval/benchmark/build_data.py task=CCLE       # substring filter
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

HERE = Path(__file__).resolve().parent          # eval/benchmark/
PARENT_EVAL = Path(os.environ["GENEPFN_EVAL"])   # the upstream curation code (not shipped)
sys.path.insert(0, str(PARENT_EVAL))

from common import (densify, gene_index, load_adata, load_config, read_gene_list,  # noqa: E402
                    reindex, to_log_cpm)
from process_data import list_tasks, load_task  # noqa: E402

# GSE81538's source expression is log2, not log1p-CPM like the rest: exponentiate then re-log1p.
LOG2_DATASETS = {"GSE81538"}
# Tiny tasks dropped from the suite (too few rows to be informative).
DROP_TASKS = {"GSE183250::metastasis", "GSE289743::pcr"}


@hydra.main(version_base="1.3", config_path=".", config_name="build_data")
def main(cfg: DictConfig) -> None:
    pcfg = load_config()                                 # parent GenePFN/eval config (curated paths etc.)
    pcfg["ccle_drugs"] = []                               # freeze ALL 20 CCLE drugs, not the parent's default 10
    ref = read_gene_list(pcfg["gene_reference"])         # 20021 genes, trained column order
    data = HERE.parent / "data" / "benchmark"
    (data / "datasets").mkdir(parents=True, exist_ok=True)
    (data / "tasks").mkdir(parents=True, exist_ok=True)
    (data / "reference_genes.txt").write_text("\n".join(ref) + "\n")

    tasks = list_tasks(pcfg)
    if cfg.task:
        tasks = [t for t in tasks if cfg.task in t.task_id]
    print(f"[build] {len(tasks)} tasks | {len(ref)} reference genes -> {data}", flush=True)

    frozen_datasets: dict[str, int] = {}     # dataset -> n_rows already written
    index_tasks = []

    for meta in tasks:
        if meta.task_id in DROP_TASKS:
            print(f"[drop-task] {meta.task_id}: excluded from suite", flush=True)
            continue
        try:
            _, df = load_task(meta.task_id, pcfg)
        except Exception as e:
            print(f"[skip-task] {meta.task_id}: {type(e).__name__}: {e}", flush=True)
            continue

        # freeze the dataset expression once (deduped): X = log1p-CPM over the FULL gene set, then
        # reindex to the 20021 reference genes (GenePFN column order).
        if meta.dataset not in frozen_datasets:
            adata = load_adata(pcfg, meta.dataset)
            obs_names = list(adata.obs_names)
            idx = gene_index(list(adata.var_names), ref)
            expr = densify(adata, obs_names)
            if meta.dataset in LOG2_DATASETS:
                lin = np.power(2.0, expr)                          # log2 -> linear
                expr = np.log1p(lin / lin.sum(1, keepdims=True) * 1e6)   # CPM -> log1p
            else:
                expr = to_log_cpm(expr)
            X = reindex(expr, idx).astype(np.float16)
            assert X.shape[1] == len(ref) and np.isfinite(X).all(), f"{meta.dataset}: bad X"
            np.savez(data / "datasets" / f"{meta.dataset}.npz",
                     X=X, obs_names=np.array(obs_names, dtype=object))
            frozen_datasets[meta.dataset] = X.shape[0]
            print(f"[dataset] {meta.dataset}: {X.shape[0]}x{X.shape[1]} f16 "
                  f"({X.nbytes/1e6:.0f} MB) | {int((idx >= 0).sum())}/{len(ref)} genes matched", flush=True)

        # keep only rows present in the frozen dataset matrix
        obs_set = set(np.load(data / "datasets" / f"{meta.dataset}.npz",
                              allow_pickle=True)["obs_names"].tolist())
        df = df[df.sample_id.isin(obs_set)].reset_index(drop=True)
        if len(df) == 0:
            print(f"[skip-task] {meta.task_id}: no rows overlap the frozen matrix", flush=True)
            continue

        payload = dict(sample_ids=df.sample_id.to_numpy(dtype=object),
                       y=df.value.to_numpy(),
                       task_type=meta.task_type, dataset=meta.dataset)
        if "group" in df.columns:
            payload["group"] = df["group"].to_numpy(dtype=object)
        np.savez(data / "tasks" / f"{meta.task_id.replace('/', '_')}.npz", **payload)
        index_tasks.append(dict(task_id=meta.task_id, dataset=meta.dataset,
                                task_type=meta.task_type, n=len(df),
                                file=f"tasks/{meta.task_id.replace('/', '_')}.npz"))
        print(f"[task]    {meta.task_id}: {len(df)} rows ({meta.task_type})", flush=True)

    (data / "index.json").write_text(json.dumps(
        dict(n_genes=len(ref), datasets=frozen_datasets, tasks=index_tasks), indent=2))
    total_mb = sum(f.stat().st_size for f in (data / "datasets").glob("*.npz")) / 1e6
    print(f"[build] done: {len(index_tasks)} tasks, {len(frozen_datasets)} datasets, "
          f"{total_mb:.0f} MB frozen -> {data/'index.json'}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
