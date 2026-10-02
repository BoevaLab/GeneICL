#!/usr/bin/env python
"""Shared driver of the zero-shot FM embedding extractors (*_extract.py): one <out_dir>/<dataset>.npz per dataset.

The COMPASS, BulkFormer, scFoundation and BulkRNABert extractors run in one venv outside the conda envs (py3.12):
    python3.12 -m venv external/fm-venv
    external/fm-venv/bin/pip install immuno-compass torch_geometric performer-pytorch einops local-attention \
        transformers hydra-core
scgpt_extract.py needs its own venv (see its docstring).
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
REPO = Path(__file__).resolve().parents[3]                       # repo root
WEIGHTS = REPO / "eval" / "benchmark" / "weights"     # FM checkpoints (see weights/README.md)
EXTERNAL = REPO / "external"                                     # gitignored: upstream model repos + venvs
EMB_DIR = REPO / "eval" / "benchmark" / "embeddings"  # gitignored: <EMB_DIR>/<model>/<dataset>.npz


# ── zero-shot embedding extraction driver ─────────────────────────────────────────────────────────
def run_extract(model_name, data_dir, out_dir, embed_dataset, datasets=None):
    """Extract sample-level embeddings for every dataset and save <out_dir>/<dataset>.npz with
    emb (n, D) + obs_names. `embed_dataset(ds)` returns (emb (n,D) float32, obs_names list)."""
    os.makedirs(out_dir, exist_ok=True)
    if datasets is None:
        idx = json.load(open(f"{data_dir}/index.json"))
        datasets = sorted({t["dataset"] for t in idx["tasks"]})
    for ds in datasets:
        outp = f"{out_dir}/{ds}.npz"
        if os.path.exists(outp):
            print(f"[skip] {ds} (exists)", flush=True); continue
        emb, obs = embed_dataset(ds)
        np.savez(outp, emb=np.asarray(emb, np.float32), obs_names=np.asarray(obs, dtype=object))
        print(f"[ok] {ds}: emb {np.asarray(emb).shape} -> {outp}", flush=True)
    print(f"{model_name.upper()}_EXTRACT_DONE", flush=True)
