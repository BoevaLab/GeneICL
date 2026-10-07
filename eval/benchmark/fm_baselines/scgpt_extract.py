#!/usr/bin/env python
"""Zero-shot scGPT (bowang-lab/scGPT, whole-human) embeddings for the benchmark datasets -- same role as
bf_extract.py / scf_extract.py / compass_extract.py. Uses the package's own zero-shot recipe,
scgpt.tasks.embed_data: per sample, the expressed genes found in scGPT's vocabulary (up to max_length,
randomly sub-sampled beyond that), a prepended <cls> token, per-sample rank binning into 51 bins, and the
L2-normalised <cls> output as the 512-d cell embedding. Saved per dataset for the EN/logistic probe
(benchmark.py model=scgpt emb_dir=<out>).

Input: our log1p-CPM -> linear CPM over the 20,021 HGNC symbols (scGPT's binning is rank-based per sample,
so CPM vs counts does not change the tokens). Runs in its own venv, without flash-attn:
    python3.10 -m venv external/scgpt-venv
    external/scgpt-venv/bin/pip install scgpt==0.2.4 torch==2.3.1 torchtext==0.18.0 hydra-core
Weights: the gdown --folder line in weights/README.md.

    CUDA_VISIBLE_DEVICES=0 python scgpt_extract.py                        # all datasets -> <out>/<dataset>.npz
    CUDA_VISIBLE_DEVICES=0 python scgpt_extract.py datasets=[GSE107422]   # a subset (defaults: CONFIG in this file)
"""
import os
import random
from pathlib import Path

import numpy as np
import torch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("FM_DATA", REPO / "eval" / "data" / "benchmark"))


CONFIG = {                        # defaults; override on the CLI as key=value (Hydra)
    "model_dir": str(REPO / "eval" / "benchmark" / "weights" / "scgpt" / "scGPT_human"),  # args.json + best_model.pt + vocab.json
    "out": str(REPO / "eval" / "benchmark" / "embeddings" / "scgpt"),  # -> <out>/<dataset>.npz; benchmark.py emb_dir=
    "datasets": None,             # e.g. datasets=[GSE107422]; None = all in index.json
    "batch": 64,
    "max_length": 1200,           # genes per sample incl. <cls> (scGPT's pretraining length)
    "seed": 0,                    # fixes embed_data's random gene sub-sampling when a sample expresses > max_length genes
}
ConfigStore.instance().store(name="scgpt_extract", node=CONFIG)


@hydra.main(version_base="1.3", config_path=None, config_name="scgpt_extract")
def main(a: DictConfig) -> None:
    import anndata as ad
    import pandas as pd
    from scgpt.tasks import embed_data
    from fm_common import run_extract
    genes = [g.strip() for g in open(DATA / "reference_genes.txt")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def embed_dataset(ds):
        z = np.load(DATA / "datasets" / f"{ds}.npz", allow_pickle=True)
        obs = [str(o) for o in z["obs_names"]]
        adata = ad.AnnData(X=np.expm1(np.asarray(z["X"], np.float32)),
                           obs=pd.DataFrame(index=obs), var=pd.DataFrame({"gene": genes}, index=genes))
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)   # gene sub-sampling >max_length
        out = embed_data(adata, a.model_dir, gene_col="gene", max_length=a.max_length, batch_size=a.batch,
                         device=dev, use_fast_transformer=False, return_new_adata=True)
        emb = np.asarray(out.X, np.float32)
        assert emb.shape == (len(obs), 512) and np.isfinite(emb).all(), f"{ds}: bad scGPT embeddings {emb.shape}"
        return emb, obs

    run_extract("scgpt", str(DATA), a.out, embed_dataset,
                datasets=list(a.datasets) if a.datasets is not None else None)


if __name__ == "__main__":
    main()
