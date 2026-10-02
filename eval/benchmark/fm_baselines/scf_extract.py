#!/usr/bin/env python
"""Zero-shot scFoundation (xTrimoGene-100M) embeddings for the benchmark datasets -- same role as
bf_extract.py / compass_extract.py / scgpt_extract.py. Bulk recipe from the repo (SCAD / DeepCDR:
--input_type bulk --output_type cell --pool_type all --pre_normalized F --version ce): per sample the
3072-d concat [S-token, T-token, gene max-pool, gene mean-pool] of the 768-d encoder, for the EN/logistic
probe (benchmark.py model=scfoundation emb_dir=<out>).

Input prep: our log1p-CPM -> linear CPM -> gathered into scFoundation's 19,264-gene order (absent genes 0)
-> scanpy-style normalize_total(target = median library size) + log1p, replicated in numpy -> two depth
tokens log10(total) appended (the "bulk + pre_normalized F" convention).

Paths come from CONFIG; the weights are the Hugging Face mirror of the released checkpoint
(genbio-ai/scFoundation, models.ckpt).

    CUDA_VISIBLE_DEVICES=0 python scf_extract.py                      # all datasets -> <out>/<dataset>.npz
    CUDA_VISIBLE_DEVICES=0 python scf_extract.py datasets=[GSE107422] # a subset (defaults: CONFIG in this file)

Needs: git clone https://github.com/biomap-research/scFoundation external/scFoundation &&
git -C external/scFoundation checkout 397631c, the weights
(python -c "from huggingface_hub import hf_hub_download as h; h('genbio-ai/scFoundation', 'models.ckpt',
local_dir='eval/benchmark/weights/scfoundation')") and the FM venv (fm_common.py).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from fm_common import DEV, EMB_DIR, EXTERNAL, WEIGHTS, run_extract

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("FM_DATA", REPO_ROOT / "eval" / "data" / "benchmark"))
N_GENES = 19264                                   # scFoundation gene panel (+2 depth tokens = 19266)


CONFIG = {                        # defaults; override on the CLI as key=value (Hydra)
    "repo": str(EXTERNAL / "scFoundation" / "model"),                # github.com/biomap-research/scFoundation
    "ckpt": str(WEIGHTS / "scfoundation" / "models.ckpt"),           # HF mirror genbio-ai/scFoundation
    "out": str(EMB_DIR / "scfoundation"),                            # -> <out>/<dataset>.npz; benchmark.py emb_dir=
    "datasets": None,             # e.g. datasets=[GSE107422]; None = all in index.json
}
ConfigStore.instance().store(name="scf_extract", node=CONFIG)


@hydra.main(version_base="1.3", config_path=None, config_name="scf_extract")
def main(a: DictConfig) -> None:
    sys.path.insert(0, str(a.repo))               # <clone>/model: load.py + pretrainmodels/
    from load import gatherData, load_model_frommmf

    model, cfg = load_model_frommmf(str(a.ckpt), "cell")
    model = model.to(DEV).eval()
    PAD = cfg["pad_token_id"]
    print(f"[scfoundation] loaded | pad_token_id={PAD} | encoder {cfg['encoder']['hidden_dim']}d "
          f"x{cfg['encoder']['depth']}", flush=True)

    gl = pd.read_csv(Path(a.repo) / "OS_scRNA_gene_index.19264.tsv", header=0, delimiter="\t")
    scf_genes = list(gl["gene_name"])
    assert len(scf_genes) == N_GENES, len(scf_genes)
    our = [l.strip() for l in open(DATA / "reference_genes.txt") if l.strip()]
    sym2col = {s: i for i, s in enumerate(our)}
    take = np.array([sym2col.get(g, -1) for g in scf_genes])          # -1 = gene absent from our matrix
    have = take >= 0
    print(f"[scfoundation] genes: {N_GENES} panel <- our {len(our)}: mapped {int(have.sum())}, "
          f"zero-filled {int((~have).sum())}", flush=True)
    gene_ids_full = torch.arange(N_GENES + 2, device=DEV).unsqueeze(0)

    def embed_rows(X):
        """(n, 19264) log1p-normalised expression -> (n, 3072) embeddings, one sample per forward."""
        out = []
        for i in range(len(X)):
            row = X[i]
            tot = np.log10(row.sum())                                  # bulk + pre_normalized F depth token
            gx = torch.tensor(row.tolist() + [tot, tot], device=DEV).unsqueeze(0)
            val = gx > 0
            x, x_pad = gatherData(gx, val, PAD)                         # keep expressed genes only
            pos_ids, _ = gatherData(gene_ids_full.repeat(gx.shape[0], 1), val, PAD)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                xe = model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
                xe = xe + model.pos_emb(pos_ids)
                g = model.encoder(xe, x_pad)                            # (1, L, 768)
                e1, e2 = g[:, -1, :], g[:, -2, :]                       # S / T depth tokens
                e3, _ = torch.max(g[:, :-2, :], dim=1)                  # gene max-pool
                e4 = torch.mean(g[:, :-2, :], dim=1)                    # gene mean-pool
                out.append(torch.concat([e1, e2, e3, e4], dim=1).float().cpu().numpy())
        return np.concatenate(out)

    def embed_dataset(ds):
        z = np.load(DATA / "datasets" / f"{ds}.npz", allow_pickle=True)
        obs = [str(o) for o in z["obs_names"]]
        lin = np.expm1(np.asarray(z["X"], np.float64))                  # linear CPM
        expr = np.zeros((len(lin), N_GENES), np.float64)
        expr[:, have] = lin[:, take[have]]
        counts = expr.sum(1)
        target = np.median(counts[counts > 0])                          # scanpy normalize_total default
        scale = np.where(counts > 0, target / np.where(counts > 0, counts, 1), 0.0)
        X = np.log1p(expr * scale[:, None]).astype(np.float32)
        print(f"[scfoundation] {ds}: {len(obs)} samples | panel coverage {have.mean():.1%}", flush=True)
        emb = embed_rows(X).astype(np.float32)
        assert emb.shape == (len(obs), 4 * cfg["encoder"]["hidden_dim"]), f"{ds}: bad shape {emb.shape}"
        assert np.isfinite(emb).all(), f"{ds}: non-finite embeddings"
        return emb, obs

    run_extract("scfoundation", str(DATA), a.out, embed_dataset,
                datasets=list(a.datasets) if a.datasets is not None else None)


if __name__ == "__main__":
    main()
