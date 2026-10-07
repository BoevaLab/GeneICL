#!/usr/bin/env python
"""Zero-shot BulkFormer-147M embeddings for the benchmark datasets -- same role as compass_extract.py /
scgpt_extract.py / bulkrnabert_extract.py. Per sample: the 643-d mean over the 2,000 "interested" genes of
the model's enriched gene embeddings, saved for the EN/logistic probe (benchmark.py model=bulkformer
emb_dir=<out>).

Input prep (as in the repo's extract-feature notebook): our log1p-CPM -> linear CPM -> exact CPM->TPM with
BulkFormer's own gene lengths (the count factor cancels) -> log1p -> aligned to the model's 20,010 Ensembl
genes, absent genes filled with -10 (the repo's missing-value code).

The model is called directly as `model(x, mask_prob, output_expr=False)`; paths come from CONFIG.

    CUDA_VISIBLE_DEVICES=0 python bf_extract.py                      # all datasets -> <out>/<dataset>.npz
    CUDA_VISIBLE_DEVICES=0 python bf_extract.py datasets=[GSE107422] # a subset

Needs: a BulkFormer clone (git clone https://github.com/KangBoming/BulkFormer external/BulkFormer), the
checkpoint (gdown line in weights/README.md), the tracked Zenodo assets in weights/bulkformer/assets and the
FM venv (fm_common.py). Defaults: CONFIG in this file.
"""
import os
import sys
from collections import OrderedDict
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


def _patch_spmm():
    """torch_sparse has no wheel for this torch/GPU, so torch_geometric falls back to torch.sparse.mm, which
    only handles 2-D. BulkFormer feeds x=[batch, genes, dim] through a graph shared across the batch, so the
    product factors column-wise: [b,N,d] -> [N,b*d], mm, reshape back. (GCNConv aggr='add' with a
    pre-normalised adjacency == plain sum, exactly what torch.sparse.mm computes.)"""
    import torch_geometric.nn.conv.gcn_conv as _gcn

    def _spmm_batched(src, other, reduce="sum"):
        if other.dim() <= 2:
            return torch.sparse.mm(src, other)
        b, N, d = other.shape
        out = torch.sparse.mm(src, other.permute(1, 0, 2).reshape(N, b * d))
        return out.reshape(N, b, d).permute(1, 0, 2).contiguous()

    _gcn.spmm = _spmm_batched


def load_model(a):
    """(model, interested_gene_idx, gene_list, keep_idx, keep_ensg, lengths) — frozen BulkFormer + gene vocab."""
    sys.path.insert(0, str(a.repo))
    _patch_spmm()
    from model.config import model_params
    from utils.BulkFormer import BulkFormer

    assets = Path(a.assets)
    g = torch.load(assets / "G_tcga.pt", map_location="cpu", weights_only=False)
    w = torch.load(assets / "G_tcga_weight.pt", map_location="cpu", weights_only=False)
    N = int(model_params["gene_length"])                               # 20010 graph nodes
    # the notebook builds SparseTensor(row=g[1], col=g[0], value=w).t() -> entry (g[0], g[1]) = w
    graph = torch.sparse_coo_tensor(torch.stack([g[0].long(), g[1].long()]), w.float(),
                                    (N, N)).coalesce().to_sparse_csr().to(DEV)
    model_params["graph"] = graph
    model_params["gene_emb"] = torch.load(assets / "esm2_feature_concat.pt", map_location="cpu", weights_only=False)
    interested = torch.load(assets / "interested_gene_list.pt", weights_only=False)

    model = BulkFormer(**model_params).to(DEV)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck
    for k in ("model", "state_dict", "model_state_dict"):
        if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict):
            sd = ck[k]; break
    sd = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items())   # strip DDP prefix
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert not unexp, f"unexpected checkpoint keys: {unexp[:5]}"
    print(f"[bulkformer] loaded | missing={len(miss)} unexpected={len(unexp)} | graph N={N} "
          f"interested={len(interested)}", flush=True)
    model.eval()

    gi = pd.read_csv(assets / "bulkformer_gene_info.csv", usecols=["gene_symbol", "ensg_id", "gene_length"])
    gene_list = gi["ensg_id"].tolist()                                 # model gene order (20010)
    sym2ensg = dict(zip(gi.gene_symbol.astype(str), gi.ensg_id))
    ensg2len = dict(zip(gi.ensg_id, gi.gene_length.astype(float)))
    our = [l.strip() for l in open(DATA / "reference_genes.txt") if l.strip()]
    keep_idx, keep_ensg, keep_len = [], [], []
    for i, s in enumerate(our):
        e = sym2ensg.get(s)
        if e is not None and ensg2len.get(e, 0) > 0:
            keep_idx.append(i); keep_ensg.append(e); keep_len.append(ensg2len[e])
    print(f"[bulkformer] genes: our {len(our)} symbols -> {len(keep_idx)} mapped of {len(gene_list)}", flush=True)
    return model, list(interested), gene_list, np.array(keep_idx), keep_ensg, np.array(keep_len, np.float64)


CONFIG = {                        # defaults; override on the CLI as key=value (Hydra)
    "repo": str(EXTERNAL / "BulkFormer"),                            # github.com/KangBoming/BulkFormer
    "assets": str(WEIGHTS / "bulkformer" / "assets"),                # Zenodo 15744294
    "ckpt": str(WEIGHTS / "bulkformer" / "bulkformer_147M.ckpt"),
    "out": str(EMB_DIR / "bulkformer"),                              # -> <out>/<dataset>.npz; benchmark.py emb_dir=
    "datasets": None,             # e.g. datasets=[GSE107422]; None = all in index.json
    "batch": 8,
    "mask_prob": 0.0,             # auxiliary mask feature; 0 = deterministic embeddings (the notebook demo used 0.1)
}
ConfigStore.instance().store(name="bf_extract", node=CONFIG)


@hydra.main(version_base="1.3", config_path=None, config_name="bf_extract")
def main(a: DictConfig) -> None:
    model, interested, gene_list, keep_idx, keep_ensg, L = load_model(a)
    idx = torch.tensor(interested, dtype=torch.long, device=DEV)

    def embed_dataset(ds):
        z = np.load(DATA / "datasets" / f"{ds}.npz", allow_pickle=True)
        obs = [str(o) for o in z["obs_names"]]
        cpm = np.expm1(np.asarray(z["X"], np.float64)[:, keep_idx])    # linear CPM for mapped genes
        rate = cpm / L
        tpm = rate / (rate.sum(1, keepdims=True) + 1e-12) * 1e6        # exact CPM -> TPM (count factor cancels)
        df = pd.DataFrame(np.log1p(tpm), columns=keep_ensg)
        df = df.loc[:, ~df.columns.duplicated()]                       # duplicate symbols -> same ensg
        aligned = pd.DataFrame(-10.0, index=df.index, columns=gene_list, dtype=np.float32)  # -10 = missing
        common = [g for g in gene_list if g in df.columns]
        aligned[common] = df[common].astype(np.float32)
        X = aligned.values.astype(np.float32)
        print(f"[bulkformer] {ds}: {len(obs)} samples | vocab coverage {len(common) / len(gene_list):.1%}", flush=True)
        out = []
        for s in range(0, len(X), a.batch):
            xb = torch.from_numpy(X[s:s + a.batch]).to(DEV)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                g = model(xb, mask_prob=a.mask_prob, output_expr=False)        # (b, 20010, 643)
                out.append(g[:, idx, :].mean(1).float().cpu().numpy())         # mean over interested genes
        emb = np.concatenate(out).astype(np.float32)
        assert emb.shape[0] == len(obs) and np.isfinite(emb).all(), f"{ds}: bad embeddings {emb.shape}"
        return emb, obs

    run_extract("bulkformer", str(DATA), a.out, embed_dataset,
                datasets=list(a.datasets) if a.datasets is not None else None)


if __name__ == "__main__":
    main()
