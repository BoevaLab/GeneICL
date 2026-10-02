#!/usr/bin/env python
"""Zero-shot BulkRNABert (InstaDeepAI/BulkRNABert) embeddings for the benchmark datasets -- same role as
bf_extract.py / scf_extract.py / compass_extract.py. Saves per dataset the mean over genes of the last
(4th) transformer layer (256-d, the model card's recipe) for the EN/logistic probe
(benchmark.py model=bulkrnabert emb_dir=<out>).

Input prep (model card + tokenizer_config.json): TCGA TPM over a fixed 19,062-gene Ensembl panel ->
log10(1 + TPM) -> tokenizer (/ 5.547 max-normalisation, 64 bins, exact 0 -> token 0). Ours is log1p-CPM over
HGNC symbols, so: expm1 -> CPM -> TPM with GENCODE v36 union-exon lengths (compass_common.gene_lengths,
same formula as bf_extract.py) -> panel ENSG mapped to its GENCODE v36 symbol (TCGA/GDC uses v36) -> absent
genes TPM 0.

Attention: the released module materialises the full (heads, 19062, 19062) attention matrix (~11.6 GB per
sample per layer in fp32). It is swapped for F.scaled_dot_product_attention -- the same function, computed
without the matrix -- and checked against the original on real samples (check_sdpa=true).

    CUDA_VISIBLE_DEVICES=0 python bulkrnabert_extract.py                  # all datasets -> <out>/<dataset>.npz
    CUDA_VISIBLE_DEVICES=0 python bulkrnabert_extract.py datasets=[GSE107422] check_sdpa=true

Needs: the tracked weights/bulkrnabert snapshot, the GENCODE v36 GTF (see compass_common.py) and the FM venv
(fm_common.py). Defaults: CONFIG in this file.
"""
import gzip
import re
import types

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

import compass_common as C                                         # GENCODE v36 lengths, benchmark data paths
from fm_common import DEV, EMB_DIR, WEIGHTS, run_extract

ENSG_MAP = C.MODEL_DIR / "gencode_v36_ensg_symbol.tsv"


def ensg_to_symbol():
    """Unversioned Ensembl gene id -> gene symbol from the GENCODE v36 GTF (cached)."""
    if ENSG_MAP.exists():
        return pd.read_csv(ENSG_MAP, sep="\t", index_col=0)["symbol"]
    rows = {}
    rx = re.compile(r'gene_id "([^".]+)[^"]*";.*gene_name "([^"]+)"')
    with gzip.open(C.GTF, "rt") as f:
        for line in f:
            if "\tgene\t" in line:
                m = rx.search(line)
                rows.setdefault(m.group(1), m.group(2))
    s = pd.Series(rows, name="symbol"); s.to_csv(ENSG_MAP, sep="\t", index_label="ensg")
    return s


def _sdpa_forward(self, query, key, value, attention_mask=None, attention_weight_bias=None):
    """Drop-in for the released MultiHeadAttention.forward (inference, no mask/bias)."""
    # the model always passes an all-True (B,1,T,T) mask when none is given -> no masking; refuse anything else
    assert not attention_weight_bias and (attention_mask is None or bool(attention_mask.all()))
    q = self.w_q(query).reshape(*query.shape[:-1], self.num_heads, self.key_size).transpose(-3, -2)
    k = self.w_k(key).reshape(*key.shape[:-1], self.num_heads, self.key_size).transpose(-3, -2)
    v = self.w_v(value).reshape(*value.shape[:-1], self.num_heads, self.value_size).transpose(-3, -2)
    out = F.scaled_dot_product_attention(q, k, v).transpose(-3, -2)
    return {"attention_weights": None, "embeddings": self.output(out.reshape(*out.shape[:-2], -1))}


def load(model_dir, sdpa=True):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg.embeddings_layers_to_save = (cfg.num_layers,)
    model = AutoModel.from_pretrained(model_dir, config=cfg, trust_remote_code=True).to(DEV).eval()
    if sdpa:
        for m in model.modules():
            if type(m).__name__ == "MultiHeadAttention":
                m.forward = types.MethodType(_sdpa_forward, m)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    panel = list(pd.read_csv(f"{model_dir}/data/tcga_sample.csv", nrows=0).columns.drop("identifier"))
    assert len(panel) == cfg.n_genes
    return model, tok, panel, f"embeddings_{cfg.num_layers}"


def panel_tpm(ds, panel):
    """(log10(1+TPM) over the panel (n, 19062), obs_names, coverage)."""
    z = np.load(C.DATA / "datasets" / f"{ds}.npz", allow_pickle=True)
    ref = [g.strip() for g in open(C.DATA / "reference_genes.txt")]
    L = C.gene_lengths()
    keep = [i for i, g in enumerate(ref) if L.get(g, 0) > 0]
    sym = [ref[i] for i in keep]
    rate = np.expm1(np.asarray(z["X"], np.float64)[:, keep]) / L.loc[sym].values[None, :]
    tpm = pd.DataFrame(rate / np.clip(rate.sum(1, keepdims=True), 1e-12, None) * 1e6, columns=sym)
    tpm = tpm.loc[:, ~tpm.columns.duplicated()]
    e2s = ensg_to_symbol()
    cols = [e2s.get(e, None) for e in panel]
    X = np.zeros((len(tpm), len(panel)), np.float64)
    for j, s in enumerate(cols):
        if s is not None and s in tpm.columns:
            X[:, j] = tpm[s].values
    coverage = float((X > 0).any(0).mean())                        # panel genes expressed in >=1 sample
    return np.log10(1 + X), [str(o) for o in z["obs_names"]], coverage


CONFIG = {                        # defaults; override on the CLI as key=value (Hydra)
    "model_dir": str(WEIGHTS / "bulkrnabert"),                       # HF snapshot of InstaDeepAI/BulkRNABert
    "out": str(EMB_DIR / "bulkrnabert"),                             # -> <out>/<dataset>.npz; benchmark.py emb_dir=
    "datasets": None,             # e.g. datasets=[GSE107422]; None = all in index.json
    "batch": 4,
    "check_sdpa": False,          # also run the released (naive) attention on 2 samples and report the max difference
}
ConfigStore.instance().store(name="bulkrnabert_extract", node=CONFIG)


@hydra.main(version_base="1.3", config_path=None, config_name="bulkrnabert_extract")
def main(a: DictConfig) -> None:
    model, tok, panel, key = load(a.model_dir)

    def embed(ids):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            return model(ids.to(DEV))[key].float().mean(1).cpu().numpy()

    if a.check_sdpa:                                               # SDPA swap == released attention (fp32, 2 samples)
        ref_model, _, _, _ = load(a.model_dir, sdpa=False)
        X, _, _ = panel_tpm(list(a.datasets)[0], panel)
        ids = tok.batch_encode_plus(X[:2], return_tensors="pt")["input_ids"].to(DEV)
        with torch.no_grad():
            r = ref_model(ids)[key].mean(1); s = model(ids)[key].mean(1)
        print(f"[bulkrnabert] SDPA vs released attention: max |diff| = {(r - s).abs().max():.2e} "
              f"(embedding scale {r.abs().mean():.2e})", flush=True)
        del ref_model; torch.cuda.empty_cache()

    def embed_dataset(ds):
        X, obs, cov = panel_tpm(ds, panel)
        print(f"[bulkrnabert] {ds}: {len(obs)} samples | panel coverage {cov:.1%}", flush=True)
        out = []
        for s in range(0, len(X), a.batch):
            out.append(embed(tok.batch_encode_plus(X[s:s + a.batch], return_tensors="pt")["input_ids"]))
        emb = np.concatenate(out).astype(np.float32)
        assert np.isfinite(emb).all(), f"{ds}: non-finite embeddings"
        return emb, obs

    run_extract("bulkrnabert", str(C.DATA), a.out, embed_dataset,
                datasets=list(a.datasets) if a.datasets is not None else None)


if __name__ == "__main__":
    main()
