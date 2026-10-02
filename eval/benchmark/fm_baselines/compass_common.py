#!/usr/bin/env python
"""COMPASS (mims-harvard/COMPASS, `pip install immuno-compass`) input alignment + model loading, shared by
compass_extract.py (zero-shot features).

COMPASS input = [cancer-code column, 15,672 gene-symbol TPM columns]; its Datascaler applies log2(x+1) and a
MinMax fitted on TCGA. Our datasets are log1p-CPM over 20,021 HGNC symbols, so per dataset:
    expm1 -> CPM -> TPM with GENCODE v36 union-exon gene lengths (COMPASS's own annotation; same formula as
    bf_extract.py) -> reindex to COMPASS's genes (absent -> TPM 0) -> prepend the cancer token.

Cancer token: COMPASS has no normal/unknown token -- the encoder is nn.Embedding(33, d) over the TCGA codes
and cancer_code.json's NORMAL=-1 raises IndexError. Every sample therefore gets a neutral MEAN token: a 34th
embedding row set to the mean of the 33 learned cancer-type embeddings (code MEAN_CODE=33). num_cancer_types
only sizes that table (the cancer concept projector reads the token's encoding, not its index), so the rest
of the checkpoint loads unchanged.

Checkpoint: example/model/pretrainer.pt = the TCGA self-supervised model, not finetuner_pft_all.pt /
pft_leave_*.pt, which were fine-tuned on immunotherapy-response labels (outside supervision).
"""
import gzip
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]                                   # GeneICL repo root
DATA = Path(os.environ.get("FM_DATA", REPO / "eval" / "data" / "benchmark"))
# pretrainer.pt is tracked here; the GTF (and the TSV caches built from it) are downloaded / written next to it:
#   curl -L -o eval/benchmark/weights/compass/gencode.v36.annotation.gtf.gz \
#       https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_36/gencode.v36.annotation.gtf.gz
MODEL_DIR = Path(os.environ.get("COMPASS_DIR", REPO / "eval" / "benchmark" / "weights" / "compass"))
CKPT = MODEL_DIR / "pretrainer.pt"
GTF = MODEL_DIR / "gencode.v36.annotation.gtf.gz"
LENGTHS = MODEL_DIR / "compass_gene_lengths.tsv"
MEAN_CODE = 33                                                              # appended neutral cancer token
MIN_COVERAGE = 0.90


def gene_lengths():
    """Union-exon length (bp) per gene symbol from GENCODE v36, cached as a TSV."""
    if LENGTHS.exists():
        return pd.read_csv(LENGTHS, sep="\t", index_col=0)["length"]
    exons = defaultdict(list)
    name_re = re.compile(r'gene_name "([^"]+)"')
    with gzip.open(GTF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 8)
            if p[2] != "exon":
                continue
            exons[(name_re.search(p[8]).group(1), p[0])].append((int(p[3]), int(p[4])))
    length = defaultdict(int)
    for (name, _chrom), iv in exons.items():                               # union of exons, per chromosome
        iv.sort(); s, e = iv[0]
        for a, b in iv[1:]:
            if a > e + 1:
                length[name] += e - s + 1; s, e = a, b
            else:
                e = max(e, b)
        length[name] += e - s + 1
    out = pd.Series(length, name="length").sort_index()
    out.to_csv(LENGTHS, sep="\t", index_label="gene")
    return out


def load_compass(ckpt=CKPT, device="cpu"):
    """Load the COMPASS PreTrainer and add the neutral MEAN cancer token (row MEAN_CODE) in place, so .extract()
    (which rebuilds the network from saver.inMemorySave) sees it."""
    from compass.utils import loadcompass
    m = loadcompass(str(ckpt), map_location="cpu")
    save = m.saver.inMemorySave
    key = "inputencoder.cancer_token_embedder.weight"
    W = save["model_state_dict"][key]
    assert W.shape[0] == MEAN_CODE, f"expected {MEAN_CODE} TCGA cancer tokens, got {tuple(W.shape)}"
    save["model_state_dict"][key] = torch.cat([W, W.mean(0, keepdim=True)], 0)
    save["model_args"]["num_cancer_types"] = MEAN_CODE + 1
    m.device = device
    return m


def align_dataset(ds, genes):
    """(DataFrame [cancer_code, *genes] of TPM, {obs_name: row}, coverage) for one benchmark (eval/data/benchmark) dataset;
    coverage = fraction of COMPASS genes expressed in >=1 sample."""
    z = np.load(DATA / "datasets" / f"{ds}.npz", allow_pickle=True)
    ref = [g.strip() for g in open(DATA / "reference_genes.txt")]
    L = gene_lengths()
    keep = [i for i, g in enumerate(ref) if L.get(g, 0) > 0]
    sym = [ref[i] for i in keep]
    cpm = np.expm1(np.asarray(z["X"], np.float64)[:, keep])
    rate = cpm / L.loc[sym].values[None, :]
    tpm = rate / np.clip(rate.sum(1, keepdims=True), 1e-12, None) * 1e6
    df = pd.DataFrame(tpm, columns=sym)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.reindex(columns=list(genes), fill_value=0.0)
    coverage = float((df.values > 0).any(0).mean())            # COMPASS genes expressed in >=1 sample (absent
                                                                # from the source -> all-zero in the benchmark data)
    df.insert(0, "cancer_code", MEAN_CODE)
    obs = [str(o) for o in z["obs_names"]]
    df.index = obs
    return df, {o: i for i, o in enumerate(obs)}, coverage
