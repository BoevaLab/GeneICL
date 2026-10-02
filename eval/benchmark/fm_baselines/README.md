# Transcriptomics FM baselines — scFoundation, BulkFormer, COMPASS, scGPT, BulkRNABert

Reproduction code for the zero-shot transcriptomics foundation-model baselines in the main table: frozen
backbone → sample-level embeddings → the benchmark's elastic-net / logistic-regression probe, under the same
5-fold CV as GeneICL.

| model | extractor | results |
|-------|-----------|---------|
| BulkFormer | `bf_extract.py` | `results/bulkformer_{classification,regression}.*` |
| scFoundation | `scf_extract.py` | `results/scfoundation_{classification,regression}.*` |
| COMPASS | `compass_extract.py` (+ `compass_common.py`) | `results/compass_{classification,regression}.*` |
| scGPT | `scgpt_extract.py` | `results/scgpt_{classification,regression}.*` |
| BulkRNABert | `bulkrnabert_extract.py` | `results/bulkrnabert_{classification,regression}.*` |

`fm_common.run_extract` is the shared driver. Fine-tuned (LoRA) variants were not reported; they are on the
`experimentation` branch.

## External dependencies

These scripts drive the upstream model repos and their weights. Weights go to `../weights/` (download lines in
`../weights/README.md`), upstream clones and venvs to the gitignored `external/` at the repo root.

| model | repo + weights | venv | zero-shot embedding |
|-------|----------------|------|-----------|
| BulkFormer | BulkFormer repo + `bulkformer_147M.ckpt` + Zenodo assets (`G_tcga` graph, ESM2 gene embeddings, `interested_gene_list`, gene info) | FM venv | 643-d, mean over the 2,000 interested genes |
| scFoundation | scFoundation repo `model/` + xTrimoGene-100M weights (`models.ckpt`, HF `genbio-ai/scFoundation`; bulk mode) | FM venv | 3072-d `[S-tok, T-tok, gene-max, gene-mean]` |
| COMPASS | `pip install immuno-compass` + `example/model/pretrainer.pt` + GENCODE v36 GTF (gene lengths) | FM venv | 177-d: 44 concept + 133 gene-set scores |
| scGPT | `scGPT_human` bundle (`args.json`, `best_model.pt`, `vocab.json`) | scGPT venv (scgpt 0.2.4, torch 2.3.1, no flash-attn) | 512-d L2-normalised `<cls>` output via `scgpt.tasks.embed_data` |
| BulkRNABert | HF snapshot of `InstaDeepAI/BulkRNABert` + GENCODE v36 GTF | FM venv | 256-d mean over the 19,062-gene panel of the last (4th) layer |

The FM venv (py3.12) is set up by the commands in `fm_common.py`, the scGPT venv by those in `scgpt_extract.py`;
the upstream clone commands are in each extractor's docstring. The extractors take Hydra keys (`repo`, `ckpt`,
`model_dir`, `out`, ...); the defaults in each script's `CONFIG` point inside this repo, with embeddings written to
`../embeddings/<model>/`.

## Zero-shot (frozen-embedding probe)

1. **Extract** sample embeddings once per model (32 datasets → `<out>/<dataset>.npz` with `emb`, `obs_names`):
   ```bash
   python bf_extract.py            # BulkFormer: log1p-CPM -> exact CPM->TPM (gene lengths) -> Ensembl align -> forward -> mean-pool
   python scf_extract.py           # scFoundation: bulk mode (SCAD/DeepCDR recipe: --input_type bulk --output_type cell --pool_type all --pre_normalized F --version ce)
   python compass_extract.py       # COMPASS (see below)
   python scgpt_extract.py         # scGPT: embed_data, <cls> embedding
   python bulkrnabert_extract.py   # BulkRNABert: TPM -> log10(1+TPM) -> 64-bin tokens -> last-layer gene mean
   ```
2. **Evaluate** the frozen embeddings with the standard EN/logistic probe under the shared CV, via the
   benchmark's frozen-embedding branch (`EMB_MODELS` + `emb_dir=`; the probe is an untuned elastic net / EN-logistic
   on the PCA-95% front-end). Outputs go to `results/<model>_<task_type>.csv`:
   ```bash
   cd ..   # eval/benchmark
   for M in bulkformer scfoundation compass scgpt bulkrnabert; do
     python benchmark.py model=$M emb_dir=embeddings/$M
   done
   ```

## COMPASS

[COMPASS](https://www.nature.com/articles/s41591-026-04502-7) (Nature Medicine 2026;
code [mims-harvard/COMPASS](https://github.com/mims-harvard/COMPASS), `pip install immuno-compass`, MIT; paper
CC BY-NC-ND) is a ~1M-parameter pan-cancer concept-bottleneck transformer: gene expression -> 133 gene-set
scores -> 44 tumour-immune concepts, self-supervised on 10,184 TCGA tumours. None of our 32 datasets is TCGA
(TARGET is the paediatric cohort), so pretraining does not overlap the evaluation data.

- **Checkpoint:** `example/model/pretrainer.pt` (TCGA self-supervised) only. `finetuner_pft_all.pt` and
  `pft_leave_*.pt` are fine-tuned on immunotherapy-response labels and would add outside supervision.
- **Input** (`compass_common.py`): COMPASS takes TPM over 15,672 gene symbols (its scaler applies
  `log2(x+1)` + a TCGA-fitted MinMax). Ours is log1p-CPM, so `expm1` -> CPM -> TPM with GENCODE v36
  union-exon lengths (COMPASS's annotation; same formula as `bf_extract.py`) -> reindex to COMPASS's genes,
  absent genes = TPM 0. Coverage (COMPASS genes expressed in >=1 sample) is 94.3-98.3% per dataset.
- **Cancer token:** COMPASS requires a TCGA cancer-type token and has no normal/unknown one (`NORMAL=-1`
  in `cancer_code.json` is an out-of-range index for the 33-row embedding). Every sample gets a neutral
  **mean token**: a 34th row = the mean of the 33 learned cancer-type embeddings.
- **Zero-shot:** 177-d features (44 concepts + 133 gene sets) -> the `en` probe. `features=vector` gives
  `project()`'s 32-d vector per gene set/concept instead (flattened, 5600-d).

```bash
../../../external/fm-venv/bin/python compass_extract.py                 # -> ../embeddings/compass
cd .. && python benchmark.py model=compass emb_dir=embeddings/compass   # -> results/compass_{regression,classification}.csv
```

All extractors are configured with Hydra (`key=value` overrides) from the in-code `CONFIG` defaults, so each venv
needs `hydra-core>=1.3`.
