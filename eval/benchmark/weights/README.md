# Baseline model weights

Pretrained competitor models used by `../benchmark.py` and `../fm_baselines/`, where each came from, and whether
git tracks it. **Tracked** = committed (large files through git LFS, rules in `/.gitattributes`; run
`git lfs pull`). **Not committed** = re-download from the source below. Hugging Face revisions are pinned by
commit. SHA-256 = first 16 hex characters of the file's checksum (`sha256sum <file>`).

## Tabular foundation-model baselines

| model | file(s) | size | status | source (pinned revision) | licence | SHA-256 |
|---|---|---|---|---|---|---|
| TabICL v2 | `tabicl-classifier-v2-20260212.ckpt`, `tabicl-regressor-v2-20260212.ckpt` | 110 MB, 114 MB | **tracked (LFS)** | https://huggingface.co/jingang/TabICL @ `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` | BSD-3-Clause | `bdc7dbd5e4ff21f8`, `0db9cb538f114e79` |
| LimiX-16M | `LimiX-16M.ckpt` | 66 MB | **tracked** | https://huggingface.co/stable-ai/LimiX-16M @ `55a7699eb81c78db2a1ed1d5bc80c8391c0ee051` | see model card ("other") | `ee6d6ae865821ca7` |
| TabDPT-Turbo (TabDPT v1.2) | `tabdpt/tabdpt1_2.safetensors` | 254 MB | **tracked (LFS)** | https://huggingface.co/Layer6/TabDPT @ `4462ffbd1d8dea25d4862d30beed4b70cd596ae5` (unchanged at `a5ca6e01`; the default of the `tabdpt` 1.2 package) | Apache-2.0 | `06680220fd66c452` |
| TabFM 1.0.0 (PyTorch) | `tabfm/{classification,regression}/model.safetensors` | 6.6 GB each | **not vendored** (over the 2 GB LFS per-file limit); `_tabfm_worker.py` downloads it from Hugging Face unless `TABFM_CKPT` points at a local snapshot | https://huggingface.co/google/tabfm-1.0.0-pytorch @ `77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83`; code https://github.com/google-research/tabfm | tabfm-non-commercial-v1.0 | — |
| TabPFN v3 (clf/reg) | `tabpfn-v3-{classifier,regressor}-v3_default.ckpt` | — | **not vendored** (gated: accept the Prior Labs licence on Hugging Face and log in to download) | https://huggingface.co/Prior-Labs; `pip install tabpfn` auto-downloads v3 to `~/.cache/tabpfn` | Prior Labs licence | — |

## Transcriptomics foundation-model baselines (`eval/benchmark/fm_baselines/`)

| model | file(s) | size | status | source | licence | SHA-256 |
|---|---|---|---|---|---|---|
| COMPASS (TCGA self-supervised) | `compass/pretrainer.pt` | 26 MB | **tracked** | https://github.com/mims-harvard/COMPASS `example/model/pretrainer.pt` (raw: https://raw.githubusercontent.com/mims-harvard/COMPASS/main/example/model/pretrainer.pt); paper https://www.nature.com/articles/s41591-026-04502-7. Not the immunotherapy-fine-tuned `finetuner_pft_all.pt` / `pft_leave_*.pt` | MIT (code) | `661ae4d838775c55` |
| BulkFormer-147M | `bulkformer/bulkformer_147M.ckpt` | 530 MB | not committed; `bulkformer/assets/*` **tracked** (`.pt` via LFS) | Google Drive https://drive.google.com/file/d/1UtqN_vCh3669Fs-GU5CTE7F7UnuQCAzN (listed in https://github.com/KangBoming/BulkFormer/blob/main/model/README.md); assets (`esm2_feature_concat.pt`, gene info, graphs) on Zenodo https://zenodo.org/records/15744294 | see repo | `07b69eed9b74154b` |
| scGPT whole-human | `scgpt/scGPT_human/{best_model.pt,args.json,vocab.json}` | 205 MB | not committed | Google Drive folder https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y (the "whole-human (recommended)" row of https://github.com/bowang-lab/scGPT) | see repo | `6cb5d451ab5c4b33` |
| BulkRNABert | `bulkrnabert/` (full HF snapshot: `model.safetensors`, `jax_params/`, tokenizer, `bulkrnabert.py`) | 24 MB | **tracked** (weights via LFS) | https://huggingface.co/InstaDeepAI/BulkRNABert @ `7bcae6ac4f68fe1c781e8c1a626d8ee023747228` | see model card | `20769ab08873906e` |
| scFoundation (xTrimoGene 100M) | `scfoundation/models.ckpt` | — | not committed | Hugging Face mirror https://huggingface.co/genbio-ai/scFoundation (original: the SharePoint folder in https://github.com/biomap-research/scFoundation/blob/main/model/README.md) | see repo | — |

## Where the benchmark loads weights from

`benchmark.py` reads TabICL v2, LimiX-16M and TabDPT-Turbo from this folder (`WEIGHTS`, `TABDPT_CKPT`), TabPFN v3
from `~/.cache/tabpfn`. The `fm_baselines/*_extract.py` scripts default to the paths in this folder (their `CONFIG`,
overridable as Hydra `key=value`).

## Re-downloading

```bash
cd eval/benchmark/weights
python -c "from huggingface_hub import snapshot_download as s; s('InstaDeepAI/BulkRNABert',local_dir='bulkrnabert'); s('google/tabfm-1.0.0-pytorch',local_dir='tabfm')"
curl -L -o compass/pretrainer.pt https://raw.githubusercontent.com/mims-harvard/COMPASS/main/example/model/pretrainer.pt
gdown 1UtqN_vCh3669Fs-GU5CTE7F7UnuQCAzN -O bulkformer/bulkformer_147M.ckpt
gdown --folder https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y -O scgpt/scGPT_human
python -c "from huggingface_hub import hf_hub_download as h; h('genbio-ai/scFoundation','models.ckpt',local_dir='scfoundation')"
curl -L -o compass/gencode.v36.annotation.gtf.gz https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_36/gencode.v36.annotation.gtf.gz
```
