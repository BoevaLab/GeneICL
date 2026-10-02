# Full benchmark including classical-ML baselines

GeneICL, in-context tabular foundation models, frozen transcriptomic-FM probes and classical-ML baselines on
the frozen benchmark (`../data/benchmark/`: 52 tasks = 27 classification + 25 regression over 32 datasets; provenance
`build_data.py` + `build_data.yaml`, which need the upstream curation code, not part of this repo). It was never used during training;
the training-time monitor used the non-benchmark tasks in `../data/non_benchmark` (`data_dir=` points
`benchmark.py` there). One
harness, `benchmark.py` (Hydra, `key=value`; defaults in its `CONFIG`), scores every model with the same
5-fold CV (patient-grouped when the task carries a `group`), `seed=42`. Each model does regression *and*
classification (auto-selected per task).

## Models (`model=`)

**GeneICL** — in-context on raw genes, no tuning:

| key | default | role |
|---|---|---|
| `geneicl_ckpt` | `../../geneicl/checkpoints/trm_segmented8_s0.pt` | GeneICL checkpoint. The reported runs use `trm_segmented8_s{0,1,2}.pt` (nhead=8, `enc_cls_ffn=false`, 4.24M params) |
| `geneicl_support_ens` | off | test-time ensemble over K support views: the full support + K-1 seeded 90% class-stratified subsets |
| `geneicl_pca_ens` | off | average over the model's internal `pca_var` thresholds, e.g. `[0.9,0.95,0.98]` |
| `query_chunk` | 256 | test rows per forward pass |

The reported configuration is 3 seeds x {plain, ensembled} = 6 GPU jobs, each over the
whole suite -> `results/geneicl_trm_s<seed>_<task_type>.csv` (plain) and
`results/geneicl_trm_supp_thresh_k32_s<seed>_<task_type>.csv` (`geneicl_support_ens=32
'geneicl_pca_ens=[0.9,0.95,0.98]'`: 32 views x 3 thresholds, averaged within the seed).

**Classical baselines** (`en`, `rf`, `catboost`, `lightgbm`, `mlp`), always tuned: every model on the PCA-90%
input (`TUNE_PCA_VAR`), 50 candidates x inner 5-fold CV per outer fold (TabArena-aligned search spaces), pick by
mean inner AUROC / R², refit on the full support. One search (`_tune`) for all five: the same candidates and the
same shuffled (stratified) inner folds. `catboost` / `lightgbm` additionally early-stop each inner fit on its
validation fold (≤1000 rounds, patience 50) and are refit at the mean best round count. EN-classification = elastic-net-penalized logistic
regression. All classical models run on CPUs.

**In-context tabular FMs** (never tuned), via a small `_ICL` adapter (`fit` stashes the
support set; the forward pass — and its timing — happens in `predict`):

| `model=` | model | weights / code |
|---|---|---|
| `tabpfn` | TabPFN-3 | `~/.cache/tabpfn/tabpfn-v3-*-v3_default.ckpt` (pinned via `model_path`: tabpfn>=9 defaults to v3.5) |
| `tabpfn_api` | TabPFN-3 via the Prior Labs API | same `v3_default` checkpoint server-side; token from `TABPFN_TOKEN` or `~/.config/tabpfn/token`; used only to re-score TARGET::relapse in the TabPFN-3 row |
| `tabicl` | TabICL v2 | `weights/tabicl-*-v2-20260212.ckpt` |
| `tabdpt` | TabDPT-Turbo (TabDPT v1.2, Layer6) | `weights/tabdpt/tabdpt1_2.safetensors`, `tabdpt` package 1.2 |
| `limix` | LimiX-16M | `weights/LimiX-16M.ckpt` (`stable-ai/LimiX-16M`), LimiX clone @61ede97 in `external/LimiX` (`LIMIX_REPO`; clone command in `benchmark.py`), its `*_noretrieval.json` configs |
| `tabfm` | Google TabFM | own py3.11 venv in `external/tabfm-venv` (`TABFM_PY`; install command in `benchmark.py`) via `_tabfm_worker.py` |

**Frozen transcriptomic-FM probes** (`EMB_MODELS`: `compass`, `bulkformer`, `scfoundation`, `scgpt`,
`bulkrnabert`; need `emb_dir=<dir of <dataset>.npz>` from `fm_baselines/*_extract.py`): the embeddings replace the
raw genes and the probe is an untuned `en` (default hyperparameters) on the PCA-95% front-end. See
`fm_baselines/README.md`.

## Inputs

| models | input |
|---|---|
| in-context tabular FMs | `Impute(mean) → StandardScaler → PCA` (full SVD, data-dependent #PCs), fitted on the support of each outer fold |
| every classical model | the same pipeline with PCA, fitted on the support of each outer fold (not inside the tuning search) |
| `geneicl` | raw genes; the model does its own per-fold standardization + internal PCA |
| `EMB_MODELS` | frozen embeddings → the same PCA front-end + untuned EN / EN-logistic probe |

## Environments

| script | Python |
|---|---|
| `benchmark.py` (every `model=` except the two below), `make_main_table.py`, `pooled_oof_table.py` | `eval/environment.yaml` |
| `model=tabfm` | `eval/environment.yaml` + TabFM's own venv as a subprocess (`TABFM_PY`, see `benchmark.py`) |
| `model=tabpfn_api` | a separate env with `tabpfn-client==0.3.3` (its pins conflict with `eval/environment.yaml`) |
| `fm_baselines/*_extract.py` | their own venvs (recipes in `fm_baselines/fm_common.py` and `scgpt_extract.py`) |

## Logged per fold

`train_seconds` (fit + predict, plus search, wall-time — for the in-context models the forward pass is in
`predict`, so this is their real inference cost), `peak_ram_mb` and `peak_vram_mb` — peak memory over that same
window, sampled across the whole process tree (loky search workers and the TabFM subprocess count).
`peak_ram_mb` is total RSS (includes torch/pandas, comparable since each model is its own job); `peak_vram_mb`
is per-process GPU memory via NVML, `NaN` when the job has no GPU/NVML (the classical baselines). Metrics:

| | metrics |
|---|---|
| classification | `auroc`, `auprc`, `f1_macro`, `f1_weighted`, `balanced_accuracy`, `accuracy`, `mcc`, `nll`, `brier`, `ece` |
| regression | `pearson`, `spearman`, `r2`, `mape` |

Non-applicable metrics are `NaN`.

## Output

One run scores every task of the suite, regression and classification, in one job. `out=` is an output prefix
(default `results/<model>`): the run writes `<out>_regression.csv` and `<out>_classification.csv`, plus per-sample
companions `<out>_regression.oof.csv` (regression predictions → pooled-OOF R²) and
`<out>_classification.clfoof.npz` (class probabilities).

Layout of `results/`: see `results/README.md`.


## Tables

- `make_main_table.py` — the main results table from `make_main_table.yaml` → `results/main_table.csv`, plus
  `results/main_table_extra.csv`: the remaining logged metrics and `train_seconds` / `peak_ram_mb` / `peak_vram_mb`
  for every row, averaged over folds → tasks → seeds on the same task set.
- `pooled_oof_table.py` — paper protocol (mean-of-folds, pooled-OOF R²) for `dir=` + `'models=[[name,tag,...],...]'`
  (Hydra; one line).

`make_main_table.py` reads `make_main_table.yaml`; the other scripts take their defaults from `CONFIG` in the code
(Hydra `key=value` overrides on the command line).
