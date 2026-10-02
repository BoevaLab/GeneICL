<p align="center">
<img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
<img src="https://img.shields.io/pypi/v/geneicl" alt="PyPI version">
<a href="https://arxiv.org/abs/REPLACE_WITH_ARXIV_ID"><img src="https://img.shields.io/badge/arXiv-paper-b31b1b" alt="arXiv paper"></a>
<img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

# GeneICL: A tabular foundation model for transcriptomics		 <img src="logo.svg" alt="" width="56" height="56" valign="middle">

GeneICL is a tabular foundation model for bulk gene expression data that does **classification, regression
and survival analysis with one model**. Given a labelled support set and unlabelled query rows (gene expression
profiles), it predicts the query labels in a single forward pass, with no per-task training. 
GeneICL was pretrained only on synthetic tasks.

The repository has two parts: `geneicl/`, a small pip-installable library (the model, its three checkpoints and
a scikit-learn-style API; needs only `torch` and `numpy`), and `eval/`, the reproduction of the manuscript's
classification/regression benchmark.

## Install

```bash
pip install geneicl              # from PyPI after the 1.0.0 release
# Alternatively, from a clone: pip install .
python -m geneicl                # self-check: runs all three modes on synthetic data
```

The checkpoints (25 MB each) ship inside the package. `eval/` data needs `git lfs pull` but the library does not.

## Usage

`quickstart.ipynb` walks you through the usage of GeneICL with example datasets for classification, regression
and survival analysis.
Cells marked `# >>> REPLACE WITH YOUR DATA` show where your own data goes. The notebook also needs `scikit-learn`, `scipy` and `matplotlib`, and `git lfs pull` for the data.

```python
from geneicl import GeneICL

clf = GeneICL(mode="classification").fit(X_train, y_train)
clf.predict(X_test)                          # class labels; predict_proba -> (n_test, n_classes)

reg = GeneICL(mode="regression").fit(X_train, y_train)
reg.predict(X_test)                          # continuous values

cox = GeneICL(mode="survival").fit(X_train, time, event=event)
cox.predict(X_test)                          # Cox risk score (higher = shorter survival)
cox.predict_survival_function(X_test)        # (n_test, len(cox.times_)): S(t) at cox.times_
```

`X` is samples x genes in log1p-CPM. The training and test rows must share the same columns; any gene set works,
since the model standardizes and PCA-reduces every task itself. `fit` only stores the labelled support set, and
`predict` runs one in-context forward pass, without training. Classification supports up to 10 classes. Optional
arguments: `ckpt=` (default `geneicl.CKPT`, seed 0; seeds 1 and 2 are next to it in `geneicl/checkpoints/`),
`device=` (default: CUDA if available) and `ensemble_seed=` (default `None`: one forward pass; an int, e.g.
`GeneICL(mode, ensemble_seed=42)`, averages a label-free test-time ensemble of 8 seeded support views x 3 internal
PCA thresholds (0.9/0.95/0.98), i.e. 24 forward passes). The ensemble sometimes helps, but the benefit is
generally small and not consistent, and it is 24 times slower.

For survival, GeneICL regresses the support's Cox martingale residuals in one in-context pass and the prediction is
the log-risk. The Breslow baseline hazard is fit on 5-fold out-of-fold scores of the support set.

Lower level: `geneicl.Runner` runs one checkpoint in context (support + query chunks per forward, bf16 on GPU) and
is what both `GeneICL` and the benchmark use; `geneicl.model.load_ckpt` rebuilds the network from a checkpoint.

## Reproducing the benchmark in the paper (`eval/`)

`eval/` imports `geneicl` from the checkout, so it runs without installing the package. The pip package needs
only `torch` and `numpy`, but **running the benchmark needs the extra packages in `eval/environment.yaml`**
(pinned numpy/scipy/scikit-learn/pandas, hydra, the classical baselines and the in-context tabular FMs):

```bash
conda env create -f eval/environment.yaml && conda activate geneicl-benchmark
git lfs pull                             # the task suites and some baseline weights
cd eval/benchmark
python benchmark.py model=geneicl geneicl_ckpt=../../geneicl/checkpoints/trm_segmented8_s0.pt out=results/geneicl_trm_s0
```

`eval/benchmark/README.md` documents every model, its options and the reported configuration (3 seeds x plain and
ensembled GeneICL). No result files are shipped: every run writes to `eval/benchmark/results/`.

## Repository structure

```
pyproject.toml      # the pip package: geneicl/ only
logo.svg            # the GeneICL logo
quickstart.ipynb    # example notebook: classification, regression and survival
geneicl/
  __init__.py       # the API: GeneICL(mode=classification|regression|survival).fit / predict, and Runner
  model.py          # the GeneICL network + load_ckpt (rebuild a checkpoint from its saved `arch`)
  checkpoints/      # three GeneICL checkpoints, trained with different seeds (trm_segmented8_s{0,1,2}.pt)
eval/               # manuscript reproduction (not part of the package)
  data/             # frozen task suites: benchmark/ (52 tasks) and non_benchmark/ (6 tasks)
  benchmark/        # benchmark harness, baselines and the results table script (results/ is written by the runs)
  environment.yaml  # conda env for everything under eval/ (not needed for the pip package)
```

`geneicl/checkpoints/`, `eval/` and the `eval/` subfolders have their own `README.md` with details.
