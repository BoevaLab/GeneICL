<div align="center">

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2610.08694&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2610.08694) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) [![PyPI version](https://badge.fury.io/py/geneicl.svg)](https://badge.fury.io/py/geneicl) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

# GeneICL: A tabular foundation model for transcriptomics

GeneICL is a tabular foundation model for bulk gene expression data that does **classification, regression
and survival analysis**. Given a labelled support set and unlabelled query rows (gene expression profiles), it
predicts the query labels in context, with no per-task training.
GeneICL was pretrained only on synthetic tasks.

The repository has two parts: `geneicl/`, a small pip-installable library (the model, its checkpoints and
a scikit-learn-style API; needs only `torch` and `numpy`), and `eval/`, the reproduction of the manuscript's
classification/regression benchmark. What changed between versions: [`CHANGELOG.md`](CHANGELOG.md).

## Install

```bash
pip install geneicl
python -m geneicl                # self-check: runs every task in both inference modes on synthetic data
```

The seed-0 checkpoints (25 MB each) ship inside the package; seeds 1 and 2 are in this repository
(`geneicl/checkpoints/`). `eval/` data needs `git lfs pull` but the library does not.

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

fast = GeneICL(mode="classification", inference="fast")   # GeneICL-Fast: 2 forward passes, no inner CV
```

`X` is samples x genes in log1p-CPM. The training and test rows must share the same columns; any gene set works,
since the model encodes and PCA-reduces every task itself. `fit` only stores the labelled support set, and
`predict` runs the model in context, without training. Classification with more than 10 classes runs one-vs-rest.

**Inference** (`inference=`, same for every task):
- `"normal"` (default, **GeneICL**): the model reads the task under 6 input configurations (PCA whitening
  strength x all genes / the 5000 most variable ones), each the mean of the plain and a gene-weighted input, and
  weights them by a 3-fold cross-validation on the support set (macro-F1 for classification, R² for regression).
  For classification the same inner CV picks how to read out the classes (all classes at once, one-vs-rest or
  one-vs-one). Nothing is trained; the inner CV only chooses how to read the frozen model.
- `"fast"` (**GeneICL-Fast**): the mean of the plain and the gene-weighted input, 2 forward passes, no inner CV.
  GeneICL-Fast is about 30 times faster and slightly less accurate (see the results in `CHANGELOG.md`).

**Checkpoints** (`ckpt=`): the default is `geneicl.CKPT_CLS` (classification) or `geneicl.CKPT_REG` (regression and
survival), seed 0. Seeds 1 and 2 are next to them in the repository, and `geneicl.CKPT_V1` is the v1.0.0 model:
`GeneICL(mode, ckpt=geneicl.CKPT_V1, inference="fast")` reproduces the v1.0.0 default. `device=` defaults to CUDA
if available.

For survival, GeneICL regresses the support's Cox martingale residuals in context and the prediction is
the log-risk. The Breslow baseline hazard is fit on 5-fold out-of-fold scores of the support set.

Lower level: `geneicl.Runner` runs one checkpoint in context (support + query chunks per forward, bf16 on GPU) in
either inference mode and is what both `GeneICL` and the benchmark use; `geneicl.model.load_ckpt` rebuilds the
network from a checkpoint.

## Reproducing the benchmark (`eval/`)

`eval/` imports `geneicl` from the checkout, so it runs without installing the package. The pip package needs
only `torch` and `numpy`, but **running the benchmark needs the extra packages in `eval/environment.yaml`**
(pinned numpy/scipy/scikit-learn/pandas, hydra, the classical baselines and the in-context tabular FMs):

```bash
conda env create -f eval/environment.yaml && conda activate geneicl-benchmark
git lfs pull                             # the task suites and some baseline weights
cd eval/benchmark
python benchmark.py model=geneicl out=results/geneicl_s0                              # GeneICL, seed 0
python benchmark.py model=geneicl geneicl_inference=fast out=results/geneicl_fast_s0  # GeneICL-Fast
```

`eval/benchmark/README.md` documents every model, its options and the reported configuration (3 seeds x
GeneICL and GeneICL-Fast). No result files are shipped: every run writes to `eval/benchmark/results/`. The paper's
numbers (v1.0.0) are reproduced by the `v1.0.0` tag of this repository.

## Repository structure

```
pyproject.toml      # the pip package: geneicl/ only
CHANGELOG.md        # release notes
logo.svg            # the GeneICL logo
quickstart.ipynb    # example notebook: classification, regression and survival
geneicl/
  __init__.py       # the API: GeneICL(mode=classification|regression|survival, inference=normal|fast).fit / predict, and Runner
  model.py          # the GeneICL network + load_ckpt (rebuild a checkpoint from its saved `arch`)
  checkpoints/      # classification (C20RFbal), regression (R20RFs20Yv) and v1.0.0 (trm_segmented8) checkpoints, seeds 0/1/2
eval/               # manuscript reproduction (not part of the package)
  data/             # frozen task suites: benchmark/ (52 tasks) and non_benchmark/ (6 tasks)
  benchmark/        # benchmark harness, baselines and the results table script (results/ is written by the runs)
  environment.yaml  # conda env for everything under eval/ (not needed for the pip package)
```

`geneicl/checkpoints/`, `eval/` and the `eval/` subfolders have their own `README.md` with details.
