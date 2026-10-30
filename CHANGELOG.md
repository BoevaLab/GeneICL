# Changelog

## 1.1.0 (2026-10-30)

New checkpoints, one for classification and one for regression and survival, and two inference modes: **GeneICL**
(the default) and **GeneICL-Fast**. GeneICL improves on v1.0.0 on every benchmark metric below; GeneICL-Fast keeps
most of the gain at the cost of 2 forward passes.

### Added
- `inference=` for `GeneICL` and `geneicl.Runner`, the same for every task:
  - `"normal"` (default, **GeneICL**): the model reads the task under 6 input configurations (PCA whitening
    exponent .25 / .5 / .75 x all genes / the 5000 of highest support variance), each the mean of the plain and a
    gene-weighted input (each rank-encoded gene scaled by its support SD over the mean SD, capped at 5). The
    configurations are weighted by exp((inner score - best) / (.4 / sqrt(n support rows))), where the inner score
    is the macro-F1 (classification) / R² (regression) of a 3-fold cross-validation on the support set. For
    classification the same inner CV picks the read-out of each configuration: all classes at once, one-vs-rest or
    one-vs-one. Nothing is trained; the inner CV only chooses how to read the frozen model.
  - `"fast"` (**GeneICL-Fast**): the mean of the plain and the gene-weighted input (2 forward passes), no inner CV.
- Checkpoints `C20RFbal_s{0,1,2}` (classification) and `R20RFs20Yv_s{0,1,2}` (regression and survival). Both use a
  rank-Gauss input encoding (each gene's ECDF against the support rows, probit-transformed) and learned Fourier
  features of the PC scores. The classification model is trained on classification tasks only with class-balanced
  cross-entropy; the regression model on regression tasks only, with a rank + value label encoding.
- `geneicl.CKPT_CLS`, `geneicl.CKPT_REG` (the defaults, seed 0) and `geneicl.CKPT_V1` (the v1.0.0 model).
  `GeneICL(mode, ckpt=geneicl.CKPT_V1, inference="fast")` is the v1.0.0 default.
- Classification with more than 10 classes (one-vs-rest).

### Changed
- `ckpt=None` picks the checkpoint by task: `CKPT_CLS` for classification, `CKPT_REG` for regression and survival.
- The pip package ships the seed-0 checkpoint of each family (`C20RFbal_s0`, `R20RFs20Yv_s0`, `trm_segmented8_s0`);
  seeds 1 and 2 are in the repository (`geneicl/checkpoints/`), because all nine would exceed PyPI's file-size limit.
- `geneicl.Runner(ckpt, device=None, query_chunk=256, inference="normal")`: `ckpt` is required, and
  `predict_fold` / `predict_proba_fold` no longer take `pca_var`.
- Benchmark (`eval/benchmark/benchmark.py`): `geneicl_inference`, `geneicl_ckpt_cls` / `geneicl_ckpt_reg` (defaults
  per task type; `geneicl_ckpt` still sets one checkpoint for every task). The reported configuration is 3 seeds x
  {GeneICL, GeneICL-Fast}.

### Removed
- `ensemble_seed=` (the label-free test-time ensemble of support subsets x PCA thresholds) and the benchmark's
  `geneicl_support_ens` / `geneicl_pca_ens`. GeneICL-Fast is already better than the v1.0.0 ensemble on most
  metrics, at a fraction of its cost (table below).
- `geneicl.CKPT`: use `CKPT_CLS` / `CKPT_REG`, or `CKPT_V1` for the v1.0.0 model.

### Results
Benchmark (`eval/data/benchmark`: 27 classification + 25 regression tasks, 5-fold CV), mean over checkpoint seeds
0-2 (SD over seeds in brackets). AUROC, AUPRC, balanced accuracy, macro-F1, MCC, Pearson and Spearman are the mean
over folds, then tasks; R² is pooled out-of-fold per task.

| | AUROC | AUPRC | Bal. acc. | Macro-F1 | MCC | Pearson | Spearman | R² |
|---|---|---|---|---|---|---|---|---|
| v1.0.0 | .741 (.004) | .689 (.002) | .588 (.003) | .569 (.003) | .283 (.006) | .521 (.001) | .533 (.001) | .271 (.004) |
| v1.0.0, test-time ensemble¹ | .742 (.002) | .695 (.004) | .592 (.006) | .576 (.006) | .292 (.008) | .528 (.001) | .537 (.001) | .274 (.005) |
| 1.1.0 GeneICL-Fast | .745 (.002) | .692 (.004) | .635 (.003) | .603 (.006) | .346 (.003) | .530 (.000) | .544 (.001) | .288 (.005) |
| **1.1.0 GeneICL** | **.751** (.002) | **.703** (.002) | **.648** (.005) | **.616** (.007) | **.364** (.009) | **.532** (.001) | **.545** (.001) | **.289** (.006) |

¹ The benchmark's setting: 32 support views x 3 PCA thresholds (the API's `ensemble_seed` used 8 x 3).

The class-balanced classification model trades plain accuracy for the minority classes: accuracy .680 (v1.0.0) ->
.656 (GeneICL), and the predicted probabilities are less sharp on imbalanced tasks (log loss .661 -> .717, Brier
.413 -> .459), while balanced accuracy, macro-F1 and MCC improve the most.

Survival (GeneICL-Cox on GSE244807, intrahepatic cholangiocarcinoma overall survival, 225 patients; 5-fold CV x 5
repeats, Harrell's C, mean over seeds 0-2): v1.0.0 .730, GeneICL-Fast .735, GeneICL .735.

Cost (mean GPU seconds per CV fold on the benchmark, classification / regression): v1.0.0 0.03 / 0.10, v1.0.0
test-time ensemble 2.2 / 8.1, GeneICL-Fast 0.10 / 0.24, GeneICL 3.6 / 6.4. GeneICL-Fast is about 30 times faster
than GeneICL.

The paper's results (v1.0.0) are reproduced by the `v1.0.0` tag of this repository.

## 1.0.0 (2026-10-07)

Initial release, the model of the paper ([arXiv 2610.08694](https://arxiv.org/abs/2610.08694)): one checkpoint
family (`trm_segmented8_s{0,1,2}`) for classification, regression and survival (GeneICL-Cox), a scikit-learn-style
API (`GeneICL(mode=...)`), an optional label-free test-time ensemble (`ensemble_seed=`), and the benchmark
reproduction in `eval/`.
