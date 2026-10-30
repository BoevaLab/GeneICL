# GeneICL checkpoints

The GeneICL checkpoints. The pip package ships the seed-0 file of each family (`geneicl.CKPT_CLS`, `geneicl.CKPT_REG`,
`geneicl.CKPT_V1`); seeds 1 and 2 are only in this repository (PyPI's file-size limit), so use a git checkout for them.
Baseline weights live in `eval/benchmark/weights/` (see its README).
SHA-256 = first 16 hex characters of the file's checksum (`sha256sum <file>`).

## GeneICL (ours) — tracked

| file | in the pip package | notes | SHA-256 |
|---|---|---|---|
| `C20RFbal_s0.pt` / `_s1.pt` / `_s2.pt` | `_s0` (`CKPT_CLS`, default for classification) | **GeneICL classification** (since 1.1.0): trained on classification tasks only, rank-Gauss input encoding (each gene's ECDF against the support rows, probit-transformed), 32 learned Fourier features of the PC scores, class-balanced cross-entropy, 20k steps, seeds 0/1/2 | `50aba0534331e081` / `cfb53907272b8db2` / `3b18877b6ccc529e` |
| `R20RFs20Yv_s0.pt` / `_s1.pt` / `_s2.pt` | `_s0` (`CKPT_REG`, default for regression and survival) | **GeneICL regression** (since 1.1.0): trained on regression tasks only (MSE), the same input encoding and Fourier features, rank + value label encoding (Gaussian rank scores plus the standardized label value), 20k steps, seeds 0/1/2 | `326fc5968ffe50e6` / `3f9e874e0c44aec9` / `063814b70b330954` |
| `trm_segmented8_s0.pt` / `_s1.pt` / `_s2.pt` | `_s0` (`CKPT_V1`) | **GeneICL v1.0.0** (the paper's model): one model for classification and regression, z-scored inputs, segmented recurrent model, depth 8, nhead=8, encoder without the gene->CLS FFN (`enc_cls_ffn=false`), seeds 0/1/2 | `52609d58f5bb7185` / `b6cbb95328eff8bb` / `db3e478f38c66cec` |

All three families share the architecture (4.25M used parameters; 4.24M for `trm_segmented8`). "Used params" excludes
the unused looped ICL block (`loop_layers`, ~2.1M), which is still in every state dict (see `../model.py`); the full
state dicts hold 6.4M parameters. A classification checkpoint is not meant for regression and vice versa;
`trm_segmented8` does both. `GeneICL(mode, ckpt=geneicl.CKPT_V1, inference="fast")` is the v1.0.0 default (one
forward pass): the gene-weighted view of the inference modes only acts on the rank-Gauss checkpoints, so on
`trm_segmented8` the runner skips it.

`eval/benchmark/benchmark.py` defaults to `C20RFbal_s0.pt` on classification tasks and `R20RFs20Yv_s0.pt` on
regression tasks (`GENEICL_DEFAULT_CKPT`).
