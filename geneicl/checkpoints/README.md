# GeneICL checkpoints

The GeneICL checkpoints, shipped inside the `geneicl` package (`geneicl.CKPT` = seed 0, the default).
Baseline weights live in `eval/benchmark/weights/` (see its README).
SHA-256 = first 16 hex characters of the file's checksum (`sha256sum <file>`).

## GeneICL (ours) — tracked

| file | size | notes | SHA-256 |
|---|---|---|---|
| `trm_segmented8_s0.pt` / `_s1.pt` / `_s2.pt` | 25 MB each | **GeneICL**: segmented recurrent model, depth 8, seeds 0/1/2, nhead=8, encoder without the gene->CLS FFN (`enc_cls_ffn=false`), 4.24M used params | `52609d58f5bb7185` / `b6cbb95328eff8bb` / `db3e478f38c66cec` |

"Used params" excludes the unused looped ICL block (`loop_layers`, ~2.1M), which is still in every state
dict (see `../model.py`); the full state dicts hold 6.39M parameters. `trm_segmented8_s0.pt` is the default
checkpoint of `eval/benchmark/benchmark.py` (`GENEICL_DEFAULT_CKPT`).
