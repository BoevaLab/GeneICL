# eval/

The manuscript reproduction: the frozen task suites and the benchmark harness with all baselines (no result files
are shipped; the scripts write them). GeneICL itself is the `geneicl` package at the repo root (`geneicl.Runner` runs the checkpoints);
the scripts here import it from the checkout, so it does not need to be installed.

```
eval/
  data/
    benchmark/       # the frozen 52-task benchmark (27 classification + 25 regression over 32 datasets)
    non_benchmark/   # 5 further tasks (the training-time monitor) + 1 survival task
  benchmark/         # benchmark.py harness, baselines (fm_baselines/, weights/) and the results table script
```

`benchmark/` has its own README.

## Task suites (`data/`)

Both suites use the same frozen format: `index.json` (task list, dataset row counts, `n_genes`),
`reference_genes.txt` (the 20021 gene symbols, i.e. the column order), `datasets/<dataset>.npz` (raw log1p-CPM, via
git LFS) and `tasks/<task_id>.npz` (sample ids, labels, task type, dataset, optional `group`; survival tasks add `event`). `benchmark.py` reads
`data/benchmark/` by default; `data_dir=../data/non_benchmark` points it at the other suite.

`non_benchmark/` was the training-time monitor; the benchmark was never used during training.
- Classification: GSE193677 `historemiss`, GSE251778 `mdd_diagnosis`, GSE102556 `brain_region`.
- Regression: GSE80655 `age_at_death`, GSE124284 `age_at_draw` (day-0 visit only).
- Survival: GSE244807 `os` (intrahepatic cholangiocarcinoma, overall survival in months; `y` = time, `event` = 1 for
  death, 0 for censored). `benchmark.py` skips survival tasks.
- GSE193677, GSE102556 and GSE80655 carry `group` (patient/donor).
