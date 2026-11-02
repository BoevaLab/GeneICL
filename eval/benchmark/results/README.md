# results/

No result files are shipped; this folder is where the runs write them.

    results/                   the runs read by make_main_table.yaml (GeneICL, in-context FMs, FM probes, tuned classical)
    results/main_table.csv, main_table_extra.csv   make_main_table.py output

One file per (model, task type), written directly by `benchmark.py`:

    <model>_<task_type>.csv          per-fold metric panel (the table/aggregation input)
    <model>_regression.oof.csv       per-sample OOF predictions   -> pooled-OOF R2
    <model>_classification.clfoof.npz  per-sample class probabilities -> ensembles, calibration

`overwrite=true` (default) deletes both `out=` CSVs before running. `overwrite=false` resumes: cached folds are
kept, missing folds and folds whose CV split changed since the cached run are re-scored. Companions are always merged per (model, task_id, fold),
never wiped.

GeneICL tags: `geneicl_s<seed>` (`geneicl_inference=normal`) and `geneicl_fast_s<seed>` (`geneicl_inference=fast`),
seeds 0/1/2 (`geneicl/checkpoints/C20RFbal_s<seed>.pt` on classification tasks, `R20RFs20Yv_s<seed>.pt` on regression
tasks).
