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

GeneICL tags: `geneicl_trm_s<seed>` (single model) and `geneicl_trm_supp_thresh_k32_s<seed>` (32 support views =
full support + 31 seeded 90% class-stratified subsets, x 3 PCA thresholds {0.9, 0.95, 0.98}, averaged within
the seed), seeds 0/1/2 (`geneicl/checkpoints/trm_segmented8_s<seed>.pt`).