#!/usr/bin/env python
"""Zero-shot COMPASS features for the benchmark datasets -- same role as bf_extract.py / scgpt_extract.py.
Frozen TCGA-pretrained COMPASS (neutral MEAN cancer token, see compass_common.py) over each sample's TPM
profile; saves per dataset the 44 concept + 133 gene-set scores (177-d; with features=vector, project()'s
32-d vector per gene set/concept, flattened to 5600-d) for the downstream EN/logistic probe
(benchmark.py model=compass emb_dir=<out>).

    CUDA_VISIBLE_DEVICES=0 python compass_extract.py                    # all datasets -> <out>/<dataset>.npz (emb, obs_names)
    CUDA_VISIBLE_DEVICES=0 python compass_extract.py datasets=[BeatAML] # a subset (defaults: CONFIG in this file)

Needs the FM venv (fm_common.py) and the GENCODE v36 GTF (see compass_common.py).

Then:  python benchmark.py model=compass emb_dir=<out>   # -> results/compass_{regression,classification}.csv
"""
import numpy as np
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

import compass_common as C
from fm_common import DEV, EMB_DIR, run_extract


CONFIG = {                        # defaults; override on the CLI as key=value (Hydra)
    "ckpt": str(C.CKPT),                  # TCGA self-supervised checkpoint (not the ICI-finetuned ones)
    "out": str(EMB_DIR / "compass"),      # -> <out>/<dataset>.npz; benchmark.py emb_dir=
    "datasets": None,             # e.g. datasets=[BeatAML]; None = all in index.json
    "batch": 256,
    # concepts_genesets (44 concept + 133 gene-set scores = 177-d) | vector (32-d per gene set/concept = 5600-d)
    "features": "concepts_genesets",
}
ConfigStore.instance().store(name="compass_extract", node=CONFIG)


@hydra.main(version_base="1.3", config_path=None, config_name="compass_extract")
def main(a: DictConfig) -> None:
    assert a.features in ("concepts_genesets", "vector"), f"features must be concepts_genesets|vector, got {a.features}"
    m = C.load_compass(a.ckpt, device=DEV)

    def embed_dataset(ds):
        df, pos, cov = C.align_dataset(ds, m.feature_name)
        print(f"[compass] {ds}: {len(df)} samples | gene coverage {cov:.1%}"
              + ("  <-- LOW" if cov < C.MIN_COVERAGE else ""), flush=True)
        if a.features == "vector":
            gs, ct = m.project(df, batch_size=a.batch)
            emb = np.concatenate([np.asarray(gs, np.float32).reshape(len(df), -1),
                                  np.asarray(ct, np.float32).reshape(len(df), -1)], 1)
        else:
            gs, ct = m.extract(df, batch_size=a.batch)
            emb = np.concatenate([ct.values, gs.values], 1).astype(np.float32)   # 44 concepts, 133 gene sets
        assert np.isfinite(emb).all(), f"{ds}: non-finite COMPASS features"
        return emb, list(df.index)

    run_extract("compass", str(C.DATA), a.out, embed_dataset,
                datasets=list(a.datasets) if a.datasets is not None else None)


if __name__ == "__main__":
    main()
