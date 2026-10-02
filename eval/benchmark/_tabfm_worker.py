#!/usr/bin/env python
"""Persistent TabFM inference worker — runs in the TabFM py3.11 venv (its torch clobbers the stack
env's). benchmark.py's `tabfm` model starts ONE of these and streams folds to it, so the 6.2GB model
loads ONCE instead of per fold.

Protocol (stdin/stdout, one request per line):
  in  : a line = path to an input .npz holding Xtr (n,d) f32, y (n,), Xte (m,d) f32, kind, seed
  out : '__TABFM__ OK <path>.out.npz'  (pred (m,) reg or (m,C) proba)  |  '__TABFM__ ERR <repr>'
Only the protocol goes to stdout; all library chatter (HF/tqdm/warnings) is redirected to stderr so
it can't corrupt the framing. Models are loaded lazily per kind and cached, so a regression-only job
never loads the classification model (halves VRAM under the reg/clf split)."""
import os
import sys

import numpy as np
import torch

_OUT = sys.stdout              # protocol channel
sys.stdout = sys.stderr        # library prints (tqdm, warnings) go to the job log, not the protocol
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_models = {}                   # kind -> loaded backbone (loaded once, reused across folds)


def _emit(msg):
    print(msg, file=_OUT, flush=True)


def _model(kind):
    if kind not in _models:
        from tabfm import tabfm_v1_0_0_pytorch as T
        # TABFM_CKPT: local snapshot of google/tabfm-1.0.0-pytorch (dir with classification/ regression/); unset = HF download
        _models[kind] = T.load(model_type="regression" if kind == "reg" else "classification",
                               checkpoint_path=os.environ.get("TABFM_CKPT"), device=DEV)
    return _models[kind]


def _run(path):
    z = np.load(path, allow_pickle=True)
    Xtr, Xte = z["Xtr"].astype(np.float32), z["Xte"].astype(np.float32)
    y, kind, seed = z["y"], str(z["kind"]), int(z["seed"])
    backbone = _model(kind)
    if kind == "reg":
        from tabfm import TabFMRegressor
        m = TabFMRegressor(model=backbone, random_state=seed)
        m.fit(Xtr, np.asarray(y, np.float64))
        pred = np.asarray(m.predict(Xte), np.float64)
    else:
        from tabfm import TabFMClassifier
        m = TabFMClassifier(model=backbone, random_state=seed)
        m.fit(Xtr, np.asarray(y))              # y already relabeled 0..C-1 by _ICL
        pred = np.asarray(m.predict_proba(Xte), np.float64)   # cols 0..C-1
    out = path + ".out.npz"
    np.savez(out, pred=pred)
    return out


for line in sys.stdin:
    p = line.strip()
    if not p:
        continue
    try:
        _emit("__TABFM__ OK " + _run(p))
    except Exception as e:                     # keep the worker alive across a bad fold
        _emit("__TABFM__ ERR " + repr(e))
