#!/usr/bin/env python
"""GeneICL (segmented) benchmark harness — reproduce the paper's GeneICL results on the frozen eval suite.

Scores a GeneICL checkpoint (geneicl/checkpoints/trm_segmented8_s*.pt) in-context on RAW genes (the model does its own per-fold
standardization + internal PCA). Optional test-time ensembles (all label-free, default off):
    geneicl_support_ens=K   support-view ensemble: K class-stratified 90% support subsets, averaged
    geneicl_pca_ens=[...]   PCA-threshold ensemble: average over internal pca_var thresholds

One run scores the whole suite (regression and classification tasks). Per fold it records the metric panel
(clf: auroc/auprc/f1/balanced_accuracy/accuracy/mcc/nll/brier/ece; reg: pearson/spearman/r2/mape) in
`<out>_{regression,classification}.csv` AND writes per-sample companions so pooled-OOF metrics are recomputable
offline: `<out>_regression.oof.csv` (regression per-sample preds -> pooled-OOF R2) and
`<out>_classification.clfoof.npz` (classification per-sample class probabilities). Aggregate with
pooled_oof_table.py.

    python benchmark.py model=geneicl geneicl_ckpt=../../geneicl/checkpoints/trm_segmented8_s0.pt out=results/geneicl_trm_s0
    # ensembled variant: add geneicl_support_ens=32 geneicl_pca_ens=[0.9,0.95,0.98]
    # classical baselines (en|rf|catboost|lightgbm|mlp; always tuned on the PCA-90% input): python benchmark.py model=catboost
    # in-context FM competitors (tabpfn|tabpfn_api|tabicl|tabdpt|limix|tabfm; PCA-95% front-end, no tuning; GPU): python benchmark.py model=tabdpt
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time
import traceback
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import pandas as pd
import psutil
from omegaconf import DictConfig
from scipy.stats import loguniform, pearsonr, randint, spearmanr, uniform
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             matthews_corrcoef, mean_absolute_percentage_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import (GroupKFold, KFold,
                                     StratifiedGroupKFold, StratifiedKFold)
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data" / "benchmark"     # the frozen benchmark: eval/data/benchmark
GENEICL_REPO = HERE.parents[1]                         # repo root: the geneicl/ package lives here
GENEICL_DEFAULT_CKPT = GENEICL_REPO / "geneicl" / "checkpoints" / "trm_segmented8_s0.pt"   # GeneICL, seed 0
# In-context FMs (no tuning). All consume the same PCA(95% var) front-end as the sklearn baselines.
WEIGHTS = HERE / "weights"                             # FM checkpoints (see weights/README.md)
# LimiX has no pip package; its code is imported from a clone via sys.path:
#   git clone https://github.com/limix-ldm-ai/LimiX external/LimiX && git -C external/LimiX checkout 61ede97
LIMIX_REPO = os.environ.get("LIMIX_REPO", str(GENEICL_REPO / "external" / "LimiX"))
LIMIX_CKPT = WEIGHTS / "LimiX-16M.ckpt"      # stable-ai/LimiX-16M
TABDPT_CKPT = WEIGHTS / "tabdpt" / "tabdpt1_2.safetensors"   # TabDPT v1.2 = TabDPT-Turbo (Layer6/TabDPT)
LIMIX_REG_CFG = f"{LIMIX_REPO}/config/reg_default_noretrieval.json"
LIMIX_CLS_CFG = f"{LIMIX_REPO}/config/cls_default_noretrieval.json"
# TabFM runs in its own py3.11 venv (its torch pin conflicts with eval/environment.yaml's):
#   python3.11 -m venv external/tabfm-venv && external/tabfm-venv/bin/pip install \
#       "tabfm[pytorch] @ git+https://github.com/google-research/tabfm@f9cdd395aa8b744912d8a308fd95680459e565a8"
TABFM_PY = os.environ.get("TABFM_PY", str(GENEICL_REPO / "external" / "tabfm-venv" / "bin" / "python"))
TABFM_WORKER = HERE / "_tabfm_worker.py"
TABPFN3_CKPT_DIR = Path.home() / ".cache" / "tabpfn"   # tabpfn-v3-{classifier,regressor}-v3_default.ckpt
ICL_MODELS = {"tabpfn", "tabpfn_api", "tabicl", "tabdpt", "limix", "tabfm"}   # never tuned
N_SPLITS, N_ITER, INNER_CV, PCA_VAR = 5, 50, 5, 0.95   # tuned: 50-iter random search, inner 5-fold
TUNE_PCA_VAR = 0.90   # classical models (trees included) are tuned + scored on PCA-90%
CLASSICAL = {"en", "rf", "catboost", "lightgbm", "mlp"}           # always tuned (_tune: random search, inner CV)
# frozen-FM embeddings (fm_baselines/*_extract.py -> <emb_dir>/<dataset>.npz {emb, obs_names}) replace the raw
# genes; the probe is an untuned `en` (shared PCA-95% front-end + elastic net / EN-logistic with default
# hyperparameters).
EMB_MODELS = {"compass", "bulkformer", "scfoundation", "scgpt", "bulkrnabert"}
METRIC_COLS = ["auroc", "auprc", "f1_macro", "f1_weighted", "balanced_accuracy",
               "accuracy", "mcc", "nll", "brier", "ece", "pearson", "spearman", "r2", "mape"]
COLUMNS = (["model", "dataset", "task_id", "task_type", "fold",
            "n_train", "n_test", "n_classes", "train_seconds",
            "peak_ram_mb", "peak_vram_mb"] + METRIC_COLS)


# ── peak RAM / VRAM sampling (whole process tree, incl. the TabFM subprocess) ────
_NVML = None   # None=uninit, False=unavailable, else list of device handles


def _nvml_handles():
    """Init NVML once; return device handles, or False if no GPU / no NVML (e.g. login node)."""
    global _NVML
    if _NVML is None:
        try:
            import pynvml
            pynvml.nvmlInit()
            _NVML = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            _NVML = False
    return _NVML


def _vram_bytes(pids):
    """GPU memory used by our process tree across all visible devices; None if NVML unavailable."""
    hs = _nvml_handles()
    if not hs:
        return None
    import pynvml
    tot = 0
    for h in hs:
        try:
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                if p.pid in pids and p.usedGpuMemory:
                    tot += p.usedGpuMemory
        except Exception:
            pass
    return tot


class _MemMonitor:
    """Sample peak RSS (self + children) and peak per-process VRAM over a code block. A daemon thread
    polls every `interval`s — so it captures the TabFM subprocess and loky workers, which are alive
    only while the (blocking) fit/predict call runs. peak_vram_mb is NaN when no GPU/NVML."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self.peak_rss = 0
        self.peak_vram = None

    def _sample(self):
        pids = {self._proc.pid}
        try:
            pids |= {c.pid for c in self._proc.children(recursive=True)}
        except psutil.Error:
            pass
        rss = 0
        for pid in pids:
            try:
                rss += psutil.Process(pid).memory_info().rss
            except psutil.Error:
                pass
        self.peak_rss = max(self.peak_rss, rss)
        v = _vram_bytes(pids)
        if v is not None:
            self.peak_vram = max(self.peak_vram or 0, v)

    def _run(self):
        self._sample()
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()

    @property
    def peak_ram_mb(self):
        return self.peak_rss / 1e6

    @property
    def peak_vram_mb(self):
        return np.nan if self.peak_vram is None else self.peak_vram / 1e6


# ── in-context FM adapter ───────────────────────────────────────────────────────
class _ICL:
    """sklearn-shaped facade over one-shot in-context models (TabDPT/LimiX/TabFM). fit() only
    stashes the support set — the forward pass is in predict()/predict_proba(), so it's timed as
    inference. reg_fn/proba_fn(Xtr_f32, y, Xte_f32) do the actual call. Handles float32 casting,
    class relabel -> classes_ (models assume contiguous 0..C-1), the <=max_classes guard (raised
    BEFORE any model/CUDA op so a too-many-classes fold is skipped cleanly), and — for models whose
    regression head emits standardized targets (LimiX) — per-fold y standardize + inverse-transform."""

    def __init__(self, is_cls, reg_fn=None, proba_fn=None, reg_std=False, max_classes=None):
        self.is_cls, self.reg_fn, self.proba_fn = is_cls, reg_fn, proba_fn
        self.reg_std, self.max_classes = reg_std, max_classes

    def fit(self, X, y):
        self.X = np.asarray(X, np.float32)
        if self.is_cls:
            self.classes_ = np.unique(y)
            if self.max_classes and len(self.classes_) > self.max_classes:
                raise ValueError(f"{len(self.classes_)} classes > {self.max_classes}-class cap")
            m = {c: i for i, c in enumerate(self.classes_)}          # -> contiguous 0..C-1
            self.y = np.array([m[v] for v in y], np.int64)
        else:
            y = np.asarray(y, np.float64)
            self.mu, self.sd = float(y.mean()), (float(y.std()) or 1.0)
            self.y = (y - self.mu) / self.sd if self.reg_std else y
        return self

    def predict(self, Xte):
        out = np.asarray(self.reg_fn(self.X, self.y, np.asarray(Xte, np.float32)), np.float64)
        return out * self.sd + self.mu if self.reg_std else out

    def predict_proba(self, Xte):                                    # cols ordered like classes_
        return np.asarray(self.proba_fn(self.X, self.y, np.asarray(Xte, np.float32)), np.float64)


_TABFM_PROC = None   # persistent TabFM worker (loads the 6.2GB model ONCE, then serves every fold)


def _tabfm_proc():
    global _TABFM_PROC
    if _TABFM_PROC is None or _TABFM_PROC.poll() is not None:
        import subprocess
        _TABFM_PROC = subprocess.Popen([TABFM_PY, str(TABFM_WORKER)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       text=True, bufsize=1)   # stderr inherits -> job log
    return _TABFM_PROC


def _tabfm_close():
    if _TABFM_PROC is not None and _TABFM_PROC.poll() is None:
        try:
            _TABFM_PROC.stdin.close()          # EOF -> worker loop exits
        except Exception:
            pass
        _TABFM_PROC.terminate()


atexit.register(_tabfm_close)


def _tabfm_call(kind, Xtr, y, Xte, seed):
    """Send one fold to the persistent TabFM worker (model stays resident across folds). Arrays go
    over a temp npz; the worker replies '__TABFM__ OK <outpath>' on stdout (chatter is on stderr)."""
    import tempfile
    proc = _tabfm_proc()
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        np.savez(path, Xtr=Xtr, y=y, Xte=Xte, kind=kind, seed=seed)
        proc.stdin.write(path + "\n")
        proc.stdin.flush()
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("tabfm worker died (see job stderr)")
            if line.startswith("__TABFM__ "):
                status = line[len("__TABFM__ "):].strip()
                break
        if not status.startswith("OK "):
            raise RuntimeError(f"tabfm worker: {status}")
        outp = status[3:]
        pred = np.load(outp)["pred"]
        os.remove(outp)
        return pred
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _tabpfn_api_auth():
    """Authenticate tabpfn-client once: TABPFN_TOKEN env var, else the file named by TABPFN_TOKEN_FILE
    (default ~/.config/tabpfn/token). The token is never printed or written to the results."""
    import tabpfn_client
    if getattr(_tabpfn_api_auth, "done", False):
        return
    tok = os.environ.get("TABPFN_TOKEN")
    if not tok:
        f = Path(os.environ.get("TABPFN_TOKEN_FILE", Path.home() / ".config" / "tabpfn" / "token"))
        if not f.exists():
            raise RuntimeError(f"tabpfn_api: set TABPFN_TOKEN or put the API key in {f}")
        tok = f.read_text().strip()
    tabpfn_client.set_access_token(tok)
    _tabpfn_api_auth.done = True


# ── classical baselines (TabArena-aligned search spaces) and the shared PCA front-end ──
def reduce(Xtr, Xte, seed, var=PCA_VAR):
    """Impute+scale on the train fold; PCA keeping `var` of the variance (fit on train only)."""
    imp = SimpleImputer(strategy="mean", keep_empty_features=True).fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    Ztr, Zte = sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xte))
    pca = PCA(n_components=var, svd_solver="full", random_state=seed).fit(Ztr)
    return pca.transform(Ztr), pca.transform(Zte)


class _LogInt:
    """Log-uniform integer sampler for the search (TabArena's Integer(..., log=True))."""
    def __init__(self, lo, hi):
        self._d = loguniform(lo, hi)

    def rvs(self, random_state=None, **kw):
        return int(round(self._d.rvs(random_state=random_state)))


def _make(model, is_cls, seed, est_jobs):
    """Return (estimator, param_distributions). EN-classification = elastic-net logistic regression."""
    if model == "en":
        est = (LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                  C=1.0, max_iter=5000, random_state=seed) if is_cls
               else ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, random_state=seed))
        dist = ({"C": loguniform(1e-1, 1e3), "l1_ratio": uniform(0, 1)} if is_cls
                else {"alpha": loguniform(1e-3, 1e1), "l1_ratio": uniform(0, 1)})
    elif model == "rf":
        RF = RandomForestClassifier if is_cls else RandomForestRegressor
        est = RF(random_state=seed, n_jobs=est_jobs)
        rf_shared = {"n_estimators": randint(200, 600),
                     "max_features": uniform(0.4, 0.6),            # Float(0.4, 1.0)
                     "min_samples_split": randint(2, 5),          # Int(2, 4)
                     "min_impurity_decrease": loguniform(1e-5, 1e-3)}
        dist = [{**rf_shared, "bootstrap": [True], "max_samples": uniform(0.5, 0.5)},   # Float(0.5, 1.0)
                {**rf_shared, "bootstrap": [False]}]
    elif model == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        CB = CatBoostClassifier if is_cls else CatBoostRegressor
        est = CB(random_seed=seed, verbose=0, allow_writing_files=False, thread_count=est_jobs)   # CPU only
        # TabArena catboost/hpo.py ranges minus categorical-feature params; iterations capped at 1000 (tuned:
        # early-stopped per inner fold). Single-choice lists pin settings that are not searched.
        dist = {"iterations": [1000], "bootstrap_type": ["Bernoulli"], "boosting_type": ["Plain"],
                "learning_rate": loguniform(5e-3, 1e-1),
                "depth": randint(4, 9),                           # Int(4, 8)
                "l2_leaf_reg": loguniform(1e-4, 5.0),
                "subsample": uniform(0.7, 0.3),                   # Float(0.7, 1.0)
                "grow_policy": ["SymmetricTree", "Depthwise"],
                "leaf_estimation_iterations": randint(1, 21),     # Int(1, 20)
                "model_size_reg": loguniform(0.1, 1.5),
                "colsample_bylevel": uniform(0.85, 0.15)}         # Float(0.85, 1.0)
    elif model == "lightgbm":                         # TabArena lightgbm/hpo.py minus categorical params
        from lightgbm import LGBMClassifier, LGBMRegressor
        LGB = LGBMClassifier if is_cls else LGBMRegressor
        # histogram_pool_size caps LightGBM's per-model histogram cache (MB). Uncapped it is num_leaves x
        # 20k genes x 255 bins (~16 GB at 200 leaves), which OOMs parallel candidates; capping only trades
        # memory for recomputation, the fitted model is unchanged.
        est = LGB(random_state=seed, n_jobs=est_jobs, verbose=-1, histogram_pool_size=1024)   # CPU only
        dist = {"learning_rate": loguniform(5e-3, 1e-1),
                "feature_fraction": uniform(0.4, 0.6),        # Float(0.4, 1.0)
                "bagging_fraction": uniform(0.7, 0.3),        # Float(0.7, 1.0)
                "bagging_freq": [1],
                "num_leaves": _LogInt(2, 200),
                "min_data_in_leaf": _LogInt(1, 64),
                "extra_trees": [False, True],
                "lambda_l1": uniform(1e-4, 1.0),              # Float(1e-4, 1.0)
                "lambda_l2": uniform(1e-4, 2.0)}              # Float(1e-4, 2.0)
    elif model == "mlp":                              # plain sklearn MLP on the PCA front-end
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        MLP = MLPClassifier if is_cls else MLPRegressor
        est = MLP(hidden_layer_sizes=(256, 256), activation="relu", solver="adam",
                  alpha=1e-4, learning_rate_init=1e-3, early_stopping=True,
                  max_iter=300, random_state=seed)
        dist = {"hidden_layer_sizes": [(256,), (256, 256), (512, 256)],
                "alpha": loguniform(1e-5, 1e-2),
                "learning_rate_init": loguniform(1e-4, 1e-2)}
    elif model == "tabpfn":                           # local open-weights TabPFN v3, no tuning (dist={})
        import torch
        from tabpfn import TabPFNClassifier, TabPFNRegressor   # local package, weights in ~/.cache/tabpfn
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        TabPFN = TabPFNClassifier if is_cls else TabPFNRegressor
        # ignore_pretraining_limits: accept the >100 PCs our PCA yields (auto_scale_n_estimators +
        # memory_saving_mode='auto' are the package defaults -> wide inputs are split + chunked).
        # model_path pins v3 explicitly: tabpfn>=9 defaults to v3.5.
        ckpt = TABPFN3_CKPT_DIR / f"tabpfn-v3-{'classifier' if is_cls else 'regressor'}-v3_default.ckpt"
        est = TabPFN(model_path=str(ckpt), device=dev, random_state=seed, ignore_pretraining_limits=True)
        dist = {}
    elif model == "tabpfn_api":                       # TabPFN-3 through the Prior Labs API, no tuning (dist={})
        from tabpfn_client import TabPFNClassifier, TabPFNRegressor
        _tabpfn_api_auth()
        TabPFN = TabPFNClassifier if is_cls else TabPFNRegressor
        # "v3_default" = the same checkpoint the local `tabpfn` run loads (tabpfn-v3-*-v3_default.ckpt)
        est = TabPFN(model_path="v3_default", random_state=seed, ignore_pretraining_limits=True)
        dist = {}
    elif model == "tabicl":                           # in-context FM, no tuning (dist={})
        import torch
        from tabicl import TabICLClassifier, TabICLRegressor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = WEIGHTS / f"tabicl-{'classifier' if is_cls else 'regressor'}-v2-20260212.ckpt"
        TabICL = TabICLClassifier if is_cls else TabICLRegressor
        est = TabICL(n_estimators=8, device=dev, random_state=seed,
                     model_path=str(ckpt), allow_auto_download=False)   # offline: use local weights
        dist = {}
    elif model == "tabdpt":                           # in-context FM, no tuning (dist={})
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # flash/triton JIT fails on these nodes
        os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        def _reg(Xtr, y, Xte):
            from tabdpt import TabDPTRegressor
            r = TabDPTRegressor(device=dev, use_flash=False, model_weight_path=str(TABDPT_CKPT)); r.fit(Xtr, y)
            return r.predict(Xte)                     # default n_ensembles=8
        def _proba(Xtr, y, Xte):
            from tabdpt import TabDPTClassifier
            c = TabDPTClassifier(device=dev, use_flash=False, model_weight_path=str(TABDPT_CKPT)); c.fit(Xtr, y)
            return c.predict_proba(Xte)
        est, dist = _ICL(is_cls, _reg, _proba), {}
    elif model == "limix":                            # in-context FM, no tuning (dist={})
        import sys
        if LIMIX_REPO not in sys.path:
            sys.path.insert(0, LIMIX_REPO)
        for k, v in (("RANK", "0"), ("WORLD_SIZE", "1"),
                     ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29500")):
            os.environ.setdefault(k, v)               # LimiX has DDP-aware paths; pin single-process
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def _limix(cfg_json, task, Xtr, y, Xte):
            from inference.predictor import LimiXPredictor
            p = LimiXPredictor(device=dev, model_path=str(LIMIX_CKPT), inference_config=cfg_json)
            out = p.predict(Xtr, y, Xte, task_type=task)
            return out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
        est = _ICL(is_cls,
                   reg_fn=lambda a, b, c: _limix(LIMIX_REG_CFG, "Regression", a, b, c),
                   proba_fn=lambda a, b, c: _limix(LIMIX_CLS_CFG, "Classification", a, b, c),
                   reg_std=True, max_classes=10)       # reg head emits standardized y; cls caps at 10
        dist = {}
    elif model == "tabfm":                            # in-context FM in its own venv (subprocess), no tuning
        est = _ICL(is_cls,
                   reg_fn=lambda a, b, c: _tabfm_call("reg", a, b, c, seed),
                   proba_fn=lambda a, b, c: _tabfm_call("cls", a, b, c, seed))
        dist = {}
    else:
        raise ValueError(f"unknown model {model}")
    return est, dist


def _tune(model, est, dist, is_cls, Ztr, ytr, n_classes, seed, n_jobs):
    """Hyperparameter search shared by every classical model: N_ITER candidates x INNER_CV shuffled (stratified for
    classification) folds of the support set, scored by AUROC / R2 on each validation fold. The candidate with the
    best mean score is refit on the full support. CatBoost / LightGBM early-stop every inner fit on its validation
    fold (patience 50, <=1000 rounds) and are refit at the mean best round count. An inner fit that fails (e.g. a
    fold missing a class) is skipped. Candidates run in parallel (n_jobs workers x 1 thread)."""
    from joblib import Parallel, delayed
    from sklearn.base import clone
    from sklearn.model_selection import ParameterSampler
    splits = list((StratifiedKFold if is_cls else KFold)(INNER_CV, shuffle=True, random_state=seed).split(Ztr, ytr))

    def _score(m, X, y):
        if not is_cls:
            return r2_score(y, m.predict(X))
        p = m.predict_proba(X)
        return (roc_auc_score(y, p[:, 1]) if n_classes == 2
                else roc_auc_score(y, p, multi_class="ovr", labels=m.classes_))

    def _fit(params, tr, va):
        """One candidate on one inner training fold -> (fitted model, best round count or None)."""
        if model == "catboost":
            m = clone(est).set_params(**params, od_type="Iter", od_wait=50, use_best_model=True)
            m.fit(Ztr[tr], ytr[tr], eval_set=(Ztr[va], ytr[va]))
            return m, m.get_best_iteration() + 1
        if model == "lightgbm":
            from lightgbm import early_stopping
            m = clone(est).set_params(**params, n_estimators=1000)
            m.fit(Ztr[tr], ytr[tr], eval_set=[(Ztr[va], ytr[va])], callbacks=[early_stopping(50, verbose=False)])
            return m, m.best_iteration_ or 1000
        return clone(est).set_params(**params).fit(Ztr[tr], ytr[tr]), None

    def _run(params):
        scores, rounds, n_fit = [], [], 0
        for tr, va in splits:
            try:
                m, r = _fit(params, tr, va)
                n_fit += 1
                if r is not None:
                    rounds.append(r)
                scores.append(_score(m, Ztr[va], ytr[va]))
            except Exception as e:                    # e.g. an inner fold missing a class (multi-class AUROC)
                print(f"[tune-fail] {model} {type(e).__name__}: {e}", flush=True)
        return ((np.mean(scores) if scores else -np.inf), (int(round(np.mean(rounds))) if rounds else None), n_fit)

    cands = list(ParameterSampler(dist, n_iter=N_ITER, random_state=seed))
    res = Parallel(n_jobs=n_jobs)(delayed(_run)(p) for p in cands)
    ok = [(sc, p, r) for (sc, r, n_fit), p in zip(res, cands) if n_fit]
    if not ok:
        raise RuntimeError(f"{model} cv search: every candidate failed")
    sc, params, r = max(ok, key=lambda x: x[0])      # first best on ties
    refit = dict(params)                              # the refit uses all cores
    if model == "catboost":
        refit.update(iterations=r, thread_count=n_jobs)
    else:
        if model == "lightgbm":
            refit["n_estimators"] = r
        if "n_jobs" in est.get_params():
            refit["n_jobs"] = n_jobs
    rounds = f" {'iterations' if model == 'catboost' else 'n_estimators'}={r}" if r is not None else ""
    print(f"[tune] {model} cv={sc:+.3f}{rounds} {params}", flush=True)
    return clone(est).set_params(**refit).fit(Ztr, ytr)


def fit_and_predict(model, tuned, is_cls, Ztr, ytr, Zte, n_classes, seed, n_jobs):
    est, dist = _make(model, is_cls, seed, 1 if tuned else n_jobs)   # tuned: the search parallelizes over candidates
    with _MemMonitor() as mon:
        t = time.perf_counter()
        est = _tune(model, est, dist, is_cls, Ztr, ytr, n_classes, seed, n_jobs) if tuned else est.fit(Ztr, ytr)
        if not is_cls:
            out = est.predict(Zte)
        else:
            proba = est.predict_proba(Zte)
            out = np.zeros((len(Zte), n_classes))
            for j, c in enumerate(est.classes_):
                out[:, int(c)] = proba[:, j]          # map fold classes -> global 0..K-1 columns
        secs = time.perf_counter() - t
    return out, secs, mon.peak_ram_mb, mon.peak_vram_mb


# ── GeneICL in-context runner + test-time ensembles ─────────────────────────────
_GENEICL_RUNNER = None


def _geneicl_runner(geneicl_cfg):
    global _GENEICL_RUNNER
    if _GENEICL_RUNNER is None:
        import sys
        import torch
        if str(GENEICL_REPO) not in sys.path:   # the checkout's geneicl/ package, installed or not
            sys.path.insert(0, str(GENEICL_REPO))
        from geneicl import Runner
        ckpt = Path(geneicl_cfg["geneicl_ckpt"])
        if not ckpt.exists():
            raise SystemExit(f"geneicl: no checkpoint at {ckpt}; pass geneicl_ckpt=/path/to.pt")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _GENEICL_RUNNER = Runner(str(ckpt), dev, geneicl_cfg.get("query_chunk", 256))
    return _GENEICL_RUNNER


def _support_views(X, y, k, seed, is_cls):
    """The support sets to average over: the full support first, then k-1 seeded 90% subsets
    (class-stratified for classification, so every view carries the same class set and order)."""
    views = [(X, y)]
    for j in range(1, k):
        r = np.random.default_rng(seed + 1009 * (j + 1))
        if is_cls:
            sel = [ci[r.permutation(len(ci))][:max(1, int(round(0.9 * len(ci))))]
                   for ci in (np.where(y == c)[0] for c in np.unique(y))]
            sub = np.sort(np.concatenate(sel))
        else:
            sub = np.sort(r.permutation(len(X))[:max(1, int(round(0.9 * len(X))))])
        views.append((X[sub], y[sub]))
    return views


def fit_and_predict_geneicl(is_cls, Xtr, ytr, Xte, n_classes, geneicl_cfg):
    """GeneICL on RAW genes (no shared PCA front-end). Times only the in-context inference. The optional
    test-time ensemble (label-free, default off) averages the model's outputs -- class probabilities or
    scalar predictions -- over the product of support views (geneicl_support_ens) and internal PCA
    thresholds (geneicl_pca_ens); with both off this is a single plain forward pass."""
    runner = _geneicl_runner(geneicl_cfg)
    seed = int(geneicl_cfg.get("seed", 0))
    ytr = np.asarray(ytr)
    views = _support_views(Xtr, ytr, int(geneicl_cfg.get("support_ens", 1) or 1), seed, is_cls)
    pca_vars = list(geneicl_cfg.get("pca_ens") or []) or [None]      # None -> the checkpoint's own pca_var
    with _MemMonitor() as mon:
        t = time.perf_counter()
        acc, cls = None, None
        for Xv, yv in views:
            pv = None                                                  # average within a view first, then over
            for v in pca_vars:                                         # views -- the summation order the
                kw = {} if v is None else {"pca_var": v}               # published results were produced with
                if is_cls:
                    p, cls = runner.predict_proba_fold(Xv, yv, Xte, **kw)   # cls = sorted str labels present
                else:
                    p = np.asarray(runner.predict_fold(Xv, yv, Xte, **kw), np.float64)
                pv = p if pv is None else pv + p
            pv = pv / len(pca_vars)
            acc = pv if acc is None else acc + pv
        out = acc / len(views)
        if is_cls:
            full = np.zeros((len(Xte), n_classes))
            for j, c in enumerate(cls):
                full[:, int(c)] = out[:, j]                           # fold classes -> global 0..K-1
            out = full
        secs = time.perf_counter() - t
    return out, secs, mon.peak_ram_mb, mon.peak_vram_mb


# ── metrics ─────────────────────────────────────────────────────────────────────
def _safe(f):
    try:
        v = float(f())
        return v if v == v else np.nan               # NaN-guard
    except Exception:
        return np.nan


def _ece(P, y_true, nb=15):
    conf = P.max(1); pred = P.argmax(1); correct = (pred == y_true).astype(float); e = 0.0
    for i in range(nb):
        lo, hi = i / nb, (i + 1) / nb; msk = (conf > lo) & (conf <= hi)
        if msk.any():
            e += float(msk.mean()) * abs(float(correct[msk].mean()) - float(conf[msk].mean()))
    return e


def clf_metrics(y_true, full, K):
    pred = full.argmax(1)
    m = {c: np.nan for c in METRIC_COLS}
    if K == 2:
        m["auroc"] = _safe(lambda: roc_auc_score(y_true, full[:, 1]))
        m["auprc"] = _safe(lambda: average_precision_score(y_true, full[:, 1]))
    else:
        Y = np.eye(K)[y_true]
        m["auroc"] = _safe(lambda: roc_auc_score(y_true, full, multi_class="ovr",
                                                 average="macro", labels=list(range(K))))
        m["auprc"] = _safe(lambda: average_precision_score(Y, full, average="macro"))
    lab = list(range(K))
    m["f1_macro"] = _safe(lambda: f1_score(y_true, pred, average="macro", labels=lab, zero_division=0))
    m["f1_weighted"] = _safe(lambda: f1_score(y_true, pred, average="weighted", labels=lab, zero_division=0))
    m["balanced_accuracy"] = _safe(lambda: balanced_accuracy_score(y_true, pred))
    m["accuracy"] = _safe(lambda: accuracy_score(y_true, pred))
    m["mcc"] = _safe(lambda: matthews_corrcoef(y_true, pred))
    # calibration/probabilistic metrics (from the probability matrix `full`)
    P = np.clip(full, 1e-12, 1.0); P = P / P.sum(1, keepdims=True)
    oh = np.eye(K)[y_true]
    m["nll"] = _safe(lambda: float(-np.log(P[np.arange(len(y_true)), y_true]).mean()))
    m["brier"] = _safe(lambda: float(((P - oh) ** 2).sum(1).mean()))
    m["ece"] = _safe(lambda: _ece(P, y_true))
    return m


def reg_metrics(y_true, pred):
    m = {c: np.nan for c in METRIC_COLS}
    m["r2"] = _safe(lambda: r2_score(y_true, pred))
    if np.std(y_true) > 0 and np.std(pred) > 0:
        m["pearson"] = _safe(lambda: pearsonr(y_true, pred).statistic)
        m["spearman"] = _safe(lambda: spearmanr(y_true, pred).statistic)
    m["mape"] = _safe(lambda: mean_absolute_percentage_error(y_true, pred))   # unreliable on zero-heavy y
    return m


# ── data ──────────────────────────────────────────────────────────────────────
def make_splits(X, y, groups, is_cls, n_splits, seed):
    if is_cls:
        if groups is not None:
            n = min(n_splits, len(np.unique(groups)))
            try:
                return list(StratifiedGroupKFold(n_splits=n).split(X, y, groups=groups))
            except ValueError:
                return list(GroupKFold(n_splits=n).split(X, y, groups=groups))
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))
    if groups is not None:
        n = min(n_splits, len(np.unique(groups)))
        return list(GroupKFold(n_splits=n).split(X, y, groups=groups))
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))


_DS_CACHE: dict = {}
_EMB_DIR = None                                       # set in main() for EMB_MODELS: rows come from embeddings


def _dataset_matrix(data_dir, dataset):
    if dataset not in _DS_CACHE:
        _DS_CACHE.clear()
        if _EMB_DIR is not None:
            z = np.load(_EMB_DIR / f"{dataset}.npz", allow_pickle=True)
            X = z["emb"]
        else:
            z = np.load(data_dir / "datasets" / f"{dataset}.npz", allow_pickle=True)
            X = z["X"]
        pos = {str(s): i for i, s in enumerate(z["obs_names"].tolist())}
        _DS_CACHE[dataset] = (X, pos)
    return _DS_CACHE[dataset]


def _load_task(data_dir, t, n_splits):
    """Return X f32 (n,G), y, groups|None, classes|None, is_cls. Rows aligned to labels."""
    is_cls = t["task_type"] == "classification"
    z = np.load(data_dir / t["file"], allow_pickle=True)
    Xall, pos = _dataset_matrix(data_dir, t["dataset"])
    missing = [s for s in z["sample_ids"].tolist() if str(s) not in pos]
    if missing:
        raise KeyError(f"{t['task_id']}: {len(missing)} samples absent from the {t['dataset']} matrix"
                       f"{f' in {_EMB_DIR}' if _EMB_DIR is not None else ''}, e.g. {missing[:3]}")
    rows_idx = np.array([pos[str(s)] for s in z["sample_ids"].tolist()])
    raw = z["y"]
    groups = z["group"] if "group" in z.files else None
    if is_cls:
        y_str = raw.astype(str)
        vc = pd.Series(y_str).value_counts()
        keep = np.isin(y_str, vc[vc >= n_splits].index)
        classes = sorted(set(y_str[keep]))
        y = np.array([classes.index(v) for v in y_str[keep]])
    else:
        y = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(np.float64)
        keep = ~np.isnan(y)
        y, classes = y[keep], None
    rows_idx = rows_idx[keep]
    groups = groups[keep] if groups is not None else None
    X = np.asarray(Xall[rows_idx], np.float32)
    return X, y, groups, classes, is_cls


# ── driver ──────────────────────────────────────────────────────────────────────
def run_task(model, t, data_dir, seed, done, geneicl_cfg, n_jobs):
    X, y, groups, classes, is_cls = _load_task(data_dir, t, N_SPLITS)
    if len(X) < 10 or (is_cls and len(classes) < 2):
        print(f"[skip] {t['task_id']}: {len(X)} usable samples", flush=True)
        return [], [], []
    rows, oof, clf_oof = [], [], []                   # oof: per-sample reg preds; clf_oof: per-sample class probs
    for fold, (tr, te) in enumerate(make_splits(X, y, groups, is_cls, N_SPLITS, seed)):
        # resume (overwrite=false) only skips a fold whose CACHED split matches the current one, so a task
        # redefined since that run (different n_train/n_test) is re-tested instead of silently kept stale
        if done.get((model, t["task_id"], fold)) == (len(tr), len(te)) or len(tr) < 2 or len(te) < 1:
            continue
        row = dict(model=model, dataset=t["dataset"], task_id=t["task_id"],
                   task_type=t["task_type"], fold=fold, n_train=len(tr), n_test=len(te),
                   n_classes=len(classes) if is_cls else 0, train_seconds=np.nan,
                   peak_ram_mb=np.nan, peak_vram_mb=np.nan,
                   **{c: np.nan for c in METRIC_COLS})
        try:
            n_cls = len(classes) if is_cls else 0
            if model == "geneicl":                    # raw genes, its own preprocessing
                pred, secs, ram_mb, vram_mb = fit_and_predict_geneicl(is_cls, X[tr], y[tr], X[te], n_cls, geneicl_cfg)
            else:
                tuned = model in CLASSICAL                # in-context FMs and the FM probes are not tuned
                Ztr, Zte = reduce(X[tr], X[te], seed, TUNE_PCA_VAR if tuned else PCA_VAR)
                fit_model = "en" if model in EMB_MODELS else model      # frozen-FM embeddings -> EN probe
                pred, secs, ram_mb, vram_mb = fit_and_predict(fit_model, tuned, is_cls, Ztr, y[tr], Zte, n_cls, seed, n_jobs)
            row["train_seconds"] = round(secs, 3)
            row["peak_ram_mb"] = round(ram_mb, 1) if ram_mb == ram_mb else np.nan
            row["peak_vram_mb"] = round(vram_mb, 1) if vram_mb == vram_mb else np.nan
            row.update(clf_metrics(y[te], pred, len(classes)) if is_cls else reg_metrics(y[te], pred))
        except Exception as e:                       # one bad fold must not kill the sweep
            print(f"[fail] {model} {t['task_id']} f{fold}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue
        key = "auroc" if is_cls else "spearman"
        print(f"[ok] {model} {t['task_id']} f{fold}: {key}={row[key]:+.3f} ({row['train_seconds']:.1f}s)", flush=True)
        rows.append(row)
        if not is_cls:                                # per-sample OOF preds -> pooled-OOF R2 offline
            p = np.asarray(pred, np.float64).ravel()
            oof.extend({"model": model, "task_id": t["task_id"], "fold": fold,
                        "y_true": float(a), "pred": float(b)} for a, b in zip(y[te], p))
        else:                                         # per-sample class PROBABILITIES -> ensemble/diversity offline
            clf_oof.append({"model": model, "task_id": t["task_id"], "fold": fold,
                            "classes": list(map(str, classes)), "y_true": np.asarray(y[te], np.int64),
                            "proba": np.asarray(pred, np.float64)})
    return rows, oof, clf_oof


KEY = ["model", "task_id", "fold"]        # identifies one scored fold, in rows and companions alike


def _flush(path, rows):
    new = pd.DataFrame(rows, columns=COLUMNS)
    both = pd.concat([pd.read_csv(path), new], ignore_index=True) if path.exists() else new
    both = both.drop_duplicates(subset=KEY, keep="last")
    both.sort_values(["model", "dataset", "task_id", "fold"]).to_csv(path, index=False)


def _merge_oof(path, new):
    """Per-sample regression OOF preds: keep the folds this run did NOT rescore (a resumed run must not drop
    the raw predictions of the folds it skipped), then append the fresh ones."""
    fresh = pd.DataFrame(new)
    if path.exists():
        old = pd.read_csv(path)
        keys = set(map(tuple, fresh[KEY].drop_duplicates().values.tolist()))
        old = old[[tuple(r) not in keys for r in old[KEY].values.tolist()]]
        fresh = pd.concat([old, fresh], ignore_index=True)
    fresh.sort_values(KEY, kind="stable").to_csv(path, index=False)


def _merge_clfoof(path, new):
    """Same for the per-sample class probabilities (one record per scored classification fold)."""
    recs = list(new)
    if path.exists():
        keys = {tuple(r[k] for k in KEY) for r in recs}
        recs = [r for r in np.load(path, allow_pickle=True)["data"] if tuple(r[k] for k in KEY) not in keys] + recs
    recs.sort(key=lambda r: tuple(str(r[k]) for k in KEY))
    np.savez(path, data=np.array(recs, dtype=object))


CONFIG = {                        # defaults; override on the CLI as key=value (Hydra)
    # en|rf|catboost|lightgbm|mlp | tabpfn|tabpfn_api|tabicl|tabdpt|limix|tabfm | geneicl (GeneICL ckpt)
    # | compass|bulkformer|scfoundation|scgpt|bulkrnabert (frozen-FM embeddings + EN probe; needs emb_dir)
    "model": "en",
    "data_dir": None,             # None -> ../data/benchmark (the benchmark)
    "out": None,                  # output prefix: <out>_<task_type>.csv + companions; None -> results/<model>
    "overwrite": True,            # re-run recomputes from scratch; False = resume unfinished / changed folds
    "seed": 42,
    # GeneICL (model=geneicl): score a GeneICL checkpoint on RAW genes (no shared PCA). GPU.
    "geneicl_ckpt": None,         # None -> ../../geneicl/checkpoints/trm_segmented8_s0.pt
    "query_chunk": 256,           # predict geneicl test rows in chunks
    "geneicl_support_ens": None,  # test-time ensemble (label-free; None/1 = off): e.g. 32 support views
    "geneicl_pca_ens": None,      # e.g. [0.9,0.95,0.98] -> PCA-threshold ensemble
    "emb_dir": None,              # frozen-FM embedding dir (<dataset>.npz with emb, obs_names) for the embedding models
}
ConfigStore.instance().store(name="benchmark", node=CONFIG)


@hydra.main(version_base="1.3", config_path=None, config_name="benchmark")
def main(cfg: DictConfig) -> None:
    assert cfg.model in CLASSICAL | {"geneicl"} | ICL_MODELS | EMB_MODELS, f"model={cfg.model}"
    global _EMB_DIR
    if cfg.model in EMB_MODELS:
        assert cfg.get("emb_dir"), f"model={cfg.model} reads frozen embeddings: pass emb_dir=<dir of <dataset>.npz>"
        _EMB_DIR = Path(cfg.emb_dir)
    geneicl_cfg = {"seed": int(cfg.seed),
                   "query_chunk": int(cfg.get("query_chunk", 256)),
                   "geneicl_ckpt": cfg.get("geneicl_ckpt") or str(GENEICL_DEFAULT_CKPT),
                   "pca_ens": list(cfg.get("geneicl_pca_ens") or []),
                   "support_ens": int(cfg.get("geneicl_support_ens", 1) or 1)}
    data_dir = Path(cfg.data_dir) if cfg.get("data_dir") else DEFAULT_DATA
    tasks = [t for t in json.load(open(data_dir / "index.json"))["tasks"]
             if t["task_type"] in ("classification", "regression")]   # survival tasks: quickstart only
    out = Path(cfg.out) if cfg.get("out") else HERE / "results" / cfg.model
    assert out.suffix != ".csv", f"out={out} is a prefix: the run writes <out>_<task_type>.csv"
    csv = {tt: out.parent / f"{out.name}_{tt}.csv" for tt in ("regression", "classification")}
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}                                     # (model, task_id, fold) -> (n_train, n_test) already on disk
    for path in csv.values():
        if not path.exists():
            continue
        if bool(cfg.get("overwrite", True)):
            path.unlink()
        else:
            d = pd.read_csv(path)
            done.update({(m, ti, f): (int(ntr), int(nte)) for m, ti, f, ntr, nte
                         in zip(d.model, d.task_id, d.fold, d.n_train, d.n_test)})
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    print(f"[{cfg.model}] {len(tasks)} tasks | n_jobs={n_jobs} -> {out}_<task_type>.csv", flush=True)

    all_oof, all_clf_oof = [], []
    n_clf_rows = n_reg_rows = 0                        # fresh folds scored THIS run, by task type
    for t in tasks:
        rows, oof, clf_oof = run_task(cfg.model, t, data_dir, int(cfg.seed), done, geneicl_cfg, n_jobs)
        if rows:
            _flush(csv[t["task_type"]], rows)
        n_clf_rows += sum(r["task_type"] == "classification" for r in rows)
        n_reg_rows += sum(r["task_type"] == "regression" for r in rows)
        all_oof.extend(oof); all_clf_oof.extend(clf_oof)
    if all_oof:                                       # companion: per-sample OOF reg preds (pooled-OOF R2)
        oof_path = out.parent / f"{out.name}_regression.oof.csv"
        _merge_oof(oof_path, all_oof)                 # merged like the CSV rows, never overwritten wholesale
        print(f"[{cfg.model}] OOF preds -> {oof_path}", flush=True)
    if all_clf_oof:                                   # companion: per-sample class probs
        clf_path = out.parent / f"{out.name}_classification.clfoof.npz"
        _merge_clfoof(clf_path, all_clf_oof)
        print(f"[{cfg.model}] clf proba -> {clf_path}", flush=True)
    # HARD INVARIANT: any run that freshly scored folds MUST persist the matching per-sample companion,
    # so every metric / aggregation / ensemble stays recomputable offline without re-running the model.
    if n_clf_rows and len(all_clf_oof) != n_clf_rows:
        raise RuntimeError(f"clfoof invariant violated: {n_clf_rows} fresh classification folds but "
                           f"{len(all_clf_oof)} per-sample proba records — raw predictions would be lost.")
    if n_reg_rows and not all_oof:
        raise RuntimeError(f"oof invariant violated: {n_reg_rows} fresh regression folds but no per-sample "
                           f"OOF predictions saved — raw predictions would be lost.")
    print(f"[{cfg.model}] done -> {out} "
          f"(raw: {len(all_clf_oof)} clf-fold proba, {len(all_oof)} reg-sample preds)", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
