"""GeneICL: one in-context model for classification, regression and survival (Cox PH) on bulk expression.

    from geneicl import GeneICL
    m = GeneICL(mode="classification")         # or "regression" / "survival"
    m.fit(X_train, y_train)                    # survival: m.fit(X_train, time, event=event)
    m.predict(X_test)                          # class labels / values / Cox risk scores (higher = higher risk)
    m.predict_proba(X_test)                    # classification: (n_test, n_classes), columns in m.classes_
    m.predict_survival_function(X_test)        # survival: (n_test, len(m.times_)), S(t) at m.times_

X: samples x genes, log1p-CPM. Train and test rows must share the columns (any gene set: the model standardizes
and PCA-reduces every task itself). `fit` only stores the labelled support set; each `predict` is one in-context
forward pass per query chunk, with no training.

GeneICL(mode, ensemble_seed=42): label-free test-time ensemble. Outputs are averaged over 8 support views (the full
support + 7 seeded 90% subsets, class-stratified for classification) x 3 internal PCA thresholds (0.9/0.95/0.98),
i.e. 24 forward passes per predict. ensemble_seed=None (default) is a single forward pass.

Survival = GeneICL-Cox with one boosting round at lr=1: GeneICL regresses the support's Cox martingale residuals
(at a zero score), and the prediction is the log-risk. The Breslow baseline hazard is fit on 5-fold out-of-fold
support scores, so S(t | x) = exp(-H0(t) * exp(risk)).
"""
import contextlib
import warnings
from pathlib import Path

import numpy as np
import torch

from . import model as M

__all__ = ["GeneICL", "Runner", "CKPT"]
__version__ = "1.0.0"

CKPT = Path(__file__).resolve().parent / "checkpoints" / "trm_segmented8_s0.pt"   # seeds 0/1/2 are shipped
MODES = ("classification", "regression", "survival")
ENS_VIEWS, ENS_PCA = 8, (0.9, 0.95, 0.98)   # test-time ensemble: support views x internal PCA thresholds


class Runner:
    """One checkpoint, run in context: support + one query chunk per forward pass, bf16 autocast on GPU."""

    def __init__(self, ckpt=CKPT, device=None, query_chunk=256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.query_chunk = int(query_chunk)
        self.model = M.load_ckpt(ckpt, self.device)
        self._cuda = self.device.startswith("cuda")                  # bf16 autocast on GPU; plain float on CPU

    @staticmethod
    def _prep(X):
        # Fortran order: the memory layout the published results were computed with. The layout changes the BLAS
        # summation order, i.e. outputs at the ~1e-6 level.
        return np.asfortranarray(X, dtype=np.float32)

    def _run(self, Xtr, y_sup, Xte, is_cls, pca_var=None):
        """Support + one query chunk per forward; pca_var: inference-time override of the model's PCA threshold."""
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self._cuda else contextlib.nullcontext()
        old_pv = self.model.pca_var
        if pca_var is not None:   # set on the shared model and restored below, so not safe for concurrent calls
            self.model.pca_var = float(pca_var)
        try:
            preds = []
            yt = torch.tensor(y_sup, dtype=torch.float32, device=self.device)[None]
            for s in range(0, len(Xte), self.query_chunk):
                X = np.concatenate([Xtr, Xte[s:s + self.query_chunk]], axis=0)
                Xt = torch.tensor(X, dtype=torch.float32, device=self.device)[None]
                with torch.no_grad(), amp:
                    out = self.model(Xt, yt, is_cls=is_cls).float()
                preds.append(out[0].cpu().numpy())
        finally:
            self.model.pca_var = old_pv
        return np.concatenate(preds)                                 # (n_test, out_dim)

    def predict_fold(self, Xtr, ytr, Xte, pca_var=None):
        """Regression: support labels z-scored, point predictions mapped back to the raw target scale."""
        ytr = np.asarray(ytr, np.float64)
        mu, sd = float(ytr.mean()), float(ytr.std()) + 1e-8
        out = self._run(self._prep(Xtr), (ytr - mu) / sd, self._prep(Xte), False, pca_var)
        return out[:, 0] * sd + mu

    def predict_proba_fold(self, Xtr, ytr, Xte, pca_var=None):
        """Classification: returns (proba (n_test, K), classes = sorted str labels present in the support)."""
        classes = sorted(set(map(str, ytr)))
        y_int = np.array([classes.index(str(v)) for v in ytr], dtype=np.float64)
        logits = self._run(self._prep(Xtr), y_int, self._prep(Xte), True, pca_var)
        # the head has MAX_K logits and support labels are 0..K-1 (sorted), so the softmax is over the K present classes
        return torch.tensor(logits[:, :len(classes)]).softmax(-1).numpy(), classes


def _breslow(T, E, f):
    """Breslow cumulative baseline hazard for log-risks f: (unique times, H0 at them)."""
    order = np.argsort(T, kind="mergesort")
    risk_set = np.cumsum(np.exp(f[order])[::-1])[::-1]
    uniq, first, inv = np.unique(T[order], return_index=True, return_inverse=True)
    deaths = np.bincount(inv, weights=E[order], minlength=len(uniq))
    return uniq, np.cumsum(deaths / np.maximum(risk_set[first], 1e-300))


def _support_views(X, y, k, seed, is_cls):
    """The support sets to average over: the full support first, then k-1 seeded 90% subsets
    (class-stratified for classification, so every view carries the same class set and order).
    Same as eval/benchmark/benchmark.py, so a seed gives the same views."""
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


class GeneICL:
    def __init__(self, mode="classification", ckpt=CKPT, device=None, ensemble_seed=None):
        if mode not in MODES:
            raise ValueError(f"mode={mode!r}, expected one of {MODES}")
        self.mode = mode
        self.ensemble_seed = ensemble_seed
        self._runner = Runner(ckpt, device)

    def _run(self, Xtr, ytr, Xte, is_cls):
        """Class probabilities / point predictions: one forward pass, or the view x PCA-threshold ensemble."""
        def one(Xs, ys, **kw):
            if is_cls:
                return self._runner.predict_proba_fold(Xs, ys, Xte, **kw)[0]
            return np.asarray(self._runner.predict_fold(Xs, ys, Xte, **kw), np.float64)
        if self.ensemble_seed is None:
            return one(Xtr, ytr)
        views = _support_views(Xtr, np.asarray(ytr), ENS_VIEWS, int(self.ensemble_seed), is_cls)
        # average within a view first, then over views: the benchmark's summation order
        return sum(sum(one(Xv, yv, pca_var=v) for v in ENS_PCA) / len(ENS_PCA) for Xv, yv in views) / len(views)

    def fit(self, X, y, event=None):
        """y: class labels / targets / survival times; event (survival only): 1 = event observed, 0 = censored."""
        self.X_ = np.asarray(X, np.float32)
        if self.X_.max() > 20:   # log1p(CPM) <= log1p(1e6) ~ 13.8
            warnings.warn("X looks like raw counts; GeneICL expects log1p-CPM")
        if self.mode == "classification":
            self.classes_, self.y_ = np.unique(y, return_inverse=True)
            if len(self.classes_) > M.MAX_K:
                raise ValueError(f"{len(self.classes_)} classes; the classification head has {M.MAX_K}")
        elif self.mode == "regression":
            self.y_ = np.asarray(y, np.float64)
        else:
            if event is None:
                raise ValueError("mode='survival' needs fit(X, time, event=event)")
            T, E = np.asarray(y, np.float64), np.asarray(event, np.float64)
            uniq, H0 = _breslow(T, E, np.zeros(len(T)))
            self.y_ = E - H0[np.searchsorted(uniq, T)]                 # martingale residuals: the regression target
            oof = np.empty(len(T))                                     # out-of-fold support scores
            for hold in np.array_split(np.random.default_rng(42).permutation(len(T)), 5):
                keep = np.setdiff1d(np.arange(len(T)), hold)
                oof[hold] = self._run(self.X_[keep], self.y_[keep], self.X_[hold], False)
            self._shift = oof.mean()                                   # partial likelihood is shift-invariant
            self._uniq, self._H0 = _breslow(T, E, oof - self._shift)
            self.times_ = np.unique(T[E > 0])
        return self

    def predict_proba(self, X):
        proba = self._run(self.X_, self.y_, np.asarray(X, np.float32), True)
        return proba   # support labels are 0..K-1 with K <= 10, so the runner's str-sorted classes are in order

    def predict(self, X):
        if self.mode == "classification":
            return self.classes_[self.predict_proba(X).argmax(1)]
        pred = self._run(self.X_, self.y_, np.asarray(X, np.float32), False)
        return pred if self.mode == "regression" else pred - self._shift

    def predict_survival_function(self, X, times=None):
        times = self.times_ if times is None else np.asarray(times, np.float64)
        idx = np.searchsorted(self._uniq, times, side="right") - 1
        H0 = np.where(idx >= 0, self._H0[np.clip(idx, 0, None)], 0.0)
        return np.exp(-np.outer(np.exp(self.predict(X)), H0))
