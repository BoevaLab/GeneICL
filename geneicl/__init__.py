"""GeneICL: in-context learning on bulk expression: classification, regression and survival (Cox PH).

    from geneicl import GeneICL
    m = GeneICL(mode="classification")         # or "regression" / "survival"; inference="normal" (default) / "fast"
    m.fit(X_train, y_train)                    # survival: m.fit(X_train, time, event=event)
    m.predict(X_test)                          # class labels / values / Cox risk scores (higher = higher risk)
    m.predict_proba(X_test)                    # classification: (n_test, n_classes), columns in m.classes_
    m.predict_survival_function(X_test)        # survival: (n_test, len(m.times_)), S(t) at m.times_

X: samples x genes, log1p-CPM. Train and test rows must share the columns (any gene set: the model encodes and
PCA-reduces every task itself). `fit` only stores the labelled support set; `predict` runs the model in context,
with no training.

Inference (`inference=`), the same for every task:
    "normal" (default)  GeneICL: 6 configurations, PCA whitening exponent {.25, .5, .75} x genes {all, the 5000 of
                        highest support variance}, each the mean of the plain and the gene-weighted input (after the
                        per-gene rank encoding, each gene scaled by its support SD over the mean SD, capped at 5). The
                        configurations are weighted by exp((inner score - best) / (.4 / sqrt(n support rows))), the inner
                        score being the macro-F1 (classification) / R2 (regression) of a 3-fold cross-validation on the
                        support set; for classification the same inner CV picks the read-out of each configuration
                        (K-way, one-vs-rest or one-vs-one). Nothing is fitted: the CV only chooses how to read the model.
    "fast"              GeneICL-Fast: the mean of the plain and the gene-weighted input (2 forward passes), K-way
                        read-out, no inner CV.
Classification with more than MAX_K (10) classes runs one-vs-rest.

Checkpoints (`ckpt=`): CKPT_CLS (classification) and CKPT_REG (regression and survival) are the defaults (seed 0);
CKPT_V1 is the v1.0.0 model (one checkpoint for every task), and GeneICL(mode, ckpt=CKPT_V1, inference="fast") is the
v1.0.0 default (one forward pass).

Survival = GeneICL-Cox with one boosting round at lr=1: GeneICL regresses the support's Cox martingale residuals
(at a zero score), and the prediction is the log-risk. The Breslow baseline hazard is fit on 5-fold out-of-fold
support scores, so S(t | x) = exp(-H0(t) * exp(risk)).
"""
import contextlib
import itertools
import warnings
from pathlib import Path

import numpy as np
import torch

from . import model as M

__all__ = ["GeneICL", "Runner", "CKPT_CLS", "CKPT_REG", "CKPT_V1"]
__version__ = "1.1.0"

_CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"   # seed 0 of each is in the package, seeds 1 / 2 in git
CKPT_CLS = _CKPT_DIR / "C20RFbal_s0.pt"           # classification
CKPT_REG = _CKPT_DIR / "R20RFs20Yv_s0.pt"         # regression and survival
CKPT_V1 = _CKPT_DIR / "trm_segmented8_s0.pt"      # v1.0.0: one model for every task
MODES = ("classification", "regression", "survival")
INFERENCE = ("normal", "fast")

# A view is (PCA whitening exponent, top-n genes by support variance, gene-weight power); None keeps the model's
# setting / all genes / no weights. A configuration is a list of views whose outputs are averaged.
BASE = (None, None, None)
FAST = [BASE, (None, None, 1.0)]         # GeneICL-Fast: plain + gene-weighted input; normal: FAST ...
WHITEN = (0.25, 0.5, 0.75)               # ... under every whitening exponent (model.PCA_WHITEN) ...
HVG = (None, 5000)                       # ... x genes: all, the 5000 of highest support variance
SOFT_C = 0.4                             # configuration weights exp((inner score - best) / (SOFT_C / sqrt(n support rows)))


def _kfold(n, k):
    """Shuffled k-fold (train, test) index pairs, seed 0: scikit-learn's KFold(k, shuffle=True, random_state=0)."""
    idx = np.arange(n)
    np.random.RandomState(0).shuffle(idx)
    sizes = np.full(k, n // k)
    sizes[:n % k] += 1
    test_fold = np.empty(n, dtype=int)
    for i, (a, b) in enumerate(zip(np.cumsum(sizes) - sizes, np.cumsum(sizes))):
        test_fold[idx[a:b]] = i
    return [(np.flatnonzero(test_fold != i), np.flatnonzero(test_fold == i)) for i in range(k)]


def _stratified_kfold(y, k):
    """Shuffled stratified k-fold (train, test) index pairs, seed 0: scikit-learn's StratifiedKFold(k, shuffle=True,
    random_state=0)."""
    rng = np.random.RandomState(0)
    _, first, inv = np.unique(y, return_index=True, return_inverse=True)
    y_enc = np.unique(first, return_inverse=True)[1][inv]          # classes numbered by first appearance
    n_cls = len(first)
    y_order = np.sort(y_enc)
    alloc = np.asarray([np.bincount(y_order[i::k], minlength=n_cls) for i in range(k)])
    test_fold = np.empty(len(y), dtype=int)
    for c in range(n_cls):
        folds_c = np.arange(k).repeat(alloc[:, c])
        rng.shuffle(folds_c)
        test_fold[y_enc == c] = folds_c
    return [(np.flatnonzero(test_fold != i), np.flatnonzero(test_fold == i)) for i in range(k)]


def _macro_f1(y, pred, K):
    """Macro-F1 over the labels 0..K-1 (0 for a label never true nor predicted)."""
    tp = np.bincount(y[y == pred], minlength=K).astype(np.float64)
    denom = np.bincount(y, minlength=K).astype(np.float64) + np.bincount(pred, minlength=K)
    f1 = np.zeros(K)
    np.divide(2.0 * tp, denom, out=f1, where=denom > 0)
    return float(np.mean(f1))


def _r2(y, pred):
    """Coefficient of determination; 1 for a perfect fit, 0 for a constant target that is missed, NaN for < 2 rows."""
    if len(y) < 2:
        return float("nan")
    num, den = np.sum((y - pred) ** 2), np.sum((y - np.mean(y)) ** 2)
    if num == 0:
        return 1.0
    return float(1 - num / den) if den != 0 else 0.0


class Runner:
    """One checkpoint, run in context: support + one query chunk per forward pass, bf16 autocast on GPU, with the
    GeneICL ("normal") or GeneICL-Fast ("fast") inference (module docstring)."""

    def __init__(self, ckpt, device=None, query_chunk=256, inference="normal"):
        if inference not in INFERENCE:
            raise ValueError(f"inference={inference!r}, expected one of {INFERENCE}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.query_chunk = int(query_chunk)
        self.inference = inference
        self.model = M.load_ckpt(ckpt, self.device)
        self._cuda = self.device.startswith("cuda")                  # bf16 autocast on GPU; plain float on CPU
        # gene weights act after the rank encoding only: on a z-score model (CKPT_V1) that view equals the plain one
        self._fast = FAST if self.model.feat_transform == "rank_gauss" else [BASE]

    @staticmethod
    def _prep(X):
        # Fortran order: the memory layout the published results were computed with. The layout changes the BLAS
        # summation order, i.e. outputs at the ~1e-6 level.
        return np.asfortranarray(X, dtype=np.float32)

    def _configs(self, n_genes):
        if self.inference == "fast":
            return [self._fast]
        hvg = [h for h in HVG if h is None or h < n_genes]           # the top n of n genes or fewer = all genes
        return [[(w, h, gw) for _, _, gw in self._fast] for w in WHITEN for h in hvg]

    def _run(self, Xtr, y_sup, Xte, is_cls, view=BASE):
        """Support + one query chunk per forward, under one view -> (n_test, out_dim)."""
        whiten, hvg, gw = view
        if hvg is not None:
            var = np.nan_to_num(np.nanvar(Xtr, axis=0), nan=-1.0)
            g = np.sort(np.argsort(-var, kind="stable")[:hvg])
            Xtr, Xte = self._prep(Xtr[:, g]), self._prep(Xte[:, g])
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self._cuda else contextlib.nullcontext()
        # set on the shared model / module and restored below, so not safe for concurrent calls
        old = M.PCA_WHITEN, getattr(self.model, "gene_w_pow", None)
        if whiten is not None:
            M.PCA_WHITEN = float(whiten)
        if gw is not None:
            self.model.gene_w_pow = float(gw)
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
            M.PCA_WHITEN, self.model.gene_w_pow = old
        return np.concatenate(preds)

    def predict_fold(self, Xtr, ytr, Xte):
        """Regression: point predictions on the raw target scale; normal weights the configurations by inner R2."""
        ytr = np.asarray(ytr, np.float64)
        Xs, Xq = self._prep(Xtr), self._prep(Xte)
        cfgs = self._configs(Xtr.shape[1])
        if len(cfgs) == 1 or len(ytr) < 6:                           # < 6 rows: no 3 inner folds of >= 2 rows
            return self._reg(cfgs[0] if len(cfgs) == 1 else self._fast, Xs, ytr, Xq)
        folds = _kfold(len(ytr), 3)
        s = np.array([np.nanmean([_r2(ytr[iva], self._reg(cfg, self._prep(Xs[itr]), ytr[itr], self._prep(Xs[iva])))
                                  for itr, iva in folds]) for cfg in cfgs])
        w = np.nan_to_num(np.exp((s - np.nanmax(s)) / (SOFT_C / np.sqrt(len(ytr)))))
        return sum(wi * self._reg(cfg, Xs, ytr, Xq) for wi, cfg in zip(w, cfgs)) / w.sum()

    def _reg(self, cfg, Xs, y, Xq):
        """Mean over the configuration's views; support labels z-scored, predictions mapped back to the raw scale."""
        mu, sd = float(y.mean()), float(y.std()) + 1e-8
        return np.mean([self.model.reg_point(self._run(Xs, (y - mu) / sd, Xq, False, v)) * sd + mu for v in cfg], 0)

    def predict_proba_fold(self, Xtr, ytr, Xte):
        """Classification: returns (proba (n_test, K), classes = sorted str labels present in the support)."""
        classes = sorted(set(map(str, ytr)))
        K = len(classes)
        y_int = np.array([classes.index(str(v)) for v in ytr], dtype=np.int64)
        cands = ["native"] if K <= self.model.max_k else []
        if K > 2:
            cands += ["ovr", "ovo"] if K <= 6 else ["ovr"]           # one-vs-one: K(K-1)/2 forwards, so small K only
        if self.inference == "fast":                                 # GeneICL-Fast: no inner CV
            cands = cands[:1]
        cfgs = self._configs(Xtr.shape[1])
        sc = self._inner_f1(cfgs, cands, Xtr, y_int, K) if len(cfgs) > 1 or len(cands) > 1 else None
        if sc is None:                                               # nothing to choose, or every inner fold lost a class
            cfg = cfgs[0] if len(cfgs) == 1 else self._fast
            return self._decode(cands[0], cfg, Xtr, y_int, Xte, K), classes
        best = sc.argmax(1)                                          # read-out per configuration (ties -> the first)
        s = sc[np.arange(len(cfgs)), best]
        w = np.exp((s - s.max()) / (SOFT_C / np.sqrt(len(y_int))))
        P = sum(wi * self._decode(cands[b], cfg, Xtr, y_int, Xte, K) for wi, b, cfg in zip(w, best, cfgs))
        return P / w.sum(), classes

    def _inner_f1(self, cfgs, cands, Xtr, y_int, K):
        """(configurations, read-outs) mean macro-F1 over stratified CV folds of the support (None: no fold kept every
        class). Picks how to read the frozen model; no parameters are fitted."""
        n_splits = int(max(2, min(3, np.bincount(y_int).min())))
        score = []
        for itr, iva in _stratified_kfold(y_int, n_splits):
            if len(np.unique(y_int[itr])) < K:                       # this inner fold lost a class
                continue
            score.append([[_macro_f1(y_int[iva], self._decode(c, cfg, Xtr[itr], y_int[itr], Xtr[iva], K).argmax(1), K)
                           for c in cands] for cfg in cfgs])
        return np.mean(score, 0) if score else None

    def _native(self, Xtr, y_int, Xte, K, view):
        """K-way read-out of support labels 0..K-1 -> (n_test, K) probabilities."""
        logits = self._run(self._prep(Xtr), y_int.astype(np.float64), self._prep(Xte), True, view)
        # the head has MAX_K logits and support labels are 0..K-1, so the softmax is over the K present classes
        return torch.tensor(logits[:, :K]).softmax(-1).numpy()

    def _decode(self, name, cfg, Xtr, y_int, Xte, K):
        """Read-out `name` (native | ovr | ovo), averaged over the configuration's views -> (n_test, K)."""
        P = np.zeros((len(Xte), K))
        for v in cfg:
            if name == "native":
                P += self._native(Xtr, y_int, Xte, K, v)
            elif name == "ovr":
                for c in range(K):
                    P[:, c] += self._native(Xtr, (y_int == c).astype(np.int64), Xte, 2, v)[:, 1]
            else:
                for i, j in itertools.combinations(range(K), 2):
                    m = (y_int == i) | (y_int == j)
                    pj = self._native(Xtr[m], (y_int[m] == j).astype(np.int64), Xte, 2, v)[:, 1]
                    P[:, j] += pj
                    P[:, i] += 1.0 - pj
        return P / P.sum(1, keepdims=True).clip(1e-12)


def _breslow(T, E, f):
    """Breslow cumulative baseline hazard for log-risks f: (unique times, H0 at them)."""
    order = np.argsort(T, kind="mergesort")
    risk_set = np.cumsum(np.exp(f[order])[::-1])[::-1]
    uniq, first, inv = np.unique(T[order], return_index=True, return_inverse=True)
    deaths = np.bincount(inv, weights=E[order], minlength=len(uniq))
    return uniq, np.cumsum(deaths / np.maximum(risk_set[first], 1e-300))


class GeneICL:
    def __init__(self, mode="classification", ckpt=None, device=None, inference="normal"):
        if mode not in MODES:
            raise ValueError(f"mode={mode!r}, expected one of {MODES}")
        self.mode = mode
        self.inference = inference
        self.ckpt = Path(ckpt) if ckpt is not None else (CKPT_CLS if mode == "classification" else CKPT_REG)
        self._runner = Runner(self.ckpt, device, inference=inference)

    def _regress(self, Xtr, ytr, Xte):
        return np.asarray(self._runner.predict_fold(Xtr, ytr, Xte), np.float64)

    def fit(self, X, y, event=None):
        """y: class labels / targets / survival times; event (survival only): 1 = event observed, 0 = censored."""
        self.X_ = np.asarray(X, np.float32)
        if self.X_.max() > 20:   # log1p(CPM) <= log1p(1e6) ~ 13.8
            warnings.warn("X looks like raw counts; GeneICL expects log1p-CPM")
        if self.mode == "classification":
            self.classes_, self.y_ = np.unique(y, return_inverse=True)
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
                oof[hold] = self._regress(self.X_[keep], self.y_[keep], self.X_[hold])
            self._shift = oof.mean()                                   # partial likelihood is shift-invariant
            self._uniq, self._H0 = _breslow(T, E, oof - self._shift)
            self.times_ = np.unique(T[E > 0])
        return self

    def predict_proba(self, X):
        p, cls = self._runner.predict_proba_fold(self.X_, self.y_, np.asarray(X, np.float32))
        proba = np.empty_like(p)
        proba[:, [int(c) for c in cls]] = p                            # the runner's str-sorted labels -> classes_ order
        return proba

    def predict(self, X):
        if self.mode == "classification":
            return self.classes_[self.predict_proba(X).argmax(1)]
        pred = self._regress(self.X_, self.y_, np.asarray(X, np.float32))
        return pred if self.mode == "regression" else pred - self._shift

    def predict_survival_function(self, X, times=None):
        times = self.times_ if times is None else np.asarray(times, np.float64)
        idx = np.searchsorted(self._uniq, times, side="right") - 1
        H0 = np.where(idx >= 0, self._H0[np.clip(idx, 0, None)], 0.0)
        return np.exp(-np.outer(np.exp(self.predict(X)), H0))
