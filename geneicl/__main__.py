"""`python -m geneicl`: self-check on synthetic data (every task in both inference modes runs, probabilities sum to 1,
S(t) is monotone)."""
import numpy as np

from geneicl import GeneICL

rng = np.random.default_rng(0)
X = np.log1p(rng.gamma(2.0, 50.0, size=(120, 200))).astype(np.float32)
lp = X[:, :5] @ np.array([1.0, -0.8, 0.6, -0.5, 0.4])
tr, te = slice(0, 90), slice(90, None)
y12 = np.digitize(lp, np.quantile(lp, np.linspace(0, 1, 13)[1:-1]))      # 12 classes: more than the head's 10

for inference in ("fast", "normal"):
    p = GeneICL("classification", inference=inference).fit(X[tr], np.where(lp > np.median(lp), "high", "low")[tr]).predict_proba(X[te])
    assert p.shape == (30, 2) and np.allclose(p.sum(1), 1)
    m = GeneICL("classification", inference=inference).fit(X[tr], y12[tr])
    assert m.predict_proba(X[te]).shape == (30, 12) and set(m.predict(X[te])) <= set(m.classes_)
    assert np.isfinite(GeneICL("regression", inference=inference).fit(X[tr], lp[tr]).predict(X[te])).all()
    T_true, C = rng.exponential(np.exp(-lp)), rng.exponential(np.exp(-np.median(lp)) * 1.5, 120)
    m = GeneICL("survival", inference=inference).fit(X[tr], np.minimum(T_true, C)[tr], event=(T_true <= C)[tr])
    S = m.predict_survival_function(X[te])
    assert S.shape == (30, len(m.times_)) and (S >= 0).all() and (S <= 1).all() and (np.diff(S, axis=1) <= 1e-12).all()
print("geneicl OK")
