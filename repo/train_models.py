"""Polynomial regression for BT2024066 (var1: turbine, var2: thermal mapping).

Run:  python train_models.py
Outputs: BT2024066_pred_var1.csv, BT2024066_pred_var2.csv, results.json, figures/*.png
Only polynomial-feature regression is used (L1/L2-regularised least squares on polynomial terms).
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.ticker
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
ROLL, D = "BT2024066", "data/"
SEED = 0


def load(v):
    tr = pd.read_csv(f"{D}{ROLL}_train_var{v}.csv")
    te = pd.read_csv(f"{D}{ROLL}_test_var{v}.csv")
    return tr.drop(columns="y").values, tr["y"].values, te.values


def make(kind, alpha):
    return Lasso(alpha=alpha, max_iter=100000, tol=1e-5) if kind == "lasso" else Ridge(alpha=alpha)


def cv_score(X, y, degree, kind, alpha, cv):
    """Out-of-fold MSE. Polynomial expansion + scaler are fitted inside each fold (no leakage)."""
    pf = PolynomialFeatures(degree, include_bias=False)
    errs, preds = [], np.zeros(len(y))
    for tr, va in cv.split(X):
        Ft = pf.fit_transform(X[tr]); sc = StandardScaler().fit(Ft)
        m = make(kind, alpha).fit(sc.transform(Ft), y[tr])
        p = m.predict(sc.transform(pf.transform(X[va])))
        errs.append(mean_squared_error(y[va], p)); preds[va] = p
    return float(np.mean(errs)), preds


def fit_predict(X, y, Xte, degree, kind, alpha):
    pf = PolynomialFeatures(degree, include_bias=False)
    F = pf.fit_transform(X); sc = StandardScaler().fit(F)
    m = make(kind, alpha).fit(sc.transform(F), y)
    return m.predict(sc.transform(pf.transform(Xte))), m.predict(sc.transform(F)), m


results = {}

# ---------------- var1 : 6 features, degree <= 10 ----------------
X, y, Xte = load(1)
cv = KFold(5, shuffle=True, random_state=SEED)
sweep1 = {}
for d in range(1, 8):                       # Lasso alpha tuned per degree
    best = min(((cv_score(X, y, d, "lasso", a, cv)[0], a) for a in [0.003, 0.007, 0.01, 0.02, 0.05]))
    sweep1[d] = best
    print(f"var1 degree {d}: best CV MSE {best[0]:.4f} (lasso alpha={best[1]})", flush=True)
# Degrees 8-10 not fitted: CV error already rises after degree 5 and the term count explodes (~8000 at d=10).
DEG1, ALPHA1 = 5, 0.01                      # alpha 0.01 ~ CV optimum (0.007) but more robust to range shift
mse1, _ = cv_score(X, y, DEG1, "lasso", ALPHA1, RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED))
pred1, tr_pred1, m1 = fit_predict(X, y, Xte, DEG1, "lasso", ALPHA1)
pd.DataFrame({"y": pred1}).to_csv(f"{ROLL}_pred_var1.csv", index=False)
results["var1"] = dict(degree=DEG1, model=f"Lasso(alpha={ALPHA1}) on standardised poly features",
                       cv_mse=mse1, cv_r2=float(1 - mse1 / y.var()), train_mse=float(mean_squared_error(y, tr_pred1)),
                       n_features=int(len(m1.coef_)), n_nonzero=int((m1.coef_ != 0).sum()))

# ---------------- var2 : 3 features, degree <= 20 ----------------
X2, y2, Xte2 = load(2)
sweep2 = {}
for d in range(1, 21):
    best = min(((cv_score(X2, y2, d, "ridge", a, cv)[0], a) for a in [1e-6, 0.03, 0.1, 0.3, 1, 3]))
    sweep2[d] = best
    print(f"var2 degree {d}: best CV MSE {best[0]:.4f} (ridge alpha={best[1]})", flush=True)
DEG2, ALPHA2 = 10, 1.0
mse2, _ = cv_score(X2, y2, DEG2, "ridge", ALPHA2, RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED))
pred2, tr_pred2, m2 = fit_predict(X2, y2, Xte2, DEG2, "ridge", ALPHA2)
pd.DataFrame({"y": pred2}).to_csv(f"{ROLL}_pred_var2.csv", index=False)
results["var2"] = dict(degree=DEG2, model=f"Ridge(alpha={ALPHA2}) on standardised poly features",
                       cv_mse=mse2, cv_r2=float(1 - mse2 / y2.var()), train_mse=float(mean_squared_error(y2, tr_pred2)),
                       n_features=int(len(m2.coef_)))
results["sweeps"] = {"var1": {d: v[0] for d, v in sweep1.items()}, "var2": {d: v[0] for d, v in sweep2.items()}}
json.dump(results, open("results.json", "w"), indent=2)

# ---------------- figures ----------------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for a_, sw, ch, t in [(ax[0], sweep1, DEG1, "var1 (6 features)"), (ax[1], sweep2, DEG2, "var2 (3 features)")]:
    ds = list(sw); a_.plot(ds, [sw[d][0] for d in ds], "o-"); a_.axvline(ch, color="r", ls="--", label=f"chosen = {ch}")
    a_.set_yscale("log"); a_.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)); a_.set_xlabel("polynomial degree"); a_.set_ylabel("5-fold CV MSE"); a_.set_title(t); a_.legend(); a_.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/degree_sweep.png", dpi=160); plt.close()

print(json.dumps({k: v for k, v in results.items() if k != "sweeps"}, indent=2))
