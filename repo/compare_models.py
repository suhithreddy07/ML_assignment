"""Compare OLS, Ridge, Lasso, ElasticNet over polynomial degrees (5-fold CV, same folds for all)."""
import json, sys, warnings, numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import enet_path
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
ROLL, D = "BT2024066", "data/"
ALPHAS = np.logspace(-3, -0.3, 8)           # L1-type penalties (sklearn scale)
RIDGE = np.logspace(-3, 2, 14)
L1R = [0.3, 0.7]

def run(v, d):
    tr = pd.read_csv(f"{D}{ROLL}_train_var{v}.csv"); X = tr.drop(columns="y").values; y = tr.y.values
    pf = PolynomialFeatures(d, include_bias=False)
    res = {"ols": [], "ridge": {a: [] for a in RIDGE}, "lasso": {a: [] for a in ALPHAS},
           "enet": {(r, a): [] for r in L1R for a in ALPHAS}}
    for tri, vai in KFold(5, shuffle=True, random_state=0).split(X):
        F = pf.fit_transform(X[tri]); sc = StandardScaler().fit(F); Ft = sc.transform(F); Fv = sc.transform(pf.transform(X[vai]))
        ym = y[tri].mean(); yt = y[tri] - ym
        mse = lambda p: float(np.mean((y[vai] - (p + ym)) ** 2))
        res["ols"].append(mse(Fv @ np.linalg.lstsq(Ft, yt, rcond=None)[0]))
        U, s, Vt = np.linalg.svd(Ft, full_matrices=False); Uy = U.T @ yt
        for a in RIDGE: res["ridge"][a].append(mse(Fv @ (Vt.T @ (s / (s**2 + a * 1.0) * Uy))))
        # sklearn Ridge(alpha) minimises ||y-Xw||^2 + alpha||w||^2 -> matches formula above
        _, c, _ = enet_path(Ft, yt, l1_ratio=1.0, alphas=ALPHAS, max_iter=3000, tol=1e-3)
        for i, a in enumerate(sorted(ALPHAS, reverse=True)): res["lasso"][a].append(mse(Fv @ c[:, i]))
        for r in L1R:
            _, c, _ = enet_path(Ft, yt, l1_ratio=r, alphas=ALPHAS, max_iter=3000, tol=1e-3)
            for i, a in enumerate(sorted(ALPHAS, reverse=True)): res["enet"][(r, a)].append(mse(Fv @ c[:, i]))
    out = {"ols": (float(np.mean(res["ols"])), None)}
    for k in ["ridge", "lasso", "enet"]:
        key = min(res[k], key=lambda q: np.mean(res[k][q]))
        out[k] = (float(np.mean(res[k][key])), key if k != "enet" else list(key))
    return v, d, out

if __name__ == "__main__":
    jobs = [(2, d) for d in range(1, 21)] + [(1, d) for d in range(1, 8)]
    with open("compare_results.jsonl", "w") as f:
        for v, d in jobs:
            _, _, o = run(v, d)
            f.write(json.dumps({"var": v, "deg": d, "res": o}) + "\n"); f.flush()
    print("done")
