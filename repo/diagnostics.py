"""All diagnostic plots for the chosen models (var1: Lasso d=5 a=0.01 ; var2: Ridge d=10 a=1)."""
import json, re, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge, enet_path
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
D = "data/BT2024066_"
CH = {1: ("lasso", 5, 0.01), 2: ("ridge", 10, 1.0)}
rows = [json.loads(l) for l in open("compare_results.jsonl")]
def load(v):
    t = pd.read_csv(f"{D}train_var{v}.csv"); return t.drop(columns="y").values, t.y.values, pd.read_csv(f"{D}test_var{v}.csv").values
def est(kind, a): return Lasso(alpha=a, max_iter=20000, tol=1e-4) if kind == "lasso" else Ridge(alpha=a)
def fitpred(Xtr, ytr, Xev, d, kind, a):
    pf = PolynomialFeatures(d, include_bias=False); F = pf.fit_transform(Xtr); sc = StandardScaler().fit(F)
    m = est(kind, a).fit(sc.transform(F), ytr); return m.predict(sc.transform(pf.transform(Xev))), m, pf, sc
def oof(v):
    X, y, _ = load(v); k, d, a = CH[v]; p = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=0).split(X): p[va] = fitpred(X[tr], y[tr], X[va], d, k, a)[0]
    return p
nm = lambda s: re.sub(r"x(\d)", lambda m: "x" + str(int(m.group(1)) + 1), s).replace(" ", "*")
data = {v: load(v) for v in (1, 2)}; O = {v: oof(v) for v in (1, 2)}
print("oof done", flush=True)

# 1. train vs CV error vs degree (best penalty per degree, from compare_results)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
for a_, v in zip(ax, (1, 2)):
    X, y, _ = data[v]; kind = CH[v][0]; ds, tr_e, cv_e = [], [], []
    for r in [r for r in rows if r["var"] == v]:
        al = r["res"][kind][1]; p = fitpred(X, y, X, r["deg"], kind, al)[0]
        ds.append(r["deg"]); tr_e.append(np.mean((y - p) ** 2)); cv_e.append(r["res"][kind][0])
    a_.plot(ds, tr_e, "o-", ms=3, label="train MSE (in-sample)"); a_.plot(ds, cv_e, "s-", ms=3, label="5-fold CV MSE")
    a_.axvline(CH[v][1], color="k", ls="--", lw=1, label=f"chosen d={CH[v][1]}"); a_.set_yscale("log")
    a_.xaxis.set_major_locator(mt.MaxNLocator(integer=True)); a_.set_xlabel("polynomial degree"); a_.set_ylabel("MSE")
    a_.set_title(f"var{v} ({kind.capitalize()}): train vs CV error"); a_.grid(alpha=.3); a_.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/tuning_curves.png", dpi=150); plt.close(); print("fig1", flush=True)

# 2. residual diagnostics per problem (out-of-fold)
for v in (1, 2):
    X, y, _ = data[v]; p = O[v]; res = y - p; sd = res.std()
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.4))
    ax[0, 0].scatter(y, p, s=5, alpha=.5); lo, hi = y.min(), y.max(); ax[0, 0].plot([lo, hi], [lo, hi], "r-", lw=1)
    ax[0, 0].set(xlabel="actual y", ylabel="out-of-fold predicted y", title="Actual vs predicted")
    ax[0, 1].scatter(p, res, s=5, alpha=.5); ax[0, 1].axhline(0, color="r", lw=1)
    ax[0, 1].set(xlabel="predicted y", ylabel="residual", title="Residuals vs predicted")
    ax[0, 2].hist(res, 40, density=True, alpha=.7); g = np.linspace(res.min(), res.max(), 200)
    ax[0, 2].plot(g, stats.norm.pdf(g, res.mean(), sd), "r-"); ax[0, 2].set(xlabel="residual", ylabel="density", title=f"Residual histogram (sd={sd:.3f})")
    stats.probplot(res, dist="norm", plot=ax[1, 0]); ax[1, 0].set_title("Normal Q-Q plot of residuals")
    ax[1, 1].scatter(p, np.abs(res), s=5, alpha=.5); ax[1, 1].set(xlabel="predicted y", ylabel="|residual|", title="Spread of residuals (heteroscedasticity)")
    ax[1, 2].plot(res, lw=.5); ax[1, 2].axhline(0, color="r", lw=1); ax[1, 2].set(xlabel="row index", ylabel="residual", title="Residuals by row order")
    for a_ in ax.ravel(): a_.grid(alpha=.3)
    k, d, a = CH[v]; fig.suptitle(f"var{v}: out-of-fold diagnostics ({k.capitalize()}, degree {d}); MSE={np.mean(res**2):.3f}, R2={1-np.mean(res**2)/y.var():.4f}", fontsize=11)
    plt.tight_layout(); plt.savefig(f"figures/diag_var{v}.png", dpi=150); plt.close()
print("fig2", flush=True)
