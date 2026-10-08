"""Builds all report figures from results.json, compare_results.jsonl, shift_results.json and the data."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [json.loads(l) for l in open("compare_results.jsonl")]
names = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso", "enet": "ElasticNet"}
col = {"ols": "tab:gray", "ridge": "tab:blue", "lasso": "tab:red", "enet": "tab:green"}
chosen = {1: 5, 2: 10}
# Fig A: all four models vs degree
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
for a, v in zip(ax, (1, 2)):
    rr = [r for r in rows if r["var"] == v]; ds = [r["deg"] for r in rr]
    for k in names:
        a.plot(ds, [min(r["res"][k][0], 100) for r in rr], "o-", ms=3, color=col[k], label=names[k])
    a.axvline(chosen[v], color="k", ls="--", lw=1, label=f"chosen d={chosen[v]}")
    a.set_yscale("log"); a.set_ylim(0.2, 100); a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    a.set_xlabel("polynomial degree"); a.set_ylabel("5-fold CV MSE (capped at 100)"); a.set_title(f"var{v}"); a.grid(alpha=.3); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/model_comparison.png", dpi=160); plt.close()
# Fig B: var1 train-vs-test spread + extrapolation check
tr = pd.read_csv("data/BT2024066_train_var1.csv").drop(columns="y").values; te = pd.read_csv("data/BT2024066_test_var1.csv").values
S = json.load(open("shift_results.json"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].hist(np.linalg.norm(tr, axis=1), 40, alpha=.6, density=True, label="train"); ax[0].hist(np.linalg.norm(te, axis=1), 40, alpha=.6, density=True, label="test")
ax[0].set_xlabel("distance of input from origin"); ax[0].set_ylabel("density"); ax[0].set_title("var1: test inputs are more extreme"); ax[0].legend()
ax[1].bar([str(k) for k in S], list(S.values()), color=["tab:gray"] * 3 + ["tab:red", "tab:gray"])
ax[1].set_xlabel("degree (Lasso, alpha=0.01)"); ax[1].set_ylabel("MSE on outer 30%"); ax[1].set_title("var1: train inner 70%, test outer 30%"); ax[1].grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("figures/var1_shift.png", dpi=160); plt.close()
# Fig D: var2 feature relationships (data view)
t2 = pd.read_csv("data/BT2024066_train_var2.csv")
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3))
for a, c in zip(ax, ["x1", "x2", "x3"]):
    a.scatter(t2[c], t2.y, s=4, alpha=.4); a.set_xlabel(c); a.set_ylabel("y"); a.grid(alpha=.3)
ax[1].set_title("var2: y vs each input (training)")
plt.tight_layout(); plt.savefig("figures/var2_data.png", dpi=160); plt.close()
print("figures done")
