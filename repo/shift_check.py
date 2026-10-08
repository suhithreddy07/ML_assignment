"""var1: train on the 70% of rows closest to the origin, test on the outer 30% (mimics the wider test set)."""
import json, warnings, numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
tr = pd.read_csv("data/BT2024066_train_var1.csv"); X = tr.drop(columns="y").values; y = tr.y.values
r = np.linalg.norm(X, axis=1); inner = r < np.quantile(r, .7); out = {}
for d in [2, 3, 4, 5, 6]:
    pf = PolynomialFeatures(d, include_bias=False); F = pf.fit_transform(X[inner]); sc = StandardScaler().fit(F)
    m = Lasso(alpha=0.01, max_iter=50000, tol=1e-5).fit(sc.transform(F), y[inner])
    out[d] = float(mean_squared_error(y[~inner], m.predict(sc.transform(pf.transform(X[~inner])))))
    print(d, out[d], flush=True)
json.dump(out, open("shift_results.json", "w"))
