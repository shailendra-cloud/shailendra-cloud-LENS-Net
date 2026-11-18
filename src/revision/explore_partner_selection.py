"""EXPLORATORY (development data only): can the anchor's fusion partner be chosen from SOURCE data alone?
For each candidate partner, source-validation quantities from the Krogh-Vedelsby decomposition of the anchor+partner
average are computed (members' mean Brier error, ambiguity, accuracy gap, rank diversity) together with source-only
criteria; the nine-set fused AUC (from review/e5_fusion.json) is only used afterwards to judge the choice."""
import json, sys
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
src = open(__import__("pathlib").Path(__file__).resolve().parents[2] / "src/revision/compare_fusion_rules.py").read().split("lg = lambda")[0]
g = {"__file__": __file__}; exec(src, g)
val, yv, ANCHOR, MEMBERS, L = g["val"], g["yv"], g["ANCHOR"], g["MEMBERS"], g["L"]
e5 = json.load(open(L / "review/e5_fusion.json"))
truth = {m: e5["pairs"][f"{ANCHOR} + {m}"]["R1 mean prob"]["mean9"] for m in MEMBERS}
A = val(ANCHOR); rows = {}
for m in MEMBERS:
    P = val(m); q = {k: [] for k in ("val_auc_fused", "val_auc_partner", "brier_members", "ambiguity", "gap_auc", "rank_div", "fused_brier")}
    for i in range(3):
        a, p = A[i], P[i]; f = 0.5 * (a + p)
        q["val_auc_fused"].append(100 * roc_auc_score(yv, f)); q["val_auc_partner"].append(100 * roc_auc_score(yv, p))
        q["brier_members"].append(0.5 * (np.mean((a - yv) ** 2) + np.mean((p - yv) ** 2)))
        q["ambiguity"].append(np.mean(0.5 * ((a - f) ** 2 + (p - f) ** 2))); q["fused_brier"].append(np.mean((f - yv) ** 2))
        q["gap_auc"].append(abs(100 * roc_auc_score(yv, a) - 100 * roc_auc_score(yv, p)))
        q["rank_div"].append(1 - spearmanr(a, p).correlation)
    rows[m] = {k: float(np.mean(v)) for k, v in q.items()}; rows[m]["nine_set_fused (truth)"] = truth[m]
crit = {
    "max source fused AUC": lambda r: r["val_auc_fused"],
    "max source partner AUC": lambda r: r["val_auc_partner"],
    "min source fused Brier": lambda r: -r["fused_brier"],
    "max source ambiguity": lambda r: r["ambiguity"],
    "max source rank diversity": lambda r: r["rank_div"],
    "max diversity / (1+gap)": lambda r: r["rank_div"] / (1 + r["gap_auc"]),
}
print(f"{'partner':24s} " + " ".join(f"{k[:12]:>12s}" for k in list(rows[MEMBERS[0]])))
for m in sorted(MEMBERS, key=lambda m: -truth[m]): print(f"{m:24s} " + " ".join(f"{v:12.4f}" for v in rows[m].values()))
print()
best_truth = max(truth.values())
for name, f in crit.items():
    pick = max(MEMBERS, key=lambda m: f(rows[m])); rho = spearmanr([f(rows[m]) for m in MEMBERS], [truth[m] for m in MEMBERS]).correlation
    print(f"{name:28s} picks {pick:24s} nine-set {truth[pick]:.2f} (best possible {best_truth:.2f}); Spearman with truth {rho:+.2f}")
json.dump(rows, open(L / "review/explore_partner_select.json", "w"), indent=1)
