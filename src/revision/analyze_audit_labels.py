#!/usr/bin/env python3
"""D1 (PREREG_revision.md): human validation of the overlap criteria. Per stratum: pairs judged same source (label 1 or
2) with Wilson 95% intervals; stratum-weighted recall of the verification stage among the audit's candidates (Jeffreys
posteriors, 20,000 draws, rng 0); cross-split candidates; test-retest agreement of the second labelling pass.
Reads workspace/review/{label_key_HIDDEN,human_labels_round1,human_labels_final,human_labels_pass2}.csv.
Writes workspace/review/d1_labels_final.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint
ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); R = ROOT / "workspace/review"
K = pd.read_csv(R / "label_key_HIDDEN.csv"); L1 = pd.read_csv(R / "human_labels_final.csv")
d = K.merge(L1, on="pair_id"); d["same"] = d.label.isin([1, 2])
out = {"labeller_count": 1, "n_labelled": len(d), "definition": "same source = label 1 or 2",
       "revision": "after a 12-pair consistency check, the labeller revisited 5 flagged pairs and changed 4 (P050,P076,P139,P191: 2->3, different hockey games); P177 kept; P035 added",
       "strata": {}}
for s, x in d.groupby("stratum"):
    k, n = int(x.same.sum()), len(x); lo, hi = proportion_confint(k, n, method="wilson")
    out["strata"][s] = {"same": k, "n": n, "rate": round(k / n, 3), "wilson95": [round(lo, 3), round(hi, 3)]}
fine = pd.read_csv(ROOT / "data_tables/audit/near_duplicate_edges_fine.csv"); Nv = int((fine.grad128 >= 0.30).sum()); Nr = int((fine.grad128 < 0.30).sum())
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "split"]).set_index("segment_id").split
rej = fine[fine.grad128 < 0.30]; cross = rej[rej.segment_a.map(man) != rej.segment_b.map(man)]
rng = np.random.default_rng(0); kv, nv = out["strata"]["verified_edge"]["same"], out["strata"]["verified_edge"]["n"]
kr, nr = out["strata"]["rejected_candidate"]["same"], out["strata"]["rejected_candidate"]["n"]
rec = lambda pv, pr: Nv * pv / (Nv * pv + Nr * pr)
b = [rec(rng.beta(kv + .5, nv - kv + .5), rng.beta(kr + .5, nr - kr + .5)) for _ in range(20000)]
rs = d[d.stratum == "rejected_candidate"].copy(); rs["cross"] = rs.a.str[6:].map(man) != rs.b.str[6:].map(man)
out["weighted"] = dict(N_verified=Nv, N_rejected=Nr, recall_verification_among_candidates=round(rec(kv / nv, kr / nr), 3),
                       recall_ci95_jeffreys=[round(x, 3) for x in np.percentile(b, [2.5, 97.5])], precision_verified=round(kv / nv, 3),
                       missed_links_crossing_train_test=f"{int((rs.same & rs.cross).sum())} of {int(rs.same.sum())}",
                       rejected_population_cross_split_share=round(float(len(cross) / len(rej)), 4),
                       sampled_cross_split_rejected=f"{int(rs.cross.sum())}, all judged different source" if not (rs.same & rs.cross).any() else f"{int(rs.cross.sum())}",
                       population_cross_split_rejected=len(cross),
                       population_cross_split_rejected_train_test=int(pd.Series([{x, y} == {"train", "test"} for x, y in zip(cross.segment_a.map(man), cross.segment_b.map(man))]).sum()))
p2 = pd.read_csv(R / "human_labels_pass2.csv"); t = L1[["pair_id", "label"]].merge(p2, on="pair_id"); s1, s2 = t.label.isin([1, 2]), t.label2.isin([1, 2])
out["intra_rater"] = dict(n=len(t), agree_same_source=round(float((s1 == s2).mean()), 3), kappa_same_source=round(float(cohen_kappa_score(s1, s2)), 3),
                          agree_3class=round(float((t.label == t.label2).mean()), 3), kappa_3class=round(float(cohen_kappa_score(t.label, t.label2)), 3))
json.dump(out, open(R / "d1_labels_final.json", "w"), indent=1)
print(json.dumps(out["weighted"]), json.dumps(out["intra_rater"]))
