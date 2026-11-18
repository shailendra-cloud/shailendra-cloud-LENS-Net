"""B7 summary: ten-seed means and SDs for the frozen-feature heads and LENS-Net-A, and the seed-aware tests of the
claims that involve only these models (H2, H4, H5), recomputed with ten seeds. Writes review/ten_seeds.json."""
import json, numpy as np
from saved_predictions import Data10, SETS, UNSEEN, L
from seed_aware_bootstrap import Boot, summarize
D = Data10()
BT = Boot({s: (D.sets[s]["y"][D.sets[s]["keep"]], np.asarray(D.sets[s]["cl"])[D.sets[s]["keep"]]) for s in SETS})
out = {"per_set": {}, "tests": {}}
H = {}
for m in ("LENS-Net", "CLIP probe", "w/o motion", "Motion probe", "LENS-Net-A"):
    rows = {}
    for s in SETS:
        k = D.sets[s]["keep"]; ps = [p[k] for p in D.probs(m, s)]; A = BT.aucs(s, ps)
        rows[s] = dict(mean=round(float(A[:, 0].mean()), 2), sd=round(float(A[:, 0].std(ddof=1)), 2), n_seeds=len(ps),
                       mean3=round(float(A[:3, 0].mean()), 2))
        H[(m, s)] = BT.hier(m, A)
    rows["mean9"] = round(float(np.mean([rows[s]["mean"] for s in UNSEEN])), 2); rows["min9"] = round(float(np.min([rows[s]["mean"] for s in UNSEEN])), 2)
    rows["mean9_3seeds"] = round(float(np.mean([rows[s]["mean3"] for s in UNSEEN])), 2)
    out["per_set"][m] = rows
    print(f"{m:14s} " + " ".join(f"{s[:6]:>6s} {rows[s]['mean']:6.2f}±{rows[s]['sd']:.2f}" for s in SETS) + f" | mean9 {rows['mean9']} (3 seeds {rows['mean9_3seeds']}) min9 {rows['min9']}")
nine = lambda a, c: (np.mean([H[(a, s)] - H[(c, s)] for s in UNSEEN], 0), np.all([BT.S[s]["ok"] for s in UNSEEN], 0))
for key, (r) in {"H2 CLIP probe - LENS-Net (nine-set mean), 10 seeds": summarize(*nine("CLIP probe", "LENS-Net")),
                 "LENS-Net-A - CLIP probe (nine-set mean), 10 seeds": summarize(*nine("LENS-Net-A", "CLIP probe")),
                 "LENS-Net-A - LENS-Net (nine-set mean), 10 seeds": summarize(*nine("LENS-Net-A", "LENS-Net")),
                 "H4 LENS-Net - w/o motion on SCFD, 10 seeds": summarize(H[("LENS-Net", "SCFD")] - H[("w/o motion", "SCFD")], BT.S["SCFD"]["ok"]),
                 "H5 LENS-Net - w/o motion on SCVD, 10 seeds": summarize(H[("LENS-Net", "SCVD")] - H[("w/o motion", "SCVD")], BT.S["SCVD"]["ok"])}.items():
    out["tests"][key] = r; print(f"{key:52s} {r['diff']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] p1={r['p_one_sided']:.4f} p2={r['p_two_sided']:.4f}")
json.dump(out, open(L / "review/ten_seeds.json", "w"), indent=1)
