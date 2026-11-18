"""B3: every unseen set with and without test-side near-duplicate filtering. Writes review/filtering.json."""
import json, numpy as np, pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score
from saved_predictions import Data, MODELS, L, _ext_frame
D = Data(); ex = _ext_frame()
# XD/UCF: the unfiltered set adds back the windows excluded for contamination with the training corpora
xd_all = {}
for b, nm in (("XD-test", "XD"), ("UCF-test", "UCF")):
    m_all = (ex.binary_label.isin([0, 1]) & (ex.decode_failed == 0) & (ex.benchmark == b)).values
    xd_all[nm] = dict(rows=np.where(m_all)[0], excl=(ex.excluded_contamination.values[m_all] == 1), y=ex.binary_label.values[m_all])
models = list(MODELS) + ["LENS-Net-A"]
out = {}
for s in ("XD", "UCF", "SCFD", "SCVD", "CCTV-F", "UBI-Fights"):
    res = {}
    for m in models:
        if s in xd_all:
            S = xd_all[s]
            def full(model):
                if model == "LENS-Net-A":
                    return [(a + b) / 2 for a, b in zip(full("LENS-Net"), full("CLIP probe"))]
                from saved_predictions import _run_dir
                return [pd.read_csv(_run_dir(MODELS[model], r) / "predictions_external.csv").p_violent.values[S["rows"]] for r in D.seeds(model)]
            ps = full(m); y = S["y"]; keep = ~S["excl"]
        else:
            ps = D.probs(m, s); y = D.sets[s]["y"]; keep = D.sets[s]["keep"]
        f = np.mean([100 * roc_auc_score(y[keep], p[keep]) for p in ps]); u = np.mean([100 * roc_auc_score(y, p) for p in ps])
        res[m] = dict(filtered=round(float(f), 2), unfiltered=round(float(u), 2), change=round(float(u - f), 2))
    n_ex = int((~keep).sum()); tau = kendalltau([res[m]["filtered"] for m in models], [res[m]["unfiltered"] for m in models])[0]
    rank_f = sorted(models, key=lambda m: -res[m]["filtered"]); rank_u = sorted(models, key=lambda m: -res[m]["unfiltered"])
    out[s] = dict(n_windows_filtered=int(keep.sum()), n_removed=n_ex, kendall_tau=round(float(tau), 3), models=res,
                  top3_filtered=rank_f[:3], top3_unfiltered=rank_u[:3],
                  mean_change=round(float(np.mean([res[m]["change"] for m in models])), 2))
    print(f"{s:10s} removed {n_ex:4d} of {len(keep):5d}  tau {tau:.3f}  mean change {out[s]['mean_change']:+.2f}  "
          f"LENS-Net {res['LENS-Net']['filtered']}->{res['LENS-Net']['unfiltered']}  VideoMAE {res['VideoMAE-B FT']['filtered']}->{res['VideoMAE-B FT']['unfiltered']}  "
          f"probe {res['CLIP probe']['filtered']}->{res['CLIP probe']['unfiltered']}  top1 {rank_f[0]} / {rank_u[0]}")
json.dump(out, open(L / "review/filtering.json", "w"), indent=1)
