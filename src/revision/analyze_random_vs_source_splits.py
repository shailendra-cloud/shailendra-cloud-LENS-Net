"""C1 analysis (PREREG_revision.md): random (R) vs source-controlled (S) splits of RWF-2000 and Hockey Fights.
O1 own-test window AUC (and video accuracy for the fine-tuned models, threshold chosen on validation videos), R - S with
a bootstrap over test clusters (same-source components) and seeds; O2 linked vs unlinked R-test windows; O3 transfer to
XD-Violence and UCF-Crime of R- vs S-trained models (seed-aware paired test). All descriptive."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
from seed_aware_bootstrap import Boot, summarize, wauc
from saved_predictions import Data
L = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace"); ROOT = L.parent
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "video_id", "component_id"])
E = pd.read_csv(ROOT / "data_tables/manifests/near_duplicate_edges.csv", usecols=["segment_a", "segment_b"]); s2v = dict(zip(man.segment_id, man.video_id))
comp = dict(zip(man.segment_id, man.component_id)); D = Data()
MODELS = {"VideoMAE-B FT": ("gpu", "videomae"), "R(2+1)D-18 FT": ("gpu", "r2plus1d_18"), "CLIP probe": ("heads", "LP"), "LENS-Net": ("heads", "LENS-M")}
def run_dir(kind, m, split, seed):
    return L / (f"runs_review/{split}_{m}_seed{seed}" if kind == "gpu" else f"runs_review_heads/L14_std_{split}_{m}_{'vrex1'}_seed{seed}")
def test_preds(kind, m, split, seed):
    d = pd.read_csv(run_dir(kind, m, split, seed) / "predictions_test.csv"); return d[["segment_id", "binary_label", "p_violent"]]
def video_acc(kind, m, split, seed):
    if kind != "gpu": return np.nan
    rd = run_dir(kind, m, split, seed)
    va = pd.read_csv(rd / "predictions_val.csv"); te = pd.read_csv(rd / "predictions_test.csv")
    vv = va.groupby("video_id").agg(y=("binary_label", "max"), p=("p_violent", "mean")); tv = te.groupby("video_id").agg(y=("binary_label", "max"), p=("p_violent", "mean"))
    ths = np.unique(vv.p); acc = [((vv.p >= t) == vv.y).mean() for t in ths]; t = ths[int(np.argmax(acc))]
    return 100 * float(((tv.p >= t) == tv.y).mean())
out = {}
for corpus in ("RWF2000", "HockeyFights"):
    sp = {k: pd.read_csv(L / f"review/splits/C1_{corpus}_{k}.csv") for k in ("R", "S")}
    # linked R-test windows (direct same-source link to an R-train video)
    R = sp["R"]; tr = set(R[R.split == "train"].video_id); segs = set(R.segment_id)
    ed = E[E.segment_a.isin(segs) & E.segment_b.isin(segs)]
    linked_v = {s2v[b] for a, b in zip(ed.segment_a, ed.segment_b) if s2v[a] in tr} | {s2v[a] for a, b in zip(ed.segment_a, ed.segment_b) if s2v[b] in tr}
    res = {}
    for name, (kind, m) in MODELS.items():
        row = {}
        for k in ("R", "S"):
            ps = [test_preds(kind, m, f"C1_{corpus}_{k}", s) for s in (42, 43, 44)]
            y = ps[0].binary_label.values; cl = ps[0].segment_id.map(comp).values
            row[k] = dict(auc=[100 * roc_auc_score(y, p.p_violent) for p in ps], vacc=[video_acc(kind, m, f"C1_{corpus}_{k}", s) for s in (42, 43, 44)], y=y, cl=cl, ps=[p.p_violent.values for p in ps], seg=ps[0].segment_id.values)
        # O1: R - S, independent test sets: bootstrap each set separately (clusters + seeds), difference of the two
        bt = Boot({"R": (row["R"]["y"], row["R"]["cl"]), "S": (row["S"]["y"], row["S"]["cl"])})
        hr = bt.hier(name + "R", bt.aucs("R", row["R"]["ps"])); hs = bt.hier(name + "S", bt.aucs("S", row["S"]["ps"]))
        o1 = summarize(hr - hs, bt.S["R"]["ok"] & bt.S["S"]["ok"])
        # O2: linked vs unlinked within the R test set
        lk = np.array([s2v[x] in linked_v for x in row["R"]["seg"]]); yR = row["R"]["y"]
        def sub_auc(mask): return float(np.mean([100 * roc_auc_score(yR[mask], p[mask]) for p in row["R"]["ps"]])) if len(set(yR[mask])) == 2 else None
        res[name] = dict(R_auc=float(np.mean(row["R"]["auc"])), S_auc=float(np.mean(row["S"]["auc"])), R_minus_S=o1,
                         R_vacc=float(np.nanmean(row["R"]["vacc"])) if kind == "gpu" else None, S_vacc=float(np.nanmean(row["S"]["vacc"])) if kind == "gpu" else None,
                         R_linked_auc=sub_auc(lk), R_unlinked_auc=sub_auc(~lk), R_linked_share=float(lk.mean()))
        # O3: transfer to XD / UCF (same windows), R-trained minus S-trained
        o3 = {}
        for s in ("XD", "UCF"):
            S_ = D.sets[s]; y = S_["y"]; cl = np.asarray(S_["cl"])
            def ext(k):
                if kind == "gpu": return [pd.read_csv(run_dir(kind, m, f"C1_{corpus}_{k}", sd) / "predictions_external.csv").p_violent.values[S_["rows"]] for sd in (42, 43, 44)]
                return [pd.read_csv(run_dir(kind, m, f"C1_{corpus}_{k}", sd) / "predictions_external.csv").p_violent.values[S_["rows"]] for sd in (42, 43, 44)]
            b2 = Boot({s: (y, cl)}); a = b2.hier(name + k + "R", b2.aucs(s, ext("R"))); b = b2.hier(name + k + "S", b2.aucs(s, ext("S")))
            o3[s] = dict(R=float(a[0]), S=float(b[0]), R_minus_S=summarize(a - b, b2.S[s]["ok"]))
        res[name]["transfer"] = o3
    out[corpus] = res
json.dump(out, open(L / "review/c1_results.json", "w"), indent=1, default=float)
f = lambda r: f"{r['diff']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]"
for c, res in out.items():
    print(f"=== {c}")
    for m, r in res.items():
        va = f" | video acc R {r['R_vacc']:.2f} S {r['S_vacc']:.2f}" if r["R_vacc"] is not None else ""
        lk = f" | R-test linked {r['R_linked_auc']:.2f} vs unlinked {r['R_unlinked_auc']:.2f} ({100*r['R_linked_share']:.0f}% linked)" if r["R_linked_auc"] else ""
        print(f"{m:15s} own-test AUC R {r['R_auc']:.2f} S {r['S_auc']:.2f} diff {f(r['R_minus_S'])}{va}{lk}")
        print(f"{'':15s} transfer XD R {r['transfer']['XD']['R']:.2f} S {r['transfer']['XD']['S']:.2f} {f(r['transfer']['XD']['R_minus_S'])} | UCF R {r['transfer']['UCF']['R']:.2f} S {r['transfer']['UCF']['S']:.2f} {f(r['transfer']['UCF']['R_minus_S'])}")
