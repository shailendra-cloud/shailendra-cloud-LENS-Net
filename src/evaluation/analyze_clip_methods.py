#!/usr/bin/env python3
"""PREREG_journal Sec. 6 analysis (exploratory; see there). Same test sets, exclusions and statistics as before:
P1 window ROC-AUC on the audited test, XD, UCF, SCFD, SCVD (near-duplicates excluded), VioPeru, AIRTLab; P3 RWF official
video accuracy and transfer (analyze_baselines_rwf targets). Seed means; paired differences on seed-averaged predictions with a
cluster bootstrap (2000, rng 0). Writes logs/clip_methods_results.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L
from video_data import window_data as D
R = L.OUT.parent; ROOT = L.ROOT; SEEDS = (42, 43, 44)
auc = lambda y, p: 100 * roc_auc_score(y, p)
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv")
te = man[man.split == "test"].reset_index(drop=True)
ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
POOL = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
fw = pd.read_csv(R / "fresh/near_duplicate_exclusion.csv"); fkeep = fw.excluded_near_duplicate.values == 0
cw = pd.read_csv(R / "conf/conf_windows.csv"); cw = cw[cw.decode_failed == 0].reset_index(drop=True)
SETS = {  # name: (labels, clusters, mask, source)
    "Test": (te.binary_label.values, te.component_id.values, np.ones(len(te), bool), "test"),
    "XD": (POOL.binary_label.values, POOL.video.values, ((POOL.benchmark == "XD-test") & (POOL.excluded_precaution == 0)).values, "ext"),
    "UCF": (POOL.binary_label.values, POOL.video.values, ((POOL.benchmark == "UCF-test") & (POOL.excluded_precaution == 0)).values, "ext"),
    "SCFD": (fw.binary_label.values, fw.video.values, fkeep & (fw.benchmark == "SCFD").values, "fresh"),
    "SCVD": (fw.binary_label.values, fw.video.values, fkeep & (fw.benchmark == "SCVD").values, "fresh"),
    "VioPeru": (cw.binary_label.values, cw.cluster.values, (cw.benchmark == "VioPeru").values, "conf"),
    "AIRTLab": (cw.binary_label.values, cw.cluster.values, (cw.benchmark == "AIRTLab").values, "conf")}
def rd(p): return pd.read_csv(p).p_violent.values
def by_seg(p, ref):                                        # align a predictions_test.csv to the manifest test order
    d = pd.read_csv(p); return d.set_index("segment_id").loc[ref.segment_id, "p_violent"].values
def lens_src(run_dir, pref, name):
    return {"test": lambda s: by_seg(run_dir(s) / "predictions_test.csv", te), "ext": lambda s: rd(run_dir(s) / "predictions_external.csv"),
            "fresh": lambda s: rd(R / "fresh/preds" / f"{pref}{name}_seed{s}.csv"), "conf": lambda s: rd(R / "conf/preds" / f"{pref}{name}_seed{s}.csv")}
MODELS = {
    "LENS-Net": lens_src(lambda s: R / "runs" / f"v2_audited_LENS-M_vrex_seed{s}", "", "v2_audited_LENS-M_vrex"),
    "LENS-Net (para)": lens_src(lambda s: R / "runs_ppara" / f"v2_audited_LENS-M_vrex_seed{s}", "ppara_", "v2_audited_LENS-M_vrex"),
    "LENS-Net (surv)": lens_src(lambda s: R / "runs_psurv" / f"v2_audited_LENS-M_vrex_seed{s}", "psurv_", "v2_audited_LENS-M_vrex"),
    "LENS-Net (broad)": lens_src(lambda s: R / "runs_pbroad" / f"v2_audited_LENS-M_vrex_seed{s}", "pbroad_", "v2_audited_LENS-M_vrex"),
    "VideoMAE-B FT": {"test": lambda s: by_seg(R / "runs_baselines" / f"A1_videomae_seed{s}/predictions_test.csv", te),
                      "ext": lambda s: rd(R / "runs_baselines" / f"A1_videomae_seed{s}/predictions_external.csv"),
                      "fresh": lambda s: rd(R / "fresh/preds" / f"A1_videomae_seed{s}.csv"), "conf": lambda s: rd(R / "conf/preds" / f"A1_videomae_seed{s}.csv")},
    "X-CLIP FT": {"test": lambda s: by_seg(R / "runs_baselines" / f"A1_xclip_seed{s}/predictions_test.csv", te),
                  "ext": lambda s: rd(R / "runs_baselines" / f"A1_xclip_seed{s}/predictions_external.csv"),
                  "fresh": lambda s: rd(R / "fresh/preds" / f"A1_xclip_seed{s}.csv"), "conf": lambda s: rd(R / "conf/preds" / f"A1_xclip_seed{s}.csv")},
    "X-CLIP FT (lr 5e-6)": {"test": lambda s: by_seg(R / "runs_baselines" / f"A1_xclip_lr5e-06_seed{s}/predictions_test.csv", te),
                  "ext": lambda s: rd(R / "runs_baselines" / f"A1_xclip_lr5e-06_seed{s}/predictions_external.csv"),
                  "fresh": lambda s: rd(R / "fresh/preds" / f"A1_xclip_lr5e-06_seed{s}.csv"), "conf": lambda s: rd(R / "conf/preds" / f"A1_xclip_lr5e-06_seed{s}.csv")},
    "X-CLIP zero-shot": {"test": lambda s: by_seg(R / "runs_baselines/XCLIP-ZS/predictions_test.csv", te),
                         "ext": lambda s: rd(R / "runs_baselines/XCLIP-ZS/predictions_external.csv"),
                         "fresh": lambda s: rd(R / "fresh/preds/XCLIP-ZS.csv"), "conf": lambda s: rd(R / "conf/preds/XCLIP-ZS.csv")},
}
for bb in ("B16", "L14"):
    rdir = lambda s, bb=bb: R / "runs_vadclip" / f"VadCLIP-{bb}_audited_lr0.0001_seed{s}"
    MODELS[f"VadCLIP ({bb})"] = {"test": lambda s, r=rdir: by_seg(r(s) / "predictions_test.csv", te), "ext": lambda s, r=rdir: rd(r(s) / "predictions_external.csv"),
                                 "fresh": lambda s, r=rdir: rd(r(s) / "predictions_fresh.csv"), "conf": lambda s, r=rdir: rd(r(s) / "predictions_conf.csv")}
SINGLE = {"X-CLIP zero-shot"}
P = {}
for m, src in MODELS.items():
    P[m] = {k: [f(s) for s in ((42,) if m in SINGLE else SEEDS)] for k, f in src.items()}
res = {"P1": {}, "P1_paired": {}, "P3": {}, "P3_paired": {}}
for m in MODELS:
    row = {}
    for sn, (y, cl, mk, k) in SETS.items():
        v = [auc(y[mk], p[mk]) for p in P[m][k]]; row[sn] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, n=len(v))
    res["P1"][m] = row
def boot(pa, pb, y, cl, groups_seed=0):
    grp = pd.Series(np.arange(len(y))).groupby(cl).apply(np.array).values; rng = np.random.default_rng(groups_seed); ds = []
    for _ in range(2000):
        i = np.concatenate(grp[rng.integers(0, len(grp), len(grp))])
        if len(np.unique(y[i])) == 2: ds.append(auc(y[i], pa[i]) - auc(y[i], pb[i]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(diff=float(auc(y, pa) - auc(y, pb)), ci=[float(lo), float(hi)], excludes_zero=bool(lo > 0 or hi < 0),
                better=bool(lo > 0), worse=bool(hi < 0), non_inferior=bool(lo > -1.0))
PAIRS = [("LENS-Net", "VadCLIP (L14)"), ("LENS-Net", "VadCLIP (B16)"), ("LENS-Net", "X-CLIP FT"), ("LENS-Net", "X-CLIP FT (lr 5e-6)"), ("LENS-Net", "X-CLIP zero-shot"),
         ("LENS-Net (broad)", "LENS-Net"), ("LENS-Net (para)", "LENS-Net"), ("LENS-Net (surv)", "LENS-Net")]
for a, b in PAIRS:
    res["P1_paired"][f"{a} - {b}"] = {}
    for sn, (y, cl, mk, k) in SETS.items():
        pa, pb = np.mean(P[a][k], 0)[mk], np.mean(P[b][k], 0)[mk]
        res["P1_paired"][f"{a} - {b}"][sn] = boot(pa, pb, y[mk], cl[mk])
res["prompt_spread_P1"] = {sn: float(np.ptp([res["P1"][m][sn]["mean"] for m in ("LENS-Net", "LENS-Net (para)", "LENS-Net (surv)")])) for sn in SETS}
# ---- per-category recall on XD/UCF (original vs broad prompts), each seed's validation threshold
XDC = {"B1": "Fighting", "B2": "Shooting", "B4": "Riot", "B5": "Abuse", "B6": "Car accident", "G": "Explosion"}
cat = np.where(POOL.benchmark == "XD-test", POOL.category.str.split("+").str[0].map(lambda c: XDC.get(c, c)), POOL.category)
res["category_recall"] = {}
for m, rdir in (("LENS-Net", "runs"), ("LENS-Net (broad)", "runs_pbroad")):
    thr = [json.load(open(R / rdir / f"v2_audited_LENS-M_vrex_seed{s}/summary.json"))["threshold_selected_on_val"] for s in SEEDS]
    out = {}
    for b in ("XD-test", "UCF-test"):
        for c in sorted(set(cat[(POOL.benchmark == b) & (POOL.binary_label == 1) & (POOL.excluded_precaution == 0)])):
            mk = ((POOL.benchmark == b) & (POOL.binary_label == 1) & (POOL.excluded_precaution == 0) & (cat == c)).values
            if mk.sum() < 8: continue
            out[f"{b}|{c}"] = float(np.mean([100 * np.mean(p[mk] >= t) for p, t in zip(P[m]["ext"], thr)]))
    res["category_recall"][m] = out
# ---- P3: RWF official accuracy and transfer (analyze_baselines_rwf targets)
mm = man[["segment_id", "video_id", "dataset", "source_unit_id", "binary_label"]].merge(pd.read_csv(R / "splits/std_rwf.csv", usecols=["segment_id", "split"]), on="segment_id", how="left")
tr_units = set(mm[mm.split == "train"].source_unit_id)
e = pd.read_csv(ROOT / "data_tables/manifests/near_duplicate_edges.csv", usecols=["unit_a", "unit_b"])
linked = set(e[e.unit_a.isin(tr_units)].unit_b) | set(e[e.unit_b.isin(tr_units)].unit_a) | tr_units
tmask = (mm.split.isna() & (mm.dataset != "RWF2000") & ~mm.source_unit_id.isin(linked)).values
comp = man.component_id.values
TG = ["HockeyFights", "RLVS", "BusViolence", "VID", "XD-test", "UCF-test"]
def p3(val, test, bench, extp):
    yv, pv = L.video_level(val, val.p_violent.values); yt, pt = L.video_level(test, test.p_violent.values)
    o = {"rwf_acc": D.binary_metrics(yt, pt, D.select_threshold(yv, pv))["accuracy"]}
    for d in TG[:4]:
        k = tmask & (mm.dataset == d).values; o[d] = auc(mm.binary_label.values[k], bench[k])
    for b in TG[4:]:
        k = ((POOL.benchmark == b) & (POOL.excluded_precaution == 0)).values; o[b] = auc(POOL.binary_label.values[k], extp[k])
    o["mean_transfer"] = float(np.mean([o[t] for t in TG])); return o
P3SRC = {"LENS-Net": R / "runs", "LENS-Net (para)": R / "runs_ppara", "LENS-Net (surv)": R / "runs_psurv", "LENS-Net (broad)": R / "runs_pbroad"}
import lensnet as V, torch
fb = V.Compact("bench"); rows = np.arange(len(fb.index)); P3P = {}
for m, rdir in list(P3SRC.items()) + [(f"VadCLIP ({bb})", None) for bb in ("B16", "L14")]:
    outs, bench_l, ext_l = [], [], []
    for s in SEEDS:
        if rdir is None:
            d = R / "runs_vadclip" / f"VadCLIP-{m[9:12]}_std_std_rwf_lr0.0001_seed{s}"; bench = np.load(d / "predictions_bench.npy")
        else:
            d = rdir / f"v2_std_std_rwf_LENS-M_vrex_seed{s}"
            bench = np.load(d / "predictions_bench.npy")
        extp = rd(d / "predictions_external.csv") if rdir is None else np.load(d / "predictions_external_std.npy")  # LENS: pre-scored per prompt set
        outs.append(p3(pd.read_csv(d / "predictions_val.csv"), pd.read_csv(d / "predictions_test.csv"), bench, extp)); bench_l.append(bench); ext_l.append(extp)
    res["P3"][m] = {k: dict(mean=float(np.mean([o[k] for o in outs])), sd=float(np.std([o[k] for o in outs], ddof=1))) for k in outs[0]}
    P3P[m] = (np.mean(bench_l, 0), np.mean(ext_l, 0))
def boot_mean_transfer(a, b):                               # resample clusters within every target; mean of per-target AUC diffs
    rng = np.random.default_rng(0); tg = []
    for d in TG[:4]:
        k = np.where(tmask & (mm.dataset == d).values)[0]; tg.append((k, mm.binary_label.values[k], comp[k], "b"))
    for bn in TG[4:]:
        k = np.where(((POOL.benchmark == bn) & (POOL.excluded_precaution == 0)).values)[0]; tg.append((k, POOL.binary_label.values[k], POOL.video.values[k], "e"))
    grp = [pd.Series(np.arange(len(y))).groupby(c).apply(np.array).values for _, y, c, _ in tg]
    def diff(sel):
        v = []
        for (k, y, _, src), s in zip(tg, sel):
            pa = (P3P[a][0] if src == "b" else P3P[a][1])[k][s]; pb = (P3P[b][0] if src == "b" else P3P[b][1])[k][s]
            v.append(auc(y[s], pa) - auc(y[s], pb))
        return float(np.mean(v))
    full = diff([np.arange(len(y)) for _, y, _, _ in tg]); ds = []
    for _ in range(2000):
        sel = [np.concatenate(g[rng.integers(0, len(g), len(g))]) for g in grp]
        if all(len(np.unique(t[1][s])) == 2 for t, s in zip(tg, sel)): ds.append(diff(sel))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(diff=full, ci=[float(lo), float(hi)], excludes_zero=bool(lo > 0 or hi < 0), better=bool(lo > 0), worse=bool(hi < 0), non_inferior=bool(lo > -1.0))
for a, b in [("LENS-Net", "VadCLIP (L14)"), ("LENS-Net", "VadCLIP (B16)"), ("LENS-Net (broad)", "LENS-Net"), ("LENS-Net (para)", "LENS-Net"), ("LENS-Net (surv)", "LENS-Net")]:
    res["P3_paired"][f"{a} - {b}"] = boot_mean_transfer(a, b)
json.dump(res, open(R / "logs/clip_methods_results.json", "w"), indent=2)
cols = list(SETS)
print(f"{'P1 model':20s}" + "".join(f"{c:>9s}" for c in cols))
for m, r in res["P1"].items(): print(f"{m:20s}" + "".join(f"{r[c]['mean']:9.2f}" for c in cols))
for k, v in res["P1_paired"].items(): print(f"{k:34s}", " ".join(f"{c}:{x['diff']:+.2f}{'*' if x['better'] else ('-' if x['worse'] else '')}" for c, x in v.items()))
print("prompt spread (orig/para/surv):", {k: round(v, 2) for k, v in res["prompt_spread_P1"].items()})
for m, r in res["P3"].items(): print(f"P3 {m:18s} RWF acc {r['rwf_acc']['mean']:.2f}  mean transfer {r['mean_transfer']['mean']:.2f}  " + " ".join(f"{t[:5]} {r[t]['mean']:.1f}" for t in TG))
for k, v in res["P3_paired"].items(): print(f"P3 {k:34s} {v['diff']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]{'*' if v['better'] else ('-' if v['worse'] else '')}")
for m, r in res["category_recall"].items(): print(m, {k: round(v, 1) for k, v in r.items()})
