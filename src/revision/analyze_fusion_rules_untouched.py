#!/usr/bin/env python3
"""Part G (PREREG_revision.md): (1) claims G1-G8 for AES and AES-bal on the new untouched samples (UCF-train, HMDB51);
(2) descriptive comparison with standard ensemble methods on the nine development sets and the five confirmation samples.
Every weight or selection uses the source validation split only; members are fused seed by seed; confirmation samples
are scored at video level (max over kept windows, primary; mean of top 3, secondary). Writes review/g_partG.json.
  analyze_fusion_rules_untouched.py dev      development sets only (no confirmation data read)
  analyze_fusion_rules_untouched.py all      everything"""
import json, sys
import numpy as np, pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
MODE = sys.argv[1] if len(sys.argv) > 1 else "dev"
src = open(__import__("pathlib").Path(__file__).resolve().parents[2] / "src/revision/compare_fusion_rules.py").read().split("lg = lambda")[0]
g = {"__file__": __file__}; sys.argv = [sys.argv[0]]; exec(src, g)
val, yv, probs, D, UNSEEN, L = g["val"], g["yv"], g["probs"], g["D"], g["UNSEEN"], g["L"]
from seed_aware_bootstrap import Boot, summarize
CAND = ["CLIP probe", "LENS-Net", "w/o motion", "Motion probe", "VideoMAE-B FT (tuned)", "VideoMAE-B LoRA", "CLIP-FT", "VadCLIP"]
STEM = {"CLIP probe": "v2_audited_LP", "LENS-Net": "v2_audited_LENS-M_vrex", "w/o motion": "v2_audited_V2b", "Motion probe": "v2_audited_MP",
        "VideoMAE-B FT (tuned)": "A1_videomae_lr2e-05", "VideoMAE-B LoRA": "A1_videomae_lora_lr0.0001", "CLIP-FT": "CLIP-FT_audited",
        "VadCLIP": "VadCLIP-L14_audited_lr0.0001"}
SEEDS = (42, 43, 44); auc = lambda y, p: 100 * roc_auc_score(y, p)
lg = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
VAL = {m: val(m) for m in CAND}


# ---- ensemble rules: each returns, for seed i, a function mapping {member: probs} -> fused score (weights from val only)
def weighted(ws): return lambda P, i: sum(w * P[m] for m, w in ws.items() if w)
def caruana(i, steps=20):
    cnt = {m: 0 for m in CAND}; best0 = max(CAND, key=lambda m: auc(yv, VAL[m][i])); cnt[best0] = 1; hist = [(auc(yv, VAL[best0][i]), dict(cnt))]
    for _ in range(steps - 1):
        n = sum(cnt.values())
        cur = sum(c * VAL[m][i] for m, c in cnt.items())
        m = max(CAND, key=lambda m: auc(yv, (cur + VAL[m][i]) / (n + 1))); cnt[m] += 1
        hist.append((auc(yv, sum(c * VAL[k][i] for k, c in cnt.items()) / (n + 1)), dict(cnt)))
    c = max(hist, key=lambda h: h[0])[1]; s = sum(c.values()); return {m: v / s for m, v in c.items()}
def likelihood(i):
    ll = np.array([np.sum(yv * np.log(np.clip(VAL[m][i], 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - VAL[m][i], 1e-6, 1))) for m in CAND])
    w = np.exp(ll - ll.max()); w /= w.sum(); return dict(zip(CAND, w))
def stacking(i):
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(np.c_[[lg(VAL[m][i]) for m in CAND]].T, yv)
    return lambda P, i: clf.decision_function(np.c_[[lg(P[m]) for m in CAND]].T)
RULES = {
    "AES (anchor + LoRA + tuned, equal)": lambda i: weighted({"CLIP probe": 1/3, "VideoMAE-B LoRA": 1/3, "VideoMAE-B FT (tuned)": 1/3}),
    "AES-bal (1/2, 1/4, 1/4)": lambda i: weighted({"CLIP probe": .5, "VideoMAE-B LoRA": .25, "VideoMAE-B FT (tuned)": .25}),
    "LENS-Net-A (anchor + LENS-Net)": lambda i: weighted({"CLIP probe": .5, "LENS-Net": .5}),
    "Mean of all eight candidates": lambda i: weighted({m: 1 / len(CAND) for m in CAND}),
    "Caruana selection with replacement": lambda i: weighted(caruana(i)),
    "Best single candidate (source val)": lambda i: weighted({max(CAND, key=lambda m: auc(yv, VAL[m][i])): 1.0}),
    "Likelihood weights (softmax of val log-lik.)": lambda i: weighted(likelihood(i)),
    "Majority vote at 0.5": lambda i: (lambda P, i: np.mean([P[m] > 0.5 for m in CAND], 0)),
    "Stacking of all eight (logistic)": stacking,
}
FUSE = {r: [f(i) for i in range(3)] for r, f in RULES.items()}
res = {"weights_info": {"caruana": [caruana(i) for i in range(3)], "likelihood": [likelihood(i) for i in range(3)],
                        "best_single": [max(CAND, key=lambda m: auc(yv, VAL[m][i])) for i in range(3)]}}
# ---- development sets
dev = {}
for r in RULES:
    per = {s: float(np.mean([auc(D.sets[s]["y"][D.sets[s]["keep"]], FUSE[r][i]({m: probs(m, s)[i] for m in CAND}, i)) for i in range(3)])) for s in UNSEEN}
    dev[r] = {**per, "mean9": float(np.mean(list(per.values()))), "min9": float(min(per.values()))}
res["development"] = dev
print(f"{'rule':46s} nine-set mean  min")
for r, v in dev.items(): print(f"{r:46s} {v['mean9']:8.2f} {v['min9']:6.2f}")
if MODE == "dev":
    json.dump(res, open(L / "review/g_partG_dev.json", "w"), indent=1); sys.exit()

# ---- confirmation samples: window probabilities per member and seed, keep masks, videos
import confirm_anchor_fusion as CA, confirm_anchor_fusion_sample_b as CB
L4 = L / "conf4"
def part_f(mod, P, ks, clipdir):
    _, w = mod.windows_ok(); x = pd.read_csv(mod.C3 / ("near_duplicate_exclusion.csv" if mod is CA else "near_duplicate_exclusionB.csv"))
    keep = x.excluded_near_duplicate.values == 0
    def get(m, s):
        if m == "CLIP-FT": return np.concatenate([pd.read_csv(mod.C3 / clipdir / f"CLIP-FT_audited_seed{s}_c{k}.csv").p_violent.values for k in ks])
        return pd.read_csv(P / f"{STEM[m]}_seed{s}.csv").p_violent.values
    return w, keep, get
wA, kA, gA = part_f(CA, CA.P, [0, 1, 2, 3, 4, 5], "preds_clipft"); wB, kB, gB = part_f(CB, CB.PB, [10, 11, 12], "predsB_clipft")
x4 = pd.read_csv(L4 / "near_duplicate_exclusion.csv"); k4 = x4.excluded_near_duplicate.values == 0; K4 = sorted(x4.chunk.unique())
def g4(m, s): return np.concatenate([pd.read_csv(L4 / f"preds/{STEM[m]}_seed{s}_c{k}.csv").p_violent.values for k in K4])
vA = pd.read_csv(CA.C3 / "videos.csv").set_index("video"); vB = pd.read_csv(CB.VB).set_index("video")
v4 = pd.concat([pd.read_csv(L4 / "hmdb51_selection.csv"), pd.read_csv(L4 / "ucf_train_selection.csv").assign(
    cluster=lambda d: "UCF/" + d.video.str.replace(r"\.mp4$", "", regex=True))]).set_index("video")
SAMPLES = {"XD sample A": (wA, kA, gA, vA, "XD-train"), "AIE": (wA, kA, gA, vA, "AIE"), "XD sample B": (wB, kB, gB, vB, "XD-train-B"),
           "UCF-train": (x4, k4, g4, v4, "UCF-train"), "HMDB51": (x4, k4, g4, v4, "HMDB51")}
cache = {}
def member(sample, m, s):
    w, k, get, _, _ = SAMPLES[sample]; key = (id(w), m, s)
    if key not in cache: p = get(m, s); assert len(p) == len(w), (sample, m, s, len(p), len(w)); cache[key] = p
    return cache[key]
def vscore(sample, p, how):
    w, k, _, _, bench = SAMPLES[sample]; sel = k & (w.benchmark.values == bench)
    gg = pd.DataFrame({"video": w.video.values[sel], "p": p[sel]}).groupby("video").p
    return gg.max() if how == "max" else gg.apply(lambda v: np.sort(v.values)[-3:].mean())
META = {}
for sname, (w, k, get, vid, bench) in SAMPLES.items():
    ids = list(vscore(sname, member(sname, "LENS-Net", 42), "max").index)
    META[sname] = (ids, vid.loc[ids, "binary_label"].values.astype(int), vid.loc[ids, "cluster"].values)
res["n_videos"] = {s: len(META[s][0]) for s in SAMPLES}; res["n_violent"] = {s: int(META[s][1].sum()) for s in SAMPLES}
res["n_clusters"] = {s: int(len(set(META[s][2]))) for s in SAMPLES}
res["n_windows_excluded"] = {s: int(((SAMPLES[s][0].benchmark.values == SAMPLES[s][4]) & ~SAMPLES[s][1]).sum()) for s in SAMPLES}
BT = Boot({s: (META[s][1], META[s][2]) for s in SAMPLES})
for how in ("max", "top3"):
    per_seed = {}
    for r in RULES:
        for sname in SAMPLES:
            ids = META[sname][0]
            per_seed[r, sname] = [vscore(sname, FUSE[r][i]({m: member(sname, m, s) for m in CAND}, i), how).loc[ids].values for i, s in enumerate(SEEDS)]
    for m in ["CLIP probe", "CLIP zero-shot"]:
        for sname in SAMPLES:
            ids = META[sname][0]
            if m == "CLIP zero-shot":
                w, _, get, _, _ = SAMPLES[sname]
                zs = (pd.read_csv((CA.P if sname in ("XD sample A", "AIE") else CB.PB) / "ZS.csv").p_violent.values if sname not in ("UCF-train", "HMDB51")
                      else np.concatenate([pd.read_csv(L4 / f"preds/ZS_c{k}.csv").p_violent.values for k in K4]))
                per_seed[m, sname] = [vscore(sname, zs, how).loc[ids].values]
            else: per_seed[m, sname] = [vscore(sname, member(sname, m, s), how).loc[ids].values for s in SEEDS]
    H = {key: BT.hier(key[0], BT.aucs(key[1], v)) for key, v in per_seed.items()}
    res[f"confirmation_auc_{how}"] = {r: {s: float(H[r, s][0]) for s in SAMPLES} for r in list(RULES) + ["CLIP probe", "CLIP zero-shot"]}
    A, B_, LA, LP = "AES (anchor + LoRA + tuned, equal)", "AES-bal (1/2, 1/4, 1/4)", "LENS-Net-A (anchor + LENS-Net)", "CLIP probe"
    fam = {}
    for tag, (a, c, s, d0) in {"G1": (A, LA, "UCF-train", 0.0), "G2": (A, LA, "HMDB51", 0.0), "G3": (B_, LA, "UCF-train", 0.0), "G4": (B_, LA, "HMDB51", 0.0),
                               "G5": (A, LP, "UCF-train", -1.0), "G6": (A, LP, "HMDB51", -1.0), "G7": (B_, LP, "UCF-train", -1.0), "G8": (B_, LP, "HMDB51", -1.0)}.items():
        r = summarize(H[a, s] - H[c, s], BT.S[s]["ok"], d0); r.update(p=r["p_one_sided"], delta0=d0, contrast=f"{a} - {c} on {s}"); fam[tag] = r
    order = sorted(fam, key=lambda k: fam[k]["p"]); stop = False
    for j, k in enumerate(order):
        thr = 0.05 / (len(order) - j); fam[k]["holm_threshold"] = thr; fam[k]["holds_after_holm"] = (not stop) and fam[k]["p"] <= thr
        if not fam[k]["holds_after_holm"]: stop = True
    res[f"claims_{how}"] = fam
    res[f"decisions_{how}"] = {"AES confirmed beyond AIE (G1 and G2)": bool(fam["G1"]["holds_after_holm"] and fam["G2"]["holds_after_holm"]),
                               "AES-bal confirmed (G3 and G4)": bool(fam["G3"]["holds_after_holm"] and fam["G4"]["holds_after_holm"])}
json.dump(res, open(L / "review/g_partG.json", "w"), indent=1, default=float)
print("videos", res["n_videos"], "violent", res["n_violent"], "clusters", res["n_clusters"], "excluded windows", res["n_windows_excluded"])
for how in ("max", "top3"):
    print(f"=== confirmation, video score {how} ({'PRIMARY' if how == 'max' else 'secondary'})")
    for r, v in res[f"confirmation_auc_{how}"].items(): print(f"  {r:46s} " + " ".join(f"{s[:9]:>9s} {x:6.2f}" for s, x in v.items()))
    for k, r in res[f"claims_{how}"].items():
        print(f"  {k} {r['contrast']:70s} {r['diff']:+6.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] p={r['p']:.4f} thr={r['holm_threshold']:.4f} holds={r['holds_after_holm']}")
    print("  decisions:", res[f"decisions_{how}"])
