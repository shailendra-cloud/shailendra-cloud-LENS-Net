#!/usr/bin/env python3
"""Part H (PREREG_revision.md): fusion baselines from the information-fusion literature - calibrated averaging (H-a),
evidential Dempster-Shafer fusion with reliability discounting (H-b), confidence-weighted averaging (H-c), test-time
adaptation of the BatchNorm heads (TENT, H-d) - on the nine development sets and the five untouched samples; paired
seed-aware intervals of LENS-Net-A minus every rule (Parts G and H) per untouched sample; window-level results.
All descriptive. Parameters from the source validation split only. Writes review/h_partH.json."""
import json, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
sys.argv = [sys.argv[0], "all"]
src = open(__import__("pathlib").Path(__file__).resolve().parents[2] / "src/revision/analyze_fusion_rules_untouched.py").read()
src = src.split("# ---- development sets")[0]          # loaders, candidates and the Part G rules, without running them
g = {"__file__": __file__}; exec(src, g)
for k in ("val", "yv", "probs", "D", "UNSEEN", "L", "CAND", "STEM", "SEEDS", "auc", "lg", "VAL", "RULES", "Boot", "summarize"): globals()[k] = g[k]
import lensnet_setup as LN, lensnet as V
S1, S2, S3 = ["CLIP probe", "LENS-Net"], ["CLIP probe", "VideoMAE-B LoRA", "VideoMAE-B FT (tuned)"], CAND


# ---- H-a temperatures, H-b reliabilities (source validation)
def temp(p):
    z = lg(p); Ts = np.exp(np.linspace(np.log(0.05), np.log(20), 300))
    nll = [-np.mean(yv * np.log(expit(z / T) + 1e-12) + (1 - yv) * np.log(1 - expit(z / T) + 1e-12)) for T in Ts]
    return float(Ts[int(np.argmin(nll))])
TEMP = {m: [temp(VAL[m][i]) for i in range(3)] for m in CAND}
REL = {m: [float(np.clip(2 * roc_auc_score(yv, VAL[m][i]) - 1, 0, 1)) for i in range(3)] for m in CAND}
def calibrated(ms): return lambda P, i: np.mean([expit(lg(P[m]) / TEMP[m][i]) for m in ms], 0)
def dempster(ms):
    def f(P, i):
        mv = mn = None
        for m in ms:
            a = REL[m][i]; v, n, t = a * P[m], a * (1 - P[m]), np.full_like(P[m], 1 - a)
            if mv is None: mv, mn, mt = v, n, t; continue
            K = mv * n + mn * v; d = np.clip(1 - K, 1e-12, None)
            mv, mn, mt = (mv * v + mv * t + mt * v) / d, (mn * n + mn * t + mt * n) / d, (mt * t) / d
        return mv + mt / 2
    return f
def confidence(ms):
    def f(P, i):
        W = np.stack([np.abs(P[m] - 0.5) for m in ms]) + 1e-6; return (W * np.stack([P[m] for m in ms])).sum(0) / W.sum(0)
    return f
NEW = {}
for tag, ms in (("S1 anchor + LENS-Net", S1), ("S2 AES members", S2), ("S3 all eight", S3)):
    NEW[f"Calibrated averaging ({tag})"] = calibrated(ms); NEW[f"Dempster-Shafer, discounted ({tag})"] = dempster(ms)
    NEW[f"Confidence-weighted ({tag})"] = confidence(ms)
TENT_HEADS = {"LENS-Net": ("v2_audited_LENS-M_vrex", "LENS-M"), "w/o motion": ("v2_audited_V2b", "V2b"), "Motion probe": ("v2_audited_MP", "MP")}
for h in TENT_HEADS:
    NEW[f"TENT {h}"] = (lambda h: lambda P, i: P[f"TENT {h}"])(h)
    NEW[f"TENT {h} + anchor"] = (lambda h: lambda P, i: 0.5 * (P[f"TENT {h}"] + P["CLIP probe"]))(h)


# ---- H-d TENT on a set of feature rows (no labels)
def tent(stem, cfg, seed, feats, rows):
    torch.manual_seed(0)
    m = V.V2(cfg, LN.text_anchors()); m.load_state_dict(torch.load(LN.OUT / f"{stem}_seed{seed}" / "best.pt", weights_only=True)); m.eval()
    bns = [x for x in m.modules() if isinstance(x, nn.BatchNorm1d)]
    for p in m.parameters(): p.requires_grad_(False)
    for b in bns:
        b.reset_running_stats(); b.momentum = None; b.train(); b.weight.requires_grad_(True); b.bias.requires_grad_(True)
    r = torch.as_tensor(np.asarray(rows))
    with torch.no_grad():                                        # target statistics (cumulative average over the set)
        for j in range(0, len(r), 512): m(*feats.get(r[j:j + 512]))
    opt = torch.optim.Adam([p for b in bns for p in (b.weight, b.bias)], lr=1e-3)
    for b in bns: b.momentum = 0.1
    perm = r[torch.randperm(len(r), generator=torch.Generator().manual_seed(0))]
    for j in range(0, len(perm), 128):
        pr = torch.softmax(m(*feats.get(perm[j:j + 128])), 1); loss = -(pr * torch.log(pr + 1e-8)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): return torch.softmax(m(*feats.get(r)), 1)[:, 1].numpy()


# ---- development sets
FEAT = {"XD": "external", "UCF": "external", "SCFD": "fresh", "SCVD": "fresh", "VioPeru": "conf", "AIRTLab": "conf", "CCTV-F": "cfights", "Movies": "conf2", "UBI-Fights": "conf2"}
CF = {n: V.Compact(n) for n in set(FEAT.values())}
def dev_members(s, i):
    P = {m: probs(m, s)[i] for m in CAND}; S = D.sets[s]; rows = S["rows"][S["keep"]]
    for h, (stem, cfg) in TENT_HEADS.items(): P[f"TENT {h}"] = tent(stem, cfg, SEEDS[i], CF[FEAT[s]], rows)
    return P
res = {"temperatures": TEMP, "reliabilities": REL, "development": {}}
FU = {r: (RULES[r](i) if r in RULES else None) for r in RULES for i in [0]}
ALLR = {**{r: [RULES[r](i) for i in range(3)] for r in RULES}, **{r: [f] * 3 for r, f in NEW.items()}}
DM = {(s, i): dev_members(s, i) for s in UNSEEN for i in range(3)}
for r, fs in ALLR.items():
    per = {s: float(np.mean([auc(D.sets[s]["y"][D.sets[s]["keep"]], fs[i](DM[s, i], i)) for i in range(3)])) for s in UNSEEN}
    res["development"][r] = {**per, "mean9": float(np.mean(list(per.values()))), "min9": float(min(per.values()))}
print(f"{'rule':52s} dev nine-set mean / min")
for r, v in res["development"].items(): print(f"  {r:52s} {v['mean9']:6.2f} {v['min9']:6.2f}", flush=True)

# ---- untouched samples
_gs = open(__import__("pathlib").Path(__file__).resolve().parents[2] / "src/revision/analyze_fusion_rules_untouched.py").read()
exec("import confirm_anchor_fusion as CA, confirm_anchor_fusion_sample_b as CB\n" + _gs.split("import confirm_anchor_fusion as CA, confirm_anchor_fusion_sample_b as CB\n")[1].split('res["n_videos"]')[0], globals())
import confirm_anchor_fusion as CA, confirm_anchor_fusion_sample_b as CB
CHUNKS = {"XD sample A": [0, 1, 2, 3, 4, 5], "AIE": [0, 1, 2, 3, 4, 5], "XD sample B": [10, 11, 12], "UCF-train": K4, "HMDB51": K4}
TC = {}
def tent_member(sname, h, s):
    key = (sname, h, s)
    if key not in TC:
        w, k, _, _, bench = SAMPLES[sname]; stem, cfg = TENT_HEADS[h]
        fs = [V.Compact(f"c3_{c}") for c in CHUNKS[sname]]
        cls = torch.cat([f.cls for f in fs]); Sm = torch.cat([f.S for f in fs]); M = torch.cat([f.M for f in fs])
        class Cat:
            def get(self, r): return cls[r], Sm[r], M[r]
        assert len(cls) == len(w), (sname, len(cls), len(w))
        rows = np.where(w.benchmark.values == bench)[0]; p = np.full(len(w), np.nan); p[rows] = tent(stem, cfg, s, Cat(), rows); TC[key] = p
    return TC[key]
def members_u(sname, i):
    s = SEEDS[i]; P = {m: member(sname, m, s) for m in CAND}
    for h in TENT_HEADS: P[f"TENT {h}"] = tent_member(sname, h, s)
    return P
UM = {(sn, i): members_u(sn, i) for sn in SAMPLES for i in range(3)}
BT = Boot({sn: (META[sn][1], META[sn][2]) for sn in SAMPLES})
for how in ("max", "top3"):
    PS = {(r, sn): [vscore(sn, fs[i](UM[sn, i], i), how).loc[META[sn][0]].values for i in range(3)] for r, fs in ALLR.items() for sn in SAMPLES}
    H = {key: BT.hier(key[0], BT.aucs(key[1], v)) for key, v in PS.items()}
    res[f"untouched_auc_{how}"] = {r: {sn: float(H[r, sn][0]) for sn in SAMPLES} for r in ALLR}
    LA = "LENS-Net-A (anchor + LENS-Net)"
    res[f"lensa_minus_rule_{how}"] = {r: {sn: summarize(H[LA, sn] - H[r, sn], BT.S[sn]["ok"]) for sn in SAMPLES} for r in ALLR if r != LA}
    res[f"better_than_lensa_everywhere_{how}"] = [r for r, d in res[f"lensa_minus_rule_{how}"].items() if all(d[sn]["worse"] for sn in SAMPLES)]
# ---- window level (each window labelled by its video), primary rules only, descriptive
res["window_level"] = {}
for sn in SAMPLES:
    w, k, _, vid, bench = SAMPLES[sn]; sel = k & (w.benchmark.values == bench); y = vid.loc[w.video.values[sel], "binary_label"].values.astype(int)
    res["window_level"][sn] = {r: float(np.mean([auc(y, fs[i](UM[sn, i], i)[sel]) for i in range(3)])) for r, fs in ALLR.items()}
json.dump(res, open(L / "review/h_partH.json", "w"), indent=1, default=float)
S5 = list(SAMPLES)
print("=== untouched samples, video level (max)")
for r, v in res["untouched_auc_max"].items():
    x = [v[sn] for sn in S5]; print(f"  {r:52s} " + " ".join(f"{a:6.2f}" for a in x) + f"   mean {np.mean(x):6.2f} min {min(x):6.2f}")
print("rules better than LENS-Net-A (interval above zero) on all five samples:", res["better_than_lensa_everywhere_max"])
