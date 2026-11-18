"""E5 (PREREG_revision.md): decision-level fusion rules under domain shift, from saved predictions only.
Pairs of members fused seed by seed with fixed rules (R1 mean prob, R2 mean logit, R3 normalised geometric mean, R4 max,
R5 min, R6 rank average) and rules fitted on the source validation windows (R7 weighted mean, R8 logistic stacking).
Writes review/e5_fusion.json."""
import json, sys
import numpy as np, pandas as pd, torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import saved_predictions as RD, lensnet_setup as LN, lensnet as V
from saved_predictions import Data, SETS, UNSEEN, L
from seed_aware_bootstrap import Boot, summarize
RD.MODELS.update({"VideoMAE-B FT (tuned)": "A1_videomae_lr2e-05", "VideoMAE-B LoRA": "A1_videomae_lora_lr0.0001"})
_orig = RD._run_dir; RD._run_dir = lambda st, sd: (L / f"runs_review/{st}_seed{sd}") if "_lora" in st else _orig(st, sd)
D = Data(); SEEDS = (42, 43, 44); RR = L / "runs_review"
ANCHOR = "CLIP probe"
MEMBERS = ["LENS-Net", "w/o motion", "Motion probe", "VideoMAE-B FT (tuned)", "VideoMAE-B LoRA", "CLIP-FT", "VadCLIP"]
PAIRS = [(ANCHOR, m) for m in MEMBERS] + [("LENS-Net", "VideoMAE-B FT (tuned)"), ("LENS-Net", "VideoMAE-B LoRA")]
RULES = ["R1 mean prob", "R2 mean logit", "R3 geometric", "R4 max", "R5 min", "R6 rank", "R7 val-weighted", "R8 stacking"]
# ---- test-side predictions (per seed, kept windows)
def ft(s):
    S = D.sets[s]; out = []
    for r in SEEDS:
        d = RR / f"CLIP-FT_audited_seed{r}"
        if s == "Test": p = pd.read_csv(d / "predictions_test.csv").set_index("segment_id").loc[S["key"]].p_violent.values
        elif s in ("XD", "UCF"): p = pd.read_csv(d / "predictions_external.csv").p_violent.values[S["rows"]]
        else: q = pd.read_csv(d / f"predictions_{S['dir']}.csv").p_violent.values; assert len(q) == S["nwin"]; p = q[S["rows"]]
        out.append(p[S["keep"]])
    return out
def probs(m, s):
    if m == "CLIP-FT": return ft(s)
    k = D.sets[s]["keep"]; return [p[k] for p in D.probs(m, s)]
# ---- validation predictions (source corpora, audited val split)
base, _ = LN.load_split("audited"); va = base[base.split == "val"].reset_index(drop=True); yv = va.binary_label.values
HEAD = {"CLIP probe": ("v2_audited_LP", "LP"), "LENS-Net": ("v2_audited_LENS-M_vrex", "LENS-M"), "w/o motion": ("v2_audited_V2b", "V2b"), "Motion probe": ("v2_audited_MP", "MP")}
RUN = {"VideoMAE-B FT (tuned)": lambda r: L / f"runs_baselines/A1_videomae_lr2e-05_seed{r}", "VideoMAE-B LoRA": lambda r: RR / f"A1_videomae_lora_lr0.0001_seed{r}",
       "CLIP-FT": lambda r: RR / f"CLIP-FT_audited_seed{r}", "VadCLIP": lambda r: L / f"runs_vadclip/VadCLIP-L14_audited_lr0.0001_seed{r}"}
feats = V.Compact("bench"); VAL = {}
def val(m):
    if m not in VAL:
        if m in HEAD:
            stem, cfg = HEAD[m]; out = []
            for r in SEEDS:
                net = V.V2(cfg, LN.text_anchors()); net.load_state_dict(torch.load(LN.OUT / f"{stem}_seed{r}" / "best.pt", weights_only=True)); net.eval()
                out.append(V.predict(net, feats, va.feat_row.values))
            VAL[m] = out
        else:
            VAL[m] = [pd.read_csv(RUN[m](r) / "predictions_val.csv").set_index("segment_id").loc[va.segment_id].p_violent.values for r in SEEDS]
    return VAL[m]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
def fuse(rule, pa, pb, va_, vb_):
    if rule == "R1 mean prob": return (pa + pb) / 2
    if rule == "R2 mean logit": return (lg(pa) + lg(pb)) / 2
    if rule == "R3 geometric": g = np.sqrt(pa * pb); return g / (g + np.sqrt((1 - pa) * (1 - pb)) + 1e-12)
    if rule == "R4 max": return np.maximum(pa, pb)
    if rule == "R5 min": return np.minimum(pa, pb)
    if rule == "R6 rank": return (rankdata(pa) + rankdata(pb)) / 2
    if rule == "R7 val-weighted":
        w = max(np.round(np.arange(0, 1.0001, 0.05), 2), key=lambda w: roc_auc_score(yv, w * va_ + (1 - w) * vb_)); return w * pa + (1 - w) * pb
    if rule == "R8 stacking":
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[lg(va_), lg(vb_)], yv); return clf.decision_function(np.c_[lg(pa), lg(pb)])
def fused(pair, rule, s):
    a, b = pair; A, B = probs(a, s), probs(b, s); VA, VB = val(a), val(b)
    return [fuse(rule, A[i], B[i], VA[i], VB[i]) for i in range(3)]
auc = lambda y, p: 100 * roc_auc_score(y, p)
out = {"pairs": {}, "family": {}, "descriptive": {}, "members": {}}
for m in [ANCHOR] + MEMBERS:
    per = {s: float(np.mean([auc(D.sets[s]["y"][D.sets[s]["keep"]], p) for p in probs(m, s)])) for s in SETS}
    per["mean9"] = float(np.mean([per[s] for s in UNSEEN])); per["min9"] = float(min(per[s] for s in UNSEEN))
    per["val"] = float(np.mean([auc(yv, p) for p in val(m)])); out["members"][m] = per
for pair in PAIRS:
    key = " + ".join(pair); out["pairs"][key] = {}
    for rule in RULES:
        per = {s: float(np.mean([auc(D.sets[s]["y"][D.sets[s]["keep"]], f) for f in fused(pair, rule, s)])) for s in SETS}
        per["mean9"] = float(np.mean([per[s] for s in UNSEEN])); per["min9"] = float(min(per[s] for s in UNSEEN))
        VA, VB = val(pair[0]), val(pair[1])
        per["val"] = float(np.mean([auc(yv, fuse(rule, VA[i], VB[i], VA[i], VB[i])) for i in range(3)]))
        if rule == "R7 val-weighted":
            per["weights_on_first"] = [float(max(np.round(np.arange(0, 1.0001, 0.05), 2), key=lambda w: roc_auc_score(yv, w * VA[i] + (1 - w) * VB[i]))) for i in range(3)]
        out["pairs"][key][rule] = per
    print(key, " ".join(f"{r.split()[0]} {out['pairs'][key][r]['mean9']:.2f}" for r in RULES), flush=True)
# ---- seed-aware tests
BT = Boot({s: (D.sets[s]["y"][D.sets[s]["keep"]], np.asarray(D.sets[s]["cl"])[D.sets[s]["keep"]]) for s in UNSEEN}); cache = {}
ok9 = np.all([BT.S[s]["ok"] for s in UNSEEN], 0)
def H(tag, preds_fn):
    if tag not in cache: cache[tag] = {s: BT.hier(tag, BT.aucs(s, preds_fn(s))) for s in UNSEEN}
    return cache[tag]
def nine(ta, fa, tb, fb): A, B = H(ta, fa), H(tb, fb); return summarize(np.mean([A[s] - B[s] for s in UNSEEN], 0), ok9)
P0 = (ANCHOR, "LENS-Net"); r1 = lambda s: fused(P0, "R1 mean prob", s)
for rule in RULES[1:]:
    out["family"][f"{rule} - R1 (anchor + LENS-Net)"] = nine(f"{P0}|{rule}", lambda s, rule=rule: fused(P0, rule, s), f"{P0}|R1", r1)
fam = out["family"]; ps = {k: v["p_two_sided"] for k, v in fam.items()}; order = sorted(ps, key=ps.get); stop = False
for i, k in enumerate(order):
    thr = 0.05 / (len(order) - i); okk = (not stop) and ps[k] <= thr; stop = stop or not okk; fam[k]["holm_threshold"] = thr; fam[k]["holds_after_holm"] = okk
for pair in PAIRS[1:]:
    key = " + ".join(pair); f1 = lambda s, pair=pair: fused(pair, "R1 mean prob", s)
    for rule in ("R7 val-weighted", "R8 stacking"):
        out["descriptive"][f"{rule} - R1 ({key})"] = nine(f"{pair}|{rule}", lambda s, pair=pair, rule=rule: fused(pair, rule, s), f"{pair}|R1", f1)
for pair in PAIRS:
    key = " + ".join(pair); best = max(pair, key=lambda m: out["members"][m]["mean9"])
    out["descriptive"][f"R1 - better member ({key}; {best})"] = nine(f"{pair}|R1", lambda s, pair=pair: fused(pair, "R1 mean prob", s), best, lambda s, best=best: probs(best, s))
json.dump(out, open(L / "review/e5_fusion.json", "w"), indent=1)
for k, v in list(out["family"].items()) + list(out["descriptive"].items()):
    print(f"{k:62s} {v['diff']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] p={v['p_two_sided']:.4f}" + (f"  Holm {'HOLDS' if v.get('holds_after_holm') else 'fails'}" if 'holm_threshold' in v else ""))
