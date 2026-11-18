"""C4 and C5 analysis (PREREG_revision.md). C5: VideoMAE-B with LoRA (lr 1e-4, chosen on validation), per-set and
nine-set AUCs and descriptive seed-aware tests against VideoMAE-B FT and the frozen-feature models. C4: VideoMAE-B FT
and R(2+1)D-18 FT at their validation-selected body lr (2e-5); claims H1 and H3 re-computed with tuned VideoMAE-B,
next to the original. Writes review/c45.json. Usage: analyze_fair_baselines.py [c5|c4|both]"""
import json, sys
import numpy as np, pandas as pd
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import saved_predictions as RD
from saved_predictions import Data, SETS, UNSEEN, L
from seed_aware_bootstrap import Boot, summarize
NEW = {"VideoMAE-B LoRA": "A1_videomae_lora_lr0.0001", "VideoMAE-B FT (tuned)": "A1_videomae_lr2e-05", "R(2+1)D-18 FT (tuned)": "A1_r2plus1d_18_lr2e-05"}
RD.MODELS.update(NEW); _orig = RD._run_dir
RD._run_dir = lambda stem, seed: (L / f"runs_review/{stem}_seed{seed}") if "_lora" in stem else _orig(stem, seed)
what = sys.argv[1] if len(sys.argv) > 1 else "both"
D = Data()
evals = {s: (D.sets[s]["y"][D.sets[s]["keep"]], np.asarray(D.sets[s]["cl"])[D.sets[s]["keep"]]) for s in SETS}
BT = Boot(evals); cache = {}
def H(m, s):
    if (m, s) not in cache:
        k = D.sets[s]["keep"]; cache[(m, s)] = BT.hier(m, BT.aucs(s, [p[k] for p in D.probs(m, s)]))
    return cache[(m, s)]
ok9 = np.all([BT.S[s]["ok"] for s in UNSEEN], 0)
def nine(a, c): return summarize(np.mean([H(a, s) - H(c, s) for s in UNSEEN], 0), ok9)
def table(m):
    per = {s: float(H(m, s)[0]) for s in SETS}; per["mean9"] = float(np.mean([per[s] for s in UNSEEN])); per["min9"] = float(min(per[s] for s in UNSEEN)); return per
out = json.load(open(L / "review/c45.json")) if (L / "review/c45.json").exists() else {}
if what in ("c5", "both"):
    r = {"auc": {m: table(m) for m in ("VideoMAE-B LoRA", "VideoMAE-B FT", "CLIP probe", "LENS-Net", "LENS-Net-A")}, "tests": {}}
    r["trained_params"] = json.load(open(RD._run_dir(NEW["VideoMAE-B LoRA"], 42) / "summary.json")).get("trained_params")
    for a, c in (("VideoMAE-B LoRA", "VideoMAE-B FT"), ("CLIP probe", "VideoMAE-B LoRA"), ("LENS-Net", "VideoMAE-B LoRA"), ("LENS-Net-A", "VideoMAE-B LoRA")):
        r["tests"][f"{a} - {c} (nine-set mean)"] = nine(a, c)
    out["C5"] = r
if what in ("c4", "both"):
    r = {"auc": {m: table(m) for m in ("VideoMAE-B FT", "VideoMAE-B FT (tuned)", "R(2+1)D-18 FT", "R(2+1)D-18 FT (tuned)")}, "claims": {}}
    for k, (a, c) in {"H1 CLIP probe - VideoMAE-B FT": ("CLIP probe", "VideoMAE-B FT"), "H3 LENS-Net - VideoMAE-B FT": ("LENS-Net", "VideoMAE-B FT")}.items():
        r["claims"][k] = {"original": nine(a, c), "tuned": nine(a, c + " (tuned)")}
    r["tests"] = {f"{a} - {c} (nine-set mean)": nine(a, c) for a, c in (("LENS-Net-A", "VideoMAE-B FT (tuned)"), ("CLIP probe", "R(2+1)D-18 FT (tuned)"),
                  ("LENS-Net", "R(2+1)D-18 FT (tuned)"), ("VideoMAE-B LoRA", "VideoMAE-B FT (tuned)"), ("VideoMAE-B FT (tuned)", "VideoMAE-B FT"))}
    fam = {k: v["p"] for k, v in json.load(open(L / "review/stats.json"))["primary"].items()}
    for k in list(fam):
        for h, key in (("H1", "H1 CLIP probe - VideoMAE-B FT"), ("H3", "H3 LENS-Net - VideoMAE-B FT")):
            if k.startswith(h + " "): fam[k] = r["claims"][key]["tuned"]["p_two_sided"]
    order = sorted(fam, key=fam.get); holm = {}; stop = False
    for i, k in enumerate(order):
        thr = 0.05 / (len(order) - i); ok = (not stop) and fam[k] <= thr; stop = stop or not ok; holm[k] = dict(p=fam[k], threshold=thr, holds=ok)
    r["holm_with_tuned"] = holm
    out["C4"] = r
json.dump(out, open(L / "review/c45.json", "w"), indent=1)
for part, r in out.items():
    print("==", part)
    for m, v in r["auc"].items(): print(f"{m:24s} " + " ".join(f"{v[s]:6.2f}" for s in SETS) + f"  mean9 {v['mean9']:.2f} min9 {v['min9']:.2f}")
    for k, v in r.get("tests", {}).items(): print(f"{k:50s} {v['diff']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] p={v['p_two_sided']:.4f}")
    for k, v in r.get("claims", {}).items():
        print(k, " | ".join(f"{t}: {x['diff']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}] p={x['p_two_sided']:.4f}" for t, x in v.items()))
    for k, v in r.get("holm_with_tuned", {}).items(): print(f"  holm {k[:55]:55s} p={v['p']:.4f} thr={v['threshold']:.4f} {'HOLDS' if v['holds'] else 'fails'}")
