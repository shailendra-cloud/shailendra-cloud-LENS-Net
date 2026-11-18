"""E1 (PREREG_revision.md, Part E): CLIP-FT (CLIP probe with the last four CLIP blocks fine-tuned) on the audited test and
the nine unseen sets; descriptive seed-aware tests against the CLIP probe, tuned VideoMAE-B FT, LoRA and LENS-Net-A.
Writes review/e1_clipft.json."""
import json, sys
import numpy as np, pandas as pd
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import saved_predictions as RD
from saved_predictions import Data, SETS, UNSEEN, L
from seed_aware_bootstrap import Boot, summarize
RD.MODELS.update({"VideoMAE-B FT (tuned)": "A1_videomae_lr2e-05", "VideoMAE-B LoRA": "A1_videomae_lora_lr0.0001"})
_orig = RD._run_dir; RD._run_dir = lambda st, sd: (L / f"runs_review/{st}_seed{sd}") if "_lora" in st else _orig(st, sd)
D = Data(); RR = L / "runs_review"
def ft(s):
    S = D.sets[s]; k = S["keep"]; out = []
    for r in (42, 43, 44):
        d = RR / f"CLIP-FT_audited_seed{r}"
        if s == "Test": p = pd.read_csv(d / "predictions_test.csv").set_index("segment_id").loc[S["key"]].p_violent.values
        elif s in ("XD", "UCF"): p = pd.read_csv(d / "predictions_external.csv").p_violent.values[S["rows"]]
        else:
            q = pd.read_csv(d / f"predictions_{S['dir']}.csv").p_violent.values; assert len(q) == S["nwin"]; p = q[S["rows"]]
        out.append(p[k])
    return out
def probs(m, s):
    if m == "CLIP-FT": return ft(s)
    k = D.sets[s]["keep"]; return [p[k] for p in D.probs(m, s)]
BT = Boot({s: (D.sets[s]["y"][D.sets[s]["keep"]], np.asarray(D.sets[s]["cl"])[D.sets[s]["keep"]]) for s in SETS}); cache = {}
def H(m, s):
    if (m, s) not in cache: cache[(m, s)] = BT.hier(m, BT.aucs(s, probs(m, s)))
    return cache[(m, s)]
ok9 = np.all([BT.S[s]["ok"] for s in UNSEEN], 0)
def nine(a, c): return summarize(np.mean([H(a, s) - H(c, s) for s in UNSEEN], 0), ok9)
out = {"auc": {}, "tests": {}}
for m in ("CLIP-FT", "CLIP probe", "LENS-Net", "LENS-Net-A", "VideoMAE-B FT (tuned)", "VideoMAE-B LoRA"):
    per = {s: float(H(m, s)[0]) for s in SETS}; per["mean9"] = float(np.mean([per[s] for s in UNSEEN])); per["min9"] = float(min(per[s] for s in UNSEEN)); out["auc"][m] = per
for a, c in (("CLIP-FT", "CLIP probe"), ("CLIP-FT", "VideoMAE-B FT (tuned)"), ("CLIP-FT", "VideoMAE-B LoRA"), ("LENS-Net-A", "CLIP-FT")):
    out["tests"][f"{a} - {c} (nine-set mean)"] = nine(a, c)
out["test_set"] = {"CLIP-FT - CLIP probe": summarize(H("CLIP-FT", "Test") - H("CLIP probe", "Test"), BT.S["Test"]["ok"])}
out["best_epochs"] = [json.load(open(RR / f"CLIP-FT_audited_seed{r}/summary.json"))["best_epoch"] for r in (42, 43, 44)]
out["trained_params"] = json.load(open(RR / "CLIP-FT_audited_seed42/summary.json"))["trained_params"]
json.dump(out, open(L / "review/e1_clipft.json", "w"), indent=1)
for m, v in out["auc"].items(): print(f"{m:22s} " + " ".join(f"{v[s]:6.2f}" for s in SETS) + f"  mean9 {v['mean9']:.2f} min9 {v['min9']:.2f}")
for k, r in list(out["tests"].items()) + list(out["test_set"].items()): print(f"{k:48s} {r['diff']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] p={r['p_two_sided']:.4f}")
print("best epochs", out["best_epochs"], "params", out["trained_params"])
