"""E4 (PREREG_revision.md, Part E): CCTV-Fights (CCTV) and UBI-Fights with the ambiguous windows added as negatives.
Per model: mean per-seed AUC on the labelled windows and with the ambiguous windows; Kendall tau of the two rankings;
best model. The ambiguous windows get the test-side cosine filter (>= 0.95 to a training window). Writes review/e4_amb.json."""
import json, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import saved_predictions as RD
from saved_predictions import Data, L
import lensnet_setup as LN
RD.MODELS.update({"VideoMAE-B FT (tuned)": "A1_videomae_lr2e-05", "R(2+1)D-18 FT (tuned)": "A1_r2plus1d_18_lr2e-05", "VideoMAE-B LoRA": "A1_videomae_lora_lr0.0001"})
_orig = RD._run_dir; RD._run_dir = lambda st, sd: (L / f"runs_review/{st}_seed{sd}") if "_lora" in st else _orig(st, sd)
D = Data(); A = L / "amb"; w = pd.read_csv(A / "amb_windows.csv"); w = w[w.decode_failed == 0].reset_index(drop=True)
zb = np.load(LN.FEAT / "bench" / LN.compact_file())["cls"]; man = pd.read_csv(LN.ROOT / "data_tables/manifests/source_windows.csv", usecols=["split"])
B = F.normalize(torch.tensor(zb[(man.split == "train").values]).float(), dim=-1)
za = F.normalize(torch.tensor(np.load(LN.FEAT / "amb" / LN.compact_file())["cls"]).float(), dim=-1)
mx = torch.cat([(za[i:i + 512] @ B.t()).max(1).values for i in range(0, len(za), 512)]).numpy(); keep_a = mx < 0.95
models = list(RD.MODELS) + ["LENS-Net-A"]
def amb_probs(m):
    if m == "LENS-Net-A": return [(a + b) / 2 for a, b in zip(amb_probs("LENS-Net"), amb_probs("CLIP probe"))]
    st = RD.MODELS[m]; seeds = (42,) if m == "CLIP zero-shot" else (42, 43, 44)
    return [pd.read_csv(A / "preds" / (f"{st}.csv" if st == "ZS" else f"{st}_seed{s}.csv")).p_violent.values for s in seeds]
out = {"removed_by_filter": {}, "sets": {}}
for s, bench in (("CCTV-F", "CCTV-Fights"), ("UBI-Fights", "UBI-Fights")):
    S = D.sets[s]; k = S["keep"]; y = S["y"][k]; am = ((w.benchmark == bench).values & keep_a)
    out["removed_by_filter"][s] = int(((w.benchmark == bench).values & ~keep_a).sum())
    res = {}
    for m in models:
        lab = D.probs(m, s); amp = amb_probs(m)
        a0 = np.mean([100 * roc_auc_score(y, p[k]) for p in lab])
        a1 = np.mean([100 * roc_auc_score(np.r_[y, np.zeros(am.sum())], np.r_[p[k], q[am]]) for p, q in zip(lab, amp)])
        res[m] = dict(labelled=float(a0), with_ambiguous=float(a1))
    tau = kendalltau([res[m]["labelled"] for m in models], [res[m]["with_ambiguous"] for m in models])[0]
    best0 = max(models, key=lambda m: res[m]["labelled"]); best1 = max(models, key=lambda m: res[m]["with_ambiguous"])
    out["sets"][s] = dict(n_labelled=int(k.sum()), n_ambiguous=int(am.sum()), kendall_tau=float(tau), best_labelled=best0, best_with_ambiguous=best1, models=res)
    print(f"{s}: +{am.sum()} ambiguous (filter removed {out['removed_by_filter'][s]}), tau {tau:.3f}, best {best0} -> {best1}")
    for m in models: print(f"   {m:24s} {res[m]['labelled']:.2f} -> {res[m]['with_ambiguous']:.2f}")
json.dump(out, open(L / "review/e4_amb.json", "w"), indent=1)
