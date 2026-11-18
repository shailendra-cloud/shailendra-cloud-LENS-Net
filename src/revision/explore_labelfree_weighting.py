"""EXPLORATORY feasibility check (development data only, not pre-registered): label-free, target-adaptive decision fusion.
For every unseen set and seed, member weights are estimated from the members' scores on that set alone (no labels),
with the spectral meta-learner idea (Parisi et al., PNAS 2014): if members err independently given the class, the
off-diagonal covariance of their rank-transformed scores is rank one, and its leading vector grows with each member's
accuracy. Labels are used only to score the fused output. Writes review/explore_saaf.json."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import saved_predictions as RD
from saved_predictions import Data, UNSEEN, L
RD.MODELS.update({"VideoMAE-B FT (tuned)": "A1_videomae_lr2e-05", "VideoMAE-B LoRA": "A1_videomae_lora_lr0.0001"})
_orig = RD._run_dir; RD._run_dir = lambda st, sd: (L / f"runs_review/{st}_seed{sd}") if "_lora" in st else _orig(st, sd)
D = Data(); SEEDS = (42, 43, 44); RR = L / "runs_review"; ANCHOR = "CLIP probe"


def ft(s):
    S = D.sets[s]; out = []
    for r in SEEDS:
        d = RR / f"CLIP-FT_audited_seed{r}"
        if s in ("XD", "UCF"): p = pd.read_csv(d / "predictions_external.csv").p_violent.values[S["rows"]]
        else: q = pd.read_csv(d / f"predictions_{S['dir']}.csv").p_violent.values; p = q[S["rows"]]
        out.append(p[S["keep"]])
    return out


def probs(m, s):
    if m == "CLIP-FT": return ft(s)
    k = D.sets[s]["keep"]; return [p[k] for p in D.probs(m, s)]


def z(p): r = rankdata(p); return (r - r.mean()) / r.std()


def sml_weights(P):
    """Leading vector of the rank-one fit to the off-diagonal covariance of standardised ranks (members x windows)."""
    Z = np.stack([z(p) for p in P]); C = np.cov(Z); M = C.shape[0]; d = np.diag(C).copy()
    for _ in range(100):                      # fill the diagonal with the rank-one fit and iterate
        C2 = C.copy(); np.fill_diagonal(C2, d); w, V = np.linalg.eigh(C2); v = V[:, -1] * np.sqrt(max(w[-1], 1e-12))
        if v.sum() < 0: v = -v
        d_new = v ** 2
        if np.allclose(d_new, d, atol=1e-9): break
        d = d_new
    return v, w


def fuse(P, w): w = np.asarray(w, float); w = w / w.sum(); return sum(wi * p for wi, p in zip(w, P))


auc = lambda y, p: 100 * roc_auc_score(y, p)
SETS_M = {
    "M3 anchor+LENS-Net+LoRA": [ANCHOR, "LENS-Net", "VideoMAE-B LoRA"],
    "M4 +tuned VideoMAE": [ANCHOR, "LENS-Net", "VideoMAE-B LoRA", "VideoMAE-B FT (tuned)"],
    "M5 +motion probe": [ANCHOR, "LENS-Net", "VideoMAE-B LoRA", "VideoMAE-B FT (tuned)", "Motion probe"],
    "M3v anchor+LoRA+tuned": [ANCHOR, "VideoMAE-B LoRA", "VideoMAE-B FT (tuned)"],
}
LAMBDAS = (0.0, 0.5, 1.0); FLOORS = (0.0, 0.25)
out = {"members": {}, "configs": {}, "tracking": {}}
members = sorted({m for v in SETS_M.values() for m in v})
PR = {(m, s): probs(m, s) for m in members for s in UNSEEN}
Y = {s: D.sets[s]["y"][D.sets[s]["keep"]] for s in UNSEEN}
for m in members:
    per = {s: float(np.mean([auc(Y[s], p) for p in PR[m, s]])) for s in UNSEEN}
    out["members"][m] = {**per, "mean9": float(np.mean(list(per.values()))), "min9": float(min(per.values()))}
for name, ms in SETS_M.items():
    track = []
    for lam in LAMBDAS:
        for fl in FLOORS:
            key = f"{name} | lambda {lam} | anchor floor {fl}"; per = {}
            for s in UNSEEN:
                a = []
                for i in range(3):
                    P = [PR[m, s][i] for m in ms]; v, _ = sml_weights(P); ws = np.clip(v, 0, None)
                    ws = ws / ws.sum() if ws.sum() > 0 else np.full(len(ms), 1 / len(ms))
                    w = (1 - lam) * np.full(len(ms), 1 / len(ms)) + lam * ws
                    if fl > 0 and w[0] < fl: w = np.r_[fl, (1 - fl) * w[1:] / w[1:].sum()]
                    a.append(auc(Y[s], fuse(P, w)))
                    if lam == 1.0 and fl == 0.0:   # does the label-free weight track true accuracy within a set?
                        true = [auc(Y[s], p) for p in P]; track.append(spearmanr(v, true).correlation if len(ms) > 2 else np.nan)
                per[s] = float(np.mean(a))
            out["configs"][key] = {**per, "mean9": float(np.mean(list(per.values()))), "min9": float(min(per.values()))}
    # oracle: best fixed simplex weights per set and seed using labels (upper bound, coarse grid)
    per = {}
    grid = [np.array(g) for g in np.ndindex(*(5,) * len(ms)) if sum(g) > 0]
    for s in UNSEEN:
        per[s] = float(np.mean([max(auc(Y[s], fuse([PR[m, s][i] for m in ms], g)) for g in grid) for i in range(3)]))
    out["configs"][f"{name} | ORACLE (labels)"] = {**per, "mean9": float(np.mean(list(per.values()))), "min9": float(min(per.values()))}
    out["tracking"][name] = float(np.nanmean(track))
pair = [ANCHOR, "VideoMAE-B LoRA"]; per = {s: float(np.mean([auc(Y[s], fuse([PR[m, s][i] for m in pair], [1, 1])) for i in range(3)])) for s in UNSEEN}
out["configs"]["reference: anchor+LoRA mean (best pair, E5)"] = {**per, "mean9": float(np.mean(list(per.values()))), "min9": float(min(per.values()))}
json.dump(out, open(L / "review/explore_saaf.json", "w"), indent=1)
print(f"{'config':62s} mean9   min9")
for k, v in out["configs"].items(): print(f"{k:62s} {v['mean9']:6.2f} {v['min9']:6.2f}")
print("Spearman(label-free weight, true AUC) within set, mean:", {k: round(v, 2) for k, v in out["tracking"].items()})
