"""E6 (PREREG_revision.md): member diversity vs gain of decision-level fusion (R1 mean of probabilities) on the nine
unseen sets, for the nine member pairs of E5. Diversity: D1 = 1 - Spearman rho of the members' scores (primary),
D2 = Krogh-Vedelsby ambiguity (Brier(pbar) = mean Brier - ambiguity, checked), D3 = decision disagreement at 0.5.
Gain = AUC(R1) - max member AUC. Permutation test within sets; OLS with set fixed effects and set bootstrap.
Writes review/e6_diversity.json."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import saved_predictions as RD
from saved_predictions import Data, UNSEEN, L
RD.MODELS.update({"VideoMAE-B FT (tuned)": "A1_videomae_lr2e-05", "VideoMAE-B LoRA": "A1_videomae_lora_lr0.0001"})
_orig = RD._run_dir; RD._run_dir = lambda st, sd: (L / f"runs_review/{st}_seed{sd}") if "_lora" in st else _orig(st, sd)
D = Data(); SEEDS = (42, 43, 44); RR = L / "runs_review"; ANCHOR = "CLIP probe"
MEMBERS = ["LENS-Net", "w/o motion", "Motion probe", "VideoMAE-B FT (tuned)", "VideoMAE-B LoRA", "CLIP-FT", "VadCLIP"]
PAIRS = [(ANCHOR, m) for m in MEMBERS] + [("LENS-Net", "VideoMAE-B FT (tuned)"), ("LENS-Net", "VideoMAE-B LoRA")]
def ft(s):
    S = D.sets[s]; out = []
    for r in SEEDS:
        d = RR / f"CLIP-FT_audited_seed{r}"
        if s in ("XD", "UCF"): p = pd.read_csv(d / "predictions_external.csv").p_violent.values[S["rows"]]
        else: q = pd.read_csv(d / f"predictions_{S['dir']}.csv").p_violent.values; assert len(q) == S["nwin"]; p = q[S["rows"]]
        out.append(p[S["keep"]])
    return out
def probs(m, s):
    if m == "CLIP-FT": return ft(s)
    k = D.sets[s]["keep"]; return [p[k] for p in D.probs(m, s)]
rows = []; max_identity_err = 0.0
for a, b in PAIRS:
    for s in UNSEEN:
        y = D.sets[s]["y"][D.sets[s]["keep"]]; A, B = probs(a, s), probs(b, s); acc = []
        for pa, pb in zip(A, B):
            pbar = (pa + pb) / 2; amb = np.mean(0.5 * ((pa - pbar) ** 2 + (pb - pbar) ** 2))
            brier = lambda p: np.mean((p - y) ** 2)
            max_identity_err = max(max_identity_err, abs(brier(pbar) - (0.5 * (brier(pa) + brier(pb)) - amb)))
            aa, ab, af = (100 * roc_auc_score(y, p) for p in (pa, pb, pbar))
            acc.append(dict(D1=1 - spearmanr(pa, pb).correlation, D2=amb, D3=float(np.mean((pa > 0.5) != (pb > 0.5))),
                            gain=af - max(aa, ab), gap=abs(aa - ab), auc_fused=af))
        r = {k: float(np.mean([x[k] for x in acc])) for k in acc[0]}; r.update(pair=f"{a} + {b}", set=s); rows.append(r)
df = pd.DataFrame(rows)
rho = spearmanr(df.D1, df.gain).correlation
rng = np.random.default_rng(0); perm = []
for _ in range(10000):
    g = df.groupby("set").gain.transform(lambda x: rng.permutation(x.values)); perm.append(spearmanr(df.D1, g).correlation)
perm = np.array(perm); p_perm = (1 + np.sum(perm >= rho)) / (len(perm) + 1)
def ols(d):
    X = np.c_[np.ones(len(d)), d.D1.values, d.gap.values, pd.get_dummies(d.set, drop_first=True).values.astype(float)]
    return np.linalg.lstsq(X, d.gain.values, rcond=None)[0][1:3]
b = ols(df); sets = np.array(UNSEEN); boots = []
for _ in range(2000):
    pick = rng.choice(sets, len(sets), replace=True)
    d = pd.concat([df[df.set == st].assign(set=f"{st}#{i}") for i, st in enumerate(pick)])
    boots.append(ols(d))
boots = np.array(boots)
pairlvl = df.groupby("pair")[["D1", "D2", "D3", "gain", "gap", "auc_fused"]].mean()
out = dict(n_points=len(df), spearman_D1_gain=float(rho), perm_p_one_sided=float(p_perm),
           spearman_D2_gain=float(spearmanr(df.D2, df.gain).correlation), spearman_D3_gain=float(spearmanr(df.D3, df.gain).correlation),
           ols_b1_D1=float(b[0]), ols_b1_ci=[float(x) for x in np.percentile(boots[:, 0], [2.5, 97.5])],
           ols_b2_gap=float(b[1]), ols_b2_ci=[float(x) for x in np.percentile(boots[:, 1], [2.5, 97.5])],
           krogh_vedelsby_max_abs_error=float(max_identity_err), pairs=pairlvl.round(4).to_dict("index"), points=df.round(5).to_dict("records"))
json.dump(out, open(L / "review/e6_diversity.json", "w"), indent=1)
print(f"rho(D1,gain)={rho:.3f} perm p={p_perm:.4f} | rho D2 {out['spearman_D2_gain']:.3f} D3 {out['spearman_D3_gain']:.3f}")
print(f"OLS b1(D1)={b[0]:.2f} {out['ols_b1_ci']}  b2(gap)={b[1]:.3f} {out['ols_b2_ci']}  KV identity max err {max_identity_err:.2e}")
print(pairlvl.round(3).to_string())
