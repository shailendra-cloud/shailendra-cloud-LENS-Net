"""Revision analyses B1, B2, B5 (PREREG_revision.md). Seed-aware hierarchical cluster bootstrap on saved predictions,
Holm-Bonferroni over the declared primary claims, symmetric-fusion control. Writes review/stats.json."""
import json, re, sys
import numpy as np, pandas as pd
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
from saved_predictions import Data, SETS, UNSEEN, L
NB = 2000


def wauc(s, y, W):
    """Exact weighted Mann-Whitney AUC (ties count 1/2) for score s (n,), labels y (n,), weights W (B, n) -> (B,)."""
    o = np.argsort(s, kind="mergesort"); s, y, W = s[o], y[o], W[:, o]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    pos = np.add.reduceat(W * (y == 1), starts, axis=1); neg = np.add.reduceat(W * (y == 0), starts, axis=1)
    below = np.cumsum(neg, 1) - neg
    return 100 * (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


class Boot:
    """per evaluation set: fixed cluster resamples (rng 0, sets in order), AUC matrices per (model, seed)."""
    def __init__(self, sets):
        rng = np.random.default_rng(0); self.S = {}
        for name, (y, cl) in sets.items():
            codes, uniq = pd.factorize(cl); k = len(uniq)
            C = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(NB)]).astype(float)
            W = np.vstack([np.ones(len(y)), C[:, codes]])           # row 0: full data (point estimate)
            ok = np.r_[True, [(W[b + 1][y == 1].sum() > 0) and (W[b + 1][y == 0].sum() > 0) for b in range(NB)]]
            self.S[name] = dict(y=y, W=W, ok=ok)
        self.seed_rng = np.random.default_rng(0); self.seed_draw = {}

    def aucs(self, name, ps):                  # list of per-seed score arrays -> (n_seeds, NB+1)
        S = self.S[name]; return np.stack([wauc(p, S["y"], S["W"]) for p in ps])

    def draw(self, model, k):                  # fixed seed resample per model (shared across sets and comparisons)
        if model not in self.seed_draw: self.seed_draw[model] = self.seed_rng.integers(0, k, (NB, k))
        return self.seed_draw[model]

    def hier(self, model, A):                  # A: (k, NB+1) -> (NB+1,) point = mean over seeds, resamples = seed-resampled means
        k = A.shape[0]; d = self.draw(model, k); out = np.empty(A.shape[1]); out[0] = A[:, 0].mean()
        out[1:] = A[d, np.arange(1, A.shape[1])[:, None]].mean(1); return out


def summarize(diff, ok, delta0=0.0):
    b = diff[1:][ok[1:]]; lo, hi = np.percentile(b, [2.5, 97.5])
    p_hi = (1 + (b <= delta0).sum()) / (len(b) + 1); p_lo = (1 + (b >= delta0).sum()) / (len(b) + 1)
    return dict(diff=float(diff[0]), ci=[float(lo), float(hi)], better=bool(lo > 0), worse=bool(hi < 0), non_inferior=bool(lo > -1.0),
                p_one_sided=float(p_hi), p_two_sided=float(min(1.0, 2 * min(p_hi, p_lo))))


def main():
    D = Data()
    evals = {s: (D.sets[s]["y"][D.sets[s]["keep"]], np.asarray(D.sets[s]["cl"])[D.sets[s]["keep"]]) for s in SETS}
    # anchor-fusion confirmation sets, video level (primary: max window score), identical filter to confirm_anchor_fusion.analyze
    sys.argv = [sys.argv[0]]; import confirm_anchor_fusion as C3m
    _, w = C3m.windows_ok(); x = pd.read_csv(C3m.C3 / "near_duplicate_exclusion.csv"); keep = x.excluded_near_duplicate.values == 0
    vid = pd.read_csv(C3m.C3 / "videos.csv").set_index("video")
    def vscore(stem):
        d = pd.DataFrame({"video": w.video.values[keep], "p": pd.read_csv(C3m.P / f"{stem}.csv").p_violent.values[keep]})
        return d.groupby("video").p.max()
    cstems = {"LENS-Net": "v2_audited_LENS-M_vrex", "CLIP probe": "v2_audited_LP", "w/o motion": "v2_audited_V2b", "Motion probe": "v2_audited_MP",
              "CLIP zero-shot": "ZS", "VadCLIP": "VadCLIP-L14_audited_lr0.0001"}
    conf = {}
    for b in ("XD-train", "AIE"):
        ref = vscore("v2_audited_LENS-M_vrex_seed42").index          # videos with at least one kept window, as in confirm_anchor_fusion.analyze
        ids = [v for v in ref if vid.loc[v, "benchmark"] == b]
        conf[b] = dict(ids=ids, y=vid.loc[ids, "binary_label"].values, cl=vid.loc[ids, "cluster"].values)
        evals[b] = (conf[b]["y"], conf[b]["cl"])
    BT = Boot(evals)

    def probs(model, s):
        if s in conf:
            def one(stem, seed): return vscore(stem if stem == "ZS" else f"{stem}_seed{seed}").loc[conf[s]["ids"]].values
            if model == "LENS-Net-A": return [(one(cstems["LENS-Net"], r) + one(cstems["CLIP probe"], r)) / 2 for r in (42, 43, 44)]
            return [one(cstems[model], r) for r in ((42,) if model == "CLIP zero-shot" else (42, 43, 44))]
        k = D.sets[s]["keep"]; return [p[k] for p in D.probs(model, s)]

    cache = {}
    def H(model, s):                            # hierarchical (seed-resampled) AUC vector for model on set s
        if (model, s) not in cache: cache[(model, s)] = BT.hier(model, BT.aucs(s, probs(model, s)))
        return cache[(model, s)]
    def avgscore(model, s):                     # old test: AUC of the seed-averaged score
        return BT.aucs(s, [np.mean(probs(model, s), 0)])[0]

    comps = ["CLIP probe", "w/o motion", "Motion probe", "CLIP zero-shot", "VadCLIP", "X-CLIP FT", "VideoMAE-B FT", "Video Swin-T FT", "R(2+1)D-18 FT", "CNN-BiGRU"]
    out = {"pairwise": {}, "pairwise_seed_averaged": {}, "primary": {}, "symmetric_fusion": {}, "per_seed_means": {}}
    for A in ("LENS-Net", "LENS-Net-A"):
        for c in comps:
            for s in SETS:
                ok = BT.S[s]["ok"]
                out["pairwise"][f"{A} - {c} | {s}"] = summarize(H(A, s) - H(c, s), ok)
                out["pairwise_seed_averaged"][f"{A} - {c} | {s}"] = summarize(avgscore(A, s) - avgscore(c, s), ok)
    # nine-set mean contrasts (per-set resamples are independent; average the per-set Deltas)
    def nine(a, c):
        M = np.mean([H(a, s) - H(c, s) for s in UNSEEN], 0); ok = np.all([BT.S[s]["ok"] for s in UNSEEN], 0); return M, ok
    fam = {}
    for key, (a, c) in {"H1 CLIP probe - VideoMAE-B FT (nine-set mean)": ("CLIP probe", "VideoMAE-B FT"),
                        "H2 CLIP probe - LENS-Net (nine-set mean)": ("CLIP probe", "LENS-Net"),
                        "H3 LENS-Net - VideoMAE-B FT (nine-set mean)": ("LENS-Net", "VideoMAE-B FT")}.items():
        M, ok = nine(a, c); r = summarize(M, ok); r["p"] = r["p_two_sided"]; fam[key] = r
    for key, s in (("H4 LENS-Net - w/o motion on SCFD (better)", "SCFD"), ("H5 LENS-Net - w/o motion on SCVD (better)", "SCVD")):
        r = summarize(H("LENS-Net", s) - H("w/o motion", s), BT.S[s]["ok"]); r["p"] = r["p_one_sided"]; fam[key] = r
    for key, (c, s, d0) in {"K1 LENS-Net-A - LENS-Net on XD-train (better)": ("LENS-Net", "XD-train", 0.0),
                            "K2 LENS-Net-A - LENS-Net on AIE (non-inferior)": ("LENS-Net", "AIE", -1.0),
                            "K3 LENS-Net-A - CLIP probe on XD-train (non-inferior)": ("CLIP probe", "XD-train", -1.0),
                            "K4 LENS-Net-A - CLIP probe on AIE (better)": ("CLIP probe", "AIE", 0.0)}.items():
        r = summarize(H("LENS-Net-A", s) - H(c, s), BT.S[s]["ok"], d0); r["p"] = r["p_one_sided"]; r["delta0"] = d0; fam[key] = r
    # Holm-Bonferroni (step-down) at 0.05
    order = sorted(fam, key=lambda k: fam[k]["p"]); m = len(order); stop = False
    for i, k in enumerate(order):
        thr = 0.05 / (m - i); fam[k]["holm_threshold"] = thr
        fam[k]["holds_after_holm"] = (not stop) and fam[k]["p"] <= thr
        if not fam[k]["holds_after_holm"]: stop = True
    out["primary"] = fam
    # B5 symmetric fusion and per-seed means (nine-set mean / min)
    for model in ["LENS-Net-A", "LENS-Net", "CLIP probe"] + [f"SYM:{c}" for c in ("VadCLIP", "X-CLIP FT", "VideoMAE-B FT", "Video Swin-T FT", "R(2+1)D-18 FT", "CNN-BiGRU", "w/o motion", "Motion probe")]:
        v = [float(np.mean([BT.aucs(s, [p])[0][0] for p in probs(model, s)])) if not model.startswith("SYM:") else
             float(np.mean([BT.aucs(s, [p])[0][0] for p in [q[D.sets[s]["keep"]] for q in D.probs(model, s)]])) for s in UNSEEN]
        out["symmetric_fusion"][model] = dict(zip(UNSEEN, [round(x, 2) for x in v]), mean9=round(float(np.mean(v)), 2), min9=round(float(np.min(v)), 2))
    json.dump(out, open(L / "review/stats.json", "w"), indent=1)
    print("=== primary family (Holm) ===")
    for k in order: r = fam[k]; print(f"{k:58s} {r['diff']:+6.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] p={r['p']:.4f} thr={r['holm_threshold']:.4f} holds={r['holds_after_holm']}")
    print("=== symmetric fusion: nine-set mean / min ===")
    for k, v in out["symmetric_fusion"].items(): print(f"{k:28s} {v['mean9']:6.2f} {v['min9']:6.2f}")
    flips = [(k, out["pairwise_seed_averaged"][k], v) for k, v in out["pairwise"].items()
             if (v["better"], v["worse"]) != (out["pairwise_seed_averaged"][k]["better"], out["pairwise_seed_averaged"][k]["worse"])]
    print(f"=== significance changes, seed-aware vs seed-averaged: {len(flips)} of {len(out['pairwise'])} ===")
    for k, o, n in flips: print(f"{k:50s} old {o['diff']:+.2f} [{o['ci'][0]:+.2f},{o['ci'][1]:+.2f}] -> new {n['diff']:+.2f} [{n['ci'][0]:+.2f},{n['ci'][1]:+.2f}]")


if __name__ == "__main__":
    main()
