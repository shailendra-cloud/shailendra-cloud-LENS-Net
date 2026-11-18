#!/usr/bin/env python3
"""Part F (PREREG_revision.md): pre-registered confirmation of anchored ensemble selection (AES) on the untouched samples.
AES = equal-weight mean of window probabilities of {CLIP probe, VideoMAE-B LoRA, VideoMAE-B FT (tuned)}, same seed;
video score = max over kept windows (primary) or mean of the top 3 (secondary); filters, clusters and labels as for K1-K4.
Claims F1-F6, seed-aware hierarchical bootstrap (seed_aware_bootstrap.Boot), one-sided p, Holm at 0.05. Writes review/f_aes_confirm.json."""
import json, sys
import numpy as np, pandas as pd
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
sys.argv = [sys.argv[0]]
import confirm_anchor_fusion as CA, confirm_anchor_fusion_sample_b as CB
from seed_aware_bootstrap import Boot, summarize
L = CA.ROOT / "workspace"; SEEDS = (42, 43, 44)
STEM = {"CLIP probe": "v2_audited_LP", "LENS-Net": "v2_audited_LENS-M_vrex", "w/o motion": "v2_audited_V2b",
        "VideoMAE-B LoRA": "A1_videomae_lora_lr0.0001", "VideoMAE-B FT (tuned)": "A1_videomae_lr2e-05"}
AES = ["CLIP probe", "VideoMAE-B LoRA", "VideoMAE-B FT (tuned)"]
FUSED = {"AES": AES, "LENS-Net-A": ["CLIP probe", "LENS-Net"], "anchor + LoRA": ["CLIP probe", "VideoMAE-B LoRA"]}

# window tables, keep masks and prediction folders of the three samples
_, wA = CA.windows_ok(); xA = pd.read_csv(CA.C3 / "near_duplicate_exclusion.csv"); kA = xA.excluded_near_duplicate.values == 0
_, wB = CB.windows_ok(); xB = pd.read_csv(CB.C3 / "near_duplicate_exclusionB.csv"); kB = xB.excluded_near_duplicate.values == 0
vA = pd.read_csv(CA.C3 / "videos.csv").set_index("video"); vB = pd.read_csv(CB.VB).set_index("video")
SAMPLES = {"XD sample A": (wA, kA, CA.P, vA, "XD-train"), "AIE": (wA, kA, CA.P, vA, "AIE"), "XD sample B": (wB, kB, CB.PB, vB, "XD-train-B")}


def window_probs(sample, model, seed):
    w, k, P, _, _ = SAMPLES[sample]
    stem = "ZS" if model == "CLIP zero-shot" else f"{STEM[model]}_seed{seed}"
    p = pd.read_csv(P / f"{stem}.csv").p_violent.values; assert len(p) == len(w), (sample, model, len(p), len(w)); return p


def video_scores(sample, p, how):
    w, k, _, _, _ = SAMPLES[sample]
    g = pd.DataFrame({"video": w.video.values[k], "p": p[k]}).groupby("video").p
    return g.max() if how == "max" else g.apply(lambda v: np.sort(v.values)[-3:].mean())


def ids_of(sample):
    w, k, P, vid, bench = SAMPLES[sample]
    ref = video_scores(sample, window_probs(sample, "LENS-Net", 42), "max").index      # videos with a kept window
    ids = [v for v in ref if vid.loc[v, "benchmark"] == bench]
    return ids, vid.loc[ids, "binary_label"].values, vid.loc[ids, "cluster"].values


META = {s: ids_of(s) for s in SAMPLES}
BT = Boot({s: (META[s][1], META[s][2]) for s in SAMPLES})
res = {"n_videos": {s: len(META[s][0]) for s in SAMPLES}, "n_violent": {s: int(META[s][1].sum()) for s in SAMPLES},
       "n_clusters": {s: int(len(set(META[s][2]))) for s in SAMPLES}}
for how in ("max", "top3"):
    def per_seed(sample, model):
        ids = META[sample][0]
        if model in FUSED:
            return [video_scores(sample, np.mean([window_probs(sample, m, r) for m in FUSED[model]], 0), how).loc[ids].values for r in SEEDS]
        seeds = (42,) if model == "CLIP zero-shot" else SEEDS
        return [video_scores(sample, window_probs(sample, model, r), how).loc[ids].values for r in seeds]
    models = list(FUSED) + list(STEM) + ["CLIP zero-shot"]
    H = {(m, s): BT.hier(m, BT.aucs(s, per_seed(s, m))) for m in models for s in SAMPLES}
    res[f"auc_{how}"] = {m: {s: float(H[m, s][0]) for s in SAMPLES} for m in models}
    fam = {}
    for i, s in enumerate(SAMPLES):
        fam[f"F{i + 1} AES - LENS-Net-A on {s} (better)"] = (summarize(H["AES", s] - H["LENS-Net-A", s], BT.S[s]["ok"], 0.0), 0.0)
    for i, s in enumerate(SAMPLES):
        fam[f"F{i + 4} AES - CLIP probe on {s} (non-inferior)"] = (summarize(H["AES", s] - H["CLIP probe", s], BT.S[s]["ok"], -1.0), -1.0)
    fam = {k: dict(v[0], p=v[0]["p_one_sided"], delta0=v[1]) for k, v in fam.items()}
    order = sorted(fam, key=lambda k: fam[k]["p"]); stop = False
    for j, k in enumerate(order):
        thr = 0.05 / (len(order) - j); fam[k]["holm_threshold"] = thr; fam[k]["holds_after_holm"] = (not stop) and fam[k]["p"] <= thr
        if not fam[k]["holds_after_holm"]: stop = True
    res[f"claims_{how}"] = fam
    res[f"confirmed_{how}"] = bool(all(fam[k]["holds_after_holm"] for k in fam if k[:2] in ("F1", "F2", "F3")))
json.dump(res, open(L / "review/f_aes_confirm.json", "w"), indent=1)
print("videos", res["n_videos"], "violent", res["n_violent"], "clusters", res["n_clusters"])
for how in ("max", "top3"):
    print(f"=== video score = {how} ({'PRIMARY' if how == 'max' else 'secondary'})")
    for m, d in res[f"auc_{how}"].items(): print(f"  {m:24s} " + "  ".join(f"{s} {v:6.2f}" for s, v in d.items()))
    for k, r in sorted(res[f"claims_{how}"].items()):
        print(f"  {k:52s} {r['diff']:+6.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] p={r['p']:.4f} thr={r['holm_threshold']:.4f} holds={r['holds_after_holm']}")
    print(f"  AES CONFIRMED (F1-F3 after Holm): {res[f'confirmed_{how}']}")
