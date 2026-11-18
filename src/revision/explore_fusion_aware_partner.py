"""EXPLORATORY (development data only): fusion-aware partner training. The frozen anchor (CLIP probe, same seed) is fixed;
the motion head on frozen VideoMAE features is trained on the SOURCE corpora with
    loss = alpha * CE(p_M) + (1 - alpha) * NLL(0.5 * (p_A + p_M)),
so that the fixed average of the two, the rule that transferred best, is what is optimised. alpha = 1 is the published
motion probe recipe. Checkpoints are chosen on the source validation AUC of the fused output; the nine unseen sets are
only scored. Writes review/explore_partner.json."""
import json, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import lensnet_setup as L, lensnet as V
from video_data import window_data as DS
from saved_predictions import Data, UNSEEN
import saved_predictions as RD
RD.MODELS.update({"VideoMAE-B LoRA": "A1_videomae_lora_lr0.0001"})
_orig = RD._run_dir; RD._run_dir = lambda st, sd: (RD.L / f"runs_review/{st}_seed{sd}") if "_lora" in st else _orig(st, sd)
D = Data()
FEAT = {"XD": "external", "UCF": "external", "SCFD": "fresh", "SCVD": "fresh", "VioPeru": "conf", "AIRTLab": "conf",
        "CCTV-F": "cfights", "Movies": "conf2", "UBI-Fights": "conf2"}
CF = {n: V.Compact(n) for n in set(FEAT.values())}
feats = V.Compact("bench"); df, _ = L.load_split("audited")
tr = df[df.split == "train"].reset_index(drop=True); va = df[df.split == "val"].reset_index(drop=True)


def anchor(seed):
    m = V.V2("LP", L.text_anchors()); m.load_state_dict(torch.load(L.OUT / f"v2_audited_LP_seed{seed}" / "best.pt", weights_only=True)); m.eval()
    return m


def train(seed, alpha):
    torch.manual_seed(seed); np.random.seed(seed)
    A = anchor(seed); model = V.V2("MP", L.text_anchors())
    w = DS.strategy_weights(tr, "hybrid", cap=0.05); rows = torch.as_tensor(tr.feat_row.values); ytr = torch.tensor(tr.binary_label.values)
    g = torch.Generator().manual_seed(seed); bs = 128; steps = len(tr) // bs; epochs = 30
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.05)
    total, warm = steps * epochs, steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1))))
    with torch.no_grad(): pa_va = torch.softmax(A(*feats.get(torch.as_tensor(va.feat_row.values))), 1)[:, 1].numpy()
    best, state = -1, None
    for ep in range(epochs):
        model.train(); idx = torch.multinomial(torch.tensor(w), steps * bs, replacement=True, generator=g)
        for s in range(steps):
            b = idx[s * bs:(s + 1) * bs]; r = rows[b]; y = ytr[b]
            with torch.no_grad(): pA = torch.softmax(A(*feats.get(r)), 1)
            lm = model(*feats.get(r)); pM = torch.softmax(lm, 1)
            ce = F.cross_entropy(lm, y, label_smoothing=0.1)
            ys = F.one_hot(y, 2).float() * 0.9 + 0.05                      # same smoothing for the fused term
            fuse = -(ys * torch.log(0.5 * (pA + pM) + 1e-8)).sum(1).mean()
            loss = alpha * ce + (1 - alpha) * fuse
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        pm = V.predict(model, feats, va.feat_row.values); auc = roc_auc_score(va.binary_label, 0.5 * (pa_va + pm))
        if auc > best: best, state = auc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); return model, best


auc = lambda y, p: 100 * roc_auc_score(y, p)
res = {}
for alpha in (1.0, 0.5, 0.0):
    per_f, per_m = {s: [] for s in UNSEEN}, {s: [] for s in UNSEEN}; vals = []
    for i, seed in enumerate((42, 43, 44)):
        model, v = train(seed, alpha); vals.append(100 * v)
        for s in UNSEEN:
            S = D.sets[s]; r = S["rows"][S["keep"]]; y = S["y"][S["keep"]]
            pm = V.predict(model, CF[FEAT[s]], r); pa = D.probs("CLIP probe", s)[i][S["keep"]]
            per_f[s].append(auc(y, 0.5 * (pa + pm))); per_m[s].append(auc(y, pm))
    f = {s: float(np.mean(v)) for s, v in per_f.items()}; m = {s: float(np.mean(v)) for s, v in per_m.items()}
    res[f"alpha {alpha}"] = dict(fused=f, member=m, val_fused=float(np.mean(vals)))
    print(f"alpha {alpha}: fused mean9 {np.mean(list(f.values())):.2f} min9 {min(f.values()):.2f} | member alone {np.mean(list(m.values())):.2f} | source val (fused) {np.mean(vals):.2f}  "
          + " ".join(f"{s[:4]}{f[s]:.1f}" for s in UNSEEN), flush=True)
json.dump(res, open(RD.L / "review/explore_partner.json", "w"), indent=1)
