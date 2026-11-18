#!/usr/bin/env python3
"""C1 (frozen models), C2 (LAION encoder), C3 (DG penalties) of PREREG_revision.md: the LENS-Net training loop of
lensnet.py (AdamW 3e-4, wd 0.05 except gates/anchors, 30 epochs, batch 128, warm-up + cosine, label smoothing 0.1,
clip 1, hybrid sampler cap 0.05, best validation AUC) with a choice of cross-corpus penalty:
  vrex:<beta> (beta=1 is LENS-Net), irm (IRMv1, weight 1 after epoch 1), gdro (GroupDRO, eta 0.01), none.
Scores the test split (if any), XD/UCF and the fresh/conf/cfights/conf2 windows. Output: runs_review_heads/<name>/."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import lensnet_setup as L, lensnet as V
D = L.D
OUT = L.ROOT / "workspace/runs_review_heads"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["LP", "V2b", "LENS-M", "MP"], required=True)
    ap.add_argument("--penalty", default="vrex:1")
    ap.add_argument("--split", default="audited")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=30.0, help="E2: cosine scale s (published 30)")
    ap.add_argument("--anchor-w", type=float, default=0.5, help="E2: lambda_A (published 0.5)")
    a = ap.parse_args()
    L.SCALE, L.ANCHOR_W = a.scale, a.anchor_w
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(6)
    name = (f"{os.environ.get('LENS_BACKBONE', 'L14')}_{a.split.replace(':', '_')}_{a.config}_{a.penalty.replace(':', '')}"
            + ("" if a.scale == 30.0 else f"_s{a.scale:g}") + ("" if a.anchor_w == 0.5 else f"_la{a.anchor_w:g}") + f"_seed{a.seed}")
    od = OUT / name; od.mkdir(parents=True, exist_ok=True)
    if (od / "summary.json").exists(): print(name, "exists"); return
    df, held = L.load_split(a.split)
    tr = df[df.split == "train"].reset_index(drop=True); va = df[df.split == "val"].reset_index(drop=True); te = df[df.split == "test"].reset_index(drop=True)
    feats = V.Compact("bench"); model = V.V2(a.config, L.text_anchors())
    w = D.strategy_weights(tr, "hybrid", cap=0.05)
    rows = torch.as_tensor(tr.feat_row.values); ytr = torch.tensor(tr.binary_label.values); dsid = torch.tensor(pd.factorize(tr.dataset)[0]); K = int(dsid.max()) + 1
    g = torch.Generator().manual_seed(a.seed); bs = 128; steps = len(tr) // bs; epochs = 30
    nowd = {"w", "g", "gm"}
    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and n not in nowd], "weight_decay": 0.05},
              {"params": [p for n, p in model.named_parameters() if p.requires_grad and n in nowd], "weight_decay": 0.0}]
    opt = torch.optim.AdamW([x for x in groups if x["params"]], lr=3e-4); total, warm = steps * epochs, steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1))))
    kind, arg = (a.penalty.split(":") + ["0"])[:2]; q = torch.ones(K) / K
    best, best_state = -1, None
    for ep in range(epochs):
        model.train(); idx = torch.multinomial(torch.tensor(w), steps * bs, replacement=True, generator=g)
        for s in range(steps):
            b = idx[s * bs:(s + 1) * bs]; r = rows[b]; d = dsid[b]
            if kind == "irm":
                scale = torch.ones(1, requires_grad=True); logits = model(*feats.get(r))
                ce = F.cross_entropy(logits * scale, ytr[b], label_smoothing=0.1, reduction="none")
                loss = ce.mean() + model.anchor_penalty()
                if ep >= 1:
                    pen = sum(torch.autograd.grad(ce[d == k].mean(), scale, create_graph=True)[0].pow(2).sum() for k in d.unique())
                    loss = loss + 1.0 * pen
            else:
                ce = F.cross_entropy(model(*feats.get(r)), ytr[b], label_smoothing=0.1, reduction="none")
                per = torch.stack([ce[d == k].mean() for k in d.unique()])
                if kind == "gdro":
                    with torch.no_grad():
                        for j, k in enumerate(d.unique()): q[k] = q[k] * torch.exp(0.01 * per[j])
                        q /= q.sum()
                    loss = (q[d.unique()] / q[d.unique()].sum() * per).sum() + model.anchor_penalty()
                else:
                    loss = ce.mean() + model.anchor_penalty()
                    beta = float(arg) if kind == "vrex" else 0.0
                    if beta > 0 and len(per) > 1: loss = loss + beta * per.var(unbiased=False)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        auc = D.binary_metrics(va.binary_label, V.predict(model, feats, va.feat_row.values), 0.5)["auc"]
        if auc > best: best, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    if len(te): te.assign(p_violent=V.predict(model, feats, te.feat_row.values)).to_csv(od / "predictions_test.csv", index=False)
    for nm in ("external", "fresh", "conf", "cfights", "conf2"):
        cf = V.Compact(nm); pd.DataFrame({"p_violent": V.predict(model, cf, np.arange(len(cf.index)))}).to_csv(od / f"predictions_{nm}.csv", index=False)
    json.dump(dict(config=a.config, penalty=a.penalty, split=a.split, seed=a.seed, scale=a.scale, anchor_w=a.anchor_w, backbone=os.environ.get("LENS_BACKBONE", "L14"), best_val_auc=best,
                   gate=float(model.g) if getattr(model, "g", None) is not None else None), open(od / "summary.json", "w"), indent=2)
    print(name, "val", round(best, 2), flush=True)


if __name__ == "__main__":
    main()
