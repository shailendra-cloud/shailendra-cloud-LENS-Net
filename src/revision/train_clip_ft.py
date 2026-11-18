#!/usr/bin/env python3
"""E1 (PREREG_revision.md, Part E): CLIP-FT = the anchored CLIP classifier (CLIP probe head) on OpenAI CLIP ViT-L/14 whose
blocks 20-23, ln_post and proj are fine-tuned (lr 1e-5; head lr 3e-4), P1 audited split, frames decoded on the fly.
5 epochs, micro-batch 4 x 4 accumulation, hybrid sampler (cap 0.05), label smoothing 0.1, selection by val window AUC.
Writes runs_review/CLIP-FT_audited_seed<s>/{predictions_val,test,external,fresh,conf,cfights,conf2}.csv, summary.json."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import clip_tail as C, lensnet_setup as L, lensnet as V2m, extract_clip_tokens as E
D = L.D
PROMPTS = torch.tensor(np.load(L.FEAT / "text_prompts.npy"))
OUT = L.ROOT / "workspace/runs_review"


def to_inputs(tok):                                    # (B,16,17,768) -> (mean CLS, prompt similarities) as in the compact features
    z = F.normalize(tok.float(), dim=-1)
    return z[:, :, 0].mean(1), z @ PROMPTS.to(z.device).t()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda")
    od = OUT / f"CLIP-FT_audited{'_smoke' if a.smoke else ''}_seed{a.seed}"; od.mkdir(parents=True, exist_ok=True)
    if (od / "summary.json").exists(): print(od.name, "exists"); return
    base, _ = L.load_split("audited"); bdf, bstore, bdec = E.load_set("bench")
    assert (bdf.segment_id.values == base.segment_id.values).all()
    bench = E.Windows(bdf[["path", "start_frame", "end_frame", "v1_cache_row"]], bstore, bdec)
    tr, va, te = [base[base.split == s] for s in ("train", "val", "test")]
    if a.smoke: tr, va, te = [d.groupby("binary_label", group_keys=False).apply(lambda x: x.sample(min(len(x), n), random_state=0)) for d, n in ((tr, 32), (va, 24), (te, 24))]
    vh = C.load_clip(dev, "fp16")
    for p in vh.parameters(): p.requires_grad_(False)
    tail = C.Tail(C.load_clip(dev, "fp32")).to(dev); tail.ckpt = True
    head = V2m.V2("LP", L.text_anchors()).to(dev)
    tail_p = list(tail.parameters()); head_p = [p for n, p in head.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": tail_p, "lr": 1e-5, "weight_decay": 0.05}, {"params": head_p, "lr": 3e-4, "weight_decay": 0.0}])
    EPOCHS, MB, ACC = (1 if a.smoke else 5), 4, 4
    steps = len(tr) // (MB * ACC); total, warm = steps * EPOCHS, max(steps, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.cuda.amp.GradScaler(); g = torch.Generator().manual_seed(a.seed)
    w = D.strategy_weights(tr.reset_index(drop=True), "hybrid", cap=0.05); y_all = torch.tensor(base.binary_label.values)

    def logits(x, train):
        B = x.shape[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            h = C.head(vh, x.to(dev, non_blocking=True).half().flatten(0, 1))
        with torch.autocast("cuda", dtype=torch.float16):
            t = tail(h.float()).view(B, 16, 17, 768)
        return head(*to_inputs(t))

    @torch.no_grad()
    def predict(ds):
        tail.eval(); head.eval(); out = np.zeros(len(ds))
        for x, ii in DataLoader(ds, batch_size=8, num_workers=12, pin_memory=True):
            out[ii.numpy()] = torch.softmax(logits(x, False).float(), 1)[:, 1].cpu().numpy()
        return out

    def sub(ds, idx):                                   # Subset whose __getitem__ returns the position inside the subset
        class S(torch.utils.data.Dataset):
            def __len__(s): return len(idx)
            def __getitem__(s, k): x, _ = ds[int(idx[k])]; return x, k
        return S()
    log = open(od / "log.jsonl", "w"); t0 = time.time(); best, best_ep = -1, -1
    for ep in range(EPOCHS):
        tail.train(); head.train(); opt.zero_grad(set_to_none=True)
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), steps * MB * ACC, replacement=True, generator=g)
        order = tr.index.values[np.fromiter(sampler, dtype=np.int64)]
        dl = DataLoader(sub(bench, order), batch_size=MB, num_workers=12, pin_memory=True, drop_last=True)
        for i, (x, k) in enumerate(dl):
            yb = y_all[order[k.numpy()]].to(dev)
            loss = (F.cross_entropy(logits(x, True).float(), yb, label_smoothing=0.1) + head.anchor_penalty()) / ACC
            scaler.scale(loss).backward()
            if (i + 1) % ACC == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(tail_p + head_p, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        pv = predict(sub(bench, va.index.values)); auc = D.binary_metrics(va.binary_label, pv, 0.5)["auc"]
        log.write(json.dumps(dict(epoch=ep + 1, val_auc=auc, minutes=(time.time() - t0) / 60)) + "\n"); log.flush()
        if auc > best: best, best_ep = auc, ep + 1; torch.save({"tail": tail.state_dict(), "head": head.state_dict()}, od / "best.pt")
    ck = torch.load(od / "best.pt", map_location=dev); tail.load_state_dict(ck["tail"]); head.load_state_dict(ck["head"])
    va.assign(p_violent=predict(sub(bench, va.index.values))).to_csv(od / "predictions_val.csv", index=False)
    te.assign(p_violent=predict(sub(bench, te.index.values))).to_csv(od / "predictions_test.csv", index=False)
    for nm in ("external", "fresh", "conf", "cfights", "conf2"):
        df, st, dec = E.load_set(nm); n_c = len(pd.read_csv(L.FEAT / nm / "index.csv")); assert len(df) == n_c, (nm, len(df), n_c)
        if a.smoke: df = df.head(16)
        ds = E.Windows(df[[c for c in ("path", "start_frame", "end_frame", "v1_cache_row") if c in df]], st, dec)
        pd.DataFrame({"p_violent": predict(ds)}).to_csv(od / f"predictions_{nm}.csv", index=False); print("scored", nm, len(df), flush=True)
    json.dump(dict(seed=a.seed, best_epoch=best_ep, best_val_auc=best, minutes=(time.time() - t0) / 60,
                   trained_params=sum(p.numel() for p in tail_p + head_p)), open(od / "summary.json", "w"), indent=2)
    print(od.name, "val", round(best, 2), "epoch", best_ep, flush=True)


if __name__ == "__main__":
    main()
