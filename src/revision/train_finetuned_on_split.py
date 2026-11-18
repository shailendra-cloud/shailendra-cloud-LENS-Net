#!/usr/bin/env python3
"""C1 / C5 (PREREG_revision.md): the shared fine-tuning recipe of train_video_baseline.py on (a) a split file
(C1: review/splits/C1_*.csv) or (b) the audited pooled split P1 ("A1"); optional LoRA (C5) on VideoMAE-B.
Writes runs_review/<name>/{predictions_val,test,external}.csv and summary.json."""
import argparse, json, math, sys, time, os
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import train_video_baseline as TVB
D, FrameStore, ROOT = TVB.D, TVB.FrameStore, TVB.ROOT
OUT = ROOT / "workspace/runs_review"


class LoRA(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__(); self.base = base; self.s = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r))
    @property
    def weight(self): return self.base.weight + self.s * (self.B @ self.A)      # VideoMAE calls F.linear(x, query.weight, q_bias)
    @property
    def bias(self): return self.base.bias
    @property
    def in_features(self): return self.base.in_features
    @property
    def out_features(self): return self.base.out_features
    def forward(self, x): return F.linear(x, self.weight, self.base.bias)


def add_lora(model):
    for p in model.parameters(): p.requires_grad_(False)
    n = 0
    for mod in model.modules():
        if hasattr(mod, "query") and hasattr(mod, "value") and isinstance(mod.query, nn.Linear):
            mod.query = LoRA(mod.query); mod.value = LoRA(mod.value); n += 1
    for name, p in model.named_parameters():
        if name.startswith(model.head_names): p.requires_grad_(True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["videomae", "r2plus1d_18"], required=True)
    ap.add_argument("--split", required=True, help="C1_<corpus>_<R|S> or A1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr-body", type=float, default=5e-5)
    ap.add_argument("--lora", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda")
    name = f"{a.split}_{a.model}{'_lora' if a.lora else ''}{'' if a.lr_body == 5e-5 else f'_lr{a.lr_body:g}'}_seed{a.seed}"
    od = OUT / name; od.mkdir(parents=True, exist_ok=True); lock = od / "RUNNING.lock"
    if (od / "summary.json").exists() or lock.exists(): print(name, "done or running; skip", flush=True); return
    lock.write_text(str(os.getpid()))
    man = TVB.frames("A1")
    if a.split == "A1": df = man
    else:
        sp = pd.read_csv(ROOT / f"workspace/review/splits/{a.split}.csv", usecols=["segment_id", "split"])
        df = man.drop(columns=["split"]).merge(sp, on="segment_id", how="inner", validate="one_to_one")
    tr = df[df.split == "train"].reset_index(drop=True); va = df[df.split == "val"].reset_index(drop=True); te = df[df.split == "test"].reset_index(drop=True)
    store = FrameStore(stores=("supp",), use_v1=True); ds = D.WindowDataset(tr, store, True); g = torch.Generator().manual_seed(a.seed)
    if a.split == "A1":
        sampler = WeightedRandomSampler(torch.tensor(D.strategy_weights(tr, "hybrid", cap=0.05)), len(tr), replacement=True, generator=g)
        dl = DataLoader(ds, batch_size=4, sampler=sampler, num_workers=8, pin_memory=True, worker_init_fn=D.worker_init, drop_last=True)
    else:
        dl = DataLoader(ds, batch_size=4, shuffle=True, generator=g, num_workers=8, pin_memory=True, worker_init_fn=D.worker_init, drop_last=True)
    model = TVB.Net(a.model).to(dev); n_lora = add_lora(model) if a.lora else 0
    if a.lora: model.to(dev)
    head = [p for n, p in model.named_parameters() if n.startswith(model.head_names) and p.requires_grad]
    body = [p for n, p in model.named_parameters() if not n.startswith(model.head_names) and p.requires_grad]
    n_train = sum(p.numel() for p in head + body)
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr_body}, {"params": head, "lr": 10 * a.lr_body}], weight_decay=0.05)
    EPOCHS, ACC = 6, 4; steps = len(dl) // ACC; total, warm = steps * EPOCHS, steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.cuda.amp.GradScaler(); log = open(od / "log.jsonl", "w"); t0 = time.time(); best, best_ep = -1, -1
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x.to(dev, non_blocking=True)).float(), y.to(dev), label_smoothing=0.1) / ACC
            scaler.scale(loss).backward()
            if (i + 1) % ACC == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        pv = TVB.predict(model, va, store, dev); auc = D.binary_metrics(va.binary_label, pv, 0.5)["auc"]
        log.write(json.dumps(dict(epoch=ep + 1, val_auc=auc, minutes=(time.time() - t0) / 60)) + "\n"); log.flush()
        if auc > best: best, best_ep = auc, ep + 1; torch.save({k: v for k, v in model.state_dict().items()}, od / "best.pt")
    model.load_state_dict(torch.load(od / "best.pt", map_location=dev))
    pv = TVB.predict(model, va, store, dev); thr = D.select_threshold(va.binary_label, pv)
    va.assign(p_violent=pv).to_csv(od / "predictions_val.csv", index=False)
    te.assign(p_violent=TVB.predict(model, te, store, dev)).to_csv(od / "predictions_test.csv", index=False)
    ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
    pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
    pool.assign(p_violent=TVB.predict(model, pool, FrameStore(stores=("external",), use_v1=False), dev, decode="seq")).to_csv(od / "predictions_external.csv", index=False)
    json.dump(dict(model=a.model, split=a.split, seed=a.seed, lora=a.lora, lora_layers=n_lora, trained_params=int(n_train), lr_body=a.lr_body,
                   best_epoch=best_ep, best_val_auc=best, threshold=thr, minutes=(time.time() - t0) / 60, n_train=len(tr), n_val=len(va), n_test=len(te)),
              open(od / "summary.json", "w"), indent=2, default=float)
    lock.unlink(missing_ok=True); os.remove(od / "best.pt") if a.split != "A1" else None
    print(name, "done", round(best, 2), round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
