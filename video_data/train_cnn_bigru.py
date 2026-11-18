#!/usr/bin/env python3
"""STEP 3: train the binary VIOLENCE / NON_VIOLENCE detector on the source manifest.

Shared encoder: cnn_bigru_model.MASTNet (STEP 1, verified identical to v1) with a 2-way head.
Kept from v1: AdamW with backbone/head learning rates, linear warm-up + cosine schedule, AMP, gradient
clipping, label smoothing, EMA (evaluated each epoch), early stopping, v1 augmentation, blur augmentation.
Changed: binary labels; source-aware weighted sampling (--strategy); NO Mixup; NO domain-adversarial
branch; model selection on validation binary AUC; decision threshold chosen on validation only.
The test split is scored once at the end with the validation threshold and never influences any choice.
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video_data import cnn_bigru_model as mm
from video_data import window_data as D
from video_data.frame_store import FrameStore


class ModelEMA:  # v1 ModelEMA
    def __init__(self, model, decay=0.999, warmup=True):
        self.ema = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.buf = {k: v.detach().clone() for k, v in model.state_dict().items() if not v.dtype.is_floating_point}
        self.decay, self.warmup, self.step = decay, warmup, 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step)) if self.warmup else self.decay
        for k, v in model.state_dict().items():
            if k in self.ema: self.ema[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else: self.buf[k] = v.detach().clone()

    def copy_to(self, model):
        model.load_state_dict({**{k: v.clone() for k, v in self.buf.items()}, **{k: v.clone() for k, v in self.ema.items()}}, strict=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["unit", "dataset", "hybrid"], required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data_tables/manifests/source_windows.csv"))
    ap.add_argument("--out-root", default=str(ROOT / "source_benchmark/runs"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--blur-aug-prob", type=float, default=0.5)
    ap.add_argument("--hybrid-cap", type=float, default=0.05)
    ap.add_argument("--max-train", type=int, default=0, help="smoke test only")
    ap.add_argument("--max-val", type=int, default=0, help="smoke test only")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    out = Path(a.out_root) / (f"binary_{a.strategy}_seed{a.seed}" + ("_smoke" if a.max_train else ""))
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "config.json", "w"), indent=2)
    log = open(out / "log.jsonl", "a")
    def emit(**kw):
        print(json.dumps(kw), flush=True); log.write(json.dumps(kw) + "\n"); log.flush()

    df = pd.read_csv(a.manifest)
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    if a.max_train: tr = tr.sample(min(a.max_train, len(tr)), random_state=a.seed).reset_index(drop=True)
    if a.max_val:
        va = va.sample(min(a.max_val, len(va)), random_state=a.seed).reset_index(drop=True)
        te = te.sample(min(a.max_val, len(te)), random_state=a.seed).reset_index(drop=True)
    store = FrameStore(stores=("supp",), use_v1=True)
    weights = D.strategy_weights(tr, a.strategy, cap=a.hybrid_cap)
    emit(event="data", train=len(tr), val=len(va), test=len(te),
         train_balance={int(k): int(v) for k, v in tr.binary_label.value_counts().items()},
         strategy=a.strategy, effective_mass=D.effective_mass(tr, weights))

    g = torch.Generator(); g.manual_seed(a.seed)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(tr), replacement=True, generator=g)
    common = dict(num_workers=a.workers, pin_memory=True, worker_init_fn=D.worker_init, persistent_workers=a.workers > 0)
    tr_dl = DataLoader(D.WindowDataset(tr, store, True), batch_size=a.batch_size, sampler=sampler, drop_last=True, **common)
    va_dl = DataLoader(D.WindowDataset(va, store, False), batch_size=a.batch_size, shuffle=False, **common)
    te_dl = DataLoader(D.WindowDataset(te, store, False), batch_size=a.batch_size, shuffle=False, **common)

    model = mm.MASTNet(backbone_kind="efficientvit_b0", pretrained=True, num_classes=2,
                       blur_aug_prob=a.blur_aug_prob, grad_checkpoint=True).to(dev)
    bb = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone")]
    new = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr_backbone}, {"params": new, "lr": a.lr_head}], weight_decay=a.weight_decay)
    spe = len(tr_dl); tot, wu = max(spe * a.epochs, 1), max(spe * a.warmup_epochs, 1)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / wu if s < wu else 0.5 * (1 + math.cos(math.pi * min((s - wu) / max(tot - wu, 1), 1.0))))
    scaler = torch.amp.GradScaler("cuda")
    ema = ModelEMA(model, a.ema_decay, warmup=True)
    crit = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    best, bad, best_state, t0 = -1.0, 0, None, time.time()
    for ep in range(a.epochs):
        model.train(); run_loss = n_seen = correct = 0; te0 = time.time()
        for x, y, _ in tr_dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda"):
                lg = model(x); loss = crit(lg.float(), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], a.grad_clip)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step(); ema.update(model)
            run_loss += loss.item() * y.size(0); n_seen += y.size(0); correct += (lg.detach().argmax(1) == y).sum().item()
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        pv = D.predict(model, va_dl, dev, tta=False)
        m = D.binary_metrics(va.binary_label, pv, 0.5)
        per = D.per_group(va, pv, 0.5, "dataset"); mac, worst = D.macro_auc(per)
        emit(event="epoch", epoch=ep + 1, train_loss=run_loss / n_seen, train_acc=100 * correct / n_seen,
             val_auc=m["auc"], val_macro_dataset_auc=mac, val_worst_dataset_auc=worst,
             val_f1_at_0_5=m["f1"], val_per_dataset_auc={k: v["auc"] for k, v in per.items()},
             minutes=(time.time() - te0) / 60)
        if m["auc"] > best:
            best, bad = m["auc"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state": best_state, "arch": model.arch, "classes": D.CLASSES, "epoch": ep + 1,
                        "val_auc": best, "strategy": a.strategy, "config": vars(a)}, out / "best.pt")
        else:
            bad += 1
        model.load_state_dict(backup)
        if bad >= a.patience:
            emit(event="early_stop", epoch=ep + 1); break

    final = mm.MASTNet(backbone_kind="efficientvit_b0", pretrained=False, num_classes=2, blur_aug_prob=a.blur_aug_prob).to(dev)
    final.load_state_dict(best_state); final.eval()
    pv_val = D.predict(final, va_dl, dev, tta=True)
    thr = D.select_threshold(va.binary_label, pv_val)
    pv_test = D.predict(final, te_dl, dev, tta=True)
    per_val = D.per_group(va, pv_val, thr, "dataset"); per_test = D.per_group(te, pv_test, thr, "dataset")
    summary = dict(strategy=a.strategy, seed=a.seed, best_epoch=torch.load(out / "best.pt", weights_only=False)["epoch"],
                   threshold_selected_on_val=thr, train_minutes=(time.time() - t0) / 60,
                   val=D.binary_metrics(va.binary_label, pv_val, thr), val_per_dataset=per_val,
                   val_macro_dataset_auc=D.macro_auc(per_val)[0], val_worst_dataset_auc=D.macro_auc(per_val)[1],
                   val_within_group_auc=D.within_group_auc(va.binary_label, pv_val, va.component_id),
                   test=D.binary_metrics(te.binary_label, pv_test, thr), test_per_dataset=per_test,
                   test_macro_dataset_auc=D.macro_auc(per_test)[0],
                   test_within_group_auc=D.within_group_auc(te.binary_label, pv_test, te.component_id),
                   effective_mass=D.effective_mass(tr, weights))
    ck = torch.load(out / "best.pt", weights_only=False); ck["threshold"] = thr; torch.save(ck, out / "best.pt")
    va.assign(p_violent=pv_val).to_csv(out / "val_predictions.csv", index=False)
    te.assign(p_violent=pv_test).to_csv(out / "test_predictions.csv", index=False)
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    emit(event="done", val_auc=summary["val"]["auc"], val_macro_dataset_auc=summary["val_macro_dataset_auc"],
         test_auc=summary["test"]["auc"], threshold=thr, out=str(out))


if __name__ == "__main__":
    main()
