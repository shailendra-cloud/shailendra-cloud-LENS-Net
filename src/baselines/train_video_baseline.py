#!/usr/bin/env python3
"""Track A (PREREG deviation 4): full fine-tuning of SOTA-class video backbones under identical conditions.
  --model videomae | swin3d_t        --protocol A1 (audited pooled) | A2 (RWF-2000 official train, single source)
Writes predictions for val/test, XD/UCF external and (A2) transfer targets. Metrics are computed in analysis."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT))
from video_data import window_data as D
from video_data.frame_store import FrameStore
OUT = ROOT / "workspace/runs_baselines"


class Net(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.kind = kind
        if kind == "videomae":
            from transformers import VideoMAEForVideoClassification
            self.m = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            self.m.classifier = nn.Linear(self.m.classifier.in_features, 2)
            self.head_names = ("m.classifier",)
        elif kind == "r2plus1d_18":
            import torchvision
            self.m = torchvision.models.video.r2plus1d_18(weights=torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1)
            self.m.fc = nn.Linear(self.m.fc.in_features, 2)
            self.head_names = ("m.fc",)
            self.register_buffer("im_m", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
            self.register_buffer("im_s", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
            self.register_buffer("k_m", torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1))
            self.register_buffer("k_s", torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1))
        elif kind == "xclip":                              # PREREG_journal 6C: X-CLIP B/32 (8 frames), linear head on the video embedding
            from transformers import XCLIPModel
            self.m = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
            self.head = nn.Linear(self.m.config.projection_dim, 2)
            self.head_names = ("head",)
            self.register_buffer("im_m", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
            self.register_buffer("im_s", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
            self.register_buffer("c_m", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1))
            self.register_buffer("c_s", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1))
        else:
            import torchvision
            self.m = torchvision.models.video.swin3d_t(weights=torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1)
            self.m.head = nn.Linear(self.m.head.in_features, 2)
            self.head_names = ("m.head",)

    def forward(self, x):                                   # x: (B,T,3,H,W), ImageNet-normalized
        if self.kind == "videomae":
            return self.m(pixel_values=x).logits
        if self.kind == "xclip":
            x = ((x[:, ::2] * self.im_s + self.im_m) - self.c_m) / self.c_s     # every 2nd of 16 frames, CLIP statistics
            B, T = x.shape[:2]                                             # = XCLIPModel.get_video_features, with tuple outputs
            v = self.m.visual_projection(self.m.vision_model(pixel_values=x.flatten(0, 1), return_dict=True).pooler_output)
            mo = self.m.mit(v.view(B, T, -1), return_dict=False)
            return self.head(mo[1])
        if self.kind == "r2plus1d_18":
            B, T = x.shape[:2]
            x = x * self.im_s + self.im_m
            x = F.interpolate(x.flatten(0, 1), size=(112, 112), mode="bilinear", align_corners=False).view(B, T, 3, 112, 112)
            x = (x - self.k_m) / self.k_s
        return self.m(x.permute(0, 2, 1, 3, 4))


def frames(protocol):
    man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", dtype={"v1_split": str})
    if protocol == "A1":
        return man
    sp = pd.read_csv(ROOT / "workspace/splits/std_rwf.csv", usecols=["segment_id", "split"]).rename(columns={"split": "std_split"})
    man = man.merge(sp, on="segment_id", how="left")
    man["split"] = man.std_split.fillna("transfer")
    tr_units = set(man[man.split == "train"].source_unit_id)
    e = pd.read_csv(ROOT / "data_tables/manifests/near_duplicate_edges.csv", usecols=["unit_a", "unit_b"])
    linked = set(e[e.unit_a.isin(tr_units)].unit_b) | set(e[e.unit_b.isin(tr_units)].unit_a)
    man["linked_to_train"] = man.source_unit_id.isin(linked | tr_units).astype(int)
    man.loc[(man.split == "transfer") & (man.dataset == "RWF2000"), "split"] = "unused"   # RWF videos outside the official split
    return man


@torch.no_grad()
def predict(model, df, store, dev, decode="seek", bs=16):
    dl = DataLoader(D.WindowDataset(df, store, False, decode=decode), batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
    return D.predict(model, dl, dev, tta=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["videomae", "swin3d_t", "r2plus1d_18", "xclip"], required=True)
    ap.add_argument("--protocol", choices=["A1", "A2"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=0, help="smoke test only")
    ap.add_argument("--lr-body", type=float, default=5e-5, help="PREREG_journal 6D; head lr = 10x; default = shared recipe")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    tag = "" if a.lr_body == 5e-5 else f"_lr{a.lr_body:g}"
    od = OUT / f"{a.protocol}_{a.model}{tag}_seed{a.seed}"; od.mkdir(parents=True, exist_ok=True)
    lock = od / "RUNNING.lock"                             # never train the same run twice (two GPU queues share jobs)
    if (od / "summary.json").exists() or (lock.exists() and not a.max_train):
        print(od.name, "already done or running elsewhere; skipping", flush=True); return
    if not a.max_train: lock.write_text(str(__import__("os").getpid()))
    df = frames(a.protocol)
    tr = df[df.split == "train"].reset_index(drop=True); va = df[df.split == "val"].reset_index(drop=True)
    if a.max_train: tr = tr.sample(a.max_train, random_state=0).reset_index(drop=True); va = va.sample(min(64, len(va)), random_state=0).reset_index(drop=True)
    store = FrameStore(stores=("supp",), use_v1=True)
    ds = D.WindowDataset(tr, store, True)
    g = torch.Generator().manual_seed(a.seed)
    if a.protocol == "A1":
        sampler = WeightedRandomSampler(torch.tensor(D.strategy_weights(tr, "hybrid", cap=0.05)), len(tr), replacement=True, generator=g)
        dl = DataLoader(ds, batch_size=4, sampler=sampler, num_workers=8, pin_memory=True, worker_init_fn=D.worker_init, drop_last=True)
    else:
        dl = DataLoader(ds, batch_size=4, shuffle=True, generator=g, num_workers=8, pin_memory=True, worker_init_fn=D.worker_init, drop_last=True)
    model = Net(a.model).to(dev)
    head = [p for n, p in model.named_parameters() if n.startswith(model.head_names)]
    body = [p for n, p in model.named_parameters() if not n.startswith(model.head_names)]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr_body}, {"params": head, "lr": 10 * a.lr_body}], weight_decay=0.05)
    EPOCHS, ACC = 6, 4
    steps = len(dl) // ACC; total, warm = steps * EPOCHS, steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.cuda.amp.GradScaler()
    log = open(od / "log.jsonl", "w"); t0 = time.time(); best, best_ep = -1, -1
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x.to(dev, non_blocking=True)).float(), y.to(dev), label_smoothing=0.1) / ACC
            scaler.scale(loss).backward()
            if (i + 1) % ACC == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        pv = predict(model, va, store, dev)
        auc = D.binary_metrics(va.binary_label, pv, 0.5)["auc"]
        log.write(json.dumps(dict(epoch=ep + 1, val_auc=auc, minutes=(time.time() - t0) / 60)) + "\n"); log.flush()
        if auc > best:
            best, best_ep = auc, ep + 1
            sd = model.state_dict()
            if a.model == "xclip":                          # disk: drop the unused X-CLIP text branch, store fp16
                sd = {k: v.half() if v.is_floating_point() else v for k, v in sd.items() if not k.startswith(("m.text_model.", "m.text_projection", "m.prompts_generator."))}
            torch.save(sd, od / "best.pt")
    model.load_state_dict(torch.load(od / "best.pt", map_location=dev), strict=(a.model != "xclip"))
    pv = predict(model, va, store, dev); thr = D.select_threshold(va.binary_label, pv)
    va.assign(p_violent=pv).to_csv(od / "predictions_val.csv", index=False)
    te = df[df.split == "test"].reset_index(drop=True)
    tgt = df[df.split == "transfer"].reset_index(drop=True) if a.protocol == "A2" else df.iloc[:0]
    if a.max_train: te, tgt = te.head(64), tgt.head(64)
    te.assign(p_violent=predict(model, te, store, dev)).to_csv(od / "predictions_test.csv", index=False)
    if len(tgt): tgt.assign(p_violent=predict(model, tgt, store, dev)).to_csv(od / "predictions_transfer.csv", index=False)
    ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
    pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
    if a.max_train: pool = pool.groupby("benchmark", group_keys=False).head(32).reset_index(drop=True)
    pool.assign(p_violent=predict(model, pool, FrameStore(stores=("external",), use_v1=False), dev, decode="seq")).to_csv(od / "predictions_external.csv", index=False)
    json.dump(dict(model=a.model, protocol=a.protocol, seed=a.seed, best_epoch=best_ep, best_val_auc=best, threshold_selected_on_val=thr,
                   minutes=(time.time() - t0) / 60, gpu=torch.cuda.get_device_name(0), cuda_visible=str(__import__("os").environ.get("CUDA_VISIBLE_DEVICES")),
                   n_train=len(tr), n_val=len(va), n_test=len(te), n_transfer=len(tgt)), open(od / "summary.json", "w"), indent=2, default=float)
    lock.unlink(missing_ok=True)
    print(od.name, "done", round(best, 2), flush=True)


if __name__ == "__main__":
    main()
