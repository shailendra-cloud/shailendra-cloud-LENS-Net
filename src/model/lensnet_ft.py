#!/usr/bin/env python3
"""Track B (PREREG deviation 4): LENS-FT = CLIP blocks 20-23 + ln_post + proj fine-tuned with the V2b head on RWF-2000
official train (std_rwf). Selection on std_rwf val (window AUC); threshold on val videos; official test video accuracy.
Also scores XD/UCF (full CLIP forward from images) to show the in-domain vs transfer trade-off."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import clip_tail as C, lensnet_setup as L, lensnet as V2m, extract_clip_tokens as E
from video_data import window_data as D
from video_data.frame_store import FrameStore

CACHE = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/features/l14/rwf_block20")
PROMPTS = torch.tensor(np.load(L.FEAT / "text_prompts.npy"))


def to_inputs(tok):                                    # (B,16,17,768) -> V2 inputs
    z = F.normalize(tok.float(), dim=-1)
    return z[:, :, 0].mean(1), z @ PROMPTS.to(z.device).t()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    od = L.OUT.parent / "runs_ft" / f"LENS-FT_seed{a.seed}"; od.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(CACHE / "index.csv"); assert np.load(CACHE / "done.npy").all()
    lab = pd.read_csv(L.ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "binary_label", "video_id"])
    idx = idx.drop(columns=[c for c in ["binary_label", "video_id"] if c in idx]).merge(lab, on="segment_id", how="left")
    idx["row"] = np.arange(len(idx))
    tok = np.memmap(CACHE / "tokens.f16", dtype=np.float16, mode="r", shape=(len(idx), 16, 257, 1024))
    tr, va, te = [idx[idx.split == s].reset_index(drop=True) for s in ("train", "val", "test")]
    if a.smoke: tr, va, te = [d.groupby("binary_label", group_keys=False).apply(lambda x: x.sample(min(len(x), n), random_state=0)).reset_index(drop=True) for d, n in ((tr, 32), (va, 24), (te, 24))]
    v = C.load_clip(dev, "fp32"); tail = C.Tail(v).to(dev); tail.ckpt = True
    head = V2m.V2("V2b", L.text_anchors()).to(dev)
    tail_p = list(tail.parameters()); nowd = [head.w, head.g]
    head_p = [p for n, p in head.named_parameters() if p.requires_grad and n not in ("w", "g")]
    opt = torch.optim.AdamW([{"params": tail_p, "lr": 1e-5, "weight_decay": 0.05}, {"params": head_p, "lr": 3e-4, "weight_decay": 0.05},
                             {"params": nowd, "lr": 3e-4, "weight_decay": 0.0}])
    EPOCHS, MB, ACC = (2 if a.smoke else 10), 4, 4
    steps = len(tr) // (MB * ACC); total, warm = steps * EPOCHS, max(steps, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.cuda.amp.GradScaler(); g = torch.Generator().manual_seed(a.seed)

    def forward(rows, train):
        x = torch.from_numpy(np.asarray(tok[np.sort(rows)], dtype=np.float16)).to(dev)           # (B,16,257,1024)
        order = np.argsort(np.argsort(rows)); x = x[torch.as_tensor(order, device=dev)]
        B = x.shape[0]
        with torch.autocast("cuda", dtype=torch.float16):
            t = tail(x.flatten(0, 1).float()).view(B, 16, 17, 768)
        return head(*to_inputs(t))

    @torch.no_grad()
    def predict(df):
        tail.eval(); head.eval(); out = []
        for i in range(0, len(df), 8):
            out.append(torch.softmax(forward(df.row.values[i:i + 8], False).float(), 1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    log = open(od / "log.jsonl", "w"); t0 = time.time(); best, best_ep = -1, -1
    ytr = torch.tensor(tr.binary_label.values)
    for ep in range(EPOCHS):
        tail.train(); head.train(); perm = torch.randperm(len(tr), generator=g).numpy(); opt.zero_grad(set_to_none=True)
        for i in range(steps * ACC):
            b = perm[i * MB:(i + 1) * MB]
            loss = (F.cross_entropy(forward(tr.row.values[b], True).float(), ytr[b].to(dev), label_smoothing=0.1) + head.anchor_penalty()) / ACC
            scaler.scale(loss).backward()
            if (i + 1) % ACC == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(tail_p + head_p + nowd, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        pv = predict(va); auc = D.binary_metrics(va.binary_label, pv, 0.5)["auc"]
        log.write(json.dumps(dict(epoch=ep + 1, val_auc=auc, gate=float(head.g), minutes=(time.time() - t0) / 60)) + "\n"); log.flush()
        if best_ep < 0 or auc > best:
            best, best_ep = auc, ep + 1; torch.save({"tail": tail.state_dict(), "head": head.state_dict()}, od / "best.pt")
    ck = torch.load(od / "best.pt", map_location=dev); tail.load_state_dict(ck["tail"]); head.load_state_dict(ck["head"])
    pv, pt = predict(va), predict(te)
    yv, pvv = L.video_level(va, pv); yt, ptv = L.video_level(te, pt); vthr = D.select_threshold(yv, pvv)
    res = dict(seed=a.seed, best_epoch=best_ep, gate=float(head.g), gpu=torch.cuda.get_device_name(0),
               val_window=D.binary_metrics(va.binary_label, pv, 0.5), test_window_auc=D.binary_metrics(te.binary_label, pt, 0.5)["auc"],
               test_video=dict(threshold_selected_on_val_videos=vthr, overall=D.binary_metrics(yt, ptv, vthr)))
    va.assign(p_violent=pv).to_csv(od / "predictions_val.csv", index=False); te.assign(p_violent=pt).to_csv(od / "predictions_test.csv", index=False)
    # transfer: full CLIP forward from images (frozen head + fine-tuned tail)
    ext = pd.read_csv(L.ROOT / "data_tables/manifests/external_windows.csv")
    pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
    if a.smoke: pool = pool.groupby("benchmark", group_keys=False).head(16).reset_index(drop=True)
    dl = DataLoader(E.Windows(pool[["path", "start_frame", "end_frame"]], FrameStore(stores=("external",), use_v1=False), "seq"),
                    batch_size=4, num_workers=8, pin_memory=True)
    vh = C.load_clip(dev, "fp16"); tail.eval(); head.eval(); pe = np.zeros(len(pool))
    with torch.no_grad():
        for x, ii in dl:
            B = x.shape[0]
            with torch.autocast("cuda", dtype=torch.float16):
                t = tail(C.head(vh, x.to(dev).half().flatten(0, 1)).float()).view(B, 16, 17, 768)
            pe[ii.numpy()] = torch.softmax(head(*to_inputs(t)).float(), 1)[:, 1].cpu().numpy()
    keep = pool.excluded_precaution == 0
    res["external_auc"] = {b: D.binary_metrics(gg.binary_label, pe[gg.index.values], 0.5)["auc"] for b, gg in pool[keep].groupby("benchmark")}
    pool.assign(p_violent=pe).to_csv(od / "predictions_external.csv", index=False)
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(od / "summary.json", "w"), indent=2, default=float)
    print(od.name, "test video acc", round(res["test_video"]["overall"]["accuracy"], 2), "ext", {k: round(x, 2) for k, x in res["external_auc"].items()}, flush=True)


if __name__ == "__main__":
    main()
