#!/usr/bin/env python3
"""PREREG_journal Sec. 6A: official VadCLIP (github.com/nwpu-zxr/VadCLIP, Apache-2.0) under our protocol.
Model CLIPVAD and losses (CLAS2 + CLASM + 1e-4 x text orthogonality) as in the official xd_train.py; only the input
length (16 frames = one window = one MIL bag), the attention window (4) and the binary label map are adapted.
Input: frozen per-frame CLIP image embeddings (cls_frames.npy). Backbone B16 (native) or L14 (matched to LENS-Net).
Window score = mean of the top-2 frame probabilities of the coarse branch (k = int(16/16) + 1, as in their MIL loss).
Usage: vadclip_run.py --bb B16|L14 --split audited|std:std_rwf --seed 42 --lr 1e-5 [--allow-test]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); VC = ROOT / "raw_data/journal/ext/VadCLIP/src"
sys.path.insert(0, str(VC)); sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import model as VM
from utils.tools import get_batch_label
import lensnet_setup as L
from video_data import window_data as D
FEATS = {"B16": ROOT / "workspace/features/b16", "L14": ROOT / "workspace/features/l14"}
CLIPNAME = {"B16": "ViT-B/16", "L14": "ViT-L/14"}; EMB = {"B16": 512, "L14": 768}
LABEL_MAP = {"Normal": "normal", "Violence": "violence"}; PROMPT_TEXT = ["normal", "violence"]


def CLASM(logits, labels, lengths, device):                      # verbatim from official xd_train.py
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True); labels = labels.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)


def CLAS2(logits, labels, lengths, device):                      # verbatim from official xd_train.py
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0]); labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        instance_logits = torch.cat((instance_logits, torch.mean(tmp).view(1)))
    return F.binary_cross_entropy(instance_logits, labels)


def score(model, X, dev, bs=512):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.from_numpy(X[i:i + bs].astype(np.float32)).to(dev); ln = torch.full((len(x),), 16, device=dev)
            _, l1, _ = model(x, None, PROMPT_TEXT, ln)
            out.append(torch.sigmoid(l1.squeeze(-1)).topk(2, dim=1).values.mean(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bb", choices=["B16", "L14"], required=True); ap.add_argument("--split", default="audited")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--allow-test", action="store_true"); a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda")
    name = f"VadCLIP-{a.bb}_{a.split.replace(':', '_')}_lr{a.lr:g}_seed{a.seed}"
    od = ROOT / "workspace/runs_vadclip" / name; od.mkdir(parents=True, exist_ok=True)
    orig_load = VM.clip.load
    VM.clip.load = lambda n, device: orig_load(CLIPNAME[a.bb], device, download_root=str(ROOT / "raw_data/clip_weights"))
    model = VM.CLIPVAD(num_class=2, embed_dim=EMB[a.bb], visual_length=16, visual_width=EMB[a.bb], visual_head=1, visual_layers=1,
                       attn_window=4, prompt_prefix=10, prompt_postfix=10, device=dev).to(dev)
    Xb = np.load(FEATS[a.bb] / "bench/cls_frames.npy", mmap_mode="r")
    df, held = L.load_split(a.split)
    tr = df[df.split == "train"].reset_index(drop=True); va = df[df.split == "val"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True) if a.allow_test else df.iloc[:0]
    Xtr = np.asarray(Xb[tr.feat_row.values]); Xva = np.asarray(Xb[va.feat_row.values])
    w = D.strategy_weights(tr, "hybrid", cap=0.05); g = torch.Generator().manual_seed(a.seed)
    bs, epochs = 96, 10; steps = len(tr) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, [3, 6, 10], 0.1)             # official defaults (milestones, rate 0.1)
    texts = np.where(tr.binary_label.values == 1, "Violence", "Normal")
    best, best_ep, t0 = -1, -1, time.time(); log = open(od / "log.jsonl", "w")
    for ep in range(epochs):
        model.train(); idx = torch.multinomial(torch.tensor(w), steps * bs, replacement=True, generator=g).numpy()
        for s in range(steps):
            b = idx[s * bs:(s + 1) * bs]
            x = torch.from_numpy(Xtr[b].astype(np.float32)).to(dev); ln = torch.full((len(b),), 16, device=dev)
            lab = get_batch_label(list(texts[b]), PROMPT_TEXT, LABEL_MAP).to(dev)
            tf, l1, l2 = model(x, None, PROMPT_TEXT, ln)
            loss3 = torch.zeros(1).to(dev); tn = tf[0] / tf[0].norm(dim=-1, keepdim=True)
            for j in range(1, tf.shape[0]): loss3 += torch.abs(tn @ (tf[j] / tf[j].norm(dim=-1, keepdim=True)))
            loss3 = loss3 / 6                                                     # as in official code
            loss = CLAS2(l1, lab, ln, dev) + CLASM(l2, lab, ln, dev) + loss3 * 1e-4
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        auc = D.binary_metrics(va.binary_label, score(model, Xva, dev), 0.5)["auc"]
        log.write(json.dumps(dict(epoch=ep + 1, val_auc=auc, minutes=(time.time() - t0) / 60)) + "\n"); log.flush()
        if auc > best: best, best_ep = auc, ep + 1; torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("clipmodel.")}, od / "best.pt")  # frozen CLIP not saved
    model.load_state_dict(torch.load(od / "best.pt", weights_only=True), strict=False)
    pv = score(model, Xva, dev); thr = D.select_threshold(va.binary_label, pv)
    res = dict(model="VadCLIP", bb=a.bb, split=a.split, seed=a.seed, lr=a.lr, best_epoch=best_ep, val_auc=best, threshold_selected_on_val=thr,
               minutes=(time.time() - t0) / 60, n_train=len(tr), n_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    va.assign(p_violent=pv).to_csv(od / "predictions_val.csv", index=False)
    if a.allow_test:
        if len(te):
            pt = score(model, np.asarray(Xb[te.feat_row.values]), dev); res["test_auc"] = D.binary_metrics(te.binary_label, pt, thr)["auc"]
            te.assign(p_violent=pt).to_csv(od / "predictions_test.csv", index=False)
        for setname in ("external", "fresh", "conf"):
            X = np.load(FEATS[a.bb] / setname / "cls_frames.npy", mmap_mode="r")
            pd.DataFrame({"p_violent": score(model, np.asarray(X), dev)}).to_csv(od / f"predictions_{setname}.csv", index=False)
    json.dump(res, open(od / "summary.json", "w"), indent=2, default=float); print(name, json.dumps({k: res[k] for k in ("val_auc", "best_epoch") if k in res}), flush=True)


if __name__ == "__main__":
    main()
