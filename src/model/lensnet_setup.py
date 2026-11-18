#!/usr/bin/env python3
"""LENS-Net: training and scoring on frozen CLIP tokens (PREREG.md Secs 1-5).

Configs: ZS, LP, LENS, -M (no motion), -S (no scene residual), -A (no language anchor).
Splits:  audited                 frozen leakage-controlled train/val/test
         lodo:<D>                source_benchmark/lodo/<D>/ manifest (test = all rows of D)
Test rows are never scored unless --allow-test is given. With --pilot, a LODO run scores only D's frozen-val rows.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))
from video_data import window_data as D

# Journal E2 (PREREG_journal Sec. 2): LENS_BACKBONE=L14 (default) | B16; LENS_FRAMES=16 (default) | 8.
BACKBONE = os.environ.get("LENS_BACKBONE", "L14")
FRAMES = int(os.environ.get("LENS_FRAMES", "16"))
CLIPCFG = {"L14": dict(model="ViT-L-14-quickgelu", tok="ViT-L-14", feat=str(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/features/l14"), dim=768, grid=16, width=1024),
           "B16": dict(model="ViT-B-16-quickgelu", tok="ViT-B-16", feat=str(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/features/b16"), dim=512, grid=14, width=768),
           # revision C2 (PREREG_revision.md): OpenCLIP ViT-L/14 trained on LAION-2B (public data); compact features only
           "LAION": dict(model="ViT-L-14", tok="ViT-L-14", pretrained="laion2b_s32b_b82k", feat=str(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/features/laion"), dim=768, grid=16, width=1024)}[BACKBONE]
FEAT = Path(CLIPCFG["feat"])
OUT = ROOT / ("workspace/runs" + {"L14": "", "B16": "_b16", "LAION": "_laion"}[BACKBONE] + ("" if FRAMES == 16 else f"_f{FRAMES}"))
T, NTOK, DIM = 16, 17, CLIPCFG["dim"]
SCALE, ANCHOR_W = 30.0, 0.5
PROMPT_SETS = {"orig": {
    0: ["people walking calmly", "people talking to each other", "a peaceful everyday scene",
        "people playing a sport without fighting", "a calm crowd of people"],
    1: ["people fighting", "a violent fight", "people punching and kicking each other",
        "a brawl between people", "a physical assault on a person"],
},
    # journal Sec. 6B (PREREG_journal): alternative and broad prompt sets, written before any run
    "para": {0: ["people walking peacefully", "people having a conversation", "an ordinary everyday scene",
                 "people playing sports calmly", "a quiet group of people"],
             1: ["two people fighting each other", "a violent physical fight", "people hitting and kicking someone",
                 "a group of people brawling", "a person being attacked"]},
    "surv": {0: ["people walking past a security camera", "people chatting on CCTV", "a normal day in a public place",
                 "people queueing calmly", "a calm street scene on CCTV"],
             1: ["a fight caught on a security camera", "people assaulting someone on CCTV", "a violent struggle between people",
                 "people punching someone on the street", "an attack on a person in public"]},
}
PROMPT_SETS["broad"] = {0: PROMPT_SETS["orig"][0] + ["a person holding a tool while working", "a crowd cheering at a public event", "people dancing at a party"],
                        1: PROMPT_SETS["orig"][1] + ["a person shooting a gun at people", "a riot with people throwing objects", "a person attacking someone with a knife"]}
PSET = os.environ.get("LENS_PROMPTS", "orig")              # prompt study (PREREG_journal Sec. 6B); default = the original prompts
PROMPTS = PROMPT_SETS[PSET]
NPROMPT = len(PROMPTS[0]) + len(PROMPTS[1])
PSUF = "" if PSET == "orig" else f"_{PSET}"
if PSET != "orig": OUT = OUT.parent / (OUT.name + f"_p{PSET}")
def compact_file(): return "compact" + ("" if FRAMES == 16 else f"_f{FRAMES}") + PSUF + ".npz"

TEMPLATES = ["a video frame of {}.", "a surveillance camera image of {}."]


# ------------------------------------------------------------------ features
def text_anchors():
    f = FEAT / f"text_anchors{PSUF}.npy"
    if f.exists():
        return np.load(f)
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms(CLIPCFG["model"], pretrained="openai", device="cpu")
    tok = open_clip.get_tokenizer(CLIPCFG["tok"])
    W = []
    with torch.no_grad():
        for c in (0, 1):
            e = m.encode_text(tok([t.format(p) for p in PROMPTS[c] for t in TEMPLATES])).float()
            e = F.normalize(e, dim=-1).mean(0)
            W.append(F.normalize(e, dim=0).numpy())
    W = np.stack(W).astype(np.float32)
    np.save(f, W)
    return W


class Tokens:
    """Row-addressable view of a feature set; rows are loaded into RAM on demand."""
    def __init__(self, name):
        d = FEAT / name
        self.index = pd.read_csv(d / "index.csv")
        done = np.load(d / "done.npy")
        if not done.all():
            raise RuntimeError(f"feature set {name} incomplete: {done.sum()}/{len(done)}")
        self.mm = np.memmap(d / "tokens.f16", dtype=np.float16, mode="r", shape=(len(self.index), T, NTOK, DIM))

    def take(self, rows):
        rows = np.asarray(rows)
        order = np.argsort(rows)
        out = np.empty((len(rows), T, NTOK, DIM), np.float16)
        out[order] = self.mm[rows[order]]
        return torch.from_numpy(out)


# ------------------------------------------------------------------ model
class LENS(nn.Module):
    def __init__(self, cfg, anchors, d=384, layers=2, heads=4, drop=0.1):
        super().__init__()
        self.cfg = cfg
        self.w0 = nn.Parameter(torch.tensor(anchors), requires_grad=False)
        if cfg == "-A":
            w = F.normalize(torch.randn(2, DIM), dim=-1)
        else:
            w = torch.tensor(anchors).clone()
        self.w = nn.Parameter(w)
        if cfg in ("LP", "ZS"):
            return
        self.inp = nn.Linear(2 * DIM, d)
        self.pos_s = nn.Parameter(torch.zeros(1, 1, NTOK, d))
        self.pos_t = nn.Parameter(torch.zeros(1, T, 1, d))
        nn.init.trunc_normal_(self.pos_s, std=0.02); nn.init.trunc_normal_(self.pos_t, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, DIM)
        self.tok_drop = 0.1

    def embed(self, z):                                   # z: (B,T,17,768) float
        if self.cfg in ("LP", "ZS"):
            return F.normalize(F.normalize(z[:, :, 0], dim=-1).mean(1), dim=-1)
        scene = z.mean(1, keepdim=True)
        r = z if self.cfg == "-S" else z - scene                                       # C3
        m = torch.cat([torch.zeros_like(z[:, :1]), z[:, 1:] - z[:, :-1]], 1)           # C2
        if self.cfg == "-M":
            m = torch.zeros_like(m)
        h = self.inp(torch.cat([r, m], -1)) + self.pos_s + self.pos_t                  # (B,T,17,d)
        if self.training and self.tok_drop > 0:
            keep = (torch.rand(h.shape[:3], device=h.device) > self.tok_drop).unsqueeze(-1)
            h = h * keep
        B = h.shape[0]
        h = self.enc(h.reshape(B, T * NTOK, -1))
        return F.normalize(self.out(self.norm(h).mean(1)), dim=-1)

    def forward(self, z):
        return SCALE * self.embed(z) @ F.normalize(self.w, dim=-1).t()

    def anchor_penalty(self):
        if self.cfg == "-A":
            return torch.zeros((), device=self.w.device)
        return ANCHOR_W * ((self.w - self.w0) ** 2).sum()


def flip_grid(z):
    """Mirror the columns of the 4x4 patch grid (tokens 1..16, row-major); CLS unchanged."""
    g = z[:, :, 1:].reshape(*z.shape[:2], 4, 4, DIM).flip(3).reshape(*z.shape[:2], 16, DIM)
    return torch.cat([z[:, :, :1], g], 2)


# ------------------------------------------------------------------ data
def load_split(split):
    base = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=[
        "segment_id", "video_id", "component_id", "dataset", "split", "binary_label", "negative_kind"])
    base["feat_row"] = np.arange(len(base))
    idx = pd.read_csv(FEAT / "bench" / "index.csv", usecols=["segment_id"])     # features must be in manifest order
    assert len(idx) == len(base) and (idx.segment_id.values == base.segment_id.values).all(), "feature/label misalignment"
    if split == "audited":
        return base, None
    if split.startswith("lodo:"):
        d = split.split(":")[1]
        m = pd.read_csv(ROOT / f"source_benchmark/lodo/{d}/source_windows.csv", usecols=["segment_id", "split"])
        m = m.rename(columns={"split": "lodo_split"})
        df = m.merge(base, on="segment_id", how="left", validate="one_to_one")
        assert df.feat_row.notna().all()
        df["frozen_split"] = df["split"]; df["split"] = df["lodo_split"]
        return df, d
    if split.startswith("std:"):
        m = pd.read_csv(ROOT / f"workspace/splits/{split.split(':')[1]}.csv", usecols=["segment_id", "split"])
        df = m.merge(base.drop(columns=["split"]), on="segment_id", how="left", validate="one_to_one")
        assert df.feat_row.notna().all()
        return df, "std"
    raise ValueError(split)


def video_level(f, p):
    v = f.assign(p=p).groupby("video_id").agg(y=("binary_label", "max"), p=("p", "mean"))
    return v.y.values, v.p.values


@torch.no_grad()
def predict(model, feats, rows, dev, bs=64):
    model.eval(); out = []
    for i in range(0, len(rows), bs):
        z = feats.take(rows[i:i + bs]).to(dev).float()
        out.append(torch.softmax(model(z).float(), 1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["ZS", "LP", "LENS", "-M", "-S", "-A"], required=True)
    ap.add_argument("--split", default="audited")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pilot", action="store_true", help="LODO: score only held-out D's frozen-val rows")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    name = f"{a.split.replace(':', '_')}_{a.config}_seed{a.seed}{'_pilot' if a.pilot else ''}{a.tag}"
    od = OUT / name; od.mkdir(parents=True, exist_ok=True)
    df, held = load_split(a.split)
    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    if held is None or held == "std":
        te = df[df.split == "test"].reset_index(drop=True) if a.allow_test else df.iloc[:0]
    elif a.pilot:
        te = df[(df.split == "test") & (df.frozen_split == "val")].reset_index(drop=True)
    else:
        te = df[df.split == "test"].reset_index(drop=True) if a.allow_test else df.iloc[:0]
    feats = Tokens("bench")
    model = LENS(a.config, text_anchors()).to(dev)
    log = open(od / "log.jsonl", "w")
    t0 = time.time()

    if a.config != "ZS":
        w = D.strategy_weights(tr, "hybrid", cap=0.05)
        Xtr = feats.take(tr.feat_row.values)                               # RAM, fp16
        ytr = torch.tensor(tr.binary_label.values, dtype=torch.long)
        g = torch.Generator().manual_seed(a.seed)
        steps = len(tr) // a.batch
        no_wd = {"w", "pos_s", "pos_t"}
        params = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and n not in no_wd], "weight_decay": 0.05},
                  {"params": [p for n, p in model.named_parameters() if p.requires_grad and n in no_wd], "weight_decay": 0.0}]
        opt = torch.optim.AdamW([g_ for g_ in params if g_["params"]], lr=3e-4)
        total, warm = steps * a.epochs, steps
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                                  0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1))))
        best, best_state, best_ep = -1, None, -1
        for ep in range(a.epochs):
            model.train()
            idx = torch.multinomial(torch.tensor(w), steps * a.batch, replacement=True, generator=g)
            tl = 0.0
            for s in range(steps):
                b = idx[s * a.batch:(s + 1) * a.batch]
                z = Xtr[b].to(dev, non_blocking=True).float()
                if a.config not in ("LP",):
                    fl = torch.rand(len(b), generator=g) < 0.5
                    if fl.any(): z[fl.to(dev)] = flip_grid(z[fl.to(dev)])
                loss = F.cross_entropy(model(z), ytr[b].to(dev), label_smoothing=0.1) + model.anchor_penalty()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
                tl += loss.item()
            pv = predict(model, feats, va.feat_row.values, dev)
            auc = D.binary_metrics(va.binary_label, pv, 0.5)["auc"]
            log.write(json.dumps(dict(epoch=ep + 1, loss=tl / steps, val_auc=auc, minutes=(time.time() - t0) / 60)) + "\n"); log.flush()
            if auc > best:
                best, best_ep = auc, ep + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        torch.save(best_state, od / "best.pt")

    pv = predict(model, feats, va.feat_row.values, dev)
    thr = D.select_threshold(va.binary_label, pv)
    res = dict(config=a.config, split=a.split, seed=a.seed, pilot=a.pilot, threshold_selected_on_val=thr,
               best_epoch=None if a.config == "ZS" else best_ep, minutes=(time.time() - t0) / 60,
               n_train=len(tr), n_val=len(va), n_scored_test=len(te))
    per = D.per_group(va, pv, thr, "dataset")
    res["val"] = dict(overall=D.binary_metrics(va.binary_label, pv, thr), per_dataset=per,
                      macro_dataset_auc=D.macro_auc(per)[0], worst_dataset_auc=D.macro_auc(per)[1])
    va.assign(p_violent=pv).to_csv(od / "predictions_val.csv", index=False)
    if len(te):
        pt = predict(model, feats, te.feat_row.values, dev)
        key = "heldout_val_rows" if a.pilot else "test"
        res[key] = dict(overall=D.binary_metrics(te.binary_label, pt, thr), per_dataset=D.per_group(te, pt, thr, "dataset"))
        te.assign(p_violent=pt).to_csv(od / f"predictions_{key}.csv", index=False)
        if held == "std":                                   # video-level accuracy, threshold from validation videos
            yv, pvv = video_level(va, pv); yt, ptv = video_level(te, pt)
            vthr = D.select_threshold(yv, pvv)
            res["test_video"] = dict(threshold_selected_on_val_videos=vthr, n_videos=len(yt),
                                     overall=D.binary_metrics(yt, ptv, vthr))
    json.dump(res, open(od / "summary.json", "w"), indent=2, default=float)
    print(name, json.dumps({k: res[k]["overall"]["auc"] for k in ("val", "test", "heldout_val_rows") if k in res}), flush=True)


if __name__ == "__main__":
    main()
