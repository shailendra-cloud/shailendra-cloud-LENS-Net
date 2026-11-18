#!/usr/bin/env python3
"""Extract frozen CLIP ViT-L/14 tokens for every benchmark window and every external evaluation window (PREREG Sec. 1).

For each window: the 16 frames cnn_bigru_model.temporal_indices(32, 16) selects, resized to 224x224 with CLIP normalisation.
Per frame, 17 tokens x 768 (fp16): the projected CLS token, then the 16x16 patch tokens after ln_post, avg-pooled to 4x4
and projected with the CLIP projection. Labels are never read. Output is a resumable memmap:
  $LENSNET_ROOT/workspace/features/l14/<set>/tokens.f16  (N,16,17,768)   index.csv (row order)   done.npy (bool mask)
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))
from video_data import cnn_bigru_model as mm
from video_data.frame_store import FrameStore, content_key

import os                                             # E2: LENS_BACKBONE=L14 (default) | B16
BB = os.environ.get("LENS_BACKBONE", "L14")
CC = {"L14": dict(model="ViT-L-14-quickgelu", out=str(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/features/l14"), dim=768, grid=16, width=1024),
      "B16": dict(model="ViT-B-16-quickgelu", out=str(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/features/b16"), dim=512, grid=14, width=768)}[BB]
OUT = Path(CC["out"])
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
T, NTOK, DIM = 16, 17, CC["dim"]


class Windows(Dataset):
    def __init__(self, df, store, decode):
        self.df, self.store = df.reset_index(drop=True), store
        self.v1 = (self.df.v1_cache_row.fillna(-1).astype(int).values if "v1_cache_row" in self.df
                   else np.full(len(self.df), -1))
        self.keys = [content_key(p, s, e, decode) for p, s, e in zip(self.df.path, self.df.start_frame, self.df.end_frame)]
        missing = sum(not store.has(int(r), k) for r, k in zip(self.v1, self.keys))
        if missing:
            raise RuntimeError(f"{missing} windows have no frames")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        jp = self.store.jpegs(int(self.v1[i]), self.keys[i])
        fr = []
        for t in mm.temporal_indices(len(jp), T):
            b = jp[int(t)]
            a = np.frombuffer(b, np.uint8) if isinstance(b, (bytes, bytearray)) else np.asarray(b)
            f = cv2.cvtColor(cv2.imdecode(a, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            fr.append(cv2.resize(f, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0)
        x = (np.stack(fr) - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))), i


def load_set(name):
    if name == "bench":
        df = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=[
            "segment_id", "path", "start_frame", "end_frame", "v1_cache_row"])
        return df, FrameStore(stores=("supp",), use_v1=True), "seek"
    if name == "fresh":
        import video_data.frame_store as VF
        VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/fresh/frames")
        f = pd.read_csv(ROOT / "workspace/fresh/fresh_windows.csv")
        f = f[f.decode_failed == 0][["benchmark", "video", "path", "start_frame", "end_frame"]].reset_index(drop=True)
        return f, VF.FrameStore(stores=("fresh",), use_v1=False), "seq"
    if name == "conf":                                  # journal confirmation sets (PREREG_journal Sec. 3)
        import video_data.frame_store as VF
        VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/frames")
        f = pd.read_csv(ROOT / "workspace/conf/conf_windows.csv")
        f = f[f.decode_failed == 0][["benchmark", "video", "cluster", "path", "start_frame", "end_frame"]].reset_index(drop=True)
        return f, VF.FrameStore(stores=("conf",), use_v1=False), "seq"
    if name == "cfights":                               # CCTV-Fights confirmation (PREREG_journal Sec. 7)
        import video_data.frame_store as VF
        VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/frames")
        f = pd.read_csv(ROOT / "workspace/cfights/cfights_windows.csv")
        f = f[f.decode_failed == 0][["benchmark", "video", "cluster", "path", "start_frame", "end_frame"]].reset_index(drop=True)
        return f, VF.FrameStore(stores=("cfights",), use_v1=False), "seq"
    if name.startswith("c3_"):                          # LENS-Net-A confirmation chunk (PREREG_journal Sec. 10)
        import video_data.frame_store as VF
        VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/frames")
        f = pd.read_csv(ROOT / f"workspace/conf3/chunks/{name}.csv")
        f = f[f.decode_failed == 0][["benchmark", "video", "cluster", "path", "start_frame", "end_frame"]].reset_index(drop=True)
        return f, VF.FrameStore(stores=(name,), use_v1=False), "seq"
    if name == "conf2":                                 # LENS-FG confirmation: Movies + UBI-Fights (PREREG_journal Sec. 9)
        import video_data.frame_store as VF
        VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/frames")
        f = pd.read_csv(ROOT / "workspace/conf2/conf2_windows.csv")
        f = f[f.decode_failed == 0][["benchmark", "video", "cluster", "path", "start_frame", "end_frame"]].reset_index(drop=True)
        return f, VF.FrameStore(stores=("conf2",), use_v1=False), "seq"
    if name == "amb":                                   # E4 (PREREG_revision Part E): ambiguous windows of CCTV-Fights / UBI-Fights
        import video_data.frame_store as VF
        VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/frames")
        f = pd.read_csv(ROOT / "workspace/amb/amb_windows.csv")
        f = f[f.decode_failed == 0][["benchmark", "video", "cluster", "path", "start_frame", "end_frame"]].reset_index(drop=True)
        return f, VF.FrameStore(stores=("amb",), use_v1=False), "seq"
    if name.startswith("fl"):                           # frame-level XD/UCF chunk (PREREG_journal Sec. 8)
        import video_data.frame_store as VF
        VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/frames")
        f = pd.read_csv(ROOT / f"workspace/framelevel/chunks/{name}.csv")
        f = f[f.decode_failed == 0][["benchmark", "video", "cluster", "path", "start_frame", "end_frame"]].reset_index(drop=True)
        return f, VF.FrameStore(stores=(name,), use_v1=False), "seq"
    ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
    pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)]
    pool = pool[["benchmark", "video", "path", "start_frame", "end_frame"]].reset_index(drop=True)
    return pool, FrameStore(stores=("external",), use_v1=False), "seq"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["bench", "external", "fresh", "conf", "cfights", "conf2"], required=True)
    ap.add_argument("--batch", type=int, default=8, help="windows per batch (x16 frames)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="smoke test only")
    a = ap.parse_args()
    import open_clip
    df, store, decode = load_set(a.set)
    if a.limit: df = df.iloc[:a.limit].copy()
    od = OUT / a.set; od.mkdir(parents=True, exist_ok=True)
    df.to_csv(od / "index.csv", index=False)
    N = len(df)
    mode = "r+" if (od / "tokens.f16").exists() and (od / "done.npy").exists() else "w+"
    tok = np.memmap(od / "tokens.f16", dtype=np.float16, mode=mode, shape=(N, T, NTOK, DIM))
    done = np.load(od / "done.npy") if mode == "r+" else np.zeros(N, bool)
    todo = np.where(~done)[0]
    print(f"{a.set}: {N} windows, {len(todo)} to do", flush=True)

    dev = torch.device("cuda")
    model, _, _ = open_clip.create_model_and_transforms(CC["model"], pretrained="openai", device=dev, precision="fp16")
    v = model.visual.eval(); v.output_tokens = True
    assert type(v.transformer.resblocks[0].mlp.gelu).__name__ == "QuickGELU", "OpenAI CLIP weights need QuickGELU"
    proj = v.proj                                           # (width, dim): (1024, 768) for L14, (768, 512) for B16
    ds = torch.utils.data.Subset(Windows(df, store, decode), todo.tolist())
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True)
    t0, k = time.time(), 0
    with torch.no_grad():
        for x, idx in dl:
            B = x.shape[0]
            x = x.to(dev, non_blocking=True).half().flatten(0, 1)             # (B*16,3,224,224)
            cls, pt = v(x)                                                     # (B*16,dim), (B*16,grid^2,width) post ln_post
            g = pt.view(-1, CC["grid"], CC["grid"], CC["width"]).permute(0, 3, 1, 2)
            g = F.adaptive_avg_pool2d(g.float(), 4).flatten(2).transpose(1, 2).half() @ proj   # (B*16,16,768)
            out = torch.cat([cls.unsqueeze(1), g], 1).view(B, T, NTOK, DIM)
            ii = idx.numpy()
            tok[ii] = out.cpu().numpy(); done[ii] = True
            k += B
            if k % 800 < B:
                tok.flush(); np.save(od / "done.npy", done)
                r = k / (time.time() - t0); print(f"{k}/{len(todo)}  {r:.1f} win/s  eta {(len(todo)-k)/r/60:.1f} min", flush=True)
    tok.flush(); np.save(od / "done.npy", done)
    json.dump(dict(set=a.set, windows=N, done=int(done.sum()), model=CC["model"] + "/openai", tokens="CLS+4x4 grid, projected",
                   frames="temporal_indices(32,16)", seconds=time.time() - t0), open(od / "info.json", "w"), indent=2)
    print("finished", int(done.sum()), "/", N)


if __name__ == "__main__":
    main()
