#!/usr/bin/env python3
"""Track B: cache CLIP tokens entering block 20 for every std_rwf window. Verifies that Tail(cache) reproduces the stored
full-model tokens in $LENSNET_ROOT/workspace/features/l14/bench."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Subset
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import clip_tail as C, extract_clip_tokens as E

OUT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/features/l14/rwf_block20"); OUT.mkdir(parents=True, exist_ok=True)
ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "path", "start_frame", "end_frame", "v1_cache_row"])
man["feat_row"] = np.arange(len(man))
sp = pd.read_csv(ROOT / "workspace/splits/std_rwf.csv")
df = sp.merge(man, on="segment_id", how="left").reset_index(drop=True)
df.to_csv(OUT / "index.csv", index=False)
N = len(df)
mode = "r+" if (OUT / "done.npy").exists() else "w+"
tok = np.memmap(OUT / "tokens.f16", dtype=np.float16, mode=mode, shape=(N, 16, 257, 1024))
done = np.load(OUT / "done.npy") if mode == "r+" else np.zeros(N, bool)
dev = torch.device("cuda")
v = C.load_clip(dev, "fp16"); tail = C.Tail(v).eval()
_, store, dec = E.load_set("bench")
ds = E.Windows(df, store, dec)
todo = np.where(~done)[0]
dl = DataLoader(Subset(ds, todo.tolist()), batch_size=8, num_workers=10, pin_memory=True)
bench = np.memmap(str(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/features/l14/bench/tokens.f16"), dtype=np.float16, mode="r", shape=(len(man), 16, 17, 768))
t0, k, checked = time.time(), 0, False
with torch.no_grad():
    for x, idx in dl:
        B = x.shape[0]
        h = C.head(v, x.to(dev).half().flatten(0, 1))
        if not checked:
            ref = torch.from_numpy(np.asarray(bench[df.feat_row.values[idx.numpy()]], dtype=np.float32)).to(dev)
            got = tail(h).float().view(B, 16, 17, 768)
            cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
            print(f"verify vs stored tokens: min cos {cos.min().item():.5f}, mean {cos.mean().item():.5f}", flush=True)
            assert cos.min() > 0.999, "tail(cache) does not reproduce stored tokens"
            checked = True
        tok[idx.numpy()] = h.view(B, 16, 257, 1024).cpu().numpy(); done[idx.numpy()] = True; k += B
        if k % 400 < B:
            tok.flush(); np.save(OUT / "done.npy", done); print(f"{k}/{len(todo)} {k/(time.time()-t0):.1f} win/s", flush=True)
tok.flush(); np.save(OUT / "done.npy", done); print("finished", int(done.sum()), "/", N, flush=True)
