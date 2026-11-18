#!/usr/bin/env python3
"""C2 (PREREG_revision.md): compact LENS-Net inputs from OpenCLIP ViT-L/14 laion2b_s32b_b82k, computed exactly as for the
OpenAI encoder (extract_clip_tokens.py + make_compact.py: CLS projected, 16x16 patch tokens after ln_post avg-pooled to
4x4 and projected; per window the mean of the normalised CLS and the token-prompt similarities S), without storing the
token memmap. Same windows and frames as the OpenAI features; VideoMAE features are linked from features/l14."""
import os, sys, json, time
os.environ["LENS_BACKBONE"] = "LAION"
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import lensnet_setup as L
os.environ["LENS_BACKBONE"] = "L14"; import extract_clip_tokens as X; os.environ["LENS_BACKBONE"] = "LAION"   # only its window loader is used
import open_clip
OUT = L.FEAT; OUT.mkdir(parents=True, exist_ok=True); L14 = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/features/l14")
dev = torch.device("cuda")
model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k", device=dev, precision="fp16")
tok = open_clip.get_tokenizer("ViT-L-14")
with torch.no_grad():                                   # prompt and anchor embeddings with the LAION text encoder (lensnet_setup.py rules)
    E, A = [], []
    for c in (0, 1):
        allc = []
        for p in L.PROMPTS[c]:
            e = F.normalize(model.encode_text(tok([t.format(p) for t in L.TEMPLATES]).to(dev)).float(), dim=-1)
            E.append(F.normalize(e.mean(0), dim=0).cpu().numpy()); allc.append(e)
        A.append(F.normalize(torch.cat(allc).mean(0), dim=0).cpu().numpy())
np.save(OUT / "text_prompts.npy", np.stack(E).astype(np.float32)); np.save(OUT / "text_anchors.npy", np.stack(A).astype(np.float32))
Et = torch.tensor(np.stack(E), device=dev)
v = model.visual.eval(); v.output_tokens = True; proj = v.proj
# the frame pipeline normalises with the OpenAI CLIP statistics; these LAION weights expect mean = std = 0.5,
# so every frame is re-normalised: x_laion = (x * std_openai + mean_openai - m) / s
LM = torch.tensor(model.visual.preprocess_cfg["mean"], device=dev).view(1, 3, 1, 1); LS = torch.tensor(model.visual.preprocess_cfg["std"], device=dev).view(1, 3, 1, 1)
OM = torch.tensor(X.MEAN, device=dev).view(1, 3, 1, 1); OS = torch.tensor(X.STD, device=dev).view(1, 3, 1, 1)
print("re-normalising from", X.MEAN, X.STD, "to", model.visual.preprocess_cfg["mean"], model.visual.preprocess_cfg["std"], flush=True)
import video_data.frame_store as VF0; ORIG_ROOT = VF0.STORE_ROOT
for name in sys.argv[1:]:
    VF0.STORE_ROOT = ORIG_ROOT                    # load_set redirects the store root for journal sets; reset per set
    df, store, decode = X.load_set(name); od = OUT / name; od.mkdir(exist_ok=True)
    if (od / L.compact_file()).exists(): print(name, "exists"); continue
    df.to_csv(od / "index.csv", index=False); N = len(df)
    cls = np.zeros((N, 768), np.float32); S = np.zeros((N, 16, 17, 10), np.float16)
    dl = DataLoader(X.Windows(df, store, decode), batch_size=8, num_workers=10, pin_memory=True); t0 = time.time(); k = 0
    with torch.no_grad():
        for x, idx in dl:
            B = x.shape[0]; x = x.to(dev, non_blocking=True).flatten(0, 1); x = (((x * OS + OM) - LM) / LS).half()
            c, pt = v(x)
            g = pt.view(-1, 16, 16, 1024).permute(0, 3, 1, 2)
            g = F.adaptive_avg_pool2d(g.float(), 4).flatten(2).transpose(1, 2).half() @ proj
            z = F.normalize(torch.cat([c.unsqueeze(1), g], 1).view(B, 16, 17, 768).float(), dim=-1)
            ii = idx.numpy(); cls[ii] = z[:, :, 0].mean(1).cpu().numpy(); S[ii] = (z @ Et.t()).cpu().numpy().astype(np.float16)
            k += B
            if k % 2000 < B: print(f"{name} {k}/{N} {k/(time.time()-t0):.1f} win/s", flush=True)
    np.savez(od / L.compact_file(), cls=cls, S=S)
    if (L14 / name / "vmae.npy").exists() and not (od / "vmae.npy").exists(): os.symlink(L14 / name / "vmae.npy", od / "vmae.npy")
    json.dump(dict(set=name, windows=N, model="ViT-L-14/laion2b_s32b_b82k", minutes=(time.time() - t0) / 60), open(od / "info.json", "w"))
    print(name, N, "done", round((time.time() - t0) / 60, 1), "min", flush=True)
