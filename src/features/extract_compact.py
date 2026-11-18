#!/usr/bin/env python3
"""Space-saving re-extraction (2026-09-27, after the /home data loss). One CLIP pass per window writes only what the
experiments use, with exactly the arithmetic of extract_clip_tokens.py + make_compact.py (fp16 projected tokens ->
fp32 -> L2-normalise; cls = mean over frames of the normalised CLS; S = normalised tokens @ prompt embeddings, fp16):
  compact{_f8}{_<promptset>}.npz  for the requested prompt sets (16 frames) and 8 frames (original prompts)
  cls_frames.npy  per-frame projected CLS embeddings (N,16,D) fp16 (VadCLIP input, = CLIP encode_image)
  index.csv  row order (identical to the original feature index)
Usage: LENS_BACKBONE=L14|B16 extract_compact.py --set bench|external|fresh|conf --psets orig,para,surv,broad"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import extract_clip_tokens as E, lensnet_setup as L


def prompt_emb(model_name, tok_name, prompts):
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms(model_name, pretrained="openai", device="cpu")
    tok = open_clip.get_tokenizer(tok_name); out = []
    with torch.no_grad():
        for c in (0, 1):
            for p in prompts[c]:
                e = F.normalize(m.encode_text(tok([t.format(p) for t in L.TEMPLATES])).float(), dim=-1).mean(0)
                out.append(F.normalize(e, dim=0).numpy())
    return np.stack(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--psets", default="orig")
    ap.add_argument("--batch", type=int, default=8); a = ap.parse_args()
    assert L.BACKBONE == E.BB and L.FRAMES == 16
    od = Path(L.FEAT) / a.set; od.mkdir(parents=True, exist_ok=True)
    df, store, dec = E.load_set(a.set); df.to_csv(od / "index.csv", index=False); N = len(df)
    psets = a.psets.split(",")
    PE = {}
    for ps in psets:
        f = Path(L.FEAT) / f"text_prompts{'' if ps == 'orig' else '_' + ps}.npy"
        if not f.exists(): np.save(f, prompt_emb(L.CLIPCFG["model"], L.CLIPCFG["tok"], L.PROMPT_SETS[ps]))
        PE[ps] = torch.tensor(np.load(f))
    D = L.DIM
    outs = {ps: (np.zeros((N, D), np.float32), np.zeros((N, 16, 17, PE[ps].shape[0]), np.float16)) for ps in psets}
    f8 = (np.zeros((N, D), np.float32), np.zeros((N, 8, 17, PE["orig"].shape[0]), np.float16))
    clsf = np.zeros((N, 16, D), np.float16)
    import open_clip
    dev = torch.device("cuda")
    v = open_clip.create_model_and_transforms(L.CLIPCFG["model"], pretrained="openai", device=dev, precision="fp16")[0].visual.eval()
    assert type(v.transformer.resblocks[0].mlp.gelu).__name__ == "QuickGELU"; v.output_tokens = True
    g_, w_ = L.CLIPCFG["grid"], L.CLIPCFG["width"]; t0 = time.time(); k = 0
    with torch.no_grad():
        for x, idx in DataLoader(E.Windows(df, store, dec), batch_size=a.batch, num_workers=10, pin_memory=True):
            B = x.shape[0]; x = x.to(dev, non_blocking=True).half().flatten(0, 1)
            c, pt = v(x); g = pt.view(-1, g_, g_, w_).permute(0, 3, 1, 2)
            g = F.adaptive_avg_pool2d(g.float(), 4).flatten(2).transpose(1, 2).half() @ v.proj
            tok = torch.cat([c.unsqueeze(1), g], 1).view(B, 16, 17, D).cpu()      # fp16, as stored in tokens.f16
            ii = idx.numpy(); clsf[ii] = tok[:, :, 0].numpy()
            z = F.normalize(tok.float(), dim=-1)
            for ps in psets:
                outs[ps][0][ii] = z[:, :, 0].mean(1).numpy(); outs[ps][1][ii] = (z @ PE[ps].t()).numpy().astype(np.float16)
            z8 = F.normalize(tok[:, ::2].float(), dim=-1)
            f8[0][ii] = z8[:, :, 0].mean(1).numpy(); f8[1][ii] = (z8 @ PE["orig"].t()).numpy().astype(np.float16)
            k += B
            if k % 1600 < B: print(f"{a.set} {k}/{N} {k/(time.time()-t0):.1f} win/s", flush=True)
    for ps in psets:
        np.savez(od / f"compact{'' if ps == 'orig' else '_' + ps}.npz", cls=outs[ps][0], S=outs[ps][1])
    np.savez(od / "compact_f8.npz", cls=f8[0], S=f8[1]); np.save(od / "cls_frames.npy", clsf)
    np.save(od / "done.npy", np.ones(N, bool))
    print(a.set, "finished", N, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
