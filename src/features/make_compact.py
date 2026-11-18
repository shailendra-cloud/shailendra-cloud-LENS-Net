#!/usr/bin/env python3
"""Deterministic LENS-Net inputs (PREREG deviation 2): per window, the mean-over-frames of the L2-normalized CLS (768) and
the prompt-similarity signals s (16 frames x 17 tokens x 10 prompts). CPU only; labels are never read."""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L

def prompt_embeddings():
    f = L.FEAT / f"text_prompts{L.PSUF}.npy"
    if f.exists(): return np.load(f)
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms(L.CLIPCFG["model"], pretrained="openai", device="cpu")
    tok = open_clip.get_tokenizer(L.CLIPCFG["tok"]); E = []
    with torch.no_grad():
        for c in (0, 1):
            for p in L.PROMPTS[c]:
                e = F.normalize(m.encode_text(tok([t.format(p) for t in L.TEMPLATES])).float(), dim=-1).mean(0)
                E.append(F.normalize(e, dim=0).numpy())
    E = np.stack(E).astype(np.float32); np.save(f, E); return E

E = torch.tensor(prompt_embeddings())                       # (10,768): 0-4 non-violence, 5-9 violence
torch.set_num_threads(16)
for name in sys.argv[1:]:
    t = L.Tokens(name); N = len(t.index)
    st = 16 // L.FRAMES                                     # journal E2: 8 frames = every second of the 16
    cls = np.zeros((N, L.DIM), np.float32); S = np.zeros((N, L.FRAMES, 17, L.NPROMPT), np.float16)
    for i in range(0, N, 256):
        z = F.normalize(torch.from_numpy(np.asarray(t.mm[i:i + 256, ::st], dtype=np.float32)), dim=-1)
        cls[i:i + 256] = z[:, :, 0].mean(1).numpy()
        S[i:i + 256] = (z @ E.t()).numpy().astype(np.float16)
    np.savez(L.FEAT / name / L.compact_file(), cls=cls, S=S)
    print(name, N, "ok", flush=True)
