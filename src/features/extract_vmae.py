#!/usr/bin/env python3
"""LENS-M motion features (PREREG deviation 6): frozen VideoMAE-B (Kinetics-400 fine-tuned weights only).
Feature = fc_norm(mean of final hidden states), 768-d, from the 16-frame evaluation clip (ImageNet normalization).
Row order equals the CLIP feature index of the same set. Output: $LENSNET_ROOT/workspace/features/l14/<set>/vmae.npy"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import extract_clip_tokens as E
ROOT = E.ROOT; sys.path.insert(0, str(ROOT))
from video_data import window_data as D
from transformers import VideoMAEForVideoClassification

name = sys.argv[1]
df, store, dec = E.load_set(name)
idx = pd.read_csv(E.OUT / name / "index.csv")
assert len(idx) == len(df) and (idx.path.values == df.path.values).all() and (idx.start_frame.values == df.start_frame.values).all()
df = df.assign(binary_label=0)                              # labels are not needed; WindowDataset requires the column
dev = torch.device("cuda")
m = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to(dev).half().eval()
assert m.fc_norm is not None
dl = DataLoader(D.WindowDataset(df, store, False, decode=dec), batch_size=16, num_workers=10, pin_memory=True)
out = np.zeros((len(df), 768), np.float32)
with torch.no_grad():
    for k, (x, _, i) in enumerate(dl):
        h = m.videomae(pixel_values=x.to(dev).half())[0]
        out[i.numpy()] = m.fc_norm(h.mean(1)).float().cpu().numpy()
        if k % 100 == 0: print(name, k * 16, "/", len(df), flush=True)
np.save(E.OUT / name / "vmae.npy", out); print(name, "done", out.shape, flush=True)
