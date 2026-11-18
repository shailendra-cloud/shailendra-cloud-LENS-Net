#!/usr/bin/env python3
"""Real assets for the architecture figure: three head-blurred frames from UCF-Crime Fighting003 (inside its annotated
violent interval) and the real PPS similarity map (10 prompts x 16 frames, grid-max) of one violent window of that video."""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.argv = [sys.argv[0], "3", "1.4", "_unused.pdf"]                      # plot_qualitative expects argv; we only import helpers
sys.path.insert(0, str(Path(__file__).parent))
src = open(Path(__file__).parent / "plot_qualitative.py").read().split("n = len(res)")[0]
exec(src)                                                                  # defines anonymize(), native_frame(), style, paths
FIG = Path(__file__).parent / "figs"
ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
path = ext[ext.video == "Fighting003_x264.mp4"].path.iloc[0]
for i, t in enumerate([66.0, 70.0, 74.0]):
    im = anonymize(native_frame(path, t)); h, w = im.shape[:2]
    im = cv2.resize(im, (320, int(320 * h / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(FIG / f"arch_frame{i}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
S = np.load(str(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[1]) / "workspace/features/l14/external/compact.npz"))["S"].astype(np.float32)
rows = pool.index[(pool.video == "Fighting003_x264.mp4") & (pool.binary_label == 1)]
k = rows[0]; A = S[k, :, 1:, :].max(1).T                                  # (10 prompts, 16 frames): max over the 4x4 grid
A = A - A.mean(0, keepdims=True)                                           # centre per frame so the prompt contrast is visible
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(1.6, 1.0), dpi=300)
ax.imshow(A, cmap="RdBu_r", vmin=-np.abs(A).max(), vmax=np.abs(A).max(), aspect="auto", interpolation="nearest")
ax.axhline(4.5, color="black", lw=0.6); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
fig.savefig(FIG / "arch_pps.png", bbox_inches="tight", pad_inches=0); print("assets ok; window", pool.start_frame[k], "-", pool.end_frame[k], "anon", ANON)
