#!/usr/bin/env python3
"""Plot the qualitative figure from figs/qualitative_data.npy (make_qualitative.py). Thumbnails are re-read from the
original videos at native aspect ratio; triangles on the time axis mark the thumbnail times."""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent)); from make_figures import *   # shared style and palette
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
LN = Path(__file__).resolve().parents[1] / "workspace"; ROOT = LN.parent
res = np.load(OUT / "qualitative_data.npy", allow_pickle=True)
ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv").drop_duplicates("video").set_index("video")
thr = np.mean([json.load(open(LN / f"runs/v2_audited_LENS-M_vrex_seed{s}/summary.json"))["threshold_selected_on_val"] for s in (42, 43, 44)])

import torch, torchvision
_det_w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
_det = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=_det_w).eval()
ANON = {"frames": 0, "people_pixelated": 0}

def anonymize(im):
    """Privacy: hide heads/faces only, so the action stays visible. For each detected person (COCO 'person', score >= 0.3)
    the head region (top 24% of the box, central 70% of its width, enlarged slightly) is strongly blurred.
    Dark frames get a disclosed gamma correction so the scene is legible."""
    with torch.no_grad():
        o = _det([torch.from_numpy(im).permute(2, 0, 1).float() / 255])[0]
    H, W = im.shape[:2]; out = im.copy()
    for (x0, y0, x1, y1), lab, sc in zip(o["boxes"].numpy(), o["labels"].numpy(), o["scores"].numpy()):
        if lab != 1 or sc < 0.3: continue
        bw, bh = x1 - x0, y1 - y0
        hx0, hx1 = x0 + 0.15 * bw - 0.05 * bw, x1 - 0.15 * bw + 0.05 * bw
        hy0, hy1 = y0 - 0.04 * bh, y0 + 0.24 * bh
        hx0, hy0, hx1, hy1 = int(max(0, hx0)), int(max(0, hy0)), int(min(W, hx1)), int(min(H, hy1))
        if hx1 - hx0 < 2 or hy1 - hy0 < 2: continue
        roi = out[hy0:hy1, hx0:hx1]; k = max(5, (min(roi.shape[:2]) // 2) | 1)
        out[hy0:hy1, hx0:hx1] = cv2.GaussianBlur(cv2.GaussianBlur(roi, (k, k), 0), (k, k), 0)
        ANON["people_pixelated"] += 1
    ANON["frames"] += 1
    lum = out.mean() / 255
    if lum < 0.28:                                         # disclosed brightness correction for dark film frames
        g = max(0.6, np.log(0.33) / np.log(max(lum, 1e-3))); out = np.clip(255 * (out / 255.0) ** g, 0, 255).astype(np.uint8); ANON["gamma_corrected"] = ANON.get("gamma_corrected", 0) + 1
    return out

def thumb_times(r):
    """Disclosed rule: 10% and 90% of the video, LENS-Net's peak score, and the centre of the longest annotated interval;
    a time within 3 s of an earlier one is replaced by the video midpoint (or 30%/70% if that is also taken)."""
    gl = max(r["gt"], key=lambda g: g[1] - g[0]) if r["gt"] else (0.4 * r["dur"], 0.6 * r["dur"])
    cand = [0.1 * r["dur"], 0.9 * r["dur"], float(r["t"][int(np.argmax(r["lens"]))]), (gl[0] + gl[1]) / 2]
    out = []
    for t in cand:
        for alt in [t, 0.5 * r["dur"], 0.3 * r["dur"], 0.7 * r["dur"]]:
            if all(abs(alt - u) >= 3 for u in out): out.append(alt); break
    out = sorted(out)
    if NTH == 3:                                            # compact version: drop the 90% frame (keep peak and event)
        out = [t for t in out if abs(t - 0.9 * r["dur"]) > 1e-6] if len(out) == 4 else out[:3]
    return out

def native_frame(path, t):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read(); cap.release()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None

NTH = int(sys.argv[1]) if len(sys.argv) > 1 else 4; RH = float(sys.argv[2]) if len(sys.argv) > 2 else 1.95
OUTNAME = sys.argv[3] if len(sys.argv) > 3 else "fig_qualitative.pdf"
n = len(res); fig = plt.figure(figsize=(COL2, RH * n + 0.25))
LEFT = 0.13
top, rowh, gap = 0.955, (0.955 - 0.05) / n, 0.012
for i, r in enumerate(res):
    y0 = top - (i + 1) * rowh
    ax = fig.add_axes([LEFT, y0 + 0.11 * rowh, 0.98 - LEFT, 0.30 * rowh])
    neg = "non-violent" in r["title"]
    for a0, a1 in r["gt"]: ax.axvspan(a0, a1, color=GREY2 if neg else ORANGE, alpha=0.2 if neg else 0.15, lw=0)
    ax.plot(r["t"], r["lens"], color=BLUE, lw=1.4)
    ax.plot(r["t"], r["vmae"], color=ORANGE, lw=1.0, ls=(0, (3, 1.5)))
    ax.axhline(thr, color=INK2, lw=0.6, ls=":")
    ax.set_xlim(0, r["dur"]); ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 0.5, 1]); style(ax, "y"); ax.tick_params(labelsize=6, pad=1.5)
    ax.set_ylabel("P(violent)", fontsize=6.5, labelpad=2)
    if i == n - 1: ax.set_xlabel("time (s)", fontsize=6.5, labelpad=1)
    ax.plot(thumb_times(r), [-0.03] * len(thumb_times(r)), marker="^", ms=4, color=INK2, lw=0, clip_on=False, zorder=4)
    fig.text(0.012, y0 + 0.62 * rowh, r["title"].replace(": ", "\n").replace(" (non-violent)", "\n(non-violent)"), fontsize=7, color=INK, va="center", ha="left", fontweight="bold")
    path = ext.loc[r["video"], "path"]
    NT = NTH; tw = (0.98 - LEFT) / NT
    tts = thumb_times(r)
    for j, tt in enumerate(tts):
        im = native_frame(path, tt)
        if im is not None: im = anonymize(im)
        a = fig.add_axes([LEFT + j * tw + 0.004, y0 + 0.47 * rowh, tw - 0.008, 0.50 * rowh])
        if im is not None: a.imshow(im, aspect="auto")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_edgecolor(GRID); sp.set_linewidth(0.5)
        a.text(0.02, 0.04, f"{tt:.0f} s", transform=a.transAxes, fontsize=5.5, color="white", va="bottom",
               bbox=dict(boxstyle="round,pad=0.12", fc="black", ec="none", alpha=0.55))
handles = [Line2D([], [], color=BLUE, lw=1.4, label="LENS-Net (ours), mean of 3 seeds"),
           Line2D([], [], color=ORANGE, lw=1.0, ls=(0, (3, 1.5)), label="VideoMAE-B fine-tuned"),
           Line2D([], [], color=INK2, lw=0.6, ls=":", label="LENS-Net decision threshold"),
           Patch(color=ORANGE, alpha=0.15, label="annotated violence"), Patch(color=GREY2, alpha=0.2, label="annotated non-violent anomaly")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=5, frameon=False, fontsize=6, handlelength=1.6, columnspacing=1.0)
fig.savefig(OUT / OUTNAME); print("saved; threshold", round(thr, 3), "anonymization:", ANON)
