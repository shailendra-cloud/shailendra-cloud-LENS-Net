#!/usr/bin/env python3
"""Journal figure: LENS-Net's own spatial evidence on the unseen confirmation sets. For each set, fixed rule (no manual
choice): the most confident correct detection, the most confident false alarm and the most confident miss (seed-averaged
LENS-Net probability, P1). Overlay = the 4x4 grid of (mean violent - mean non-violent prompt similarity) that feeds PPS,
shown as its true 4x4 cells; one symmetric colour scale for the whole figure. Frames are the model's 224-px square inputs (3 of the 16),
heads blurred with the same rule as the qualitative figure."""
import json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from make_figures import L, OUT, INK, INK2, COL2
LN = L; ROOT = L.parent; sys.path.insert(0, str(ROOT)); sys.path[:0] = [str(LN.parent / "src" / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
from video_data import cnn_bigru_model as mm
import video_data.frame_store as VF
from video_data.frame_store import content_key
# head blur: copied verbatim from plot_qualitative.py (same privacy rule as the qualitative figure)
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

VF.STORE_ROOT = ROOT / "raw_data/journal/frames"; store = VF.FrameStore(stores=("conf",), use_v1=False)
w = pd.read_csv(LN / "conf/conf_windows.csv"); w = w[w.decode_failed == 0].reset_index(drop=True)
S = np.load(ROOT / "workspace/features/l14/conf/compact.npz")["S"].astype(np.float32)          # (N,16,17,10)
p = np.mean([pd.read_csv(LN / f"conf/preds/v2_audited_LENS-M_vrex_seed{s}.csv").p_violent.values for s in (42, 43, 44)], 0)
thr = np.mean([json.load(open(LN / f"runs/v2_audited_LENS-M_vrex_seed{s}/summary.json"))["threshold_selected_on_val"] for s in (42, 43, 44)])
ev = S[:, :, 1:, 5:].mean(-1) - S[:, :, 1:, :5].mean(-1)                                     # (N,16,16) grid evidence
rows = []
for b in ("VioPeru", "AIRTLab"):
    g = w.index[w.benchmark == b]
    y = w.binary_label.values
    rows.append([(b, "correct detection", g[y[g] == 1][np.argmax(p[g][y[g] == 1])]),
                 (b, "false alarm", g[y[g] == 0][np.argmax(p[g][y[g] == 0])]),
                 (b, "miss", g[y[g] == 1][np.argmin(p[g][y[g] == 1])])])
TS = (0, 7, 15); sel = [r for col in rows for r in col]
vmax = np.percentile(np.abs(np.concatenate([ev[i][list(TS)].ravel() for _, _, i in sel])), 98)
norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax); cmap = cm.get_cmap("RdBu_r")
fig, axs = plt.subplots(3, 7, figsize=(COL2, 3.55), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.12, 1, 1, 1], wspace=0.04, hspace=0.32))
for c in range(3): axs[c, 3].axis("off")
meta = []
for col, block in enumerate(rows):
    for r, (b, kind, i) in enumerate(block):
        jp = store.jpegs(-1, content_key(w.path[i], w.start_frame[i], w.end_frame[i], "seq"))
        idx = mm.temporal_indices(len(jp), 16)
        for k, t in enumerate(TS):
            im = cv2.cvtColor(cv2.imdecode(np.frombuffer(jp[int(idx[t])], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            im = anonymize(cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR))
            heat = cv2.resize(ev[i, t].reshape(4, 4), (224, 224), interpolation=cv2.INTER_NEAREST)   # true 4x4 cells
            ov = (0.66 * im / 255 + 0.34 * cmap(norm(heat))[..., :3]).clip(0, 1)
            ax = axs[r, col * 4 + k]; ax.imshow(ov); ax.set_xticks([]); ax.set_yticks([])
            for g_ in (56, 112, 168): ax.axhline(g_ - 0.5, color='white', lw=0.35, alpha=0.7); ax.axvline(g_ - 0.5, color='white', lw=0.35, alpha=0.7)
            for sp in ax.spines.values(): sp.set_linewidth(0.4); sp.set_color("#999999")
            if k == 0: ax.set_ylabel(f"{kind}\n$p$ = {p[i]:.2f}", fontsize=6.6, color=INK)
            if r == 0 and k == 1: ax.set_title(f"{b} ({'real CCTV' if b == 'VioPeru' else 'staged'})", fontsize=7.5, color=INK)
            if r == 2: ax.set_xlabel(f"frame {t + 1}/16", fontsize=6.2, color=INK2)
        meta.append(dict(set=b, kind=kind, video=w.video[i], start_frame=int(w.start_frame[i]), p=float(p[i]), label=int(w.binary_label[i])))
cax = fig.add_axes([0.35, 0.015, 0.3, 0.018])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
cb.set_ticks([-vmax, 0, vmax]); cb.set_ticklabels(["non-violent prompts", "0", "violent prompts"]); cb.ax.tick_params(labelsize=6.2, length=2)
fig.savefig(OUT / "fig_evidence.pdf", dpi=400); fig.savefig(OUT / "fig_evidence.png", dpi=400); plt.close(fig)
json.dump(dict(threshold=float(thr), vmax=float(vmax), examples=meta, anonymization=ANON), open(OUT / "fig_evidence_meta.json", "w"), indent=2)
print(json.dumps(meta, indent=0)[:900]); print("threshold", round(thr, 3), ANON)
