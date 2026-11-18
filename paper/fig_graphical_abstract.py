#!/usr/bin/env python3
"""Graphical abstract (Information Fusion): source-controlled data -> evidence -> two fusion levels -> what transfers.
Numbers read from review/e5_fusion.json and review/stats.json-derived results (no typed numbers)."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
R = Path(__file__).resolve().parents[1] / "workspace/review"; OUT = Path(__file__).resolve().parent / "figs/graphical_abstract"
OUT.mkdir(parents=True, exist_ok=True)
e5 = json.load(open(R / "e5_fusion.json")); pr = e5["pairs"]
lora = pr["CLIP probe + VideoMAE-B LoRA"]["R1 mean prob"]["mean9"]; lensa = pr["CLIP probe + LENS-Net"]["R1 mean prob"]["mean9"]
lrn = [v for k, v in e5["descriptive"].items() if k.startswith(("R7", "R8"))]; nb = sum(v["better"] for v in lrn)
BLUE, ORANGE, PURPLE, GREY, INK, INK2, BG = "#2a78d6", "#eb6834", "#6a3d9a", "#8c8a84", "#0b0b0b", "#52514e", "#f4f3ef"
# Elsevier: 1328 x 531 px (2.5:1) or proportionally more, readable at 13 x 5.2 cm -> drawn at 1.5x that size, text >= 6.3 pt when printed
plt.rcParams.update({"font.family": "serif", "font.size": 9.5})
fig = plt.figure(figsize=(7.68, 3.072)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 26); ax.set_ylim(0, 10); ax.axis("off")
TXT = []
def box(x, y, w, h, title, lines, col, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.35", fc=BG, ec=col, lw=1.4))
    TXT.append(((x, y, w, h), ax.text(x + w / 2, y + h - 0.5, title, ha="center", va="top", fontsize=10.5, weight="bold", color=tc)))
    for i, l in enumerate(lines): TXT.append(((x, y, w, h), ax.text(x + w / 2, y + h - 1.45 - 0.78 * i, l, ha="center", va="top", fontsize=9.5, color=INK2)))
def arrow(x0, y0, x1, y1, col=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=col))
box(0.3, 0.6, 4.9, 8.8, "Source control", ["5 source corpora", "same-source", "audit, checked", "by a human", "", "random splits:", "24-56% of test", "videos linked"], INK2)
box(6.3, 5.4, 5.0, 4.0, "Frozen models", ["CLIP anchor", "VideoMAE motion"], BLUE)
box(6.3, 0.6, 5.0, 4.0, "Adapted models", ["fine-tuned and", "LoRA video"], ORANGE)
box(12.4, 5.4, 6.4, 4.0, "Feature-level fusion", ["learned gates:", "unstable under shift"], PURPLE)
box(12.4, 0.6, 6.4, 4.0, "Decision-level fusion", ["8 rules x 9 pairs;", "fitted rules better", f"in {nb} of {len(lrn)} tests"], PURPLE)
h5 = json.load(open(R / "h_partH.json")); nrules = len(h5["untouched_auc_max"])
box(19.9, 0.6, 5.8, 8.8, "14 unseen test sets", ["anchored average:", f"{lensa:.1f} AUC (dev.),", "confirmed on", "untouched data", "", f"of {nrules} fusion rules,", "none consistently", "better on 5 untouched"], BLUE)
arrow(5.3, 7.4, 6.2, 7.4); arrow(5.3, 2.6, 6.2, 2.6)
arrow(11.4, 7.4, 12.3, 7.4, BLUE); arrow(11.4, 6.4, 12.3, 3.4, BLUE); arrow(11.4, 2.6, 12.3, 2.6, ORANGE)
arrow(18.9, 7.4, 19.8, 6.4); arrow(18.9, 2.6, 19.8, 2.6, PURPLE)
fig.canvas.draw(); rd = fig.canvas.get_renderer(); bad = []
for (x, y, w, h), t in TXT:
    bb = t.get_window_extent(rd).transformed(ax.transData.inverted())
    if bb.x0 < x + 0.1 or bb.x1 > x + w - 0.1 or bb.y0 < y + 0.1: bad.append(t.get_text())
assert not bad, f"text outside its box: {bad}"
for ext in ("pdf", "png"): fig.savefig(OUT / f"graphical_abstract.{ext}", dpi=625)
from PIL import Image
im = Image.open(OUT / "graphical_abstract.png"); im.convert("RGB").save(OUT / "graphical_abstract.tif", compression="tiff_lzw", dpi=(625, 625)); print(im.size)
print("written", OUT)
