#!/usr/bin/env python3
"""Journal figure: ROC-AUC vs corruption severity (PREREG_journal Sec. 5), from logs/robust_results.json.
Same palette and style as make_figures.py (LENS-Net blue, VideoMAE-B orange, others grey with distinct markers)."""
import json
import matplotlib.pyplot as plt
from make_figures import L, OUT, BLUE, ORANGE, GREY1, GREY2, INK2, COL2, style
R = json.load(open(L / "logs/robust_results.json"))
FAM = [("Gaussian blur", "blur", ["0", "1", "2", "3"], "$\\sigma$ (px)"), ("Low light", "dark", ["1", "0.5", "0.3", "0.15"], "intensity scale"),
       ("Gaussian noise", "noise", ["0", "8", "16", "32"], "$\\sigma$ (0–255)"), ("JPEG compression", "jpeg", ["–", "30", "15", "5"], "quality")]
MOD = [("LENS-Net (ours)", "LENS-Net", BLUE, "o", "-"), ("LENS-Net w/o motion", "w/o motion", BLUE, "o", "--"),
       ("VideoMAE-B FT", "VideoMAE-B FT", ORANGE, "s", "-"), ("Video Swin-T FT", "Swin-T FT", GREY1, "D", "-"), ("R(2+1)D-18 FT", "R(2+1)D-18 FT", GREY2, "^", "-")]
fig, axs = plt.subplots(2, 4, figsize=(COL2, 3.55))
for r, (s, sname) in enumerate((("test", "Audited test"), ("viop", "VioPeru (real CCTV)"))):
    for c, (title, key, ticks, xl) in enumerate(FAM):
        ax = axs[r, c]; style(ax, "y")
        for lab, m, col, mk, ls in MOD:
            a = R[s]["auc"][m]; y = [a["clean"]] + [a[f"{key}{i}"] for i in (1, 2, 3)]
            ax.plot(range(4), y, ls=ls, marker=mk, color=col, lw=1.1, ms=3.4, mfc=("white" if ls == "--" else col), mew=0.9, label=lab)
        ax.set_xticks(range(4)); ax.set_xticklabels(ticks)
        if r == 0: ax.set_title(title)
        if r == 1: ax.set_xlabel(xl)
        else: ax.tick_params(labelbottom=False)
        if c == 0: ax.set_ylabel(f"{sname}\nROC-AUC")
axs[1, 0].set_ylim(48, 86); [axs[1, i].set_ylim(48, 86) for i in range(4)]
[axs[0, i].set_ylim(80, 99) for i in range(4)]
for i in range(1, 4): axs[0, i].tick_params(labelleft=False); axs[1, i].tick_params(labelleft=False)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.04), handlelength=2.2, columnspacing=1.2)
fig.tight_layout(h_pad=0.6, w_pad=0.4)
fig.savefig(OUT / "fig_robust.pdf"); fig.savefig(OUT / "fig_robust_preview.png", dpi=160); plt.close(fig)
print("fig_robust written")
