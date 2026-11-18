#!/usr/bin/env python3
"""Scientific figures for the LENS-Net papers, drawn only from saved results (vector PDF, IEEE sizes).
Palette validated with the dataviz validator: LENS-Net blue #2a78d6, VideoMAE-B orange #eb6834; other models in
neutral greys with distinct markers (identity never carried by colour alone; direct labels where possible)."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
L = Path(__file__).resolve().parents[1] / "workspace"; OUT = Path(__file__).parent / "figs"; ROOT = L.parent
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
GREY1, GREY2 = "#7d7b76", "#a8a6a0"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL1, COL2 = 3.5, 7.16
A2 = json.load(open(L / "logs/A2_results.json"))
MODELS = [("VideoMAE-B FT", "A2_videomae", ORANGE, "s"), ("R(2+1)D-18 FT", "A2_r2plus1d_18", GREY1, "^"),
          ("Video Swin-T FT", "A2_swin3d_t", GREY2, "D"), ("LENS-Net (ours)", "A2_LENS-LENS-M_vrex", BLUE, "o")]
TGT = [("Hockey", "HockeyFights"), ("RLVS", "RLVS"), ("Bus Violence", "BusViolence"), ("VID", "VID"), ("XD-Violence", "XD-test"), ("UCF-Crime", "UCF-test")]


def runs(prefix): return [v for k, v in A2.items() if isinstance(v, dict) and re.fullmatch(prefix + r"_seed4[234]", k)]
def style(ax, grid_axis="x"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5); ax.set_axisbelow(True)


# ---- Fig: transfer to unseen corpora (P3), dot plot with seed spread
def fig_transfer():
    fig, ax = plt.subplots(figsize=(COL1, 2.9))
    rows = [t[0] for t in TGT] + ["Mean"]
    y = np.arange(len(rows))[::-1]
    off = np.linspace(-0.27, 0.27, len(MODELS))
    for (lab, pre, c, mk), dy in zip(MODELS, off):
        R = runs(pre)
        if not R: continue
        vals = np.array([[r[k] for _, k in TGT] + [np.mean([r[k] for _, k in TGT])] for r in R])
        m, lo, hi = vals.mean(0), vals.min(0), vals.max(0)
        ax.hlines(y + dy, lo, hi, color=c, lw=1.0, zorder=2)
        ax.scatter(m, y + dy, s=22 if "LENS" in lab else 16, marker=mk, color=c, edgecolor="white", linewidth=0.6, zorder=3,
                   label=f"{lab.replace(' (ours)', ', ours')} ({len(R)} seed{'s' if len(R) > 1 else ''})")
    ax.axhline(0.5, color=INK2, lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(rows); ax.set_xlabel("ROC-AUC on the unseen corpus (%)")
    ax.set_xlim(66, 100.5); style(ax)
    h, l = ax.get_legend_handles_labels(); order = [3, 0, 1, 2]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, frameon=False,
              handletextpad=0.3, columnspacing=1.0)
    fig.savefig(OUT / "fig_transfer.pdf"); plt.close(fig)


# ---- Fig: in-domain accuracy vs transfer (P3); annotate trainable parameters
PARAMS = {"A2_videomae": "86M", "A2_r2plus1d_18": "31.5M", "A2_swin3d_t": "28M", "A2_LENS-LENS-M_vrex": "32k",
          "A2_LENS-V2b": "5.7k", "A2_LENS-LP": "1.5k", "A2_LENS-MP": "28k"}
def fig_tradeoff():
    pts = MODELS + [("CLIP probe", "A2_LENS-LP", GREY2, "v"), ("Motion probe", "A2_LENS-MP", GREY2, "<"), ("LENS-Net w/o motion", "A2_LENS-V2b", GREY1, "P")]
    fig, ax = plt.subplots(figsize=(COL1, 2.45))
    offs = {"VideoMAE-B FT": (0, -22, "center"), "R(2+1)D-18 FT": (6, -12, "left"), "Video Swin-T FT": (6, -4, "left"), "LENS-Net (ours)": (-7, 6, "right"),
            "CLIP probe": (0, -21, "center"), "Motion probe": (-7, -3, "right"), "LENS-Net w/o motion": (-7, 3, "right")}
    for lab, pre, c, mk in pts:
        R = runs(pre)
        if not R: continue
        x = np.array([r["rwf_official_video_acc"] for r in R]); yv = np.array([np.mean([r[k] for _, k in TGT]) for r in R])
        ax.errorbar(x.mean(), yv.mean(), xerr=x.std(ddof=1) if len(R) > 1 else None, yerr=yv.std(ddof=1) if len(R) > 1 else None,
                    fmt=mk, ms=6.5 if "ours" in lab else 5, color=c, mec="white", mew=0.6, ecolor=c, elinewidth=0.8, capsize=0, zorder=3)
        dx, dy, ha = offs[lab]
        ax.annotate(f"{lab}\n({PARAMS[pre]} trained)", (x.mean(), yv.mean()), textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=6.5, color=INK if "ours" in lab else INK2, fontweight="bold" if "ours" in lab else "normal", linespacing=0.95)
    ax.set_xlabel("In-domain accuracy, RWF-2000 official test (%)"); ax.set_ylabel("Mean transfer ROC-AUC (%)")
    style(ax, "both"); ax.margins(x=0.25, y=0.18)
    fig.savefig(OUT / "fig_tradeoff.pdf"); plt.close(fig)


# ---- Fig: leakage in standard protocols (share of test videos linked to training footage)
def fig_leak():
    pa = json.load(open(ROOT / "results/public_split_audit.json"))["C_random_80_20"]
    std = json.load(open(L / "splits/standard_splits_report.json"))
    k5 = lambda s: np.mean([v["linked_pct"] for k, v in std.items() if k.startswith(s)])
    bars = [("RWF-2000, random 80/20", pa["RWF2000"]["test_videos_linked_pct"]["median"]),
            ("Hockey, random 80/20", pa["HockeyFights"]["test_videos_linked_pct"]["median"]),
            ("Hockey, 5-fold CV", k5("std_hockey")), ("RLVS, random 80/20", pa["RLVS"]["test_videos_linked_pct"]["median"]),
            ("RLVS, 5-fold CV", k5("std_rlvs")), ("RWF-2000, official split", std["std_rwf"]["linked_pct"]), ("Our audited split", 0.0)]
    fig, ax = plt.subplots(figsize=(COL1, 1.85))
    y = np.arange(len(bars))[::-1]
    for yi, (lab, v) in zip(y, bars):
        ax.barh(yi, v, height=0.62, color=BLUE if lab != "Our audited split" else GREY1, zorder=2)
        ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([b[0] for b in bars]); ax.set_xlim(0, 66)
    ax.set_xlabel("Test videos with a verified same-source link to training (%)"); style(ax)
    fig.savefig(OUT / "fig_leak.pdf"); plt.close(fig)


# ---- Fig (journal): PPS evidence over time, violent vs non-violent windows, per evaluation set
def pps_signal(S):
    """S: (N,16,17,10) prompt similarities; prompts 0-4 non-violent, 5-9 violent. Evidence per frame = max over grid
    cells of (mean violent-prompt sim - mean non-violent-prompt sim)."""
    d = S[:, :, 1:, 5:].mean(-1) - S[:, :, 1:, :5].mean(-1)
    return d.max(-1)
def fig_pps():
    sets = []
    ex = np.load(str(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[1]) / "workspace/features/l14/external/compact.npz"))["S"].astype(np.float32)
    ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
    pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
    keep = (pool.excluded_precaution == 0).values
    for b, name in [("XD-test", "XD-Violence"), ("UCF-test", "UCF-Crime")]:
        m = keep & (pool.benchmark == b).values; sets.append((name, pps_signal(ex[m]), pool.binary_label.values[m]))
    fr = np.load(str(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[1]) / "workspace/features/l14/fresh/compact.npz"))["S"].astype(np.float32)
    fx = pd.read_csv(L / "fresh/near_duplicate_exclusion.csv"); kf = fx.excluded_near_duplicate.values == 0
    for b in ["SCFD", "SCVD"]:
        m = kf & (fx.benchmark == b).values; sets.append((f"{b} (fresh)", pps_signal(fr[m]), fx.binary_label.values[m]))
    fig, axs = plt.subplots(2, 2, figsize=(COL1, 2.9), sharey=True, sharex=True, gridspec_kw=dict(hspace=0.55, wspace=0.12)); axs = axs.ravel()   # single column
    t = np.arange(1, 17)
    for ax, (name, sig, y) in zip(axs, sets):
        for lab, c, txt in [(1, BLUE, "violent"), (0, ORANGE, "non-violent")]:
            s = sig[y == lab]; mu = s.mean(0); ci = 1.96 * s.std(0, ddof=1) / np.sqrt(len(s))
            ax.fill_between(t, mu - ci, mu + ci, color=c, alpha=0.18, lw=0); ax.plot(t, mu, color=c, lw=1.4, label=f"{txt} windows (mean, 95% CI)")
        ax.set_title(f"{name} ($n$ = {(y == 1).sum()} / {(y == 0).sum()})", color=INK, fontsize=7); ax.set_xticks([1, 8, 16]); style(ax, "y")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    for ax in axs[2:]: ax.set_xlabel("frame")
    fig.text(0.0, 0.5, "PPS violence evidence (max over grid)", rotation=90, va="center", ha="left", fontsize=7.5, color=INK)
    fig.savefig(OUT / "fig_pps.pdf", dpi=400); fig.savefig(OUT / "fig_pps.png", dpi=400); plt.close(fig)


# ---- Fig (journal): which stream drives the decision (mean |logit-difference contribution| per stream)
def fig_streams():
    import sys, torch
    sys.path[:0] = [str(L.parent / "src" / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]; import lensnet_setup as LS, lensnet as V
    from sklearn.metrics import roc_auc_score
    sets = {}
    b = V.Compact("bench"); man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["split", "dataset"])
    for d in ["RLVS", "RWF2000", "HockeyFights"]:
        sets[{"HockeyFights": "Hockey", "RWF2000": "RWF-2000"}.get(d, d)] = (b, np.where((man.split == "test") & (man.dataset == d))[0])
    ex = V.Compact("external"); ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
    pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
    for bb, n in [("XD-test", "XD-Violence"), ("UCF-test", "UCF-Crime")]: sets[n] = (ex, np.where((pool.excluded_precaution == 0) & (pool.benchmark == bb))[0])
    fr = V.Compact("fresh"); fx = pd.read_csv(L / "fresh/near_duplicate_exclusion.csv")
    for bb in ["SCFD", "SCVD"]: sets[f"{bb} (fresh)"] = (fr, np.where((fx.excluded_near_duplicate == 0) & (fx.benchmark == bb))[0])
    comp = {k: [] for k in sets}
    for s in (42, 43, 44):
        m = V.V2("LENS-M", LS.text_anchors()); m.load_state_dict(torch.load(L / f"runs/v2_audited_LENS-M_vrex_seed{s}/best.pt")); m.eval()
        with torch.no_grad():
            for k, (F, rows) in sets.items():
                cls, S, M = F.get(torch.as_tensor(rows))
                a = LS.SCALE * torch.nn.functional.normalize(cls, dim=-1) @ torch.nn.functional.normalize(m.w, dim=-1).t()
                p = m.g * m.head(V.pps_features(S, "V2b")); mo = m.gm * m.mhead(M)
                diff = lambda z: (z[:, 1] - z[:, 0]).abs().mean().item()
                comp[k].append([diff(a), diff(p), diff(mo)])
    names = list(sets); vals = np.array([np.mean(comp[k], 0) for k in names])
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    y = np.arange(len(names))[::-1]; h = 0.26
    for j, (lab, c) in enumerate([("language anchor $\\ell^A$", GREY1), ("PPS stream $g_P\\delta^P$", BLUE), ("motion stream $g_M\\delta^M$", ORANGE)]):
        ax.barh(y + (1 - j) * h, vals[:, j], height=h - 0.03, color=c, label=lab, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlabel("mean |logit-difference contribution|"); style(ax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, fontsize=6.5, handlelength=1.2, columnspacing=0.8)
    fig.savefig(OUT / "fig_streams.pdf"); plt.close(fig)
    json.dump({k: dict(zip(["anchor", "pps", "motion"], map(float, v))) for k, v in zip(names, vals)}, open(OUT / "fig_streams_values.json", "w"), indent=2)


if __name__ == "__main__":
    for f in (fig_transfer, fig_tradeoff, fig_leak, fig_pps, fig_streams):
        f(); print("ok", f.__name__)


# ================= heatmap and parameter figures =================
def fig_heatmap():
    """Cross-dataset heatmap (P3): models x unseen corpora, cell = mean ROC-AUC over seeds; best per column bold.
    Single-hue sequential scale (light -> dark blue); values printed in every cell, so colour is never needed alone."""
    rows = [("CLIP probe", "A2_LENS-LP"), ("Motion probe", "A2_LENS-MP"), ("Video Swin-T FT", "A2_swin3d_t"),
            ("R(2+1)D-18 FT", "A2_r2plus1d_18"), ("VideoMAE-B FT", "A2_videomae"), ("LENS-Net without motion", "A2_LENS-V2b"), ("LENS-Net (ours)", "A2_LENS-LENS-M_vrex")]
    cols = [t[0] for t in TGT] + ["Mean"]
    M, n = [], []
    for lab, pre in rows:
        R = runs(pre); v = np.array([[r[k] for _, k in TGT] for r in R]).mean(0); M.append(list(v) + [v.mean()]); n.append(len(R))
    M = np.array(M)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", ["#f3f7fd", "#cfe0f6", "#8fb8ea", "#2a78d6", "#154a8a"])
    fig, ax = plt.subplots(figsize=(COL1, 2.35))
    lo, hi = 68, 100
    ax.imshow(np.clip((M - lo) / (hi - lo), 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    best = M.argmax(0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; dark = (v - lo) / (hi - lo) > 0.62
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white" if dark else INK,
                    fontweight="bold" if best[j] == i else "normal")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{l.replace(' (ours)', ', ours')} ({k})" for (l, _), k in zip(rows, n)])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.axvline(len(cols) - 1.5, color="white", lw=2); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlabel("unseen corpus (trained on RWF-2000 only)", labelpad=2)
    fig.savefig(OUT / "fig_heatmap.pdf"); plt.close(fig)


def fig_params():
    """Trained parameters (log) vs mean transfer ROC-AUC (P3), five key models, short labels."""
    pts = [("VideoMAE-B FT", "A2_videomae", 86.2e6, ORANGE, "s"), ("R(2+1)D-18 FT", "A2_r2plus1d_18", 31.3e6, INK2, "^"),
           ("Video Swin-T FT", "A2_swin3d_t", 27.9e6, GREY2, "D"), ("CLIP probe", "A2_LENS-LP", 1536, GREY2, "v"),
           ("LENS-Net (ours)", "A2_LENS-LENS-M_vrex", 31926, BLUE, "o")]
    fig, ax = plt.subplots(figsize=(COL1, 1.95))
    for lab, pre, p, c, mk in pts:
        R = runs(pre); y = np.array([np.mean([r[k] for _, k in TGT]) for r in R])
        ax.errorbar(p, y.mean(), yerr=y.std(ddof=1) if len(R) > 1 else None, fmt=mk, ms=7 if "ours" in lab else 5.5, color=c, mec="white", mew=0.7,
                    ecolor=c, elinewidth=0.9, capsize=0, zorder=4 if lab.startswith("R(2+1)D") else 3)
        ha, dx, dy = {"R(2+1)D-18 FT": ("left", 9, 5), "Video Swin-T FT": ("right", -9, -9)}.get(
            lab, ("left", 7, -3) if p < 1e6 else ("right", -7, -3))            # R(2+1)D and Swin sit almost on top of each other
        ax.annotate(lab, (p, y.mean()), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=7,
                    color=INK if "ours" in lab else INK2, fontweight="bold" if "ours" in lab else "normal")
    ax.set_xscale("log"); ax.set_xlim(5e2, 4e8)
    ax.set_xlabel("trained parameters (log scale)"); ax.set_ylabel("mean transfer\nROC-AUC (%)")
    style(ax, "both"); ax.margins(y=0.2)
    fig.savefig(OUT / "fig_params.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_heatmap(); fig_params(); print("ok heatmap/params figs")
