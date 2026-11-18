"""Check the paper's headline numbers against the repository's data tables and the released per-window predictions.

    python reproduce/verify_paper.py            # Table 6 from predictions + Table 9 from results/revision/stats.json
    python reproduce/verify_paper.py --full     # also recompute stats.json with the seed-aware bootstrap (about 30 min on a CPU)

Section 4 (share of test videos with a same-source training link under random 80/20 splits) is recomputed from
the window manifest and the verified same-source edges. Table 6 (ROC-AUC of every model on the audited test set and the nine unseen sets) is recomputed here from scratch:
per-seed window-level ROC-AUC after the test-side near-duplicate filter, averaged over seeds. Table 9 (the nine
primary claims with Holm correction) is read from stats.json, which src/revision/seed_aware_bootstrap.py writes.
Every expected value below is copied from the submitted manuscript. The script exits with status 1 on any mismatch
larger than the rounding of the printed value.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "revision"))

# ---- Table 6 of the paper: audited test, nine unseen sets, mean and minimum over the nine unseen sets ----------------
COLS = ["Test", "XD", "UCF", "SCFD", "SCVD", "VioPeru", "AIRTLab", "CCTV-F", "Movies", "UBI-Fights"]
TABLE6 = {
    "LENS-Net":       [97.94, 85.11, 77.99, 96.88, 90.87, 83.46, 78.54, 85.32, 74.68, 88.44, 84.59, 74.68],
    "LENS-Net-A":     [98.27, 90.42, 77.67, 96.93, 91.89, 83.68, 82.11, 83.85, 89.88, 92.03, 87.61, 77.67],
    "CLIP probe":     [96.98, 90.07, 72.38, 92.86, 89.44, 79.78, 84.95, 76.14, 97.82, 91.59, 86.11, 72.38],
    "w/o motion":     [97.32, 89.30, 76.80, 94.94, 88.42, 77.91, 82.74, 81.38, 29.62, 90.44, 79.06, 29.62],
    "Motion probe":   [97.53, 79.27, 75.70, 95.24, 88.08, 79.05, 71.25, 83.34, 62.24, 84.18, 79.82, 62.24],
    "CLIP zero-shot": [94.60, 84.03, 69.46, 91.53, 86.25, 72.76, 81.74, 72.63, 84.45, 86.19, 81.01, 69.46],
    "VadCLIP":        [97.65, 88.11, 74.24, 90.63, 90.00, 81.03, 84.23, 78.21, 71.85, 89.36, 83.07, 71.85],
    "X-CLIP FT":      [95.12, 79.73, 61.98, 86.94, 80.72, 68.23, 70.34, 72.93, 69.86, 86.10, 75.20, 61.98],
    "VideoMAE-B FT":  [97.27, 80.77, 73.82, 96.15, 87.87, 78.15, 75.90, 83.19, 49.13, 83.98, 78.77, 49.13],
    "Video Swin-T FT": [97.06, 78.29, 72.77, 92.93, 82.22, 74.37, 73.96, 80.19, 69.48, 83.81, 78.67, 69.48],
    "R(2+1)D-18 FT":  [96.24, 76.84, 69.35, 92.12, 85.07, 72.14, 70.96, 80.49, 64.71, 78.32, 76.67, 64.71],
    "CNN-BiGRU":      [96.10, 76.74, 52.46, 83.60, 78.30, 62.40, 68.84, 68.79, 16.56, 81.02, 65.41, 16.56],
}

# ---- Table 9 of the paper: (stats.json key prefix, diff, ci_low, ci_high, p, Holm threshold, holds) -------------------
TABLE9 = [
    ("H1", 7.34, 4.77, 9.71, 0.0010, 0.0071, True),
    ("H2", 1.53, 0.13, 2.85, 0.0290, 0.0250, False),
    ("H3", 5.81, 3.36, 8.04, 0.0010, 0.0083, True),
    ("H4", 1.94, -0.13, 4.11, 0.0330, 0.0500, False),
    ("H5", 2.45, 0.19, 4.54, 0.0160, 0.0125, False),
    ("K1", 2.61, 1.42, 3.88, 0.0005, 0.0056, True),
    ("K2", 2.84, 1.17, 4.60, 0.0005, 0.0063, True),
    ("K3", 0.45, -0.91, 1.83, 0.0200, 0.0167, False),
    ("K4", 2.97, 0.71, 5.35, 0.0050, 0.0100, True),
]

# ---- Section 4 / Fig. 3: share of test videos with a same-source training link under random 80/20 splits ---------
OVERLAP = {"RLVS": 24.4, "HockeyFights": 48.2, "RWF2000": 55.6}

failures, checks = [], 0


def check(label, got, want, decimals):
    global checks
    checks += 1
    if abs(got - want) > 0.5 * 10 ** -decimals + 1e-9:
        failures.append(f"{label}: computed {got:.{decimals + 2}f}, paper {want:.{decimals}f}")


def verify_table6():
    from saved_predictions import Data
    D = Data()
    for model, row in TABLE6.items():
        per_set = []
        for s in COLS:
            S = D.sets[s]
            keep, y = S["keep"], S["y"]
            aucs = [100 * roc_auc_score(y[keep], p[keep]) for p in D.probs(model, s)]
            per_set.append(float(np.mean(aucs)))
        unseen = per_set[1:]
        got = per_set + [float(np.mean(unseen)), float(np.min(unseen))]
        for name, g, w in zip(COLS + ["Mean", "Min"], got, row):
            check(f"Table 6 {model} / {name}", g, w, 2)
        print(f"  {model:16s} " + " ".join(f"{g:6.2f}" for g in got))


def verify_table9(stats_path):
    primary = json.load(open(stats_path))["primary"]
    for key, diff, lo, hi, p, thr, holds in TABLE9:
        match = [k for k in primary if k.startswith(key + " ")]
        if len(match) != 1:
            failures.append(f"Table 9 {key}: not found in {stats_path}"); continue
        r = primary[match[0]]
        check(f"Table 9 {key} diff", r["diff"], diff, 2)
        check(f"Table 9 {key} CI low", r["ci"][0], lo, 2)
        check(f"Table 9 {key} CI high", r["ci"][1], hi, 2)
        check(f"Table 9 {key} p", r["p"], p, 4)
        check(f"Table 9 {key} Holm threshold", r["holm_threshold"], thr, 4)
        global checks
        checks += 1
        if bool(r["holds_after_holm"]) != holds:
            failures.append(f"Table 9 {key}: holds_after_holm={r['holds_after_holm']}, paper says {holds}")
        print(f"  {match[0][:52]:52s} {r['diff']:+6.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]  p={r['p']:.4f}  "
              f"{'holds' if r['holds_after_holm'] else 'fails'}")


def verify_overlap():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(REPO / "src/audit/audit_public_splits.py")], check=True,
                       env={**os.environ, "AUDIT_OUT": tmp}, stdout=subprocess.DEVNULL)
        res = json.load(open(Path(tmp) / "public_split_audit.json"))["C_random_80_20"]
    for ds, want in OVERLAP.items():
        got = res[ds]["test_videos_linked_pct"]["median"]
        check(f"Section 4 overlap {ds}", got, want, 1)
        print(f"  {ds:13s} {got:5.1f} % of test videos linked to training footage (paper {want} %)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="recompute stats.json with the seed-aware bootstrap first")
    a = ap.parse_args()
    root = Path(os.environ.get("LENSNET_ROOT") or REPO)
    print("Section 4: same-source overlap under random 80/20 splits (recomputed from data_tables/)")
    verify_overlap()
    if not (root / "workspace" / "runs").exists():
        print(f"\n{checks - len(failures)}/{checks} values match so far.")
        sys.exit(f"No predictions under {root / 'workspace'}. Run: bash reproduce/get_predictions.sh")

    print("\nTable 6: ROC-AUC (%) recomputed from the per-window predictions")
    print("  " + " " * 16 + " ".join(f"{c[:6]:>6s}" for c in COLS + ["Mean", "Min"]))
    verify_table6()

    stats = root / "workspace" / "review" / "stats.json"
    if a.full:
        print("\nRecomputing stats.json with the seed-aware cluster bootstrap (2,000 resamples) ...")
        subprocess.run([sys.executable, str(REPO / "src/revision/seed_aware_bootstrap.py")], check=True)
    print(f"\nTable 9: primary claims with Holm correction ({stats.relative_to(root)})")
    verify_table9(stats)

    print(f"\n{checks - len(failures)}/{checks} values match the manuscript.")
    if failures:
        print("MISMATCHES:\n  " + "\n  ".join(failures)); sys.exit(1)
    print("All checked values reproduce.")


if __name__ == "__main__":
    main()
