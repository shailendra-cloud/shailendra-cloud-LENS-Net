#!/usr/bin/env python3
"""Generate every number and table in the LENS-Net papers from result JSON files (no hand-typed numbers).
Outputs paper/numbers.tex (macros) and paper/tables/*.tex. Missing results render as \\pend."""
import json, re
from pathlib import Path
import numpy as np
L = Path(__file__).resolve().parents[1] / "workspace"; OUT = Path(__file__).parent
J = lambda f: json.load(open(L / "logs" / f)) if (L / "logs" / f).exists() else {}
F, A2, FR = J("stageF_results.json"), J("A2_results.json"), J("fresh_results.json")
M = {}
def fmt(x, d=2): return "\\pend" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"
def ms(d):
    if not d: return "\\pend"
    single = d.get("n", len(d.get("per_seed", [0, 0]))) == 1 or d.get("sd", 0) == 0
    return f"{d['mean']:.2f}" + ("$^\\dagger$" if single else f"{{\\scriptsize$\\pm${d['sd']:.2f}}}")
def ci(x): return "\\pend" if not x else f"{sg(x['diff'])} [{sg(x['ci'][0])}, {sg(x['ci'][1])}]{'$^*$' if x['excludes_zero'] else ''}"
def mac(name, val): M[name] = val
def sg(v, d=2):
    t = f"{v:+.{d}f}"
    return t.replace("-", "$-$") if v < 0 else t

# ---------------- main table: audited test, XD, UCF, fresh SCFD/SCVD
fr = FR.get("models", {})
rows = [("CNN-BiGRU + motion", "MAST-Net", "MAST-Net"), ("CLIP zero-shot", "ZS", "ZS"), ("CLIP probe (anchored)", "LP", "v2_audited_LP"),
        ("Motion probe (VideoMAE feat.)", "MP", "v2_audited_MP"), ("LENS-Net w/o motion", "LENS(V2b)", "v2_audited_V2b"),
        ("\\textbf{LENS-Net (ours)}", "LENS-M", "v2_audited_LENS-M_vrex")]
base = [("R(2+1)D-18 FT~\\cite{tran2018r21d}", "A1_r2plus1d_18"), ("Video Swin-T FT~\\cite{liu2022videoswin}", "A1_swin3d_t"), ("VideoMAE-B FT~\\cite{tong2022videomae}", "A1_videomae")]
t = F.get("F1_F3_table", {})
lines = []
for lab, k, fk in rows:
    v = t.get(k, {}); f = fr.get(fk, {}) if fk else {}
    lines.append(f"{lab} & {ms(v.get('test_auc'))} & {ms(v.get('xd_auc'))} & {ms(v.get('ucf_auc'))} & {ms(f.get('SCFD'))} & {ms(f.get('SCVD'))} \\\\")
lines.append("\\midrule")
for lab, k in base:   # fine-tuned baselines: audited/external/fresh come from their own runs (filled when available)
    b = J("A1_results.json").get(k, {})
    f = fr.get(k, {})
    lines.append(f"{lab} & {ms(b.get('test_auc'))} & {ms(b.get('xd_auc'))} & {ms(b.get('ucf_auc'))} & {ms(f.get('SCFD'))} & {ms(f.get('SCVD'))} \\\\")
(OUT / "tables/main.tex").write_text("\n".join(lines) + "\n")

# ---------------- A2 single-source (train RWF official)
cols = ["rwf_official_video_acc", "HockeyFights", "RLVS", "BusViolence", "VID", "XD-test", "UCF-test"]
def agg(prefix):
    runs = [v for k, v in A2.items() if isinstance(v, dict) and re.fullmatch(prefix + r"_seed4[234]", k)]
    if not runs: return None, 0
    out = {c: float(np.mean([r[c] for r in runs])) for c in cols}
    out["mean_transfer"] = float(np.mean([out[c] for c in cols[1:]])); return out, len(runs)
a2rows = [("R(2+1)D-18 FT", "A2_r2plus1d_18"), ("Video Swin-T FT", "A2_swin3d_t"), ("VideoMAE-B FT", "A2_videomae"),
          ("CLIP probe", "A2_LENS-LP"), ("Motion probe", "A2_LENS-MP"), ("LENS-Net w/o motion", "A2_LENS-V2b"), ("\\textbf{LENS-Net (ours)}", "A2_LENS-LENS-M_vrex")]
lines = []
for lab, p in a2rows:
    o, n = agg(p)
    lines.append(f"{lab} & {n} & " + " & ".join(fmt(o[c]) if o else "\\pend" for c in cols + ["mean_transfer"]) + " \\\\")
    if p == "A2_LENS-LENS-M_vrex" and o: mac("AtwoLensMean", fmt(o["mean_transfer"])); mac("AtwoLensRWF", fmt(o["rwf_official_video_acc"]))
    if p == "A2_videomae" and o: mac("AtwoVmaeMean", fmt(o["mean_transfer"])); mac("AtwoVmaeRWF", fmt(o["rwf_official_video_acc"]))
(OUT / "tables/a2.tex").write_text("\n".join(lines) + "\n")

# ---------------- LODO
lines = []
for d, v in F.get("F2_LODO", {}).items():
    nm = {"HockeyFights": "Hockey", "RWF2000": "RWF-2000", "BusViolence": "Bus Violence"}.get(d, d)
    lines.append(f"{nm} & {v['n']:,} & {fmt(v['MAST-Net(seed42)'])}$^\\dagger$ & {ms(v.get('ZS'))} & {ms(v.get('LP'))} & {ms(v.get('MP'))} & {ms(v.get('LENS(V2b)'))} & {ms(v.get('LENS-M'))} \\\\")
(OUT / "tables/lodo.tex").write_text("\n".join(lines) + "\n")

# ---------------- ablations (audited test / XD / UCF / fresh)
abl = [("LENS-Net (full)", "LENS-M", "v2_audited_LENS-M_vrex"), ("$-$ motion stream", "LENS(V2b)", "v2_audited_V2b"),
       ("$-$ motion, $-$ V-REx", "V2a", "v2_audited_V2a"), ("$-$ motion, $-$ gate", "-G", "v2_audited_-G_vrex"),
       ("$-$ motion, temporal mean only", "-T", "v2_audited_-T_vrex"), ("$-$ motion, CLS only (no grid)", "-Sp", "v2_audited_-Sp_vrex"),
       ("$-$ motion, $-$ PPS (anchored probe)", "LP", "v2_audited_LP"), ("motion stream only", "MP", "v2_audited_MP"),
       ("large temporal-transformer head (v1)", "v1", None)]
lines = [f"{lab} & {ms(t.get(k, {}).get('test_auc'))} & {ms(t.get(k, {}).get('xd_auc'))} & {ms(t.get(k, {}).get('ucf_auc'))} & {ms(fr.get(fk, {}).get('SCFD')) if fk else '--'} & {ms(fr.get(fk, {}).get('SCVD')) if fk else '--'} \\\\" for lab, k, fk in abl]
(OUT / "tables/ablation.tex").write_text("\n".join(lines) + "\n")
mo = lambda d: "\\pend" if not d else f"{d['mean']:.2f}"
lines = [f"{lab} & {mo(t.get(k, {}).get('test_auc'))} & {mo(t.get(k, {}).get('xd_auc'))} & {mo(t.get(k, {}).get('ucf_auc'))} & {mo(fr.get(fk, {}).get('SCFD')) if fk else '--'} & {mo(fr.get(fk, {}).get('SCVD')) if fk else '--'} \\\\" for lab, k, fk in abl]
(OUT / "tables/ablation_means.tex").write_text("\n".join(lines) + "\n")

# ---------------- standard protocols
lines = []
for p, v in F.get("F4_standard", {}).items():
    lines.append(f"{p} & {v['test_videos_linked_to_train_pct']:.1f} & {ms(v.get('LP', {}).get('video_accuracy'))} & {ms(v.get('V2b', {}).get('video_accuracy'))} & {ms(v.get('LENS-M_vrex', {}).get('video_accuracy'))} \\\\")
(OUT / "tables/standard.tex").write_text("\n".join(lines) + "\n")

# ---------------- paired CIs (key claims)
P1, PF = F.get("F1_F3_paired", {}), FR.get("paired", {})
lines = []
A1P = J("A1_results.json").get("paired_vs_LENS-Net", {})
for lab, src, key, cols_ in [("LENS-Net $-$ VideoMAE-B FT", A1P, "A1_videomae", ["test", "xd", "ucf"]),
                             ("LENS-Net $-$ Video Swin-T FT", A1P, "A1_swin3d_t", ["test", "xd", "ucf"]),
                             ("LENS-Net $-$ R(2+1)D-18 FT", A1P, "A1_r2plus1d_18", ["test", "xd", "ucf"]),
                             ("LENS-Net $-$ CNN-BiGRU", P1, "LENS-M - MAST-Net", ["test", "XD-test", "UCF-test"]),
                             ("LENS-Net $-$ w/o motion", P1, "LENS-M - LENS(V2b)", ["test", "XD-test", "UCF-test"]),
                             ("LENS-Net $-$ motion probe", P1, "LENS-M - MP", ["test", "XD-test", "UCF-test"])]:
    v = src.get(key, {}); lines.append(f"{lab} & " + " & ".join(ci(v.get(c)) for c in cols_) + " \\\\")
(OUT / "tables/paired.tex").write_text("\n".join(lines) + "\n")
lines = []
for lab, key in [("LENS-Net $-$ w/o motion (pooled training)", "v2_audited_LENS-M_vrex - v2_audited_V2b"),
                 ("LENS-Net $-$ w/o motion (RWF-only training)", "v2_std_std_rwf_LENS-M_vrex - v2_std_std_rwf_V2b"),
                 ("LENS-Net $-$ CNN-BiGRU", "v2_audited_LENS-M_vrex - MAST-Net"),
                 ("LENS-Net $-$ VideoMAE-B FT (pooled)", "v2_audited_LENS-M_vrex - A1_videomae"),
                 ("LENS-Net $-$ VideoMAE-B FT (RWF-only)", "v2_std_std_rwf_LENS-M_vrex - A2_videomae")]:
    v = PF.get(key, {}); lines.append(f"{lab} & {ci(v.get('SCFD'))} & {ci(v.get('SCVD'))} \\\\")
(OUT / "tables/paired_fresh.tex").write_text("\n".join(lines) + "\n")

# ---------------- macros used in text
tt = lambda k, m: t.get(k, {}).get(m, {}).get("mean")
for k, n in [("LENS-M", "Lens"), ("LENS(V2b)", "NoMot"), ("MAST-Net", "Mast"), ("LP", "Probe"), ("MP", "Mprobe")]:
    mac(f"{n}Test", fmt(tt(k, "test_auc"))); mac(f"{n}XD", fmt(tt(k, "xd_auc"))); mac(f"{n}UCF", fmt(tt(k, "ucf_auc")))
for k, n in [("v2_audited_LENS-M_vrex", "Lens"), ("v2_audited_V2b", "NoMot")]:
    mac(f"{n}SCFD", fmt(fr.get(k, {}).get("SCFD", {}).get("mean"))); mac(f"{n}SCVD", fmt(fr.get(k, {}).get("SCVD", {}).get("mean")))
for key, n in [("LENS-M - MAST-Net", "LmM"), ("LENS-M - LENS(V2b)", "LmN")]:
    for c, cn in [("test", "Test"), ("XD-test", "XD"), ("UCF-test", "UCF")]:
        x = P1.get(key, {}).get(c); mac(f"{n}{cn}", fmt(x["diff"]) if x else "\\pend")
x = P1.get("LENS-M - LENS(V2b)", {}).get("XD-test"); mac("LmNXDabs", fmt(abs(x["diff"])) if x else "\\pend")
for key, n in [("v2_audited_LENS-M_vrex - v2_audited_V2b", "FrLmN")]:
    for b in ["SCFD", "SCVD"]:
        x = PF.get(key, {}).get(b); mac(f"{n}{b}", fmt(x["diff"]) if x else "\\pend")
mac("FreshWinSCFD", str(FR.get("n_windows", {}).get("SCFD", "\\pend"))); mac("FreshWinSCVD", str(FR.get("n_windows", {}).get("SCVD", "\\pend")))
mac("FreshVidSCFD", str(FR.get("n_videos", {}).get("SCFD", "\\pend"))); mac("FreshVidSCVD", str(FR.get("n_videos", {}).get("SCVD", "\\pend")))
for p, n in [("Hockey 5-fold", "LeakHockey"), ("RLVS 5-fold", "LeakRLVS"), ("RWF-2000 official", "LeakRWF")]:
    v = F.get("F4_standard", {}).get(p); mac(n, f"{v['test_videos_linked_to_train_pct']:.1f}" if v else "\\pend")
for d, n in [("RWF2000", "LodoRWF"), ("BusViolence", "LodoBus"), ("VID", "LodoVID")]:
    v = F.get("F2_LODO", {}).get(d, {}); mac(f"{n}Lens", fmt(v.get("LENS-M", {}).get("mean"))); mac(f"{n}Mast", fmt(v.get("MAST-Net(seed42)")))
(OUT / "numbers.tex").write_text("% generated by make_tables.py -- do not edit\n\\providecommand{\\pend}{\\textcolor{red}{\\textbf{[pending]}}}\n" +
                                  "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(M.items())))
print(len(M), "macros;", sum(v == "\\pend" for v in M.values()), "pending;", "tables:", [p.name for p in (OUT / "tables").glob("*.tex")])

# ---------------- dataset / leakage counts (appended)
import pandas as pd
ROOT = L.parent
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["split", "dataset", "video_id"])
for s in ["train", "val", "test"]: M[f"N{s.capitalize()}"] = f"{(man.split == s).sum():,}"
M["NWin"] = f"{len(man):,}"; M["NVid"] = f"{man.video_id.nunique():,}"
ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0) & (ext.excluded_precaution == 0)]
for b, n in [("XD-test", "XD"), ("UCF-test", "UCF")]:
    g = pool[pool.benchmark == b]; M[f"NWin{n}"] = f"{len(g):,}"; M[f"NVid{n}"] = f"{g.video.nunique():,}"
pa = json.load(open(ROOT / "results/public_split_audit.json"))["C_random_80_20"]
for d, n in [("RLVS", "RandLeakRLVS"), ("HockeyFights", "RandLeakHockey"), ("RWF2000", "RandLeakRWF")]:
    M[n] = f"{pa[d]['test_videos_linked_pct']['median']:.1f}"
fx = pd.read_csv(L / "fresh/near_duplicate_exclusion.csv")
for b in ["SCFD", "SCVD"]:
    g = fx[fx.benchmark == b]; M[f"FreshExcl{b}"] = str(int(g.excluded_near_duplicate.sum())); M[f"FreshAll{b}"] = str(len(g))
a1 = J("A1_results.json")
for k, n in [("A1_videomae", "VmaeAone"), ("A1_r2plus1d_18", "RtdAone"), ("A1_swin3d_t", "SwinAone")]:
    for m in ["test", "xd", "ucf"]:
        v = a1.get(k, {}).get(f"{m}_auc"); M[f"{n}{m.capitalize()}"] = fmt(v["mean"]) if v else "\\pend"
for k, n in [("A1_videomae", "LmV"), ("A1_r2plus1d_18", "LmR"), ("A1_swin3d_t", "LmS")]:
    for m, mn in [("test", "Test"), ("xd", "XD"), ("ucf", "UCF")]:
        x = A1P.get(k, {}).get(m); M[f"{n}{mn}"] = fmt(x["diff"]) if x else "\\pend"
        M[f"{n}{mn}Sig"] = ("yes" if x["excludes_zero"] else "no") if x else "\\pend"
M["VmaeAoneSeeds"] = str(len(J("A1_results.json").get("A1_videomae", {}).get("per_seed", {})) or "\\pend")
(OUT / "numbers.tex").write_text("% generated by make_tables.py -- do not edit\n\\providecommand{\\pend}{\\textcolor{red}{\\textbf{[pending]}}}\n" +
                                  "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(M.items())))
print("total macros", len(M), "pending", [k for k, v in M.items() if v == "\\pend"])

# ---------------- fresh-set comparisons vs baselines (appended)
PF = FR.get("paired", {})
for key, n in [("v2_audited_LENS-M_vrex - A1_videomae", "FrV"), ("v2_audited_LENS-M_vrex - A1_r2plus1d_18", "FrR"),
               ("v2_audited_LENS-M_vrex - A1_swin3d_t", "FrS"), ("v2_audited_LENS-M_vrex - MAST-Net", "FrC"),
               ("v2_std_std_rwf_LENS-M_vrex - A2_videomae", "FrVtwo"), ("v2_std_std_rwf_LENS-M_vrex - A2_r2plus1d_18", "FrRtwo")]:
    for b in ["SCFD", "SCVD"]:
        x = PF.get(key, {}).get(b); M[f"{n}{b}"] = (sg(x["diff"]) + ("$^*$" if x["excludes_zero"] else "")) if x else "\\pend"
for k, n in [("A1_videomae", "VmaeFr"), ("A1_r2plus1d_18", "RtdFr"), ("MAST-Net", "MastFr")]:
    for b in ["SCFD", "SCVD"]: M[f"{n}{b}"] = fmt(fr.get(k, {}).get(b, {}).get("mean"))
(OUT / "numbers.tex").write_text("% generated by make_tables.py -- do not edit\n\\providecommand{\\pend}{\\textcolor{red}{\\textbf{[pending]}}}\n" +
                                  "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(M.items())))
print("fresh macros added; pending:", [k for k, v in M.items() if v == "\\pend"])

# ---------------- P3 mean-transfer significance and efficiency (appended)
AB = J("a2_mean_boot.json")
for k, n in [("A2_videomae", "PmVmae"), ("A2_r2plus1d_18", "PmRtd"), ("A2_swin3d_t", "PmSwin")]:
    x = AB.get(k); M[n] = (f"{sg(x['diff'])} [{sg(x['ci'][0])}, {sg(x['ci'][1])}]" + ("$^*$" if x["excludes_zero"] else "")) if x else "\\pend"
EF = J("efficiency.json")
if EF:
    M["EffLensGF"] = f"{EF['LENS-Net']['gflops']:,.0f}"; M["EffClipGF"] = f"{EF['LENS-Net']['detail']['clip16']:,.0f}"
    M["EffVmaeGF"] = f"{EF['videomae FT']['gflops']:.0f}"; M["EffRtdGF"] = f"{EF['r2plus1d_18 FT']['gflops']:.0f}"
    M["EffSwinGF"] = f"{EF['swin3d_t FT']['gflops']:.0f}"; M["EffLensTot"] = f"{EF['LENS-Net']['params_total']/1e6:.0f}M"
(OUT / "numbers.tex").write_text("% generated by make_tables.py -- do not edit\n\\providecommand{\\pend}{\\textcolor{red}{\\textbf{[pending]}}}\n" +
                                  "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(M.items())))
print("P3 boot + efficiency macros; pending:", [k for k, v in M.items() if v == "\\pend"])

# ================= JOURNAL EXTENSION (PREREG_journal): confirmation sets, E1/E2, robustness, error analysis, LENS-FT
LG = L / "logs"
def JL(n):
    p = LG / n
    return json.load(open(p)) if p.exists() else {}
CR, RB, EA, EF = JL("conf_results.json"), JL("robust_results.json"), JL("error_analysis.json"), JL("efficiency_journal.json")
def cic(x):          # diff [CI] with pre-registered flags: * better (CI > 0), dagger worse (CI < 0), NI non-inferior (lower bound > -1)
    if not x: return "\\pend"
    f = "$^*$" if x.get("better") else ("$^\\dagger$" if x.get("worse") else "")
    ni = "\\,$^{\\mathrm{NI}}$" if x.get("non_inferior") and not x.get("better") else ""
    return f"{sg(x['diff'])} [{sg(x['ci'][0])}, {sg(x['ci'][1])}]{f}{ni}"
if CR:
    Mo = CR["models"]; B2 = ["VioPeru", "AIRTLab"]
    for b in B2: mac(f"Conf{b}Win", f"{CR['n_windows'][b]:,}"); mac(f"Conf{b}Cl", f"{CR['n_clusters'][b]:,}")
    rows = [("\\textbf{LENS-Net (ours)}", "v2_audited_LENS-M_vrex"), ("LENS-Net w/o motion", "v2_audited_V2b"), ("CLIP probe (anchored)", "v2_audited_LP"),
            ("Motion probe", "v2_audited_MP"), ("CLIP zero-shot", "ZS"), ("VideoMAE-B FT", "A1_videomae"), ("Video Swin-T FT", "A1_swin3d_t"),
            ("R(2+1)D-18 FT", "A1_r2plus1d_18"), ("CNN-BiGRU + motion", "MAST-Net")]
    lines = []
    for lab, k in rows:
        v = Mo.get(k, {})
        cell = lambda b: (f"{v[b]['mean']:.2f}" + ("$^\\ddagger$" if v.get("seeds", 3) == 1 else f"{{\\scriptsize$\\pm${v[b]['sd']:.2f}}}")) if b in v else "\\pend"
        lines.append(f"{lab} & {cell('VioPeru')} & {cell('AIRTLab')} \\\\")
        if k == "ZS": lines.append("\\midrule")
    (OUT / "tables/conf.tex").write_text("\n".join(lines) + "\n")
    # main results table: the unseen sets (same row order as tables/main.tex)
    order = ["MAST-Net", "ZS", "v2_audited_LP", "v2_audited_MP", "v2_audited_V2b", "v2_audited_LENS-M_vrex", None,
             "A1_r2plus1d_18", "A1_swin3d_t", "A1_videomae"]
    base_rows = (OUT / "tables/main.tex").read_text().rstrip("\n").split("\n")
    assert len(base_rows) == len(order)
    def c2(k, b):
        v = Mo[k][b]; return f"{v['mean']:.2f}" + ("$^\\dagger$" if v.get("seeds", 3) == 1 else f"{{\\scriptsize$\\pm${v['sd']:.2f}}}")
    CFR = JL("cfights_results.json")                                   # CCTV-Fights (PREREG_journal Sec. 7), CCTV subset
    c3 = lambda k: (f" & {CFR['models'][k]['CCTV']['mean']:.2f}" + ("$^\\dagger$" if CFR['models'][k]['seeds'] == 1 else f"{{\\scriptsize$\\pm${CFR['models'][k]['CCTV']['sd']:.2f}}}")) if CFR else ""
    out = [r if k is None else r[:r.rindex("\\\\")].rstrip() + f" & {c2(k, 'VioPeru')} & {c2(k, 'AIRTLab')}{c3(k)} \\\\" for r, k in zip(base_rows, order)]
    (OUT / "tables/main_journal.tex").write_text("\n".join(out) + "\n")
    S2 = CR["secondary"]; P2 = CR["primary"]
    pr = [("LENS-Net $-$ VideoMAE-B FT", "LENS-M - A1_videomae"), ("LENS-Net $-$ Video Swin-T FT", "LENS-M - A1_swin3d_t"),
          ("LENS-Net $-$ R(2+1)D-18 FT", "LENS-M - A1_r2plus1d_18"), ("LENS-Net $-$ CNN-BiGRU", "LENS-M - CNN-BiGRU"),
          ("LENS-Net $-$ w/o motion", "LENS-M - w/o motion")]
    cfk = {"LENS-M - A1_videomae": ("primary", "LENS-M (P1) - A1_videomae"), "LENS-M - A1_swin3d_t": ("primary", "LENS-M (P1) - A1_swin3d_t"),
           "LENS-M - A1_r2plus1d_18": ("primary", "LENS-M (P1) - A1_r2plus1d_18"), "LENS-M - CNN-BiGRU": ("secondary", "LENS-M - CNN-BiGRU (P1)"),
           "LENS-M - w/o motion": ("secondary", "LENS-M - w/o motion (P1)")}
    cfc = lambda k: cic(CFR[cfk[k][0]][cfk[k][1]]["CCTV"]) if CFR and k in cfk else "--"
    TCj = JL("clip_methods_results.json").get("P1_paired", {})
    tcc = lambda k, b: cic(dict(TCj[k][b], better=TCj[k][b]["ci"][0] > 0, worse=TCj[k][b]["ci"][1] < 0)) if k in TCj else "--"   # flags by direction
    rows_cp = [f"{lab} & {cic(S2.get(k, {}).get('VioPeru'))} & {cic(S2.get(k, {}).get('AIRTLab'))} & {cfc(k)} \\\\" for lab, k in pr]
    if CFR:
        for lab, tk, ck in (("LENS-Net $-$ VadCLIP (ViT-L/14)", "LENS-Net - VadCLIP (L14)", "LENS-M (P1) - VadCLIP-L14_audited_lr0.0001"),
                            ("LENS-Net $-$ X-CLIP FT", "LENS-Net - X-CLIP FT (lr 5e-6)", "LENS-M (P1) - A1_xclip_lr5e-06")):
            rows_cp.append(f"{lab} & {tcc(tk, 'VioPeru')} & {tcc(tk, 'AIRTLab')} & {cic(CFR['primary'][ck]['CCTV'])} \\\\")
    (OUT / "tables/conf_paired.tex").write_text("\n".join(rows_cp) + "\n")
    for k, nm in (("LENS-M - A1_videomae", "V"), ("LENS-M - A1_swin3d_t", "S"), ("LENS-M - A1_r2plus1d_18", "R"), ("LENS-M - CNN-BiGRU", "C"), ("LENS-M - w/o motion", "N")):
        for b in B2: mac(f"Cf{nm}{b}", cic(S2[k][b]))
    for b in B2:
        for k, nm in (("v2_audited_LENS-M_vrex", "Lens"), ("A1_videomae", "Vmae"), ("A1_swin3d_t", "Swin"), ("A1_r2plus1d_18", "Rtd"), ("MAST-Net", "Mast"),
                      ("v2_audited_V2b", "Nomo"), ("v2_audited_LP", "Probe"), ("ZS", "Zs"), ("v2_audited_LENS-AG_vrex", "Ag")):
            mac(f"Cf{nm}Auc{b}", fmt(Mo[k][b]["mean"]))
    # E1 / E2 table
    ef = lambda k: f"{EF[k]['gflops']:,.0f}" if k in EF else "\\pend"
    _LT = JL("latency.json"); _lk = {"LENS-Net": "LENS-Net (L14, 16f)", "LENS-AG": "LENS-Net (L14, 16f)", "L1 (L14, 8f)": "L1 (L14, 8f)", "L2 (B16, 16f)": "L2 (B16, 16f)", "L3 (B16, 8f)": "L3 (B16, 8f)"}
    ef = (lambda f0: (lambda k: f0(k) + (f" & {_LT[_lk[k]]:.0f}" if _LT and _lk.get(k) in _LT else " & --")))(ef)
    lite = [("LENS-Net (ViT-L/14, 16 fr.)", "LENS-Net", None), ("LENS-AG: input-adaptive gate (E1)", "LENS-AG", "H1 LENS-AG - LENS-M"),
            ("L1: 8 CLIP frames", "L1 (L14, 8f)", "H2 L1 - LENS-M"), ("L2: CLIP ViT-B/16", "L2 (B16, 16f)", "H2 L2 - LENS-M"),
            ("L3: ViT-B/16, 8 frames", "L3 (B16, 8f)", "H2 L3 - LENS-M")]
    lines = []
    for lab, ek, pk in lite:
        if pk is None: lines.append(f"{lab} & {ef(ek)} & \\multicolumn{{4}}{{c}}{{reference}} \\\\"); continue
        a, c = P2.get(f"{pk} (P1)", {}), P2.get(f"{pk} (P3)", {})
        lines.append(f"{lab} & {ef(ek)} & {cic(a.get('VioPeru'))} & {cic(a.get('AIRTLab'))} & {cic(c.get('VioPeru'))} & {cic(c.get('AIRTLab'))} \\\\")
    (OUT / "tables/lite.tex").write_text("\n".join(lines) + "\n")
    for k, nm in (("LENS-Net", "Lens"), ("L1 (L14, 8f)", "Lone"), ("L2 (B16, 16f)", "Ltwo"), ("L3 (B16, 8f)", "Lthree")):
        mac(f"Jgf{nm}", f"{EF[k]['gflops']:,.0f}")
    for k, nm in (("H1 LENS-AG - LENS-M (P1)", "AgP"), ("H1 LENS-AG - LENS-M (P3)", "AgT"), ("H2 L1 - LENS-M (P1)", "LoneP"), ("H2 L1 - LENS-M (P3)", "LoneT")):
        for b in B2: mac(f"E{nm}{b}", cic(P2[k][b]))
if RB:
    fam = {"Blur": ["blur1", "blur2", "blur3"], "Dark": ["dark1", "dark2", "dark3"], "Noise": ["noise1", "noise2", "noise3"], "JPEG": ["jpeg1", "jpeg2", "jpeg3"]}
    order = ["LENS-Net", "w/o motion", "Motion probe", "VideoMAE-B FT", "Swin-T FT", "R(2+1)D-18 FT", "CNN-BiGRU"]
    disp = {"LENS-Net": "\\textbf{LENS-Net (ours)}", "w/o motion": "LENS-Net w/o motion", "CNN-BiGRU": "CNN-BiGRU"}
    lines = []
    for m in order:
        cells = []
        for s in ("test", "viop"):
            r = RB[s]["auc"][m]
            cells += [fmt(r["clean"])] + [fmt(float(np.mean([r[c] for c in cs]))) for cs in fam.values()] + [fmt(r["mean_corrupted"]), fmt(r["drop"])]
        lines.append(f"{disp.get(m, m)} & " + " & ".join(cells) + " \\\\")
        if m == "Motion probe": lines.append("\\midrule")
    (OUT / "tables/robust.tex").write_text("\n".join(lines) + "\n")
    for s, sn in (("test", "Te"), ("viop", "Vp")):
        for m, mn in (("LENS-Net", "Lens"), ("VideoMAE-B FT", "Vmae"), ("w/o motion", "Nomo"), ("Swin-T FT", "Swin"), ("R(2+1)D-18 FT", "Rtd")):
            r = RB[s]["auc"][m]; mac(f"Rb{mn}Mean{sn}", fmt(r["mean_corrupted"])); mac(f"Rb{mn}Drop{sn}", fmt(r["drop"]))
        pp = RB[s]["paired_LENS_minus_VideoMAE"]
        for c, cn in (("dark3", "Darkthree"), ("jpeg3", "Jpegthree"), ("blur3", "Blurthree")):
            mac(f"Rb{cn}{sn}", ci(pp[c]))
if EA:
    ex = EA["external_per_category"]; Ln, Vm = ex["LENS-Net"], ex["VideoMAE-B FT"]
    keys = ["XD-test|Fighting", "XD-test|Abuse", "XD-test|Riot", "XD-test|Shooting", "XD-test|Explosion", "XD-test|Car accident",
            "UCF-test|Fighting", "UCF-test|Robbery", "UCF-test|Shooting"]
    lines = []
    for k in keys:
        b, c = k.split("|"); bn = "XD" if b.startswith("XD") else "UCF"
        lines.append(f"{bn} & {c} & {Ln[k]['n_pos']} & {fmt(Ln[k]['auc'])} & {fmt(Vm[k]['auc'])} & {fmt(Ln[k]['recall'], 1)} & {fmt(Vm[k]['recall'], 1)} \\\\")
    (OUT / "tables/errcat.tex").write_text("\n".join(lines) + "\n")
    mac("EaBetter", str(sum(Ln[k]["auc"] > Vm[k]["auc"] for k in keys))); mac("EaCats", str(len(keys)))
    A = EA["airtlab"]
    for cls, nm in (("greet", "Greet"), ("handgestures", "Hand"), ("friendly punch", "Friendly"), ("hug", "Hug"), ("jump", "Jump"), ("highfive", "Highfive")):
        mac(f"Ea{nm}L", fmt(A["LENS-Net"][f"false_alarm|{cls}"]["rate"], 1)); mac(f"Ea{nm}V", fmt(A["VideoMAE-B FT"][f"false_alarm|{cls}"]["rate"], 1))
    for cls, nm in (("gunshot", "Gun"), ("slap", "Slap"), ("fight", "Fight"), ("push", "Push")):
        mac(f"Ea{nm}MissL", fmt(A["LENS-Net"][f"miss|{cls}"]["rate"], 1)); mac(f"Ea{nm}MissV", fmt(A["VideoMAE-B FT"][f"miss|{cls}"]["rate"], 1))
    mac("EaVpFpL", fmt(EA["viop"]["LENS-Net"]["false_positives_validation"]["false_alarm_rate"], 1))
    mac("EaVpFpV", fmt(EA["viop"]["VideoMAE-B FT"]["false_positives_validation"]["false_alarm_rate"], 1))
    mac("EaVpFpN", str(EA["viop"]["LENS-Net"]["false_positives_validation"]["n_windows"]))
# LENS-FT (track B, PREREG.md deviation 4B)
ftr = [json.load(open(L / f"runs_ft/LENS-FT_seed{s}/summary.json")) for s in (42, 43, 44) if (L / f"runs_ft/LENS-FT_seed{s}/summary.json").exists()]
if len(ftr) == 3:
    acc = [r["test_video"]["overall"]["accuracy"] for r in ftr]; m_ = float(np.mean(acc)); se = 100 * np.sqrt(m_ / 100 * (1 - m_ / 100) / 398)
    mac("FtAcc", fmt(m_)); mac("FtAccSd", fmt(float(np.std(acc, ddof=1)))); mac("FtAccLo", fmt(m_ - 1.96 * se)); mac("FtAccHi", fmt(m_ + 1.96 * se))
    mac("FtXD", fmt(float(np.mean([r["external_auc"]["XD-test"] for r in ftr])))); mac("FtUCF", fmt(float(np.mean([r["external_auc"]["UCF-test"] for r in ftr]))))
    mac("FtWinAuc", fmt(float(np.mean([r["test_window_auc"] for r in ftr]))))
    o, _ = agg("A2_LENS-LENS-M_vrex")
    if o: mac("AtwoLensXD", fmt(o["XD-test"])); mac("AtwoLensUCF", fmt(o["UCF-test"]))
nd = L / "conf/near_duplicate_exclusion.csv"
if nd.exists(): mac("ConfMaxCos", f"{pd.read_csv(nd).max_cos_to_train.max():.3f}")
ag = [json.load(open(L / f"runs/v2_audited_LENS-AG_vrex_seed{s}/summary.json")) for s in (42, 43, 44)]
mac("AgXD", fmt(float(np.mean([r["external"]["XD-test"]["overall"]["auc"] for r in ag]))))
mac("AgUCF", fmt(float(np.mean([r["external"]["UCF-test"]["overall"]["auc"] for r in ag]))))
o, _ = agg("A2_LENS-V2b")
if o: mac("AtwoNomoRWF", fmt(o["rwf_official_video_acc"]))
# ================= CCTV-Fights (PREREG_journal Sec. 7)
CFR = JL("cfights_results.json")
if CFR:
    for st, sn in (("CCTV", "Cc"), ("All", "Al"), ("Testing", "Ts")):
        mac(f"Cfi{sn}Win", f"{CFR['n_windows'][st]:,}"); mac(f"Cfi{sn}Vid", f"{CFR['n_videos'][st]:,}"); mac(f"Cfi{sn}Excl", str(CFR['n_excluded_near_duplicate'][st]))
        for k, nm in (("v2_audited_LENS-M_vrex", "Lens"), ("v2_std_std_rwf_LENS-M_vrex", "LensT"), ("A1_videomae", "Vmae"), ("A2_videomae", "VmaeT"),
                      ("A1_swin3d_t", "Swin"), ("A1_r2plus1d_18", "Rtd"), ("A1_xclip_lr5e-06", "Xclip"), ("VadCLIP-L14_audited_lr0.0001", "Vadclip"),
                      ("VadCLIP-L14_std_std_rwf_lr0.0001", "VadclipT"), ("MAST-Net", "Mast"), ("ZS", "Zs"), ("v2_audited_V2b", "Nomo"),
                      ("b16_f8_v2_audited_LENS-M_vrex", "Lthree"), ("v2_audited_LENS-AG_vrex", "Ag")):
            mac(f"Cfi{nm}{sn}", fmt(CFR["models"][k][st]["mean"]))
        for k, nm in (("LENS-M (P1) - A1_videomae", "PV"), ("LENS-M (P1) - A1_swin3d_t", "PS"), ("LENS-M (P1) - A1_r2plus1d_18", "PR"),
                      ("LENS-M (P1) - A1_xclip_lr5e-06", "PX"), ("LENS-M (P1) - VadCLIP-L14_audited_lr0.0001", "PC"), ("LENS-M (P3) - A2_videomae", "TV"),
                      ("LENS-M (P3) - A2_swin3d_t", "TS"), ("LENS-M (P3) - A2_r2plus1d_18", "TR"), ("LENS-M (P3) - VadCLIP-L14_std_std_rwf_lr0.0001", "TC")):
            mac(f"Cfd{nm}{sn}", cic(CFR["primary"][k][st]))
        for k, nm in (("LENS-AG - LENS-M (P1)", "Ag"), ("LENS-M - w/o motion (P1)", "N"), ("LENS-M - CNN-BiGRU (P1)", "C"), ("L1 - LENS-M (P1)", "Lone"),
                      ("L2 - LENS-M (P1)", "Ltwo"), ("L3 - LENS-M (P1)", "Lthree")):
            mac(f"Cfd{nm}{sn}", cic(CFR["secondary"][k][st]))
# ================= Standard frame-level protocol on XD-Violence / UCF-Crime (PREREG_journal Sec. 8)
FLR = JL("framelevel_results.json")
if FLR:
    FM = FLR["models"]
    frows = [("\\textbf{LENS-Net (ours)}", "v2_audited_LENS-M_vrex", "Lens"), ("LENS-Net w/o motion", "v2_audited_V2b", "Nomo"),
             ("LENS-Net-A (anchor fusion)$^\\S$", "v2_audited_LENS-NetA", "A"), ("CLIP probe (anchor only)", "v2_audited_LP", "Lp"),
             ("LENS-AG (adaptive gate)$^\\S$", "v2_audited_LENS-AG_vrex", "Ag"), ("LENS-FG (footage gate)$^\\S$", "v2_audited_LENS-FG_vrex", "Fg"),
             ("LENS-Lite L3", "b16_f8_v2_audited_LENS-M_vrex", "Lthree"),
             ("CLIP zero-shot (our prompts)", "ZS", "Zs"), ("VadCLIP~\\cite{wu2024vadclip} (our protocol)", "VadCLIP-L14_audited_lr0.0001", "Vadclip"),
             ("VideoMAE-B FT", "A1_videomae", "Vmae"), ("Video Swin-T FT", "A1_swin3d_t", "Swin"), ("R(2+1)D-18 FT", "A1_r2plus1d_18", "Rtd"),
             ("X-CLIP FT", "A1_xclip_lr5e-06", "Xclip"), ("CNN-BiGRU", "MAST-Net", "Mast")]
    pm = lambda v, k: f"{v[k]:.2f}" + (f"{{\\scriptsize$\\pm${v[k + '_sd']:.2f}}}" if v[k + "_sd"] > 0 else "")
    lines = []
    for lab, k, nm in frows:
        v = FM[k]
        if nm not in ("Ag", "Fg", "Lthree"): lines.append(f"{lab} & {pm(v['UCF'], 'auc')} & {v['UCF-violent']['auc']:.2f} & {pm(v['XD'], 'ap')} & {v['XD']['auc']:.2f} \\\\")
        if nm == "Zs": lines.append("\\midrule")
        mac(f"Fl{nm}Ucf", fmt(v["UCF"]["auc"])); mac(f"Fl{nm}UcfV", fmt(v["UCF-violent"]["auc"])); mac(f"Fl{nm}XdAp", fmt(v["XD"]["ap"])); mac(f"Fl{nm}XdAuc", fmt(v["XD"]["auc"]))
        mac(f"Fl{nm}XdApClean", fmt(v["XD-clean"]["ap"])); mac(f"Fl{nm}UcfClean", fmt(v["UCF-clean"]["auc"]))
    (OUT / "tables/framelevel.tex").write_text("\n".join(lines) + "\n")
    for c, nm in (("A1_videomae", "V"), ("A1_swin3d_t", "S"), ("A1_r2plus1d_18", "R"), ("A1_xclip_lr5e-06", "X"), ("VadCLIP-L14_audited_lr0.0001", "C"),
                  ("MAST-Net", "M"), ("ZS", "Z"), ("v2_audited_V2b", "N"), ("b16_f8_v2_audited_LENS-M_vrex", "L"), ("v2_audited_LENS-AG_vrex", "A"), ("v2_audited_LENS-FG_vrex", "F"), ("v2_audited_LENS-NetA", "Na"), ("v2_audited_LP", "P")):
        for mk, mn in (("UCF_auc", "Ucf"), ("XD_ap", "XdAp"), ("XD_auc", "XdAuc"), ("UCF-violent_auc", "UcfV")):
            mac(f"Fld{nm}{mn}", cic(FLR["paired"][f"LENS-M - {c}"][mk]))
    du = [FM[k]["UCF"]["auc"] - FM[k]["UCF-clean"]["auc"] for _, k, _ in frows]; dx = [abs(FM[k]["XD"]["ap"] - FM[k]["XD-clean"]["ap"]) for _, k, _ in frows]
    mac("FlCleanUcfLo", fmt(min(du), 1)); mac("FlCleanUcfHi", fmt(max(du), 1)); mac("FlCleanMaxXd", fmt(max(dx)))
    mac("FlUcfVid", str(FLR["n_videos"]["UCF"])); mac("FlXdVid", str(FLR["n_videos"]["XD"]))
    mac("FlUcfFrames", f"{FLR['n_frames']['UCF']:,}"); mac("FlXdFrames", f"{FLR['n_frames']['XD']:,}")
# ================= LENS-FG and the nine unseen test sets (PREREG_journal Sec. 9)
C2R, FGD = JL("conf2_results.json"), JL("fg_dev_window.json")
if C2R and FGD and CR and CFR:
    TCp = JL("clip_methods_results.json")["P1"]; A1r = J("A1_results.json"); frm = JL("fresh_results.json")["models"]; cfm = CFR["models"]; c2m = C2R["models"]; cm = CR["models"]
    mean = lambda d: d["mean"]
    def nine(xd, ucf, kf, kc, kcf, k2, tc=None):
        v = [xd, ucf] + ([mean(TCp[tc][c]) for c in ("SCFD", "SCVD", "VioPeru", "AIRTLab")] if tc else
                         [mean(frm[kf]["SCFD"]), mean(frm[kf]["SCVD"]), mean(cm[kc]["VioPeru"]), mean(cm[kc]["AIRTLab"])])
        return v + [mean(cfm[kcf]["CCTV"]), mean(c2m[k2]["Movies"]), mean(c2m[k2]["UBI-Fights"])]
    tF = F["F1_F3_table"]; x = lambda k: (mean(tF[k]["xd_auc"]), mean(tF[k]["ucf_auc"])); a1 = lambda k: (mean(A1r[k]["xd_auc"]), mean(A1r[k]["ucf_auc"]))
    def std(k, tk): return nine(*tk, k, k, k, k)
    NR = [("\\textbf{LENS-Net (ours)}", "Lens", std("v2_audited_LENS-M_vrex", x("LENS-M"))),
          ("LENS-FG (footage gate)$^\\S$", "Fg", std("v2_audited_LENS-FG_vrex", FGD["LENS-FG"][1:])),
          ("LENS-AG (adaptive gate)", "Ag", std("v2_audited_LENS-AG_vrex", FGD["LENS-AG"][1:])),
          ("LENS-Net w/o motion", "Nomo", std("v2_audited_V2b", x("LENS(V2b)"))),
          ("CLIP zero-shot", "Zs", std("ZS", x("ZS"))), None,
          ("VadCLIP~\\cite{wu2024vadclip}", "Vadclip", nine(mean(TCp["VadCLIP (L14)"]["XD"]), mean(TCp["VadCLIP (L14)"]["UCF"]), None, None, "VadCLIP-L14_audited_lr0.0001", "VadCLIP-L14_audited_lr0.0001", tc="VadCLIP (L14)")),
          ("X-CLIP FT", "Xclip", nine(mean(TCp["X-CLIP FT (lr 5e-6)"]["XD"]), mean(TCp["X-CLIP FT (lr 5e-6)"]["UCF"]), None, None, "A1_xclip_lr5e-06", "A1_xclip_lr5e-06", tc="X-CLIP FT (lr 5e-6)")),
          ("VideoMAE-B FT", "Vmae", std("A1_videomae", a1("A1_videomae"))), ("Video Swin-T FT", "Swin", std("A1_swin3d_t", a1("A1_swin3d_t"))),
          ("R(2+1)D-18 FT", "Rtd", std("A1_r2plus1d_18", a1("A1_r2plus1d_18"))), ("CNN-BiGRU", "Mast", std("MAST-Net", x("MAST-Net")))]
    cols = list(zip(*[r[2] for r in NR if r])); best = [max(c) for c in cols]
    lines = []
    for r in NR:
        if r is None: lines.append("\\midrule"); continue
        lab, nm, v = r
        cells = [("\\textbf{%.2f}" % a) if abs(a - b) < 1e-9 else f"{a:.2f}" for a, b in zip(v, best)]
        lines.append(f"{lab} & " + " & ".join(cells) + f" & {np.mean(v):.2f} & {min(v):.2f} \\\\")
        mac(f"Ns{nm}Mean", fmt(float(np.mean(v)))); mac(f"Ns{nm}Min", fmt(float(min(v)))); mac(f"Ns{nm}Movies", fmt(v[7])); mac(f"Ns{nm}Ubi", fmt(v[8]))
        if nm in ("Fg", "Ag", "Nomo", "Lens"):
            for c, cn in zip(v, ("Xd", "Ucf", "Scfd", "Scvd", "Vp", "Airt", "Cf")): mac(f"Ns{nm}{cn}", fmt(c))
    (OUT / "tables/ninesets.tex").write_text("\n".join(lines) + "\n")
    pc = C2R["primary_LENS-FG_P1"]["claims"]
    for k, nm in zip(pc, ("One", "Two", "Three", "Four")): mac(f"FgC{nm}", cic({kk: vv for kk, vv in pc[k].items() if kk != "non_inferior"}))
    for key, nm in (("secondary_LENS-FG-post_P1", "Post"), ("secondary_LENS-FG_P3", "T")):
        for k, cn in zip(C2R[key]["claims"], ("One", "Two", "Three", "Four")): mac(f"Fg{nm}C{cn}", cic({kk: vv for kk, vv in C2R[key]["claims"][k].items() if kk != "non_inferior"}))
    for c, nm in (("A1_videomae", "V"), ("A1_swin3d_t", "S"), ("A1_r2plus1d_18", "R"), ("A1_xclip_lr5e-06", "X"), ("VadCLIP-L14_audited_lr0.0001", "C"), ("v2_audited_LENS-AG_vrex", "A")):
        for b, bn in (("Movies", "Mov"), ("UBI-Fights", "Ubi")): mac(f"Fgd{nm}{bn}", cic({kk: vv for kk, vv in C2R["vs_baselines_P1"][f"LENS-FG - {c}"][b].items() if kk != "non_inferior"}))
    mac("CtwoMovWin", str(C2R["n_windows"]["Movies"])); mac("CtwoUbiWin", f"{C2R['n_windows']['UBI-Fights']:,}"); mac("CtwoUbiVid", str(C2R["n_videos"]["UBI-Fights"]))
    mac("CtwoUbiExcl", str(C2R["n_excluded_near_duplicate"]["UBI-Fights"])); mac("CtwoMovExcl", str(C2R["n_excluded_near_duplicate"]["Movies"]))
    for k, nm in (("Movies/fights", "MovF"), ("Movies/noFights", "MovN"), ("UBI-Fights", "Ubi")): mac(f"Praw{nm}", fmt(C2R["p_raw_mean"][k]))
    mac("NsFgPostMovies", fmt(mean(c2m["v2_audited_LENS-FGpost_vrex"]["Movies"]))); mac("NsFgPostUbi", fmt(mean(c2m["v2_audited_LENS-FGpost_vrex"]["UBI-Fights"])))
    mac("NsNomoTMovies", fmt(mean(c2m["v2_std_std_rwf_V2b"]["Movies"]))); mac("NsLensTMovies", fmt(mean(c2m["v2_std_std_rwf_LENS-M_vrex"]["Movies"])))
    mac("FgTest", fmt(FGD["LENS-FG"][0]))
# ================= CLIP-based comparisons, prompt study and frame-level protocol (PREREG_journal Sec. 6)
TC, LT = JL("clip_methods_results.json"), JL("latency.json")
if TC and LT:
    S7 = ["Test", "XD", "UCF", "SCFD", "SCVD", "VioPeru", "AIRTLab"]
    cellm = lambda r: f"{r['mean']:.2f}" + ("$^\\ddagger$" if r.get("n", 3) == 1 else f"{{\\scriptsize$\\pm${r['sd']:.2f}}}")
    rows = [("\\textbf{LENS-Net (ours)}", "LENS-Net", "32k", LT.get("LENS-Net (L14, 16f)")), ("VadCLIP~\\cite{wu2024vadclip} (ViT-L/14)", "VadCLIP (L14)", "21.0M", LT.get("VadCLIP (L14)")),
            ("VadCLIP~\\cite{wu2024vadclip} (ViT-B/16)", "VadCLIP (B16)", "9.4M", None), ("X-CLIP FT~\\cite{ni2022xclip} (lr on validation)", "X-CLIP FT (lr 5e-6)", "126.9M", LT.get("X-CLIP FT")),
            ("X-CLIP FT, shared recipe", "X-CLIP FT", "126.9M", LT.get("X-CLIP FT")),
            ("X-CLIP zero-shot~\\cite{ni2022xclip}", "X-CLIP zero-shot", "0", None), ("VideoMAE-B FT~\\cite{tong2022videomae}", "VideoMAE-B FT", "86.2M", LT.get("VideoMAE-B FT"))]
    lines = [f"{lab} & {pm} & {('%.0f' % ms) if ms else '--'} & " + " & ".join(cellm(TC['P1'][k][c]) for c in S7) + " \\\\" for lab, k, pm, ms in rows]
    (OUT / "tables/clipcomp.tex").write_text("\n".join(lines) + "\n")
    pr = [("LENS-Net $-$ VadCLIP (ViT-L/14)", "LENS-Net - VadCLIP (L14)"), ("LENS-Net $-$ VadCLIP (ViT-B/16)", "LENS-Net - VadCLIP (B16)"),
          ("LENS-Net $-$ X-CLIP FT (lr on validation)", "LENS-Net - X-CLIP FT (lr 5e-6)")]
    lines = []
    for lab, k in pr:
        p3 = TC["P3_paired"].get(k.replace("LENS-Net - ", "LENS-Net - "), None)
        lines.append(f"{lab} & " + " & ".join(ci(TC['P1_paired'][k][c]) for c in S7) + f" & {ci(p3) if p3 else '--'} \\\\")
    (OUT / "tables/clippaired.tex").write_text("\n".join(lines) + "\n")
    for k, nm in (("LENS-Net - VadCLIP (L14)", "Vl"), ("LENS-Net - VadCLIP (B16)", "Vb")):
        for c in S7: mac(f"Tc{nm}{c}", ci(TC["P1_paired"][k][c]))
        mac(f"Tc{nm}PthreeMean", ci(TC["P3_paired"][k]))
    for m, nm in (("LENS-Net", "Lens"), ("VadCLIP (L14)", "Vl"), ("VadCLIP (B16)", "Vb")):
        mac(f"Tc{nm}Pthree", fmt(TC["P3"][m]["mean_transfer"]["mean"])); mac(f"Tc{nm}PthreeRwf", fmt(TC["P3"][m]["rwf_acc"]["mean"]))
        for c in S7: mac(f"Tc{nm}Auc{c}", fmt(TC["P1"][m][c]["mean"]))
    for c in S7:
        mac(f"TcXftAuc{c}", fmt(TC["P1"]["X-CLIP FT (lr 5e-6)"][c]["mean"])); mac(f"TcXshAuc{c}", fmt(TC["P1"]["X-CLIP FT"][c]["mean"]))
        mac(f"TcXzsAuc{c}", fmt(TC["P1"]["X-CLIP zero-shot"][c]["mean"])); mac(f"TcXft{c}", ci(TC["P1_paired"]["LENS-Net - X-CLIP FT (lr 5e-6)"][c]))
    mac("TcXftNsig", str(sum(TC["P1_paired"]["LENS-Net - X-CLIP FT (lr 5e-6)"][c]["better"] for c in S7)))
    # prompt study
    ps = [("Original", "LENS-Net"), ("Paraphrased", "LENS-Net (para)"), ("CCTV-worded", "LENS-Net (surv)"), ("Broad (8+8)", "LENS-Net (broad)")]
    S5 = ["Test", "XD", "UCF", "VioPeru", "AIRTLab"]
    lines = [f"{lab} & " + " & ".join(fmt(TC['P1'][k][c]['mean']) for c in S5) + f" & {fmt(TC['P3'][k]['rwf_acc']['mean'])} & {fmt(TC['P3'][k]['mean_transfer']['mean'])} \\\\" for lab, k in ps]
    (OUT / "tables/prompts.tex").write_text("\n".join(lines) + "\n")
    mac("PsSpreadMax", fmt(max(TC["prompt_spread_P1"].values()))); mac("PsSpreadTest", fmt(TC["prompt_spread_P1"]["Test"]))
    mac("PsBroadPthree", ci(TC["P3_paired"]["LENS-Net (broad) - LENS-Net"])); mac("PsBroadRwf", fmt(TC["P3"]["LENS-Net (broad)"]["rwf_acc"]["mean"]))
    cr = TC["category_recall"]
    for c, nm in (("XD-test|Abuse", "Abuse"), ("XD-test|Explosion", "Expl"), ("XD-test|Riot", "Riot"), ("XD-test|Shooting", "Shoot")):
        mac(f"PsRec{nm}O", fmt(cr["LENS-Net"][c], 1)); mac(f"PsRec{nm}B", fmt(cr["LENS-Net (broad)"][c], 1))
    for k, nm in (("LENS-Net (L14, 16f)", "Lens"), ("L1 (L14, 8f)", "Lone"), ("L2 (B16, 16f)", "Ltwo"), ("L3 (B16, 8f)", "Lthree"), ("VideoMAE-B FT", "Vmae"),
                  ("Video Swin-T FT", "Swin"), ("R(2+1)D-18 FT", "Rtd"), ("X-CLIP FT", "Xclip"), ("CNN-BiGRU", "Mast"), ("VadCLIP (L14)", "Vadclip")):
        mac(f"Ms{nm}", f"{LT[k]:.0f}")
    mac("MsGpu", LT["gpu"])
    FPv = JL("fig_prompts_values.json")
    if FPv:
        Dm = np.array(FPv["cohens_d"]); ucf = FPv["sets"].index("UCF- Crime") if "UCF- Crime" in FPv["sets"] else [i for i, x in enumerate(FPv["sets"]) if x.startswith("UCF")][0]
        mac("PdMax", f"{Dm[5:].max():.2f}"); mac("PdUcfLo", f"{Dm[5:, ucf].min():.2f}"); mac("PdUcfHi", f"{Dm[5:, ucf].max():.2f}")
(OUT / "numbers.tex").write_text("% generated by make_tables.py -- do not edit\n\\providecommand{\\pend}{\\textcolor{red}{\\textbf{[pending]}}}\n" +
                                  "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(M.items())))
print("journal extension macros written; pending:", [k for k, v in M.items() if v == "\\pend"])

# ================= CONSOLIDATED JOURNAL TABLES (2026-10-05): II all sets, III paired, IV protocols, VI variants
C2R, FGD, CR, CFR, TCJ, FRJ = JL("conf2_results.json"), JL("fg_dev_window.json"), JL("conf_results.json"), JL("cfights_results.json"), JL("clip_methods_results.json"), JL("fresh_results.json")
if all((C2R, FGD, CR, CFR, TCJ, FRJ)):
    tF, A1r, LT_ = F["F1_F3_table"], J("A1_results.json"), JL("latency.json"); TCp = TCJ["P1"]
    ANX = JL("anchor_safe_exploratory.json")["A: average(probe, LENS-Net)"]
    frm, cm, cfm, c2m = FRJ["models"], CR["models"], CFR["models"], C2R["models"]
    mn = lambda d: d["mean"]
    def lite_tx(prefix):         # audited test / XD / UCF window AUC from the Lite run summaries (three seeds)
        out = []
        for s in (42, 43, 44):
            r = json.load(open(L / prefix / f"v2_audited_LENS-M_vrex_seed{s}/summary.json"))
            out.append([r["test"]["overall"]["auc"], r["external"]["XD-test"]["overall"]["auc"], r["external"]["UCF-test"]["overall"]["auc"]])
        return list(np.mean(out, 0))
    def row(tx, key=None, tc=None, k2=None, kcf=None):
        """tx: [test, XD, UCF]; then SCFD, SCVD, VioPeru, AIRTLab, CCTV-F, Movies, UBI (None where not scored)."""
        if tc: mid = [mn(TCp[tc][c]) for c in ("SCFD", "SCVD", "VioPeru", "AIRTLab")]
        else: mid = [mn(frm[key]["SCFD"]), mn(frm[key]["SCVD"]), mn(cm[key]["VioPeru"]), mn(cm[key]["AIRTLab"])]
        kcf = kcf or key; k2 = k2 or key
        tail = [mn(cfm[kcf]["CCTV"]) if kcf in cfm else None] + ([mn(c2m[k2]["Movies"]), mn(c2m[k2]["UBI-Fights"])] if k2 in c2m else [None, None])
        return list(tx) + mid + tail
    ftx = lambda k: [mn(tF[k][c]) for c in ("test_auc", "xd_auc", "ucf_auc")]
    atx = lambda k: [mn(A1r[k][c]) for c in ("test_auc", "xd_auc", "ucf_auc")]
    ttx = lambda k: [mn(TCp[k][c]) for c in ("Test", "XD", "UCF")]
    ms_ = lambda k: f"{LT_[k]:.0f}" if k in LT_ else "--"
    # ---- Table 6: every model on the audited test split and the nine unseen test sets
    T2 = [("\\textbf{LENS-Net (ours)}", "31.9k", ms_("LENS-Net (L14, 16f)"), row(ftx("LENS-M"), "v2_audited_LENS-M_vrex")),
          ("\\textbf{LENS-Net-A (anchor fusion)}$^\\ddagger$", "33.5k", ms_("LENS-Net (L14, 16f)"), [ANX[k] for k in ("Test", "XD", "UCF", "SCFD", "SCVD", "VioPeru", "AIRTLab", "CCTV-F", "Movies", "UBI-Fights")]), None,
          ("CLIP probe (anchor only)", "1.5k", "--", row(ftx("LP"), "v2_audited_LP")),
          ("Anchor + PPS (w/o motion)", "5.7k", "--", row(ftx("LENS(V2b)"), "v2_audited_V2b")),
          ("Motion probe (motion only)", "26.2k", "--", row(ftx("MP"), "v2_audited_MP")), None,
          ("CLIP zero-shot", "0", "--", row(ftx("ZS"), "ZS")),
          ("VadCLIP~\\cite{wu2024vadclip}", "21.0M", ms_("VadCLIP (L14)"), row(ttx("VadCLIP (L14)"), tc="VadCLIP (L14)", kcf="VadCLIP-L14_audited_lr0.0001", k2="VadCLIP-L14_audited_lr0.0001")),
          ("X-CLIP FT~\\cite{ni2022xclip}", "126.9M", ms_("X-CLIP FT"), row(ttx("X-CLIP FT (lr 5e-6)"), tc="X-CLIP FT (lr 5e-6)", kcf="A1_xclip_lr5e-06", k2="A1_xclip_lr5e-06")),
          ("VideoMAE-B FT~\\cite{tong2022videomae}", "86.2M", ms_("VideoMAE-B FT"), row(atx("A1_videomae"), "A1_videomae")),
          ("Video Swin-T FT~\\cite{liu2022videoswin}", "27.9M", ms_("Video Swin-T FT"), row(atx("A1_swin3d_t"), "A1_swin3d_t")),
          ("R(2+1)D-18 FT~\\cite{tran2018r21d}", "31.3M", ms_("R(2+1)D-18 FT"), row(atx("A1_r2plus1d_18"), "A1_r2plus1d_18")),
          ("CNN-BiGRU", "6.1M", ms_("CNN-BiGRU"), row(ftx("MAST-Net"), "MAST-Net"))]
    vals = [r[3] for r in T2 if r]; best = [max(c) for c in zip(*vals)]
    summ = lambda v: (float(np.mean(v[1:])), float(min(v[1:])))
    bm, bn = max(summ(v)[0] for v in vals), max(summ(v)[1] for v in vals)
    B = lambda x, b: ("\\textbf{%.2f}" % x) if abs(x - b) < 1e-9 else f"{x:.2f}"
    lines = []
    for r in T2:
        if r is None: lines.append("\\midrule"); continue
        lab, pr, ms, v = r; m_, n_ = summ(v)
        lines.append(f"{lab} & {pr} & {ms} & " + " & ".join(B(x, b) for x, b in zip(v, best)) + f" & {B(m_, bm)} & {B(n_, bn)} \\\\")
    (OUT / "tables/t2_all.tex").write_text("\n".join(lines) + "\n")
    for lab, nm in (("CLIP probe (anchor only)", "Lp"), ("Motion probe (motion only)", "Mp"), ("\\textbf{LENS-Net-A (anchor fusion)}$^\\ddagger$", "A")):
        v = [r for r in T2 if r and r[0] == lab][0][3]; mac(f"Ns{nm}Mean", fmt(summ(v)[0])); mac(f"Ns{nm}Min", fmt(summ(v)[1])); mac(f"Ns{nm}Movies", fmt(v[8])); mac(f"Ns{nm}Ubi", fmt(v[9]))
    lv = C2R["lensnet_vs_P1"]
    for c, nm in (("v2_audited_LP", "Lp"), ("ZS", "Zs"), ("v2_audited_MP", "Mp")):
        for b, bn_ in (("Movies", "Mov"), ("UBI-Fights", "Ubi")): mac(f"Cld{nm}{bn_}", cic(lv[f"LENS-Net - {c}"][b]))
    # ---- paired differences with the seed-averaged test (t3_paired.tex; Table 7 prints t3_paired_seedaware.tex), diff with markers
    def sig(d):
        if not d: return "--"
        lo, hi = d["ci"]; return sg(d["diff"]) + ("$^*$" if lo > 0 else ("$^\\dagger$" if hi < 0 else ""))
    tcP, a1p, fP, frp, cs, cfp, cfs = TCJ["P1_paired"], A1r["paired_vs_LENS-Net"], F["F1_F3_paired"], FRJ["paired"], CR["secondary"], CFR["primary"], CFR["secondary"]
    def p_tc(k): return [tcP[k][c] for c in ("Test", "XD", "UCF", "SCFD", "SCVD", "VioPeru", "AIRTLab")]
    def p_std(a1k=None, fk=None, frk=None, ck=None):
        tx = [a1p[a1k][c] for c in ("test", "xd", "ucf")] if a1k else [fP[fk][c] for c in ("test", "XD-test", "UCF-test")]
        return tx + [frp[frk]["SCFD"], frp[frk]["SCVD"], cs[ck]["VioPeru"], cs[ck]["AIRTLab"]]
    T3 = [("VadCLIP", p_tc("LENS-Net - VadCLIP (L14)"), cfp["LENS-M (P1) - VadCLIP-L14_audited_lr0.0001"], "VadCLIP-L14_audited_lr0.0001"),
          ("X-CLIP FT", p_tc("LENS-Net - X-CLIP FT (lr 5e-6)"), cfp["LENS-M (P1) - A1_xclip_lr5e-06"], "A1_xclip_lr5e-06"),
          ("VideoMAE-B FT", p_std("A1_videomae", frk="v2_audited_LENS-M_vrex - A1_videomae", ck="LENS-M - A1_videomae"), cfp["LENS-M (P1) - A1_videomae"], "A1_videomae"),
          ("Video Swin-T FT", p_std("A1_swin3d_t", frk="v2_audited_LENS-M_vrex - A1_swin3d_t", ck="LENS-M - A1_swin3d_t"), cfp["LENS-M (P1) - A1_swin3d_t"], "A1_swin3d_t"),
          ("R(2+1)D-18 FT", p_std("A1_r2plus1d_18", frk="v2_audited_LENS-M_vrex - A1_r2plus1d_18", ck="LENS-M - A1_r2plus1d_18"), cfp["LENS-M (P1) - A1_r2plus1d_18"], "A1_r2plus1d_18"),
          ("CNN-BiGRU", p_std(fk="LENS-M - MAST-Net", frk="v2_audited_LENS-M_vrex - MAST-Net", ck="LENS-M - CNN-BiGRU"), cfs["LENS-M - CNN-BiGRU (P1)"], "MAST-Net"),
          ("Anchor + PPS (w/o motion)", p_std(fk="LENS-M - LENS(V2b)", frk="v2_audited_LENS-M_vrex - v2_audited_V2b", ck="LENS-M - w/o motion"), cfs["LENS-M - w/o motion (P1)"], "v2_audited_V2b")]
    lines = [f"{lab} & " + " & ".join(sig(d) for d in seven + [cf["CCTV"], lv[f"LENS-Net - {k2}"]["Movies"], lv[f"LENS-Net - {k2}"]["UBI-Fights"]]) + " \\\\"
             for lab, seven, cf, k2 in T3]
    (OUT / "tables/t3_paired.tex").write_text("\n".join(lines) + "\n")
    # ---- Table 8a: single-source transfer (P3) incl. VadCLIP
    cols = ["rwf_official_video_acc", "HockeyFights", "RLVS", "BusViolence", "VID", "XD-test", "UCF-test"]
    lines = []
    for lab, p in (("CLIP probe (anchor only)", "A2_LENS-LP"), ("Anchor + PPS (w/o motion)", "A2_LENS-V2b"), ("Motion probe (motion only)", "A2_LENS-MP"),
                   ("\\textbf{LENS-Net (ours)}", "A2_LENS-LENS-M_vrex"), (None, None), ("VadCLIP", "VAD"), ("VideoMAE-B FT", "A2_videomae"),
                   ("Video Swin-T FT", "A2_swin3d_t"), ("R(2+1)D-18 FT", "A2_r2plus1d_18")):
        if lab is None: lines.append("\\midrule"); continue
        if p == "VAD":
            t = TCJ["P3"]["VadCLIP (L14)"]; v = [mn(t["rwf_acc"])] + [mn(t[c]) for c in cols[1:]] + [mn(t["mean_transfer"])]
        else:
            o, _ = agg(p)
            if o is None: lines.append(f"{lab} & " + " & ".join(["--"] * 8) + " \\\\"); continue
            v = [o[c] for c in cols] + [o["mean_transfer"]]
        lines.append(f"{lab} & " + " & ".join(fmt(x) for x in v) + " \\\\")
    (OUT / "tables/t4_p3.tex").write_text("\n".join(lines) + "\n")
    # ---- Table 15: ablations and variants of LENS-Net
    EFJ = JL("efficiency_journal.json"); gf = lambda k: f"{EFJ[k]['gflops']:,.0f}" if k in EFJ else "--"
    def vrow(tx, key, cf_ok=True):
        v = list(tx) + [mn(frm[key]["SCFD"]), mn(frm[key]["SCVD"]), mn(cm[key]["VioPeru"]), mn(cm[key]["AIRTLab"])]
        return v + [mn(cfm[key]["CCTV"]) if cf_ok and key in cfm else None]
    def prow(k):
        return [mn(TCp[k][c]) for c in ("Test", "XD", "UCF", "SCFD", "SCVD", "VioPeru", "AIRTLab")] + [None]
    T6 = [("\\textbf{LENS-Net (reference)}", "31.9k", gf("LENS-Net"), ms_("LENS-Net (L14, 16f)"), vrow(ftx("LENS-M"), "v2_audited_LENS-M_vrex")),
          ("\\emph{Components}", None),
          ("w/o motion (anchor + PPS)", "5.7k", "--", "--", vrow(ftx("LENS(V2b)"), "v2_audited_V2b")),
          ("\\quad and w/o V-REx", "5.7k", "--", "--", vrow(ftx("V2a"), "v2_audited_V2a")),
          ("\\quad and w/o gate", "5.7k", "--", "--", vrow(ftx("-G"), "v2_audited_-G_vrex")),
          ("\\quad and temporal mean only", "2.7k", "--", "--", vrow(ftx("-T"), "v2_audited_-T_vrex")),
          ("\\quad and CLS only (no grid)", "3.0k", "--", "--", vrow(ftx("-Sp"), "v2_audited_-Sp_vrex")),
          ("\\emph{Motion gates}", None),
          ("LENS-AG (learned gate)", "34.4k", gf("LENS-AG"), ms_("LENS-Net (L14, 16f)"), vrow(FGD["LENS-AG"], "v2_audited_LENS-AG_vrex")),
          ("LENS-FG (footage gate)$^\\S$", "31.9k", gf("LENS-Net"), ms_("LENS-Net (L14, 16f)"), vrow(FGD["LENS-FG"], "v2_audited_LENS-FG_vrex")),
          ("\\emph{Efficient variants (E2)}", None),
          ("L1: 8 CLIP frames", "31.9k", gf("L1 (L14, 8f)"), ms_("L1 (L14, 8f)"), vrow(lite_tx("runs_f8"), "f8_v2_audited_LENS-M_vrex")),
          ("L2: CLIP ViT-B/16", "31.4k", gf("L2 (B16, 16f)"), ms_("L2 (B16, 16f)"), vrow(lite_tx("runs_b16"), "b16_v2_audited_LENS-M_vrex")),
          ("L3: ViT-B/16, 8 frames", "31.4k", gf("L3 (B16, 8f)"), ms_("L3 (B16, 8f)"), vrow(lite_tx("runs_b16_f8"), "b16_f8_v2_audited_LENS-M_vrex")),
          ("\\emph{Fixed prompt set}", None),
          ("Paraphrased prompts", "31.9k", "--", "--", prow("LENS-Net (para)")),
          ("CCTV-worded prompts", "31.9k", "--", "--", prow("LENS-Net (surv)")),
          ("Broad prompts (8+8)", "34.4k", "--", "--", prow("LENS-Net (broad)"))]
    ref = T6[0][4]; lines = []
    mac("NsAblTCf", fmt(ref[7] - [r for r in T6 if r[0] == "\\quad and temporal mean only"][0][4][7]))
    for r in T6:
        if r[1] is None: lines.append(f"\\multicolumn{{12}}{{l}}{{{r[0]}}}\\\\"); continue
        lab, pr, g, ms, v = r
        cells = ["--" if x is None else (f"{x:.2f}" if lab.startswith("\\textbf") else f"{x:.2f}") for x in v]
        lines.append(f"{lab} & {pr} & {g} & {ms} & " + " & ".join(cells) + " \\\\")
    (OUT / "tables/t6_variants.tex").write_text("\n".join(lines) + "\n")
    print("consolidated journal tables written", flush=True)
# ================= LENS-Net-A confirmation (PREREG_journal Sec. 10)
C3R = JL("conf3_results.json")
if C3R:
    mm = lambda how, g, b: C3R["models"][how][g][b]["mean"]
    rows7 = [("\\textbf{LENS-Net-A}", "LENS-Net-A"), ("LENS-Net", "v2_audited_LENS-M_vrex"), ("CLIP probe (anchor only)", "v2_audited_LP"),
             ("Anchor + PPS (w/o motion)", "v2_audited_V2b"), ("Motion probe", "v2_audited_MP"),
             ("CLIP zero-shot", "ZS"), ("VadCLIP~\\cite{wu2024vadclip}", "VadCLIP-L14_audited_lr0.0001")]
    lines = []
    for lab, g in rows7:
        lines.append(f"{lab} & " + " & ".join(f"{mm(h, g, b):.2f}" for h in ("max", "top3") for b in ("XD-train", "AIE")) + " \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{l}{\\emph{Pre-registered claims (LENS-Net-A minus comparator, 95\\% CI)}}\\\\")
    for k, lab in zip(C3R["claims_max"]["claims"], ("K1: $-$ LENS-Net, XD-train ($>0$)", "K2: $-$ LENS-Net, AIE ($>-1$)", "K3: $-$ probe, XD-train ($>-1$)", "K4: $-$ probe, AIE ($>0$)")):
        cp, cs = C3R["claims_max"]["claims"][k], C3R["claims_top3"]["claims"][k]
        hp, hs = C3R["claims_max"]["holds"][k], C3R["claims_top3"]["holds"][k]
        OK, NO = "\\checkmark", "$\\times$"
        f = lambda d, h: "\\multicolumn{2}{c}{" + f"{sg(d['diff'])} [{sg(d['ci'][0])}, {sg(d['ci'][1])}] " + (OK if h else NO) + "}"
        lines.append(f"{lab} & {f(cp, hp)} & {f(cs, hs)} \\\\")
    (OUT / "tables/t7_confirm.tex").write_text("\n".join(lines) + "\n")
    for i, k in enumerate(C3R["claims_max"]["claims"], 1):
        mac(f"AK{'ABCD'[i-1]}", cic(C3R["claims_max"]["claims"][k])); mac(f"AKs{'ABCD'[i-1]}", cic(C3R["claims_top3"]["claims"][k]))
    for g, nm in (("LENS-Net-A", "A"), ("v2_audited_LENS-M_vrex", "Lens"), ("v2_audited_LP", "Lp"), ("v2_audited_V2b", "Nomo"), ("ZS", "Zs"),
                  ("VadCLIP-L14_audited_lr0.0001", "Vadclip"), ("LENS-Net-A (P3)", "AT"), ("v2_audited_LENS-FG_vrex", "Fg")):
        mac(f"Ac{nm}Xd", fmt(mm("max", g, "XD-train"))); mac(f"Ac{nm}Aie", fmt(mm("max", g, "AIE")))
    vo = C3R["vs_others_max"]
    for c, nm in (("v2_audited_V2b", "Nomo"), ("ZS", "Zs"), ("VadCLIP-L14_audited_lr0.0001", "Vadclip"), ("v2_audited_LENS-FG_vrex", "Fg")):
        for b, bn in (("XD-train", "Xd"), ("AIE", "Aie")): mac(f"Acd{nm}{bn}", cic(vo[f"A - {c}"][b]))
    mac("AcXdVid", str(C3R["n_videos"]["XD-train"])); mac("AcAieVid", str(C3R["n_videos"]["AIE"]))
    mac("AcXdCl", str(C3R["claims_max"]["claims"]["K1 A - LENS-Net on XD-train (better)"]["n_clusters"]))
    mac("AcExcl", str(sum(C3R["n_excluded_windows"].values())))
# write the macros again so that those added by the consolidated tables are included
(OUT / "numbers.tex").write_text("% generated by make_tables.py -- do not edit\n\\providecommand{\\pend}{\\textcolor{red}{\\textbf{[pending]}}}\n" +
                                  "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(M.items())))
print("journal extension macros written; pending:", [k for k, v in M.items() if v == "\\pend"])


# ---- emphasis: best value per column in bold, second best underlined (higher is better for every scored column);
# in the paired table, every significant gain of LENS-Net ($^*$) is bold. Runs last, so it applies to every table.
NUM = re.compile(r"^(\s*)((?:\$-\$|[+\-])?\d[\d,]*\.\d+)")


def _val(cell):
    m = NUM.match(cell)
    return None if not m else float(m.group(2).replace("$-$", "-").replace(",", ""))


def _strip(cell):
    return re.sub(r"\\(?:textbf|underline)\{((?:\$-\$|[+\-])?\d[\d,]*\.\d+)\}", r"\1", cell)


def _wrap(cell, cmd):
    return NUM.sub(lambda m: f"{m.group(1)}\\{cmd}{{{m.group(2)}}}", cell, count=1)


def emphasise(name, skip=(), paired=False, byrow=False):
    p = OUT / "tables" / name
    lines = p.read_text().split("\n")
    rows = [i for i, l in enumerate(lines) if "&" in l and "\\multicolumn" not in l and l.rstrip().endswith("\\\\")]
    cells = {i: [_strip(c) if j else c for j, c in enumerate(lines[i].rstrip()[:-2].split("&"))] for i in rows}
    if byrow:  # models are columns: rank within each row
        for i in rows:
            js = [j for j in range(1, len(cells[i])) if j not in skip and _val(cells[i][j]) is not None]
            vals = sorted({_val(cells[i][j]) for j in js}, reverse=True)
            for j in js:
                v = _val(cells[i][j])
                cells[i][j] = _wrap(cells[i][j], "textbf") if v == vals[0] else _wrap(cells[i][j], "underline") if len(vals) > 2 and v == vals[1] else cells[i][j]
    elif paired:
        for i in rows:
            cells[i] = [c if j == 0 or "$^*$" not in c else _wrap(c, "textbf") for j, c in enumerate(cells[i])]
    else:
        ncol = max(len(c) for c in cells.values())
        for j in range(1, ncol):
            if j in skip:
                continue
            vals = sorted({_val(cells[i][j]) for i in rows if j < len(cells[i]) and _val(cells[i][j]) is not None}, reverse=True)
            if len(vals) < 3:
                continue
            for i in rows:
                if j < len(cells[i]) and _val(cells[i][j]) is not None:
                    v = _val(cells[i][j])
                    cells[i][j] = _wrap(cells[i][j], "textbf") if v == vals[0] else _wrap(cells[i][j], "underline") if v == vals[1] else cells[i][j]
    for i in rows:
        lines[i] = "&".join(cells[i]) + "\\\\"
    p.write_text("\n".join(lines))


for name, skip in (("main.tex", ()), ("ablation_means.tex", ()),
                   ("t2_all.tex", (1, 2)), ("t4_p3.tex", ()), ("t6_variants.tex", (1, 2, 3)), ("framelevel.tex", ()),
                   ("t7_confirm.tex", ())):
    emphasise(name, skip)
emphasise("t3_paired.tex", paired=True)
emphasise("lodo.tex", (1,), byrow=True)
emphasise("standard.tex", (1,), byrow=True)
# House style of the manuscript: "without" is spelled out in every table label.
for _t in (OUT / "tables").glob("*.tex"):
    _t.write_text(_t.read_text().replace("w/o", "without"))
print("emphasis applied: best bold, second underlined; significant paired gains bold")
