"""Revision tables and macros from review/*.json: seed-aware paired table, primary-claims (Holm) table,
and review_numbers.tex. Every number in the revised text that is new comes from here."""
import json, re
from pathlib import Path
L = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace"); P = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "paper")
ST = json.load(open(L / "review/stats.json")); TS = json.load(open(L / "review/ten_seeds.json")); FI = json.load(open(L / "review/filtering.json"))
SETS = ["Test", "XD", "UCF", "SCFD", "SCVD", "VioPeru", "AIRTLab", "CCTV-F", "Movies", "UBI-Fights"]
mac = {}
f2 = lambda x: f"{x:.2f}"; sg = lambda x: (f"+{x:.2f}" if x >= 0 else f"$-${abs(x):.2f}")
def ci(r): return f"{sg(r['diff'])} [{sg(r['ci'][0])}, {sg(r['ci'][1])}]" + ("$^*$" if r["better"] else "$^\\dagger$" if r["worse"] else "")
# ---- seed-aware paired table (LENS-Net minus comparator)
rows = [("CLIP probe", "CLIP probe (anchor only)"), ("w/o motion", "Anchor + PPS (w/o motion)"), ("VadCLIP", "VadCLIP"), ("X-CLIP FT", "X-CLIP FT"),
        ("VideoMAE-B FT", "VideoMAE-B FT"), ("Video Swin-T FT", "Video Swin-T FT"), ("R(2+1)D-18 FT", "R(2+1)D-18 FT"), ("CNN-BiGRU", "CNN-BiGRU")]
lines = []
for k, lab in rows:
    cells = []
    for s in SETS:
        r = ST["pairwise"][f"LENS-Net - {k} | {s}"]; v = sg(r["diff"])
        cells.append((f"\\textbf{{{v}}}$^*$" if r["better"] else f"{v}$^\\dagger$" if r["worse"] else v))
    lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
(P / "tables/t3_paired_seedaware.tex").write_text("\n".join(lines) + "\n")
n_better = sum(v["better"] for k, v in ST["pairwise"].items() if k.startswith("LENS-Net - ")); n_worse = sum(v["worse"] for k, v in ST["pairwise"].items() if k.startswith("LENS-Net - "))
n_tot = sum(1 for k in ST["pairwise"] if k.startswith("LENS-Net - "))
flips = sum(1 for k, v in ST["pairwise"].items() if (v["better"], v["worse"]) != (ST["pairwise_seed_averaged"][k]["better"], ST["pairwise_seed_averaged"][k]["worse"]))
mac.update(SaBetter=str(n_better), SaWorse=str(n_worse), SaTotal=str(n_tot), SaFlips=str(flips), SaAll=str(len(ST["pairwise"])))
# ---- primary family table
labels = {"H1": ("CLIP probe $-$ VideoMAE-B FT", "nine unseen sets (mean)", "two-sided"),
          "H2": ("CLIP probe $-$ LENS-Net", "nine unseen sets (mean)", "two-sided"),
          "H3": ("LENS-Net $-$ VideoMAE-B FT", "nine unseen sets (mean)", "two-sided"),
          "H4": ("LENS-Net $-$ w/o motion", "SCFD", "better"), "H5": ("LENS-Net $-$ w/o motion", "SCVD", "better"),
          "K1": ("LENS-Net-A $-$ LENS-Net", "XD-train sample", "better"), "K2": ("LENS-Net-A $-$ LENS-Net", "AIE", "non-inferior"),
          "K3": ("LENS-Net-A $-$ CLIP probe", "XD-train sample", "non-inferior"), "K4": ("LENS-Net-A $-$ CLIP probe", "AIE", "better")}
plines = []
for k in sorted(ST["primary"], key=lambda x: (x[0] != "H", x[:2])):
    r = ST["primary"][k]; code = k[:2]; a, s, kind = labels[code]
    plines.append(f"{code} & {a} & {s} & {kind} & {sg(r['diff'])} [{sg(r['ci'][0])}, {sg(r['ci'][1])}] & {r['p']:.4f} & {r['holm_threshold']:.4f} & " + ("\\checkmark" if r["holds_after_holm"] else "$\\times$") + " \\\\")
    w = code[0] + {"1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five"}[code[1]]
    mac[f"Pr{w}"] = ci(r); mac[f"Pr{w}d"] = sg(r["diff"]); mac[f"Pr{w}p"] = f"{r['p']:.3f}"
(P / "tables/primary.tex").write_text("\n".join(plines) + "\n")
mac["PrNhold"] = str(sum(r["holds_after_holm"] for r in ST["primary"].values())); mac["PrN"] = str(len(ST["primary"]))
# ---- symmetric fusion, ten seeds, filtering
sf = ST["symmetric_fusion"]
for k, nm in (("LENS-Net-A", "A"), ("SYM:VideoMAE-B FT", "Vmae"), ("SYM:Motion probe", "Mp"), ("SYM:VadCLIP", "Vadclip"), ("SYM:Video Swin-T FT", "Swin"),
              ("SYM:w/o motion", "Nomo"), ("SYM:R(2+1)D-18 FT", "Rtd"), ("SYM:X-CLIP FT", "Xclip"), ("SYM:CNN-BiGRU", "Mast")):
    mac[f"Sf{nm}Mean"] = f2(sf[k]["mean9"]); mac[f"Sf{nm}Min"] = f2(sf[k]["min9"])
ps = TS["per_set"]
for k, nm in (("LENS-Net", "Lens"), ("CLIP probe", "Lp"), ("w/o motion", "Nomo"), ("Motion probe", "Mp"), ("LENS-Net-A", "A")):
    mac[f"Ts{nm}Mean"] = f2(ps[k]["mean9"]); mac[f"Ts{nm}Min"] = f2(ps[k]["min9"])
    mac[f"Ts{nm}MoviesSd"] = f2(ps[k]["Movies"]["sd"]); mac[f"Ts{nm}XdSd"] = f2(ps[k]["XD"]["sd"])
t = TS["tests"]
mac["TsHtwo"] = ci(t["H2 CLIP probe - LENS-Net (nine-set mean), 10 seeds"]); mac["TsALp"] = ci(t["LENS-Net-A - CLIP probe (nine-set mean), 10 seeds"])
mac["TsALens"] = ci(t["LENS-Net-A - LENS-Net (nine-set mean), 10 seeds"]); mac["TsHfour"] = ci(t["H4 LENS-Net - w/o motion on SCFD, 10 seeds"])
mac["TsHfive"] = ci(t["H5 LENS-Net - w/o motion on SCVD, 10 seeds"])
fs = {k: v for k, v in FI.items() if v["n_removed"] > 0}
mac["FiMaxChange"] = f2(max(abs(v["mean_change"]) for v in fs.values())); mac["FiMinTau"] = f"{min(v['kendall_tau'] for v in fs.values()):.2f}"
mac["FiMaxRemoved"] = str(max(v["n_removed"] for v in fs.values()))
mac["FiLensMaxChange"] = f2(max(abs(v["models"]["LENS-Net"]["change"]) for v in fs.values()))
old = ST["pairwise_seed_averaged"]["LENS-Net - VadCLIP | Movies"]; new = ST["pairwise"]["LENS-Net - VadCLIP | Movies"]
mac["ExOldVadMov"] = ci(old); mac["ExNewVadMov"] = ci(new)
o = ST["pairwise_seed_averaged"]["LENS-Net - VideoMAE-B FT | CCTV-F"]; n = ST["pairwise"]["LENS-Net - VideoMAE-B FT | CCTV-F"]
mac["ExOldVmaeCf"] = ci(o); mac["ExNewVmaeCf"] = ci(n)
for s, nm in (("XD", "XD"), ("VioPeru", "Vp"), ("AIRTLab", "Airt"), ("CCTV-F", "Cf"), ("SCFD", "SCFD"), ("SCVD", "SCVD"), ("UCF", "UCF"), ("Movies", "Mov"), ("UBI-Fights", "Ubi"), ("Test", "Test")):
    for c, cn in (("VideoMAE-B FT", "V"), ("Video Swin-T FT", "S"), ("R(2+1)D-18 FT", "R"), ("CNN-BiGRU", "C"), ("VadCLIP", "Vad"), ("X-CLIP FT", "X"), ("w/o motion", "N"), ("CLIP probe", "P")):
        mac[f"Sa{cn}{nm}"] = ci(ST["pairwise"][f"LENS-Net - {c} | {s}"])
# fixed audit facts (frozen audit outputs; B4/B6 of PREREG_revision.md)
mac.update(AuCand="127{,}752", AuEdges="132{,}962", AuStageThreeRej="173", AuComp="3{,}232", AuCompSingle="2{,}758", AuCompMulti="474",
           AuGiantWin="11{,}158", AuGiantVid="3{,}006", AuCalN="46", AuCalPrec="0.84", AuCalRec="0.97", AuCalTP="31", AuCalFP="6", AuCalFN="1",
           AmbCfAll="9.7", AmbCfCc="7.3", AmbUbi="0.7")
(P / "review_numbers.tex").write_text("% generated by src/revision/make_revision_macros_part_b.py -- do not edit\n" +
                                       "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(mac.items())))
print(len(mac), "macros;", "paired seed-aware better/worse/total", n_better, n_worse, n_tot, "flips", flips)

# House style of the manuscript: "without" is spelled out in every table label.
for _t in (P / "tables").glob("*.tex"):
    _t.write_text(_t.read_text().replace("w/o", "without"))
