"""Revision parts C and D: LaTeX macros (digit-free names) and two tables from review/{c1_results,c2_laion,c3_dg,
c6_confirmB,c7_filter090,d1_labels_final}.json. Writes paper/revision_c_numbers.tex and
the tables of the control, fusion and audit studies under paper/tables/."""
import json
import pandas as pd
from pathlib import Path
R = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/review"); O = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "paper")
(O / "tables").mkdir(parents=True, exist_ok=True)
J = lambda f: json.load(open(R / f))
c1, c2, c3, c6, c7, d1 = (J(f) for f in ("c1_results.json", "c2_laion.json", "c3_dg.json", "c6_confirmB.json", "c7_filter090.json", "d1_labels_final.json"))
M = {}
def f2(x): return f"{x:.2f}"
def sg(x): return f"{x:+.2f}".replace("-", "$-$")
def ci(r): return f"{sg(r['diff'])} [{sg(r['ci'][0])}, {sg(r['ci'][1])}]"
# D1
st = d1["strata"]; w = d1["weighted"]
M.update(DoVerK=st["verified_edge"]["same"], DoVerN=st["verified_edge"]["n"], DoVerLo=f"{100*st['verified_edge']['wilson95'][0]:.0f}",
         DoRandK=st["random_pair"]["same"], DoRandN=st["random_pair"]["n"], DoRejK=st["rejected_candidate"]["same"], DoRejN=st["rejected_candidate"]["n"],
         DoCosK=st["cosine_removed"]["same"], DoCosN=st["cosine_removed"]["n"], DoMidK=st["cosine_0.90-0.95"]["same"], DoMidN=st["cosine_0.90-0.95"]["n"],
         DoN=d1["n_labelled"], DoRecall=f"{w['recall_verification_among_candidates']:.3f}",
         DoRecallCI=f"[{w['recall_ci95_jeffreys'][0]:.3f}, {w['recall_ci95_jeffreys'][1]:.3f}]",
         DoNVer=f"{w['N_verified']:,}".replace(",", "{,}"), DoNRej=f"{w['N_rejected']:,}".replace(",", "{,}"),
         DoCrossPop=w["population_cross_split_rejected"], DoCrossPopTT=w["population_cross_split_rejected_train_test"])
# C1
for ds, tag in (("RWF2000", "Rwf"), ("HockeyFights", "Hoc")):
    for m, mt in (("VideoMAE-B FT", "Vmae"), ("R(2+1)D-18 FT", "Rtd"), ("CLIP probe", "Lp"), ("LENS-Net", "Lens")):
        r = c1[ds][m]; M[f"Co{tag}{mt}RmS"] = ci(r["R_minus_S"])
        M[f"Co{tag}{mt}Link"] = f2(r["R_linked_auc"]); M[f"Co{tag}{mt}Unl"] = f2(r["R_unlinked_auc"])
        for t in ("XD", "UCF"): M[f"Co{tag}{mt}Tr{t.capitalize()}"] = ci(r["transfer"][t]["R_minus_S"])
    M[f"Co{tag}Share"] = f"{100*c1[ds]['VideoMAE-B FT']['R_linked_share']:.0f}"
# C2
a = c2["auc"]; t = c2["tests"]
for m, mt in (("LAION CLIP probe", "Lp"), ("LAION LENS-Net", "Lens"), ("LAION LENS-Net-A", "A"), ("LAION w/o motion", "Nomo")):
    M[f"Ct{mt}Mean"] = f2(a[m]["mean9"]); M[f"Ct{mt}Min"] = f2(a[m]["min9"])
M.update(CtLpVmae=ci(t["LAION CLIP probe - VideoMAE-B FT (nine-set mean)"]), CtLensVmae=ci(t["LAION LENS-Net - VideoMAE-B FT (nine-set mean)"]),
         CtAVmae=ci(t["LAION LENS-Net-A - VideoMAE-B FT (nine-set mean)"]), CtLpLens=ci(t["LAION CLIP probe - LAION LENS-Net (nine-set mean)"]),
         CtALens=ci(t["LAION LENS-Net-A - LAION LENS-Net (nine-set mean)"]), CtLpOai=ci(t["LAION CLIP probe - CLIP probe (nine-set mean)"]),
         CtLensOai=ci(t["LAION LENS-Net - LENS-Net (nine-set mean)"]), CtAOai=ci(t["LAION LENS-Net-A - LENS-Net-A (nine-set mean)"]),
         CtNomoMovies=f2(a["LAION w/o motion"]["Movies"]))
# C3
v = {k: c3[f"L14_audited_LENS-M_{k}"] for k in ("vrex0", "vrex0.3", "vrex3", "vrex10", "irm", "gdro")}
means = [c3["published LENS-Net (beta=1)"]["mean9"]] + [v[k]["mean9"] for k in ("vrex0", "vrex0.3", "vrex3", "vrex10")]
M.update(CdBetaLo=f2(min(means)), CdBetaHi=f2(max(means)), CdIrm=f2(v["irm"]["mean9"]), CdGdro=f2(v["gdro"]["mean9"]),
         CdLpIrm=f2(c3["L14_audited_LP_irm"]["mean9"]), CdLpGdro=f2(c3["L14_audited_LP_gdro"]["mean9"]))
# C6
cl = c6["claims"]; M.update(CsN=c6["n_videos"], CsClusters=c6["n_clusters"], CsExcl=c6["n_windows_excluded"],
                            CsKOne=ci(cl["K1 LENS-Net-A - LENS-Net (better)"]), CsKThree=ci(cl["K3 LENS-Net-A - CLIP probe (non-inferior)"]),
                            CsA=f2(c6["auc"]["LENS-Net-A"]), CsLens=f2(c6["auc"]["LENS-Net"]), CsLp=f2(c6["auc"]["CLIP probe"]), CsZs=f2(c6["auc"]["CLIP zero-shot"]))
# C7
rm = c7["removed"]; ch = [c7["mean9"][m]["at090"] - c7["mean9"][m]["at095"] for m in c7["mean9"]]
M.update(CfUbiRm=f"{rm['UBI-Fights']['at090']:,}".replace(",", "{,}"), CfUbiN=f"{rm['UBI-Fights']['windows']:,}".replace(",", "{,}"),
         CfMaxDrop=f"{-min(ch):.2f}", CfMinTau=f"{min(c7['tau'].values()):.2f}",
         CfHOne=ci(c7["claims"]["H1 CLIP probe - VideoMAE-B FT"]["at090"]), CfHTwo=ci(c7["claims"]["H2 CLIP probe - LENS-Net"]["at090"]),
         CfHThree=ci(c7["claims"]["H3 LENS-Net - VideoMAE-B FT"]["at090"]),
         CfHTwoP=f'{c7["claims"]["H2 CLIP probe - LENS-Net"]["at090"]["p_two_sided"]:.3f}')
# C4 / C5
c45 = J("c45.json"); c4, c5 = c45["C4"], c45["C5"]; e1 = J("e1_clipft.json")
M.update(FbVmaeOld=f2(c4["auc"]["VideoMAE-B FT"]["mean9"]), FbVmaeTuned=f2(c4["auc"]["VideoMAE-B FT (tuned)"]["mean9"]),
         FbRtdOld=f2(c4["auc"]["R(2+1)D-18 FT"]["mean9"]), FbRtdTuned=f2(c4["auc"]["R(2+1)D-18 FT (tuned)"]["mean9"]),
         FbLora=f2(c5["auc"]["VideoMAE-B LoRA"]["mean9"]), FbLoraMin=f2(c5["auc"]["VideoMAE-B LoRA"]["min9"]),
         FbLoraParams=f"{c5['trained_params']/1000:.0f}k",
         FbHOneTuned=ci(c4["claims"]["H1 CLIP probe - VideoMAE-B FT"]["tuned"]), FbHThreeTuned=ci(c4["claims"]["H3 LENS-Net - VideoMAE-B FT"]["tuned"]),
         FbHOneTunedP=f'{c4["claims"]["H1 CLIP probe - VideoMAE-B FT"]["tuned"]["p_two_sided"]:.3f}',
         FbHThreeTunedP=f'{c4["claims"]["H3 LENS-Net - VideoMAE-B FT"]["tuned"]["p_two_sided"]:.3f}',
         FbHOneThr=f'{c4["holm_with_tuned"]["H1 CLIP probe - VideoMAE-B FT (nine-set mean)"]["threshold"]:.4f}',
         FbAVmaeTuned=ci(c4["tests"]["LENS-Net-A - VideoMAE-B FT (tuned) (nine-set mean)"]),
         FbLpRtdTuned=ci(c4["tests"]["CLIP probe - R(2+1)D-18 FT (tuned) (nine-set mean)"]),
         FbLensRtdTuned=ci(c4["tests"]["LENS-Net - R(2+1)D-18 FT (tuned) (nine-set mean)"]),
         FbTunedGain=ci(c4["tests"]["VideoMAE-B FT (tuned) - VideoMAE-B FT (nine-set mean)"]),
         FbLoraTuned=ci(c4["tests"]["VideoMAE-B LoRA - VideoMAE-B FT (tuned) (nine-set mean)"]),
         FbLoraFt=ci(c5["tests"]["VideoMAE-B LoRA - VideoMAE-B FT (nine-set mean)"]), FbLpLora=ci(c5["tests"]["CLIP probe - VideoMAE-B LoRA (nine-set mean)"]),
         FbLensLora=ci(c5["tests"]["LENS-Net - VideoMAE-B LoRA (nine-set mean)"]), FbALora=ci(c5["tests"]["LENS-Net-A - VideoMAE-B LoRA (nine-set mean)"]))
rows = []
for m, lab, src in (("VideoMAE-B FT", "VideoMAE-B FT, shared lr $5\\times10^{-5}$", c4), ("VideoMAE-B FT (tuned)", "VideoMAE-B FT, tuned lr $2\\times10^{-5}$", c4),
                    ("VideoMAE-B LoRA", "VideoMAE-B LoRA (rank 8)", c5), ("R(2+1)D-18 FT", "R(2+1)D-18 FT, shared lr", c4), ("R(2+1)D-18 FT (tuned)", "R(2+1)D-18 FT, tuned lr $2\\times10^{-5}$", c4),
                    ("CLIP-FT", "CLIP-FT (last four blocks)", e1), ("CLIP probe", "CLIP probe", c5), ("LENS-Net", "LENS-Net", c5), ("LENS-Net-A", "LENS-Net-A", c5)):
    vv = src["auc"][m]; rows.append(f"{lab} & {f2(vv['Test'])} & {f2(vv['mean9'])} & {f2(vv['min9'])}\\\\")
    if m in ("VideoMAE-B LoRA", "R(2+1)D-18 FT (tuned)"): rows.append("\\midrule")
(O / "tables/fair.tex").write_text("\n".join(rows) + "\n")
# E2 / E3
e = J("e23.json"); mm9 = lambda k: e[k]["mean9"]
lp = [mm9(f"L14_audited_LP_none_{x}") for x in ("s15", "s60", "la0", "la0.1", "la2")] + [e["published CLIP probe"]["mean9"]]
ln = [mm9(f"L14_audited_LENS-M_vrex1_{x}") for x in ("s15", "s60", "la0", "la0.1", "la2")] + [e["published LENS-Net"]["mean9"]]
mp = [mm9(f"L14_audited_MP_{x}") for x in ("irm", "gdro", "vrex1")]
M.update(EsLpLo=f2(min(lp)), EsLpHi=f2(max(lp)), EsLensLo=f2(min(ln)), EsLensHi=f2(max(ln)),
         EsLensValSel=f2(mm9("L14_audited_LENS-M_vrex1_la0")), EsLpValSel=f2(mm9("L14_audited_LP_none_la0.1")),
         EsLensSFifteenMovies=f2(e["L14_audited_LENS-M_vrex1_s15"]["Movies"]),
         EdMpLo=f2(min(mp)), EdMpHi=f2(max(mp)), EdMpNone=f2(mm9("L14_audited_MP_none")), EdMpPub=f2(e["published Motion probe"]["mean9"]))
# E4
am = J("e4_amb.json")["sets"]
dr = [v["labelled"] - v["with_ambiguous"] for st in am.values() for v in st["models"].values()]
M.update(AmCfAdd=am["CCTV-F"]["n_ambiguous"], AmUbiAdd=am["UBI-Fights"]["n_ambiguous"], AmCfTau=f'{am["CCTV-F"]["kendall_tau"]:.2f}',
         AmUbiTau=f'{am["UBI-Fights"]["kendall_tau"]:.2f}', AmDropLo=f"{min(dr):.1f}", AmDropHi=f"{max(dr):.1f}",
         AmCfLens=f'{am["CCTV-F"]["models"]["LENS-Net"]["with_ambiguous"]:.2f}', AmUbiA=f'{am["UBI-Fights"]["models"]["LENS-Net-A"]["with_ambiguous"]:.2f}')
ir = d1["intra_rater"]
M.update(DoIrN=ir["n"], DoIrAgree=f"{100*ir['agree_same_source']:.0f}", DoIrKappa=f"{ir['kappa_same_source']:.2f}", DoIrKappaThree=f"{ir['kappa_3class']:.2f}")
e1 = J("e1_clipft.json"); t1 = e1["tests"]
M.update(FtcMean=f2(e1["auc"]["CLIP-FT"]["mean9"]), FtcMin=f2(e1["auc"]["CLIP-FT"]["min9"]), FtcParams=f"{e1['trained_params']/1e6:.1f}M",
         FtcLp=ci(t1["CLIP-FT - CLIP probe (nine-set mean)"]), FtcVmae=ci(t1["CLIP-FT - VideoMAE-B FT (tuned) (nine-set mean)"]),
         FtcLora=ci(t1["CLIP-FT - VideoMAE-B LoRA (nine-set mean)"]), FtcA=ci(t1["LENS-Net-A - CLIP-FT (nine-set mean)"]),
         FtcTest=ci(e1["test_set"]["CLIP-FT - CLIP probe"]))
fi = json.load(open(str(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "paper/figs/fig_idood_data.json")))
M.update(IoTestLo=f"{min(v[0] for v in fi.values()):.1f}", IoTestHi=f"{max(v[0] for v in fi.values()):.1f}",
         IoNineLo=f"{min(v[1] for v in fi.values()):.1f}", IoNineHi=f"{max(v[1] for v in fi.values()):.1f}")
# E5 fusion rules
e5 = J("e5_fusion.json"); fam5 = e5["family"]; pr = e5["pairs"]; de = e5["descriptive"]
k0 = "CLIP probe + LENS-Net"
M.update(FrMin=ci(fam5["R5 min - R1 (anchor + LENS-Net)"]), FrStack=ci(fam5["R8 stacking - R1 (anchor + LENS-Net)"]),
         FrLogit=ci(fam5["R2 mean logit - R1 (anchor + LENS-Net)"]), FrValw=ci(fam5["R7 val-weighted - R1 (anchor + LENS-Net)"]),
         FrMax=ci(fam5["R4 max - R1 (anchor + LENS-Net)"]), FrRank=ci(fam5["R6 rank - R1 (anchor + LENS-Net)"]),
         FrHoldsN=sum(v["holds_after_holm"] for v in fam5.values()),
         FrLora=f2(pr["CLIP probe + VideoMAE-B LoRA"]["R1 mean prob"]["mean9"]), FrLoraMin=f2(pr["CLIP probe + VideoMAE-B LoRA"]["R1 mean prob"]["min9"]),
         FrVmae=f2(pr["CLIP probe + VideoMAE-B FT (tuned)"]["R1 mean prob"]["mean9"]),
         FrLoraGain=ci(de["R1 - better member (CLIP probe + VideoMAE-B LoRA; CLIP probe)"]),
         FrVmaeGain=ci(de["R1 - better member (CLIP probe + VideoMAE-B FT (tuned); CLIP probe)"]),
         FrNoAnchorVmae=f2(pr["LENS-Net + VideoMAE-B FT (tuned)"]["R1 mean prob"]["mean9"]), FrNoAnchorLora=f2(pr["LENS-Net + VideoMAE-B LoRA"]["R1 mean prob"]["mean9"]),
         FrValMean=f2(pr[k0]["R1 mean prob"]["val"]), FrValStack=f2(pr[k0]["R8 stacking"]["val"]))
lrn = {k: v for k, v in de.items() if k.startswith(("R7", "R8"))}
M.update(FrLearnedBetter=sum(v["better"] for v in lrn.values()), FrLearnedWorse=sum(v["worse"] for v in lrn.values()), FrLearnedN=len(lrn))
wts = [w for kk, v in pr.items() if kk.startswith("CLIP probe +") for w in v["R7 val-weighted"]["weights_on_first"]]
M.update(FrWLo=f"{min(wts):.2f}", FrWHi=f"{max(wts):.2f}")
short = {"CLIP probe + LENS-Net": "Anchor + LENS-Net (LENS-Net-A)", "CLIP probe + w/o motion": "Anchor + w/o motion", "CLIP probe + Motion probe": "Anchor + motion probe",
         "CLIP probe + VideoMAE-B FT (tuned)": "Anchor + VideoMAE-B (tuned)", "CLIP probe + VideoMAE-B LoRA": "Anchor + VideoMAE-B LoRA", "CLIP probe + CLIP-FT": "Anchor + CLIP-FT",
         "CLIP probe + VadCLIP": "Anchor + VadCLIP", "LENS-Net + VideoMAE-B FT (tuned)": "LENS-Net + VideoMAE-B (tuned)", "LENS-Net + VideoMAE-B LoRA": "LENS-Net + VideoMAE-B LoRA"}
RL = ["R1 mean prob", "R2 mean logit", "R3 geometric", "R4 max", "R5 min", "R6 rank", "R7 val-weighted", "R8 stacking"]
rows = []
for kk, lab in short.items():
    vals = [pr[kk][r]["mean9"] for r in RL]; best = max(vals)
    cells = [("\\textbf{%s}" % f2(v)) if abs(v - best) < 1e-9 else f2(v) for v in vals]
    rows.append(f"{lab} & " + " & ".join(cells) + "\\\\")
    if kk == "CLIP probe + VadCLIP": rows.append("\\midrule")
(O / "tables/fusion_rules.tex").write_text("\n".join(rows) + "\n")
# E6 diversity
e6 = J("e6_diversity.json"); pp = e6["pairs"]
pv = e6["perm_p_one_sided"]
M.update(DvRho=f"{e6['spearman_D1_gain']:.2f}", DvPerm=("<0.001" if pv < 0.001 else f"{pv:.3f}"), DvN=e6["n_points"],
         DvBone=f"{e6['ols_b1_D1']:+.2f} [{e6['ols_b1_ci'][0]:+.2f}, {e6['ols_b1_ci'][1]:+.2f}]".replace("-", "$-$"),
         DvBtwo=f"{e6['ols_b2_gap']:+.2f} [{e6['ols_b2_ci'][0]:+.2f}, {e6['ols_b2_ci'][1]:+.2f}]".replace("-", "$-$"),
         DvKV=f"{e6['krogh_vedelsby_max_abs_error']:.0e}".replace("e-", "\\times10^{-") + "}",
         DvLoraDone=f"{pp['CLIP probe + VideoMAE-B LoRA']['D1']:.2f}", DvVmaeDone=f"{pp['CLIP probe + VideoMAE-B FT (tuned)']['D1']:.2f}",
         DvLensDone=f"{pp['CLIP probe + LENS-Net']['D1']:.2f}", DvVadDone=f"{pp['CLIP probe + VadCLIP']['D1']:.2f}",
         DvMpDone=f"{pp['CLIP probe + Motion probe']['D1']:.2f}", DvMpGap=f"{pp['CLIP probe + Motion probe']['gap']:.1f}",
         DvMpGain=f"{pp['CLIP probe + Motion probe']['gain']:+.2f}".replace("-", "$-$"),
         DvVmaeGain=f"{pp['CLIP probe + VideoMAE-B FT (tuned)']['gain']:+.2f}", DvLoraGain=f"{pp['CLIP probe + VideoMAE-B LoRA']['gain']:+.2f}",
         DvLensLoraGain=f"{pp['LENS-Net + VideoMAE-B LoRA']['gain']:+.2f}",
         DvNGained=["no", "one", "two", "three", "four", "five"][sum(v['gain'] > 0 for v in pp.values())])
# Part F: anchored ensemble selection (AES): development (exploratory), confirmation (pre-registered), diagnosis, fix
from scipy.stats import spearmanr
ae_sel = J("explore_anchored_selection.json")["tol 0.05"]; ae_ps = J("explore_partner_select.json"); ae_fx = J("explore_aes_fixes_dev.json")
ae_cf = J("f_aes_confirm.json"); ae_dg = J("f_aes_diagnosis.json")
ae_rho = spearmanr([v["val_auc_fused"] for v in ae_ps.values()], [v["nine_set_fused (truth)"] for v in ae_ps.values()]).correlation
fxA, fx0 = ae_fx["(a) balanced 1/2, 1/4, 1/4"], ae_fx["AES (equal weights, as tested)"]
au = ae_cf["auc_max"]; cl = ae_cf["claims_max"]; ck = {k.split()[0]: v for k, v in cl.items()}
gap = lambda s: au["CLIP probe"][s] - (au["VideoMAE-B LoRA"][s] + au["VideoMAE-B FT (tuned)"][s]) / 2
M.update(AeSvAnchor=f2(ae_sel[0]["source_val_auc"]), AeSvPair=f2(ae_sel[1]["source_val_auc"]), AeSvTriple=f2(ae_sel[2]["source_val_auc"]),
         AeDevMean=f2(ae_sel[2]["nine_mean"]), AeDevMin=f2(ae_sel[2]["nine_min"]), AePartnerRho=f"{ae_rho:+.2f}", AeNCand=["no", "one", "two", "three", "four", "five", "six", "seven"][len(ae_ps)],
         AeFixMean=f2(fxA["mean9"]), AeFixMin=f2(fxA["min9"]), AeFixXd=sg(fxA["XD"] - fx0["XD"]), AeFixMovies=sg(fxA["Movies"] - fx0["Movies"]),
         AeFixAirt=sg(fxA["AIRTLab"] - fx0["AIRTLab"]), AeFixUcf=sg(fxA["UCF"] - fx0["UCF"]), AeFixCctv=sg(fxA["CCTV-F"] - fx0["CCTV-F"]),
         AeRankMean=f2(ae_fx["(b) rank-normalised, equal"]["mean9"]),
         AeFOne=ci(ck["F1"]), AeFTwo=ci(ck["F2"]), AeFThree=ci(ck["F3"]), AeFFour=ci(ck["F4"]), AeFFive=ci(ck["F5"]), AeFSix=ci(ck["F6"]),
         AeAieAes=f2(au["AES"]["AIE"]), AeAieA=f2(au["LENS-Net-A"]["AIE"]), AeAieLp=f2(au["CLIP probe"]["AIE"]),
         AeGapXdA=f"{gap('XD sample A'):.1f}", AeGapXdB=f"{gap('XD sample B'):.1f}", AeGapAie=sg(gap("AIE")),
         AeLoudLpLo=f"{min(v['CLIP probe'] for v in ae_dg.values()):.2f}", AeLoudLpHi=f"{max(v['CLIP probe'] for v in ae_dg.values()):.2f}",
         AeLoudVLo=f"{min(min(v['VideoMAE-B LoRA'], v['VideoMAE-B FT (tuned)']) for v in ae_dg.values()):.2f}",
         AeLoudVHi=f"{max(max(v['VideoMAE-B LoRA'], v['VideoMAE-B FT (tuned)']) for v in ae_dg.values()):.2f}")
SAM = ["XD sample A", "AIE", "XD sample B"]
rows = []
for m, lab in (("AES", "\\textbf{AES} (anchor + LoRA + tuned)"), ("LENS-Net-A", "LENS-Net-A"), ("CLIP probe", "CLIP probe (anchor)"),
               ("VideoMAE-B LoRA", "VideoMAE-B LoRA"), ("VideoMAE-B FT (tuned)", "VideoMAE-B FT (tuned)"), ("CLIP zero-shot", "CLIP zero-shot")):
    rows.append(f"{lab} & " + " & ".join(f2(au[m][s]) for s in SAM) + "\\\\")
rows.append("\\midrule")
rows.append("\\multicolumn{4}{l}{\\emph{Pre-registered claims (95\\% CI, \\checkmark: holds after Holm correction, $\\times$: fails)}}\\\\")
mark = lambda t: "\\checkmark" if ck[t]["holds_after_holm"] else "$\\times$"
rows.append("AES $-$ LENS-Net-A ($>0$) & " + " & ".join(ci(ck[t]) + " " + mark(t) for t in ("F1", "F2", "F3")) + "\\\\")
rows.append("AES $-$ CLIP probe ($>-1$) & " + " & ".join(ci(ck[t]) + " " + mark(t) for t in ("F4", "F5", "F6")) + "\\\\")
(O / "tables/aes_confirm.tex").write_text("\n".join(rows) + "\n")
# Part G: new untouched samples (UCF-train, HMDB51) and ensemble baselines on all five confirmation samples
pg = J("g_partG.json"); pa = pg["confirmation_auc_max"]; pd_ = pg["development"]; pc = pg["claims_max"]
GS = ["XD sample A", "AIE", "XD sample B", "UCF-train", "HMDB51"]
RN = {"LENS-Net-A (anchor + LENS-Net)": "LENS-Net-A", "AES-bal (1/2, 1/4, 1/4)": "AES-bal", "AES (anchor + LoRA + tuned, equal)": "AES",
      "Mean of all eight candidates": "Mean of all eight models", "Caruana selection with replacement": "Caruana selection (with replacement)",
      "Stacking of all eight (logistic)": "Stacking (logistic, all eight)", "Majority vote at 0.5": "Majority vote",
      "Likelihood weights (softmax of val log-lik.)": "Likelihood weights", "Best single candidate (source val)": "Best single model (source val.)"}
m5 = lambda r: sum(pa[r][s] for s in GS) / len(GS); n5 = lambda r: min(pa[r][s] for s in GS)
LA, BAL, AES_, ALL, STK = "LENS-Net-A (anchor + LENS-Net)", "AES-bal (1/2, 1/4, 1/4)", "AES (anchor + LoRA + tuned, equal)", "Mean of all eight candidates", "Stacking of all eight (logistic)"
ux = pd.read_csv(str(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/conf4/near_duplicate_exclusion.csv"))
M.update(PgUcfN=pg["n_videos"]["UCF-train"], PgUcfViol=pg["n_violent"]["UCF-train"], PgHmdbN=pg["n_videos"]["HMDB51"],
         PgHmdbViol=pg["n_violent"]["HMDB51"], PgHmdbClust=pg["n_clusters"]["HMDB51"], PgUcfExcl=f"{pg['n_windows_excluded']['UCF-train']:,}".replace(",", "{,}"),
         PgUcfWin=f"{int((ux.benchmark == 'UCF-train').sum()):,}".replace(",", "{,}"),
         **{f"PgG{w}": ci(pc[f"G{i}"]) for i, w in enumerate(["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"], 1)},
         PgLaMeanFive=f2(m5(LA)), PgBalMeanFive=f2(m5(BAL)), PgAllMeanFive=f2(m5(ALL)), PgAesMeanFive=f2(m5(AES_)), PgStackMeanFive=f2(m5(STK)),
         PgBestMeanFive=f2(m5("Best single candidate (source val)")), PgLaMinFive=f2(n5(LA)), PgAllMinFive=f2(n5(ALL)), PgStackMinFive=f2(n5(STK)), PgBalMinFive=f2(n5(BAL)),
         PgLaUcf=f2(pa[LA]["UCF-train"]), PgLaHmdb=f2(pa[LA]["HMDB51"]), PgAesUcf=f2(pa[AES_]["UCF-train"]), PgAesHmdb=f2(pa[AES_]["HMDB51"]),
         PgBalUcf=f2(pa[BAL]["UCF-train"]), PgBalHmdb=f2(pa[BAL]["HMDB51"]), PgLpUcf=f2(pa["CLIP probe"]["UCF-train"]), PgLpHmdb=f2(pa["CLIP probe"]["HMDB51"]),
         PgZsHmdb=f2(pa["CLIP zero-shot"]["HMDB51"]), PgAllUcf=f2(pa[ALL]["UCF-train"]), PgStackAie=f2(pa[STK]["AIE"]), PgLaAie=f2(pa[LA]["AIE"]),
         PgLaBeatsAnchor=["no", "one", "two", "three", "four", "five"][sum(pa[LA][s] > pa["CLIP probe"][s] for s in GS)],
         PgLaTopTwo=["no", "one", "two", "three", "four", "five"][sum(sorted([r for r in pa if r in RN], key=lambda r: -pa[r][s]).index(LA) < 2 for s in GS)])
fus = [r for r in pa if r in RN]
best = {s: max(fus, key=lambda r: pa[r][s]) for s in GS}; bestm, bestn = max(fus, key=m5), max(fus, key=n5)
bestd = max(fus, key=lambda r: pd_[r]["mean9"])
cell = lambda v, b: ("\\textbf{" + f2(v) + "}") if b else f2(v)
rows = []
for r in [LA, BAL, AES_, ALL, "Caruana selection with replacement", STK, "Majority vote at 0.5", "Likelihood weights (softmax of val log-lik.)", "Best single candidate (source val)"]:
    rows.append(f"{RN[r]} & {cell(pd_[r]['mean9'], r == bestd)} & " + " & ".join(cell(pa[r][s], r == best[s]) for s in GS) + f" & {cell(m5(r), r == bestm)} & {cell(n5(r), r == bestn)}\\\\")
rows.append("\\midrule")
for r, lab in (("CLIP probe", "CLIP probe (anchor alone)"), ("CLIP zero-shot", "CLIP zero-shot")):
    rows.append(f"{lab} & -- & " + " & ".join(f2(pa[r][s]) for s in GS) + f" & {f2(m5(r))} & {f2(n5(r))}\\\\")
(O / "tables/fusion_untouched.tex").write_text("\n".join(rows) + "\n")
# Part H: fusion-literature baselines, intervals against LENS-Net-A, window-level results, backbone passes
ph = J("h_partH.json"); hu = ph["untouched_auc_max"]; hd = ph["development"]; hi = ph["lensa_minus_rule_max"]; hw = ph["window_level"]
GS = ["XD sample A", "AIE", "XD sample B", "UCF-train", "HMDB51"]
BB = {"CLIP probe": {"CLIP"}, "LENS-Net": {"CLIP", "VMAE"}, "w/o motion": {"CLIP"}, "Motion probe": {"VMAE"}, "VideoMAE-B FT (tuned)": {"VMAE-tuned"},
      "VideoMAE-B LoRA": {"VMAE-LoRA"}, "CLIP-FT": {"CLIP-FT"}, "VadCLIP": {"CLIP"}}
ALL8 = list(BB); S1_, S2_ = ["CLIP probe", "LENS-Net"], ["CLIP probe", "VideoMAE-B LoRA", "VideoMAE-B FT (tuned)"]
car = set().union(*[{m for m, w in c.items() if w} for c in pg["weights_info"]["caruana"]])
MEM = {"LENS-Net-A (anchor + LENS-Net)": S1_, "AES-bal (1/2, 1/4, 1/4)": S2_, "AES (anchor + LoRA + tuned, equal)": S2_, "Mean of all eight candidates": ALL8,
       "Caruana selection with replacement": sorted(car), "Stacking of all eight (logistic)": ALL8, "Majority vote at 0.5": ALL8,
       "Likelihood weights (softmax of val log-lik.)": ["VideoMAE-B LoRA"], "Best single candidate (source val)": ["VideoMAE-B LoRA"],
       "TENT LENS-Net": ["LENS-Net"], "TENT LENS-Net + anchor": S1_, "TENT w/o motion": ["w/o motion"], "TENT w/o motion + anchor": ["w/o motion", "CLIP probe"],
       "TENT Motion probe": ["Motion probe"], "TENT Motion probe + anchor": ["Motion probe", "CLIP probe"]}
for tag, ms in (("S1 anchor + LENS-Net", S1_), ("S2 AES members", S2_), ("S3 all eight", ALL8)):
    for kind in ("Calibrated averaging", "Dempster-Shafer, discounted", "Confidence-weighted"): MEM[f"{kind} ({tag})"] = ms
passes = lambda r: len(set().union(*[BB[m] for m in MEM[r]]))
sig = lambda r: (sum(hi[r][s]["worse"] for s in GS), sum(hi[r][s]["better"] for s in GS)) if r in hi else (0, 0)
h5 = lambda r: sum(hu[r][s] for s in GS) / len(GS); hn = lambda r: min(hu[r][s] for s in GS)
NAMES = {**RN, "Calibrated averaging (S1 anchor + LENS-Net)": "Calibrated mean, anchor + LENS-Net", "Calibrated averaging (S3 all eight)": "Calibrated mean, all eight",
         "Calibrated averaging (S2 AES members)": "Calibrated mean, AES members", "Dempster-Shafer, discounted (S1 anchor + LENS-Net)": "Dempster-Shafer, anchor + LENS-Net",
         "Dempster-Shafer, discounted (S2 AES members)": "Dempster-Shafer, AES members", "Dempster-Shafer, discounted (S3 all eight)": "Dempster-Shafer, all eight",
         "Confidence-weighted (S1 anchor + LENS-Net)": "Confidence-weighted, anchor + LENS-Net", "Confidence-weighted (S2 AES members)": "Confidence-weighted, AES members",
         "Confidence-weighted (S3 all eight)": "Confidence-weighted, all eight", "TENT LENS-Net": "TENT LENS-Net", "TENT LENS-Net + anchor": "TENT LENS-Net + anchor",
         "TENT w/o motion": "TENT w/o motion", "TENT w/o motion + anchor": "TENT w/o motion + anchor", "TENT Motion probe": "TENT motion probe",
         "TENT Motion probe + anchor": "TENT motion probe + anchor", "Likelihood weights (softmax of val log-lik.)": "Likelihood weights", "Best single candidate (source val)": "Best single model"}
fusr = [r for r in hu if r in MEM]
def row(r, label=None):
    sa, sb = sig(r)
    return (f"{label or NAMES[r]} & {passes(r)} & {cell(hd[r]['mean9'], r == bd)} & " + " & ".join(cell(hu[r][s], r == bcol[s]) for s in GS) +
            f" & {cell(h5(r), r == bm)} & {cell(hn(r), r == bn)} & " + ("-- & --" if r == LA else f"{sa} & {sb}") + "\\\\")
GROUPS = [("Anchored averages", [LA, BAL, AES_]),
          ("Standard ensembles", [ALL, "Caruana selection with replacement", STK, "Majority vote at 0.5", "Best single candidate (source val)"]),
          ("Information-fusion baselines", ["Calibrated averaging (S1 anchor + LENS-Net)", "Calibrated averaging (S3 all eight)", "Dempster-Shafer, discounted (S1 anchor + LENS-Net)",
                                            "Dempster-Shafer, discounted (S3 all eight)", "Confidence-weighted (S1 anchor + LENS-Net)", "TENT LENS-Net + anchor", "TENT w/o motion + anchor"])]
shown = [r for _, rs in GROUPS for r in rs]
bcol = {s: max(shown, key=lambda r: hu[r][s]) for s in GS}; bm, bn = max(shown, key=h5), max(shown, key=hn); bd = max(shown, key=lambda r: hd[r]["mean9"])
rows = []
for gname, rs in GROUPS:
    rows.append(f"\\multicolumn{{12}}{{@{{}}l}}{{\\emph{{{gname}}}}}\\\\")
    for r in rs: rows.append(row(r, "Best single model (= likelihood weights)" if r.startswith("Best single") else None))
(O / "tables/fusion_untouched.tex").write_text("\n".join(rows) + "\n")
rows = []
for r in sorted(fusr, key=lambda r: -h5(r)):
    sa, sb = sig(r)
    rows.append(f"{NAMES[r]} & {passes(r)} & " + " & ".join(f2(hu[r][s]) for s in GS) + f" & {f2(h5(r))} & " + " & ".join(f2(hw[s][r]) for s in GS) + " & " + ("-- & --" if r == LA else f"{sa} & {sb}") + "\\\\")
(O / "tables/fusion_all.tex").write_text("\n".join(rows) + "\n")
mx = max(sig(r)[0] for r in fusr if r != LA)
M.update(PhNRules=len(fusr), PhMaxSigBetter=["no", "one", "two", "three", "four", "five"][mx], PhCalAllMean=f2(h5("Calibrated averaging (S3 all eight)")),
         PhCalAMean=f2(h5("Calibrated averaging (S1 anchor + LENS-Net)")), PhDsAMean=f2(h5("Dempster-Shafer, discounted (S1 anchor + LENS-Net)")),
         PhDsAllMean=f2(h5("Dempster-Shafer, discounted (S3 all eight)")), PhConfAMean=f2(h5("Confidence-weighted (S1 anchor + LENS-Net)")),
         PhTentLens=f2(h5("TENT LENS-Net")), PhTentLensA=f2(h5("TENT LENS-Net + anchor")), PhTentNomo=f2(h5("TENT w/o motion")),
         PhTentNomoMin=f2(hn("TENT w/o motion")), PhTentNomoHmdb=f2(hu["TENT w/o motion"]["HMDB51"]),
         PhDevCalAll=f2(hd["Calibrated averaging (S3 all eight)"]["mean9"]), PhDevDsAll=f2(hd["Dempster-Shafer, discounted (S3 all eight)"]["mean9"]),
         PhDevTentLens=f2(hd["TENT LENS-Net"]["mean9"]), PhNOthers=len([r for r in fusr if r != LA]),
         PhLaSigTwoN=sum(sig(r)[1] >= 2 for r in fusr if r != LA),
         PhSigSets=" or ".join(sorted({s.replace("XD sample", "XD sample") for r in fusr if r != LA for s in GS if hi[r][s]["worse"]})))
pi = J("i_partI.json"); GI = ["XD sample A", "AIE", "XD sample B", "UCF-train", "HMDB51"]   # Part I: anchor and LENS-Net
one = {g: 100 * (pi["pairs"][g]["anchor_only"] + pi["pairs"][g]["lens_only"]) for g in GI}; rs = {g: 100 * pi["pairs"][g]["resolved"] for g in GI}
M.update(PiRhoLo=f"{min(pi['spearman'].values()):.2f}", PiRhoHi=f"{max(pi['spearman'].values()):.2f}",
         PiOneLo=f"{min(one.values()):.1f}", PiOneHi=f"{max(one.values()):.1f}", PiOneAie=f"{one['AIE']:.1f}",
         PiResLo=f"{min(rs.values()):.0f}", PiResHi=f"{max(rs.values()):.0f}", PiRhoAie=f"{pi['spearman']['AIE']:.2f}")
with open(O / "revision_c_numbers.tex", "w") as fh:
    fh.write("% generated by src/revision/make_revision_macros.py -- do not edit\n")
    for k, x in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{x}}}\n")
# Table: C1
rows = []
for ds, nm in (("RWF2000", "RWF-2000"), ("HockeyFights", "Hockey")):
    for m in ("VideoMAE-B FT", "R(2+1)D-18 FT", "CLIP probe", "LENS-Net"):
        r = c1[ds][m]; tr = r["transfer"]
        rows.append(f"{nm} & {m} & {f2(r['R_auc'])} & {f2(r['S_auc'])} & {ci(r['R_minus_S'])} & {f2(r['R_linked_auc'])} & {f2(r['R_unlinked_auc'])} & {ci(tr['XD']['R_minus_S'])} & {ci(tr['UCF']['R_minus_S'])}\\\\")
    rows.append("\\midrule")
(O / "tables/c1_splits.tex").write_text("\n".join(rows[:-1]) + "\n")
# Table: controls (nine-set mean / min)
def mm(d): return f"{f2(d['mean9'])} & {f2(d['min9'])}"
L = [f"CLIP probe & {mm(a['CLIP probe'])} & {mm(a['LAION CLIP probe'])} & {mm(c3['L14_audited_LP_irm'])} & {mm(c3['L14_audited_LP_gdro'])}\\\\",
     f"w/o motion & {mm(a['w/o motion'])} & {mm(a['LAION w/o motion'])} & -- & -- & -- & --\\\\",
     f"LENS-Net & {mm(a['LENS-Net'])} & {mm(a['LAION LENS-Net'])} & {mm(v['irm'])} & {mm(v['gdro'])}\\\\",
     f"LENS-Net-A & {mm(a['LENS-Net-A'])} & {mm(a['LAION LENS-Net-A'])} & -- & -- & -- & --\\\\",
     f"VideoMAE-B FT & {mm(a['VideoMAE-B FT'])} & -- & -- & -- & -- & -- & --\\\\"]
(O / "tables/controls.tex").write_text("\n".join(L) + "\n")
print(len(M), "macros")
# Supplement tables: D1 strata, C2 per set, C7 per set
from saved_predictions import UNSEEN
names = {"verified_edge": "Verified same-source links", "rejected_candidate": "Candidates rejected by verification",
         "random_pair": "Random cross-video pairs", "cosine_removed": "Unseen windows removed (cosine $\\ge$ 0.95)",
         "cosine_0.90-0.95": "Unseen windows kept (cosine 0.90--0.95)"}
rows = [f"{names[k]} & {st[k]['n']} & {st[k]['same']} & {100*st[k]['rate']:.1f} & [{100*st[k]['wilson95'][0]:.1f}, {100*st[k]['wilson95'][1]:.1f}]\\\\" for k in names]
(O / "tables/d1_strata.tex").write_text("\n".join(rows) + "\n")
short = {"XD": "XD", "UCF": "UCF", "SCFD": "SCFD", "SCVD": "SCVD", "VioPeru": "VioP.", "AIRTLab": "AIRT.", "CCTV-F": "CCTV-F.", "Movies": "Movies", "UBI-Fights": "UBI"}
rows = []
for m in ("CLIP probe", "LAION CLIP probe", "w/o motion", "LAION w/o motion", "LENS-Net", "LAION LENS-Net", "LENS-Net-A", "LAION LENS-Net-A", "VideoMAE-B FT"):
    rows.append(f"{m} & " + " & ".join(f2(a[m][s]) for s in UNSEEN) + f" & {f2(a[m]['mean9'])}\\\\")
(O / "tables/c2_perset.tex").write_text("\n".join(rows) + "\n")
rows = [f"Windows removed (0.95, 0.90) & " + " & ".join(f"{rm[s]['at095']}, {rm[s]['at090']}" for s in UNSEEN) + " & --\\\\", "\\midrule"]
for m in ("CLIP probe", "LENS-Net", "LENS-Net-A", "VideoMAE-B FT", "VadCLIP"):
    rows.append(f"{m} & " + " & ".join(f"{f2(c7['auc'][s][m]['at090'])}" for s in UNSEEN) + f" & {f2(c7['mean9'][m]['at090'])}\\\\")
rows.append("\\midrule")
rows.append("Kendall $\\tau$ & " + " & ".join(f"{c7['tau'][s]:.2f}" for s in UNSEEN) + " & --\\\\")
(O / "tables/c7_perset.tex").write_text("\n".join(rows) + "\n")
print("supplement tables written")

# House style of the manuscript: "without" is spelled out in every table label.
for _t in (O / "tables").glob("*.tex"):
    _t.write_text(_t.read_text().replace("w/o", "without"))
