# Result summaries

Every number, table and figure of the paper is generated from these files and from the run summaries in the
predictions archive (`reproduce/get_predictions.sh`) by `paper/make_tables.py`,
`src/revision/make_revision_macros*.py` and the `paper/fig_*.py` scripts. Short file names follow the section IDs of
the pre-registration documents in `prereg/` (for example `c1` is Part C1 of `PREREG_revision.md`), so each result
can be traced to the plan that fixed it before it was computed.

## Main study (`PREREG.md`, `PREREG_journal.md`)

| File | Contents | Paper |
|---|---|---|
| `clip_methods_results.json` | pooled training (P1) and single-source (P3) results of every model; VadCLIP, X-CLIP, prompt study | Tables 6, 8a, 15 |
| `A1_results.json` | fine-tuned video baselines under pooled training (VideoMAE-B, Video Swin-T, R(2+1)D-18) | Table 6 |
| `A2_results.json` | single-source training on RWF-2000 (P3), per seed | Table 8a, Fig. 6 |
| `a2_mean_boot.json` | P3: bootstrap test of the mean-transfer differences | §5.6 |
| `stageF_results.json` | leave-one-dataset-out (P2), LENS-AG, efficient variants, prompt sets | Tables 8b, 15 |
| `fresh_results.json` | SCFD and SCVD (fresh sets for the motion stream) | §5.7 |
| `conf_results.json` | VioPeru and AIRTLab | §5.8 |
| `cfights_results.json` | CCTV-Fights | §5.8 |
| `conf2_results.json` | LENS-FG confirmation on Movies and UBI-Fights | §8.2 |
| `fg_dev_window.json` | LENS-FG on the development sets (exploratory) | Table 15 |
| `conf3_results.json` | LENS-Net-A confirmation on the XD-Violence training sample and AIE | Table 16 |
| `anchor_safe_exploratory.json`, `ensemble_control.txt` | anchor fusion on the development sets; seed-ensembling and symmetric-fusion control | §6.2 |
| `framelevel_results.json` | frame-level protocol on XD-Violence and UCF-Crime | Table 14 |
| `public_split_audit.json` | same-source overlap under the public splits | §4, Fig. 3, Table 8c |
| `robust_results.json` | robustness to blur, low light, noise and JPEG | §7.3 |
| `error_analysis.json` | per-category results | §7.2 |
| `efficiency.json`, `efficiency_journal.json`, `latency.json`, `fig_efficiency_values.json` | GFLOPs, latency, accuracy versus cost | §8.3, Fig. 18 |
| `fig_prompts_values.json` | prompt discriminability before training | Fig. 10 |
| `pilot_decision.json`, `design_selection.json` | pre-registered pilot gate and design selection | §3.8 |
| `prereg_md5.txt` | dated checksum log of the plans | §3.8 |

## Controls and fusion studies (`results/revision/`, `PREREG_revision.md`)

| File | Plan part | Contents | Paper |
|---|---|---|---|
| `stats.json` | B1, B2, B5 | seed-aware paired tests, primary claims with Holm correction, symmetric fusion | Tables 7, 9 |
| `filtering.json` | B3 | every unseen set with and without the test-side filter | §4.2 |
| `ten_seeds.json` | B7 | ten seeds for the frozen-feature heads | §5.13 |
| `c1_results.json`, `splits/` | C1 | random versus source-controlled splits | Table 4 |
| `c2_laion.json` | C2 | OpenCLIP (LAION-2B) control | Tables 11, 12 |
| `c3_dg.json` | C3 | V-REx sweep, IRM, GroupDRO | Table 11 |
| `c45.json` | C4, C5 | learning-rate search and LoRA for the fine-tuned baselines | Table 10, Fig. 5 |
| `c6_confirmB.json` | C6 | second untouched XD-Violence sample | §6.2 |
| `c7_filter090.json` | C7 | stricter test-side filter (cosine 0.90) | Table 13 |
| `d1_labels_final.json`, `d1_labels_round1.json`, `human_labels_*.csv`, `label_key_HIDDEN.csv` | D1 | human validation of the audit; the key that maps anonymous pair IDs to strata was hidden from the labeller | §4.3, Table 5 |
| `e1_clipft.json` | E1 | CLIP probe with the last four CLIP blocks fine-tuned | Table 10 |
| `e23.json` | E2, E3 | sensitivity to s and λ_A; invariance objectives on the motion probe | §5.3 |
| `e4_amb.json` | E4 | ambiguous windows added as negatives | §4.2 |
| `e5_fusion.json` | E5 | eight decision-level fusion rules on nine member pairs | Table 17 |
| `e6_diversity.json` | E6 | member diversity versus fusion gain | §6.4, Fig. 7 |
| `explore_anchored_selection.json`, `explore_partner_select.json` | F (exploratory) | anchored ensemble selection on the development sets | §6.5 |
| `f_aes_confirm.json`, `f_aes_diagnosis.json` | F | pre-registered confirmation of AES and its diagnosis | Table 18, §6.5 |
| `explore_aes_fixes_dev.json` | F (exploratory) | AES-bal and rank normalisation on the development sets | §6.5 |
| `explore_saaf*.json`, `explore_coadapt*.json`, `explore_partner.json` | exploratory | label-free weighting, cross-view co-adaptation, fusion-aware partner (no robust gain) | §6.5 |
| `g_partG.json`, `g_partG_dev.json` | G | fusion rules on the two further untouched samples | Table 19, §6.6 |
| `h_partH.json` | H | calibrated, Dempster–Shafer, confidence-weighted fusion and TENT | Tables 19, 20, Fig. 8 |
| `i_partI.json` | I | complementarity of the anchor and LENS-Net | Fig. 9 |

## Names used in the files

Run folders, prediction files and result keys use the working names of the project.

| In the paper | In the files |
|---|---|
| LENS-Net | `v2_audited_LENS-M_vrex` (config `LENS-M`) |
| CLIP probe (anchor only) | `v2_audited_LP` (config `LP`) |
| Anchor + PPS (without motion) | `v2_audited_V2b` (config `V2b`) |
| Motion probe | `v2_audited_MP` (config `MP`) |
| LENS-Net-A | no file of its own: the same-seed average of the LENS-Net and CLIP-probe probabilities |
| CLIP zero-shot | `ZS` |
| VadCLIP | `VadCLIP-L14_audited_lr0.0001` |
| X-CLIP FT | `A1_xclip_lr5e-06` |
| VideoMAE-B FT, Video Swin-T FT, R(2+1)D-18 FT | `A1_videomae`, `A1_swin3d_t`, `A1_r2plus1d_18` (`A1` = pooled training P1, `A2` = RWF-2000 only, P3) |
| CNN-BiGRU | `MAST-Net` |
| SCFD and SCVD | set `fresh` |
| VioPeru and AIRTLab | set `conf` |
| CCTV-Fights | set `cfights` |
| Movies and UBI-Fights | set `conf2` |
| XD-Violence training samples A and B, AIE | set `conf3` |
| UCF-Crime training subset, HMDB51 | set `conf4` |
