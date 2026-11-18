#!/usr/bin/env bash
# Stage 8 - revision experiments and re-analyses (PREREG_revision.md, Parts B-E, D1). Every number of the revised paper
# comes from the result files written here (workspace/review = results/revision). GPU steps are marked.
set -e; cd "$(dirname "$0")/.."; PY=${PY:-python}; R=src/revision
# B: seed-aware statistics, Holm-corrected primary claims, ten seeds, with/without test-side filter
$PY $R/seed_aware_bootstrap.py; $PY $R/score_extra_seeds.py; $PY $R/ten_seed_analysis.py; $PY $R/analyze_test_filter.py
# C1: random vs source-controlled splits (GPU for the fine-tuned models)
$PY $R/make_random_vs_source_splits.py
for s in 42 43 44; do for sp in C1_RWF2000_R C1_RWF2000_S C1_HockeyFights_R C1_HockeyFights_S; do
  for m in videomae r2plus1d_18; do CUDA_VISIBLE_DEVICES=0 $PY $R/train_finetuned_on_split.py --model $m --split $sp --seed $s; done
  for c in LP LENS-M; do CUDA_VISIBLE_DEVICES="" $PY $R/train_frozen_heads.py --config $c --split std:$sp --seed $s; done; done; done
$PY $R/analyze_random_vs_source_splits.py
# C2: OpenCLIP (LAION-2B) control
CUDA_VISIBLE_DEVICES=0 $PY $R/extract_laion_features.py
for s in 42 43 44; do LENS_BACKBONE=LAION $PY $R/train_frozen_heads.py --config LP --penalty none --seed $s
  LENS_BACKBONE=LAION $PY $R/train_frozen_heads.py --config V2b --penalty vrex:1 --seed $s; LENS_BACKBONE=LAION $PY $R/train_frozen_heads.py --config LENS-M --penalty vrex:1 --seed $s; done
$PY $R/analyze_laion_control.py
# C3 / E2 / E3: invariance objectives, s and lambda_A sensitivity, DG on the motion features
for s in 42 43 44; do for p in vrex:0 vrex:0.3 vrex:3 vrex:10 irm gdro; do $PY $R/train_frozen_heads.py --config LENS-M --penalty $p --seed $s; done
  for p in irm gdro; do $PY $R/train_frozen_heads.py --config LP --penalty $p --seed $s; done
  for c in "LP --penalty none" "LENS-M --penalty vrex:1"; do for x in "--scale 15" "--scale 60" "--anchor-w 0" "--anchor-w 0.1" "--anchor-w 2"; do $PY $R/train_frozen_heads.py --config $c $x --seed $s; done; done
  for p in none irm gdro vrex:1; do $PY $R/train_frozen_heads.py --config MP --penalty $p --seed $s; done; done
$PY $R/analyze_dg_objectives.py; $PY $R/analyze_hyperparameters_and_motion_dg.py
# C4 / C5 / E1: learning-rate search, LoRA, fine-tuned CLIP (GPU)
for m in videomae swin3d_t r2plus1d_18; do for lr in 2e-5 1.25e-4; do CUDA_VISIBLE_DEVICES=0 $PY src/baselines/train_video_baseline.py --model $m --protocol A1 --lr-body $lr --seed 42; done; done
for s in 43 44; do for m in videomae r2plus1d_18; do CUDA_VISIBLE_DEVICES=0 $PY src/baselines/train_video_baseline.py --model $m --protocol A1 --lr-body 2e-5 --seed $s; done; done
for lr in 1e-4 5e-4; do CUDA_VISIBLE_DEVICES=0 $PY $R/train_finetuned_on_split.py --model videomae --split A1 --lora --lr-body $lr --seed 42; done
for s in 43 44; do CUDA_VISIBLE_DEVICES=0 $PY $R/train_finetuned_on_split.py --model videomae --split A1 --lora --lr-body 1e-4 --seed $s; done
CUDA_VISIBLE_DEVICES=0 $PY $R/score_finetuned_unseen.py workspace/runs_baselines/A1_*_lr2e-05_seed4* workspace/runs_review/A1_videomae_lora_lr0.0001_seed4*
for s in 42 43 44; do CUDA_VISIBLE_DEVICES=0 $PY $R/train_clip_ft.py --seed $s; done
$PY $R/analyze_fair_baselines.py both; $PY $R/analyze_clip_ft.py
# C6: second untouched XD-Violence sample; C7: stricter test-side filter
$PY $R/download_xd_sample_b.py
for K in 10 11 12; do $PY $R/confirm_anchor_fusion_sample_b.py build $K; CUDA_VISIBLE_DEVICES=0 $PY src/features/extract_compact.py --set c3_$K
  CUDA_VISIBLE_DEVICES=0 $PY src/features/extract_vmae.py c3_$K; $PY src/datasets/confirm_anchor_fusion.py clean $K; done
for st in dedup score analyze; do $PY $R/confirm_anchor_fusion_sample_b.py $st; done
$PY $R/analyze_stricter_filter.py
# E4: ambiguous windows as negatives
$PY $R/build_ambiguous_windows.py; CUDA_VISIBLE_DEVICES=0 LENS_BACKBONE=L14 $PY src/features/extract_compact.py --set amb
CUDA_VISIBLE_DEVICES=0 $PY src/features/extract_vmae.py amb; $PY $R/score_ambiguous_windows.py cpu; CUDA_VISIBLE_DEVICES=0 $PY $R/score_ambiguous_windows.py gpu
$PY $R/analyze_ambiguous_windows.py
# E5: decision-level fusion rules (fixed vs fitted on source validation data), from saved predictions
$PY $R/compare_fusion_rules.py
# E6: member diversity vs fusion gain (needs e5_fusion.json)
$PY $R/analyze_member_diversity.py
# Part F: anchored ensemble selection - development (exploratory), confirmation (pre-registered; needs GPU + videos)
$PY $R/explore_anchored_selection.py; $PY $R/explore_partner_selection.py
CUDA_VISIBLE_DEVICES=0 $PY $R/score_aes_confirmation.py; $PY $R/analyze_aes_confirmation.py; $PY $R/analyze_aes_diagnosis.py
$PY $R/explore_aes_balanced_weights.py
# Part G: further untouched samples (download from Kaggle, needs ~20 GB free + GPU) and ensemble baselines
CUDA_VISIBLE_DEVICES=0 $PY $R/confirm_fusion_rules_untouched.py all; $PY $R/analyze_fusion_rules_untouched.py all
# Part H: calibrated, Dempster-Shafer and confidence-weighted fusion and TENT on the same samples; Part I: complementarity
$PY $R/analyze_fusion_baselines_untouched.py; $PY $R/analyze_anchor_complementarity.py
# D1: human validation of the audit (labels collected with a web form from the pairs of make_audit_label_sample.py)
$PY $R/analyze_audit_labels.py
# macros, tables and the two revision figures
(cd paper && $PY fig_claims_idood.py)   # writes paper/figs/fig_idood_data.json, read by make_revision_macros.py
$PY $R/make_revision_macros_part_b.py; $PY $R/make_revision_macros.py
