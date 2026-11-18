#!/usr/bin/env bash
# Stage 6 - every statistic in the paper, recomputed from saved predictions (cluster bootstrap, 2000 resamples).
set -e; cd "$(dirname "$0")/.."; PY=${PY:-python}; export CUDA_VISIBLE_DEVICES=""
$PY src/audit/audit_public_splits.py       # same-source overlap under public splits (Section 4, Fig. 3)
$PY src/evaluation/analyze_main.py            # pooled results, leave-one-dataset-out (Table 8b), standard protocols (Table 8c)
$PY src/evaluation/analyze_baselines_pooled.py; $PY src/evaluation/analyze_baselines_rwf.py; $PY src/evaluation/test_rwf_mean_transfer.py   # fine-tuned baselines, P3
$PY src/evaluation/score_footage_gate_dev.py   # LENS-FG on the development sets and the frame-level chunks (exploratory; results/fg_dev_window.json)
$PY src/evaluation/analyze_scfd_scvd.py; $PY src/evaluation/analyze_vioperu_airtlab.py; $PY src/evaluation/analyze_cctv_fights.py
$PY src/evaluation/analyze_clip_methods.py      # VadCLIP, X-CLIP, prompt study
$PY src/evaluation/analyze_framelevel_xd_ucf.py # Table 14 (frame-level protocol)
$PY src/datasets/confirm_footage_gate.py analyze        # LENS-FG confirmation
$PY src/evaluation/anchor_fusion_dev.py; $PY src/evaluation/ensemble_control.py   # anchor fusion on development sets
$PY src/datasets/confirm_anchor_fusion.py analyze        # LENS-Net-A confirmation (Section 6.2, Table 16)
C="clean blur1 blur2 blur3 dark1 dark2 dark3 noise1 noise2 noise3 jpeg1 jpeg2 jpeg3"
CUDA_VISIBLE_DEVICES=0 $PY src/evaluation/robustness.py test $C; CUDA_VISIBLE_DEVICES=0 $PY src/evaluation/robustness.py viop $C   # 12 corruptions
$PY src/evaluation/analyze_robust.py; $PY src/evaluation/error_analysis.py
$PY src/evaluation/efficiency_lensnet.py; $PY src/evaluation/efficiency_variants.py; CUDA_VISIBLE_DEVICES=0 $PY src/evaluation/latency.py
