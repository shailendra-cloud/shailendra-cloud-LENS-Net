#!/usr/bin/env bash
# Stage 4 - the unseen test sets: build windows + frames, extract features, near-duplicate check, score every model once.
# Paper: Tables 6, 7 and 16. Each set has its own pre-registered section (prereg/PREREG*.md).
set -e; cd "$(dirname "$0")/.."; PY=${PY:-python}
feat() { SETS="$1" bash pipelines/01_features.sh; }       # features of a set, once its windows and frame store exist
variants() {                                               # LENS-family scores: default, Lite variants L1-L3, prompt sets
  for bb in L14 B16; do for fr in 16 8; do LENS_BACKBONE=$bb LENS_FRAMES=$fr $PY $1 lensnet; done; done
  for ps in para surv broad; do LENS_PROMPTS=$ps $PY $1 lensnet; done
}
# SCFD, SCVD (fresh CCTV sets for the motion stream, Section 5.7)
$PY src/datasets/build_scfd_scvd.py; feat fresh; $PY src/evaluation/score_scfd_scvd.py dedup
variants src/evaluation/score_scfd_scvd.py; $PY src/evaluation/score_scfd_scvd.py gpu-baselines
# VioPeru, AIRTLab (PREREG_journal.md Sec. 3)
$PY src/datasets/build_vioperu_airtlab.py; feat conf; $PY src/datasets/dedup_vioperu_airtlab.py
variants src/evaluation/score_vioperu_airtlab.py; $PY src/evaluation/score_vioperu_airtlab.py gpu-baselines
# CCTV-Fights (PREREG_journal.md Sec. 7; access request at ROSE Lab, then src/datasets/download_cctv_fights.py)
$PY src/datasets/build_cctv_fights.py; feat cfights; $PY src/datasets/dedup_cctv_fights.py
for bb in L14 B16; do for fr in 16 8; do LENS_BACKBONE=$bb LENS_FRAMES=$fr $PY src/evaluation/score_cctv_fights.py lensnet; done; done
$PY src/evaluation/score_cctv_fights.py gpu
# Movies, UBI-Fights (PREREG_journal.md Sec. 9; LENS-FG confirmation)
$PY src/datasets/confirm_footage_gate.py build; $PY src/features/extract_compact.py --set conf2; $PY src/features/extract_vmae.py conf2
$PY src/datasets/confirm_footage_gate.py dedup; $PY src/datasets/confirm_footage_gate.py score-lensnet; $PY src/datasets/confirm_footage_gate.py score-gpu; $PY src/datasets/confirm_footage_gate.py score-probes
# XD-Violence training subset, AIE (PREREG_journal.md Sec. 10; LENS-Net-A confirmation)
$PY src/datasets/download_xd_train_aie.py; $PY src/datasets/confirm_anchor_fusion.py plan
N=$($PY -c "import pandas as pd; print(int(pd.read_csv('${LENSNET_ROOT:-.}/workspace/conf3/windows.csv', usecols=['chunk']).chunk.max()) + 1)")
for K in $(seq 0 $((N - 1))); do
  $PY src/datasets/confirm_anchor_fusion.py build $K; $PY src/features/extract_compact.py --set c3_$K; $PY src/features/extract_vmae.py c3_$K; $PY src/datasets/confirm_anchor_fusion.py clean $K
done
$PY src/datasets/confirm_anchor_fusion.py dedup; $PY src/datasets/confirm_anchor_fusion.py score
