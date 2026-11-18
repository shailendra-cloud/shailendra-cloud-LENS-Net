#!/usr/bin/env bash
# Stage 2 - LENS-Net and all its variants (CPU, minutes per run). Paper: Tables 6, 8 and 15.
#   P1 pooled = --split audited | P2 leave-one-dataset-out = --split lodo:<corpus> | P3 RWF-2000 only = --split std:std_rwf
#   P4 standard 5-fold = --split std:std_hockey_f<k> / std_rlvs_f<k>
set -e; cd "$(dirname "$0")/.."; PY=${PY:-python}; export CUDA_VISIBLE_DEVICES=""
SPLITS="audited lodo:HockeyFights lodo:RLVS lodo:RWF2000 lodo:BusViolence lodo:VID std:std_rwf"
for k in 0 1 2 3 4; do SPLITS="$SPLITS std:std_hockey_f$k std:std_rlvs_f$k"; done
for s in 42 43 44; do
  for sp in $SPLITS; do
    for c in LENS-M LP V2b MP; do $PY src/model/lensnet.py --config $c --split $sp --seed $s --allow-test; done   # LENS-Net, CLIP probe, w/o motion, motion probe
  done
  $PY src/model/lensnet.py --config V2a --split audited --seed $s --allow-test                                   # w/o V-REx
  for c in -G -T -Sp; do $PY src/model/lensnet.py --config=$c --vrex 1 --split audited --seed $s --allow-test; done # ablations
  for sp in audited std:std_rwf; do
    for c in LENS-AG LENS-FG; do $PY src/model/lensnet.py --config $c --split $sp --seed $s --allow-test; done     # learned gate, footage gate
    LENS_FRAMES=8 $PY src/model/lensnet.py --config LENS-M --split $sp --seed $s --allow-test                     # Lite L1
    LENS_BACKBONE=B16 $PY src/model/lensnet.py --config LENS-M --split $sp --seed $s --allow-test                 # Lite L2
    LENS_BACKBONE=B16 LENS_FRAMES=8 $PY src/model/lensnet.py --config LENS-M --split $sp --seed $s --allow-test   # Lite L3
    for ps in para surv broad; do LENS_PROMPTS=$ps $PY src/model/lensnet.py --config LENS-M --split $sp --seed $s --allow-test; done
  done
done
# LENS-Net-A (anchor fusion) needs no training: it averages the probabilities of the LP and LENS-M models of the same seed.
# The first design (temporal-transformer head, Sections 3.8 and 5.12) and CLIP zero-shot (--config ZS) use
# src/model/lensnet_setup.py, which reads the token features written by src/features/extract_clip_tokens.py.
# src/evaluation/score_xd_ucf_windows.py scores these runs on XD-Violence and UCF-Crime. The pre-registered pilot
# (PREREG.md Sec. 5 and deviation 2) trained both scripts with --pilot and was evaluated by
# src/evaluation/pilot_decision.py and select_design.py (results/pilot_decision.json, design_selection.json).
for s in 42 43 44; do $PY src/model/lensnet_setup.py --config LENS --split audited --seed $s --allow-test --tag _v1; done
