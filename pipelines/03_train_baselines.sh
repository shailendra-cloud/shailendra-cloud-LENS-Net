#!/usr/bin/env bash
# Stage 3 - baselines (GPU). Paper: Tables 6-8.  A1 = pooled (P1), A2 = RWF-2000 only (P3).
set -e; cd "$(dirname "$0")/.."; PY=${PY:-python}
for s in 42 43 44; do for m in videomae swin3d_t r2plus1d_18; do for p in A1 A2; do
  $PY src/baselines/train_video_baseline.py --model $m --protocol $p --seed $s
done; done; done
# X-CLIP: learning rate chosen on validation (seed 42) from {5e-6, 1e-5, 2e-5, 5e-5}; 5e-6 was selected
for lr in 5e-6 1e-5 2e-5; do $PY src/baselines/train_video_baseline.py --model xclip --protocol A1 --seed 42 --lr-body $lr; done
for s in 43 44; do $PY src/baselines/train_video_baseline.py --model xclip --protocol A1 --seed $s --lr-body 5e-6; done
# Official VadCLIP on the same frozen CLIP features: lr chosen on validation (1e-4 for both backbones)
for bb in B16 L14; do for lr in 1e-5 1e-4 3e-4; do $PY src/baselines/vadclip_run.py --bb $bb --split audited --seed 42 --lr $lr; done; done
for bb in B16 L14; do for sp in audited std:std_rwf; do for s in 42 43 44; do
  $PY src/baselines/vadclip_run.py --bb $bb --split $sp --seed $s --lr 1e-4 --allow-test
done; done; done
$PY src/baselines/vadclip_score_bench.py                         # VadCLIP on every source window (transfer targets of P3)
$PY src/baselines/xclip_zs.py                                    # X-CLIP zero-shot
# CNN-BiGRU detector: video_data/train_cnn_bigru.py (checkpoints source_benchmark/runs/binary_hybrid_seed4x)
# LENS-FT (encoder fine-tuning): src/model/cache_clip_block20.py, then src/model/lensnet_ft.py --seed {42,43,44}
