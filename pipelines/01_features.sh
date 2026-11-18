#!/usr/bin/env bash
# Stage 1 - frozen features (paper: Section 3.5.1).
# CLIP ViT-L/14 (and ViT-B/16 for the Lite variants) prompt signals + VideoMAE-B motion features for a window set.
# By default the source corpora (bench) and the XD-Violence and UCF-Crime test windows (external). The other unseen sets
# need their windows and frame stores first, so stage 4 calls this script for them: SETS=fresh bash pipelines/01_features.sh
set -e; cd "$(dirname "$0")/.."; PY=${PY:-python}
SETS=${SETS:-"bench external"}                                  # c3_* and fl* chunks are built by stages 4 and 5
for s in $SETS; do
  LENS_BACKBONE=L14 $PY src/features/extract_compact.py --set $s --psets orig,para,surv,broad   # prompt sets: Section 7.1
  LENS_BACKBONE=B16 $PY src/features/extract_compact.py --set $s                                # Lite variants L2/L3
  $PY src/features/extract_vmae.py $s
done
