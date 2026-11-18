#!/usr/bin/env bash
# Stage 5 - standard frame-level protocol on the full XD-Violence and UCF-Crime test sets (Section 5.9, Table 14).
set -e; cd "$(dirname "$0")/.."; PY=${PY:-python}
$PY src/datasets/framelevel_xd_ucf.py plan
N=$($PY -c "import pandas as pd; print(int(pd.read_csv('${LENSNET_ROOT:-.}/workspace/framelevel/windows.csv', usecols=['chunk']).chunk.max()) + 1)")
for K in $(seq 0 $((N - 1))); do
  $PY src/datasets/framelevel_xd_ucf.py build $K
  LENS_BACKBONE=L14 $PY src/features/extract_compact.py --set fl$K; LENS_BACKBONE=B16 $PY src/features/extract_compact.py --set fl$K
  $PY src/features/extract_vmae.py fl$K
  LENS_BACKBONE=L14 LENS_FRAMES=16 $PY src/datasets/framelevel_xd_ucf.py score-lensnet $K; LENS_BACKBONE=B16 LENS_FRAMES=8 $PY src/datasets/framelevel_xd_ucf.py score-lensnet $K
  $PY src/datasets/framelevel_xd_ucf.py score-gpu $K; $PY src/datasets/framelevel_xd_ucf.py clean $K
done
