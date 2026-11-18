#!/usr/bin/env bash
# Stage 7 - tables (numbers.tex, tables/*.tex) and 400-dpi figures for the paper, from the result files only.
set -e; cd "$(dirname "$0")/../paper"; PY=${PY:-python}
$PY make_tables.py
$PY make_figures.py; $PY fig_efficiency.py; $PY fig_robust.py; $PY fig_prompts.py; $PY fig_categories.py; $PY fig_evidence.py; $PY tsne_fig.py
bash figs_to_png.sh
# Not run here, because they read the videos and need a GPU: the qualitative figure (make_qualitative.py, then
# plot_qualitative.py) and the frames of the architecture figure (make_arch_assets.py). fig_graphical_abstract.py draws
# the graphical abstract. The revision tables, macros and figures are written at the end of stage 8.
