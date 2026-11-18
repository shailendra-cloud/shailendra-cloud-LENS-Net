#!/usr/bin/env bash
# Every paper figure as a 600-dpi PNG: TikZ diagrams via their standalone files, plots from the vector PDFs written by
# the figure scripts.
set -e; shopt -s nullglob
# TikZ diagrams need the tectonic LaTeX engine (https://tectonic-typesetting.github.io); set TECTONIC=/path/to/tectonic if it is not on PATH.
cd "$(dirname "$0")"
TT=${TECTONIC:-tectonic}
for s in arch paradigm; do
  $TT ${s}_standalone.tex > /dev/null 2>&1
  pdftoppm -r 600 -png -singlefile ${s}_standalone.pdf figs/fig_${s}
done
for f in figs/fig_*.pdf; do                     # every plot written by the figure scripts
  b=${f%.pdf}
  case $b in figs/fig_arch|figs/fig_paradigm) continue;; esac
  pdftoppm -r 600 -png -singlefile $f $b
done
echo "600-dpi PNGs written: $(ls figs/fig_*.png | wc -l)"
