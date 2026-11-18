#!/usr/bin/env bash
# Prepare a clone for re-running the full pipeline: the scripts read and write under workspace/. This links the
# window tables, results and pre-registrations of this repository into workspace/. Folders that already exist
# (for example after reproduce/get_predictions.sh) are left untouched.
# Paths: every script works from the repository root, so no file needs editing. Leave LENSNET_ROOT unset: the scripts
# under src/ honour it, the video_data utilities and the table and figure generators always use the clone.
set -e
cd "$(dirname "$0")/.."
mkdir -p workspace/features source_benchmark raw_data

link() {  # link <target relative to workspace/> <name in workspace/>
  if [ -e "workspace/$2" ] || [ -L "workspace/$2" ]; then echo "  keep  workspace/$2"; else ln -s "$1" "workspace/$2"; echo "  link  workspace/$2 -> $1"; fi
}
link ../data_tables/splits splits
link ../data_tables/scfd_scvd fresh
link ../data_tables/vioperu_airtlab conf
link ../data_tables/cctv_fights cfights
link ../data_tables/movies_ubi_fights conf2
link ../data_tables/xd_train_aie conf3
link ../data_tables/untouched_fusion conf4
link ../data_tables/framelevel_xd_ucf framelevel
link ../data_tables/ambiguous_windows amb
link ../results logs
link ../results/revision review
link ../prereg/PREREG.md PREREG.md
link ../prereg/PREREG_journal.md PREREG_journal.md
link ../prereg/revision_timestamped/PREREG_revision.md PREREG_revision.md
echo "Layout ready. Put the datasets under raw_data/ (see docs/DATA.md)."
