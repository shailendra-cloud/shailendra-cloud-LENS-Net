# Protocol 5b: audit of the public evaluation splits (CPU only, no training)

Status: written 2026-09-23 under the user's instruction to "proceed" (item 4 of the extension plan:
"write the protocols for 3a and 5b and start the CPU work"). The md5 is frozen in `logs/protocol_md5.txt`
before any count is computed. Later changes go in a "Deviations" section at the end.

Purpose: the journal's multi-setting evaluation reports what a conventional split permits on our
benchmark. This item asks the same question of the splits that published results actually use, with the
frozen near-duplicate edges and no new model.

## Inputs (read-only)
- Frozen manifest `dataset_v2/manifests/segments_v2_binary.csv` (md5 `c3a7c92a…`) and frozen edge list
  `dataset_v2/manifests/near_duplicate_edges.csv` (retained pairs between different source units, fine
  score ≥ 0.30).
- Name chain used to recover original file names: v2 `path` → `data/FINAL_DATASET/manifest_5class.csv`
  (`final_path` → `original_filename`) → `VID-Detect/data/manifest.csv` (file name → source
  `original_filename`).
- Official RWF-2000 split: the file listing of the Kaggle mirror `vulamnguyen/rwf2000` (the mirror the
  project downloaded from), retrieved via the Kaggle API on 2026-09-23. Folder `train/` or `val/` = the
  official split. Saved as `inputs/rwf2000_kaggle_files.json`.

## A. RWF-2000 official train/val split
1. Match every RWF-2000 video in the frozen benchmark to its official split by original file name.
   Report matched, unmatched and ambiguous counts. Unmatched videos are excluded from A and reported.
2. Count, using the frozen edges between two matched RWF-2000 windows:
   - retained near-duplicate window pairs that cross official train↔val;
   - of those, pairs whose inherited labels differ;
   - official val videos (and windows) with ≥ 1 direct link to an official train window;
   - near-duplicate components (restricted to RWF-2000) that span official train and val.
3. Same counts under our audited split, restricted to the same videos, for reference (expected 0).

## B. File-name prefix as a candidate source identifier (RWF-2000)
RWF-2000 names often have the form `<prefix>_<k>.avi`. Prefix = the name with a final `_<digits>`
removed. The prefix is treated as a **candidate** source identifier only; it is not assumed to be one.
Report:
- the number of prefixes, and the videos per prefix;
- prefixes spanning official train and val;
- prefixes spanning our audited splits (train/val/test);
- agreement with detected links: among pairs of RWF-2000 videos sharing a prefix, the fraction joined by
  ≥ 1 retained near-duplicate edge (same component); among pairs linked by an edge, the fraction
  sharing a prefix.
If prefixes span our audited splits, this is reported as a provenance finding for the journal. The
frozen benchmark and its split are **not** changed by this protocol; any change would be a separate,
approved decision.

## C. Random file-level splits as used for RLVS and Hockey Fights
These corpora have no official split; published work typically uses a random file-level split (e.g.
80/20). For each of RLVS and Hockey Fights separately: 1,000 random 80/20 file-level splits stratified
by label (rng seed 0). For each split, count retained near-duplicate pairs crossing train↔test and the
share of test videos with a direct link to a training window. Report the median and 2.5/97.5
percentiles. RWF-2000 is included too, for comparison with its official split.

## Not done here
- No model is trained or scored. Published models are not retrained (item 5c is out of scope for the
  journal).
- Bus Violence and VID: no official split is recoverable from the copies used here; reported as such.
- Near-duplicate detection is the frozen detector. Its recall is unmeasured (item 3), so all counts
  are counts of **detected** pairs, i.e. lower bounds on relatedness.

## Outputs
`results/public_split_audit.json`, `results/rwf_official_split_videos.csv`,
`results/rwf_prefix_groups.csv`, `logs/` with checksums of the inputs and outputs.
