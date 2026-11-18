#!/usr/bin/env python3
"""Label schema: the single source of truth for labels, provenance rules and constants.

STEP 2 scope: the primary label is binary (VIOLENCE=1 / NON_VIOLENCE=0). `violence_type` is kept
for later use; every type assigned here is PROVISIONAL (derived from a dataset's coarse label,
not from temporal type annotation) and is flagged as such in the manifest.
"""
import re
from pathlib import Path

# ---------------------------------------------------------------- labels
BINARY = {"NON_VIOLENCE": 0, "VIOLENCE": 1}
VIOLENCE_TYPES = ("FIGHT_ASSAULT", "RIOT_CROWD", "SHOOTING_WEAPON", "ABUSE", "OTHER_VIOLENCE")
TYPE_NONE = "NONE"                      # non-violent segments
# UNKNOWN is an inference outcome only; it is never a training label and never appears here.
NEGATIVE_KINDS = ("none", "same_setting", "same_dataset", "same_video_quiet", "same_film",
                  "hard_negative_anomaly", "everyday")
SETTINGS = ("sports", "public_transport", "street", "classroom", "domestic", "mixed_real_world",
            "film", "other", "unknown")
PRODUCTIONS = ("cctv", "web_mobile", "broadcast", "staged", "film", "unknown")
LABEL_SOURCES = ("dataset_label_trimmed_video", "temporal_annotation", "manual")
LABEL_CONFIDENCE = ("high", "medium", "low")

# ---------------------------------------------------------------- windows / later temporal rules
WINDOW_SECONDS = 1.5
MAX_WINDOWS_PER_VIDEO = 8          # v1 value, so v1 cached frames stay reusable
FPS_FALLBACK = 25.0                # v1 build_clip_manifest fallback
GUARD_SECONDS = {"XD": 2.0, "UCF": 1.0}     # used only once temporal annotations exist
QUIET_MIN_GAP_SECONDS = 3.0
QUIET_PER_VIOLENT_WINDOW = 2

# ---------------------------------------------------------------- perceptual hash / leakage
PHASH_FRAME_POSITIONS = (8, 16, 24)   # indices into the 32-frame v1 sampling of a window
PHASH_MAX_HAMMING = 5
PHASH_MIN_MATCHED_FRAMES = 2          # frames of segment A matching any frame of segment B
PHASH_MIN_GRAY_STD = 3.0              # near-uniform frames (black, fades) carry no identity
PHASH_CROP_P90 = 20                   # a row/column is picture content if its 90th-percentile grey > this
PHASH_CROP_MIN_FRAC = 0.15            # content box smaller than this fraction of the frame -> unusable
PHASH_THUMB = 32                      # verification thumbnail side (content-cropped, z-normalised grey)
PHASH_VERIFY_MIN_R = 0.67             # content-intersection correlation for a frame to count as matched.
                                      # Calibrated on 46 visually labelled pairs (data_tables/audit/calibration_labels.csv):
                                      # true same-source pairs 0.716-0.995, different content <= 0.633; midpoint chosen.

# ---------------------------------------------------------------- splits
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42

# ---------------------------------------------------------------- inclusion rules
# (dataset, original_label pattern) -> rule. Patterns are regexes on original_label.
_FA = "FIGHT_ASSAULT"
RULES = [
    ("RWF2000", r"^Fight$", dict(binary=1, violence_type=_FA, type_provisional=True, negative_kind="none",
        setting="mixed_real_world", production="cctv", label_confidence="medium")),
    ("RWF2000", r"^NonFight$", dict(binary=0, violence_type=TYPE_NONE, type_provisional=False,
        negative_kind="same_dataset", setting="mixed_real_world", production="cctv", label_confidence="medium")),
    ("RLVS", r"^Violence$", dict(binary=1, violence_type=_FA, type_provisional=True, negative_kind="none",
        setting="mixed_real_world", production="web_mobile", label_confidence="medium")),
    ("RLVS", r"^NonViolence$", dict(binary=0, violence_type=TYPE_NONE, type_provisional=False,
        negative_kind="same_dataset", setting="mixed_real_world", production="web_mobile", label_confidence="medium")),
    ("HockeyFights", r"^fight$", dict(binary=1, violence_type=_FA, type_provisional=True, negative_kind="none",
        setting="sports", production="broadcast", label_confidence="high")),
    ("HockeyFights", r"^no_fight$", dict(binary=0, violence_type=TYPE_NONE, type_provisional=False,
        negative_kind="same_setting", setting="sports", production="broadcast", label_confidence="high")),
    ("BusViolence", r"^Violence$", dict(binary=1, violence_type=_FA, type_provisional=True, negative_kind="none",
        setting="public_transport", production="staged", label_confidence="high")),
    ("BusViolence", r"^NoViolence$", dict(binary=0, violence_type=TYPE_NONE, type_provisional=False,
        negative_kind="same_setting", setting="public_transport", production="staged", label_confidence="high")),
    ("VID", r"^v_s_b_\d+$", dict(binary=1, violence_type=_FA, type_provisional=True, negative_kind="none",
        setting="street", production="unknown", label_confidence="medium")),
    ("VID", r"^v_c_b_\d+$", dict(binary=1, violence_type=_FA, type_provisional=True, negative_kind="none",
        setting="classroom", production="unknown", label_confidence="medium")),
    ("VID", r"^v_d_b_\d+$", dict(binary=1, violence_type=_FA, type_provisional=True, negative_kind="none",
        setting="domestic", production="unknown", label_confidence="medium")),
    ("VID", r"^v_o_b_\d+$", dict(binary=1, violence_type="OTHER_VIOLENCE", type_provisional=True,
        negative_kind="none", setting="other", production="unknown", label_confidence="medium")),
    ("VID", r"^nv_b_\d+$", dict(binary=0, violence_type=TYPE_NONE, type_provisional=False,
        negative_kind="same_dataset", setting="unknown", production="unknown", label_confidence="medium")),
    # XD normals: whole videos with label A contain no violence, so windows are valid NV. Excluded
    # by default (see EXCLUSIONS) and admitted only with --include-xd-normals.
    ("XDtrain", r"^A$", dict(binary=0, violence_type=TYPE_NONE, type_provisional=False,
        negative_kind="everyday", setting="film", production="film", label_confidence="high")),
]

EXCLUSIONS = {
    ("XDtrain", "B4"): ("untrimmed_video_level_label",
        "Untrimmed XD-train riot video (median ~128 s) with only a video-level label. XD-test "
        "annotations show riot videos are ~78% violent by duration, so fixed windows would carry "
        "~22% label noise. Requires temporal annotation before use."),
    ("XDtrain", "B1"): ("untrimmed_video_level_label",
        "Untrimmed XD-train fighting video (median ~60 s) with only a video-level label. XD-test "
        "annotations show fighting videos are ~46% violent by duration. Requires temporal annotation."),
    ("XDtrain", "A"): ("single_class_source_pending_annotation",
        "XD-train normal video. Valid non-violent footage, but with XD violent material excluded "
        "pending annotation, XD would contribute only NON_VIOLENCE and its film/web appearance "
        "would become a label shortcut. Admit with --include-xd-normals only by explicit decision."),
    ("Movies", "*"): ("held_out_unknown_film_provenance",
        "Movies/Peliculas: film titles are unknown, so film-level leakage cannot be prevented. "
        "Reserved as a held-out film-domain test set."),
}
REASONS = {
    "xd_test_title_overlap": "Film/web source title also appears in XD-test (external benchmark).",
    "shorter_than_window": "Video shorter than one 1.5 s window; no valid segment can be formed.",
    "missing_file": "File path does not exist on disk.",
    "unreadable": "OpenCV could not open the file or read a frame count / fps.",
    "duplicate_content": "Byte-identical file (MD5) already included from another row.",
    "external_benchmark": "UCF-Crime test video with temporal annotation: external benchmark, never training.",
    "no_temporal_annotation": "UCF-Crime violent-category video with only a video-level label; requires temporal annotation.",
    "single_class_source_pending_annotation_ucf": "UCF-Crime non-violent anomaly video; with no UCF violent training "
        "segments available, UCF would be a non-violence-only source.",
    "ignore_category": "Explosion / Arrest: ambiguous with respect to the violence definition; IGNORE in training, "
        "reported separately.",
    "no_rule": "No inclusion rule for this dataset/label combination.",
}
UCF_VIOLENT = ("Abuse", "Assault", "Fighting", "Robbery", "Shooting")
UCF_IGNORE = ("Explosion", "Arrest")


def rule_for(dataset, original_label):
    for ds, pat, rule in RULES:
        if ds == dataset and re.match(pat, str(original_label)):
            return rule
    return None


def xd_title(filename):
    """Source title of an XD-Violence file: film title, or 'v=<youtube id>' for web videos."""
    return re.split(r"__#", Path(str(filename)).name)[0]


def source_unit_id(dataset, original_label, original_filename, path):
    """Independent source unit and the rule used to derive it."""
    stem = Path(str(path)).stem
    if dataset == "VID":
        return f"VID:group:{original_label}", "recording_group"
    if dataset == "XDtrain":
        t = xd_title(original_filename)
        return (f"XD:web:{t[2:]}", "youtube_id") if t.startswith("v=") else (f"XD:film:{t}", "film_title")
    if dataset == "UCF":
        return f"UCF:video:{stem}", "video"
    # RWF2000 / RLVS / HockeyFights / BusViolence: upstream provenance lost in the VID-Detect copies
    return f"{dataset}:video:{stem}", "video_provenance_limited"
NEW_VIDEO_DECODE_MODE = "seek"          # set to "sequential" if seek-determinism test fails for new videos


# ---------------------------------------------------------------- STEP 2 decisions (accepted)
# Decision 2: candidate stage is permissive (recall first); verification is unchanged.
PHASH_CAND_MAX_HAMMING = 12
PHASH_CAND_MIN_FRAMES = 1
# Decision 3: datasets whose independent sources cannot be split are train-only in the main experiment.
TRAIN_ONLY_DATASETS = ("VID",)
# Decision 1: visually reviewed external-benchmark matches; confirmed groups are removed from the benchmark.
CONTAMINATION_REVIEW = "data_tables/audit/external_benchmark_matches_reviewed.csv"
CONTAMINATION_MAX_COMPONENT_WINDOWS = 500   # larger groups need a manual decision, not silent removal
REASONS["external_benchmark_contamination"] = (
    "Shares footage (visually confirmed) with an XD-test or UCF-test video, or belongs to the same "
    "near-duplicate group as such a video; removed so the external benchmark stays unseen.")

# ---- post-rebuild audit fix: fine-structure verification (stage 3) ----
# The looser candidate stage surfaced many coarse-verified (64x64 r >= 0.67) pairs that share only
# low-frequency layout; they chained HockeyFights / RLVS / RWF units into train-only VID groups.
# Stage 3 requires Pearson r of Sobel gradient magnitude on 128x128 grey (content-box intersection,
# >= 2 frames per side) >= PHASH_FINE_MIN_R. Calibration (data_tables/audit/fine_probe_*.csv):
# visually confirmed shared footage >= 0.564 internal (excluding the handheld same-event pair no image
# rule detects) and >= 0.337 external; cross-dataset coarse-verified links median 0.025, q90 0.16.
PHASH_FINE_SIDE = 128
PHASH_FINE_MIN_R = 0.30
# Unreviewed external matches at or above this fine score are excluded from external evaluation as a
# precaution (not removed from training: not visually confirmed).
EXTERNAL_PRECAUTION_MIN_R = 0.30

# ---- Phase 5 decisions: annotation kinds and the canonical per-video window cap ----
# The primary task stays binary. Only 'violent' and 'non_violent' produce training labels; everything else is
# represented in the annotation data but excluded from both classes, so ambiguity is never forced into
# VIOLENCE or NON_VIOLENCE.
ANNOTATION_KINDS = ("violent", "non_violent", "restraint", "aftermath", "ambiguous", "unusable")
KIND_TO_BINARY = {"violent": 1, "non_violent": 0}
EXCLUDED_KINDS = ("restraint", "aftermath", "ambiguous", "unusable")
ANNOTATION_STATUSES = ("complete", "partial", "unusable_source")
# Use of force (Decision 1): struck / thrown / dragged / forcefully pinned -> 'violent';
# routine handcuffing, escorting or a controlled hold with no strike -> 'restraint'.
# MAX_WINDOWS_PER_VIDEO is 8 windows TOTAL per source video, counting every class together. This is the
# canonical cap for manifest generation, annotation derivation, external evaluation and validation.
MAX_WINDOWS_TOTAL_PER_VIDEO = MAX_WINDOWS_PER_VIDEO

# ---- Video-only protocol (rule 7, revised) ----
# The study is VIDEO-ONLY. Audio is not part of the dataset or the protocol and is never used for
# annotation, labelling, training, evaluation or any decision. Final annotation requires continuous
# VISUAL playback with temporal context; still grids may only locate candidates and refine boundaries.
# An annotation's `annotator` field must record the review method and contain REVIEW_METHOD_TOKEN.
REVIEW_METHOD_TOKEN = "playback"
REVIEW_METHOD_EXAMPLE = "jdoe (visual playback, video-only)"
# Strings that assert a stills-only review are rejected even if they also say "playback".
STILLS_ONLY_MARKERS = ("stills only", "stills-only", "still frames only", "grids only", "grid only",
                       "screenshots only", "frames only")
# SHOOTING_WEAPON requires DIRECT VISUAL evidence: a firearm visibly present and visibly used/fired, or
# another unambiguous visual indication. Never inferred from running, police presence, smoke, crowd
# reaction, aftermath alone, or dataset context. Audio is never evidence.
