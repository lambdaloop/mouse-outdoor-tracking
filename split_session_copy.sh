#!/usr/bin/env bash
# Split 2026_05_19_mouse_right into two sessions at the point the camera rig moved.
#
# The rig physically moved during video_21_2026-05-20T16_00_55, so extrinsic
# calibration differs before vs after. We copy the recordings into two independent
# sessions and EXCLUDE the moved-camera slot (2026-05-20T16_00_55) from BOTH.
#
# All 12 cameras (10-21) share the same timestamp string per recording slot, and the
# ISO-8601 timestamps sort lexicographically, so a plain string compare partitions them.
#
# Run directly on a cluster node with fast storage access, or submit via LSF:
#   bsub -J split_5_19 -n 4 -o split_5_19.log "bash split_session_copy.sh"
set -euo pipefail

SRC="/groups/voigts/voigtslab/outdoor/2026_05_19_mouse_right/data"
SPLIT="2026-05-20T16_00_55"
DST1="/groups/karashchuk/karashchuklab/outdoor_data/2026_05_19_mouse_right_1/data"
DST2="/groups/karashchuk/karashchuklab/outdoor_data/2026_05_19_mouse_right_2/data"

WORK="$(mktemp -d)"
LIST1="$WORK/session1.files"
LIST2="$WORK/session2.files"
: > "$LIST1"; : > "$LIST2"

# Bucket every file by the timestamp embedded in its name.
for f in "$SRC"/video_*.avi "$SRC"/timestamps_*.cvs; do
    name="$(basename "$f")"
    ts="$(printf '%s' "$name" | sed -E 's/^(video|timestamps)_[0-9]+_//; s/\.(avi|cvs)$//')"
    if [[ "$ts" < "$SPLIT" ]]; then
        printf '%s\n' "$name" >> "$LIST1"
    elif [[ "$ts" > "$SPLIT" ]]; then
        printf '%s\n' "$name" >> "$LIST2"
    fi   # ts == SPLIT -> excluded from both
done

echo "Session 1 files: $(wc -l < "$LIST1")  (expected 504)"
echo "Session 2 files: $(wc -l < "$LIST2")  (expected 1536)"

mkdir -p "$DST1" "$DST2"

RSYNC_OPTS=(-a --partial --info=progress2 --human-readable)
echo "=== Copying session 1 -> $DST1 ==="
rsync "${RSYNC_OPTS[@]}" --files-from="$LIST1" "$SRC/" "$DST1/"
echo "=== Copying session 2 -> $DST2 ==="
rsync "${RSYNC_OPTS[@]}" --files-from="$LIST2" "$SRC/" "$DST2/"

echo "=== Verification ==="
echo "Session 1 dest count: $(find "$DST1" -maxdepth 1 -type f | wc -l)  (expected 504)"
echo "Session 2 dest count: $(find "$DST2" -maxdepth 1 -type f | wc -l)  (expected 1536)"
rm -rf "$WORK"
echo "Done."
