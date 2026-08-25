#!/usr/bin/env bash
# Download THINGS-data subsets into $DEST (default data/things).
#
# usage: download_things.sh [annotations|fmri|meg|meg-probe|tiny] [--dest DIR] [subjects...]
#   annotations  THINGSplus property ratings TSV (~10 MB)
#   fmri         betas_csv.zip -> filtered unzip -> delete archive
#                (figshare hosts one opaque 43 GB zip, so subjects ride in one transfer;
#                 includes the per-subject ResponseData.h5 and metadata CSVs)
#   meg          per-subject preprocessed epochs from OpenNeuro ds004212 (needs aws cli):
#                aws s3 cp --no-sign-request s3://openneuro.org/ds004212/derivatives/preprocessed/
#   meg-probe    list remote MEG files without downloading
#   tiny         annotations + meg-probe
set -euo pipefail

FMRI_ZIP="https://ndownloader.figshare.com/files/43635873"
MEG_S3="s3://openneuro.org/ds004212/derivatives/preprocessed/"
RATINGS="https://osf.io/download/670d5918125f07cd45015270"

WHAT=tiny; DEST=data/things; SUBJECTS=(01)
while [[ $# -gt 0 ]]; do
  case $1 in
    --dest) DEST=$2; shift 2 ;;
    annotations|fmri|meg|meg-probe|tiny) WHAT=$1; shift ;;
    *) SUBJECTS+=("$1"); shift ;;
  esac
done
SUBJECTS=${SUBJECTS[*]}
mkdir -p "$DEST"

do_annotations() {
  local out="$DEST/annotations/property-ratings.tsv"
  mkdir -p "$(dirname "$out")"
  [[ -e $out ]] && { echo "exists, skipping $out"; return; }
  curl -fsSL "$RATINGS" -o "$out"
  echo "wrote $out ($(du -m "$out" | cut -f1) MB)"
}

do_fmri() {
  local archive="$DEST/betas_csv.zip"
  curl -fsSL "$FMRI_ZIP" -o "$archive"
  local pats=()
  for s in $SUBJECTS; do pats+=("betas_csv/sub-${s}_*"); done
  unzip -o "$archive" "${pats[@]}" -d "$DEST"
  rm "$archive"
}

do_meg() {
  command -v aws >/dev/null || { echo "aws cli required: https://docs.aws.amazon.com/cli/"; exit 1; }
  local inc=()
  for s in $SUBJECTS; do inc+=(--include "preprocessed_P${s#0}-epo*"); done
  mkdir -p "$DEST/meg"
  aws s3 cp --no-sign-request --recursive --exclude "*" "${inc[@]}" "$MEG_S3" "$DEST/meg/"
}

do_meg_probe() {
  command -v aws >/dev/null || { echo "aws cli required"; return; }
  aws s3 ls --no-sign-request "$MEG_S3" | grep -E "preprocessed_P[0-9]+-epo"
}

case $WHAT in
  annotations) do_annotations ;;
  fmri) do_fmri ;;
  meg) do_meg ;;
  meg-probe) do_meg_probe ;;
  tiny) do_annotations; do_meg_probe ;;
esac
