#!/usr/bin/env bash
# Download all THINGS annotations, fMRI betas, and preprocessed MEG epochs.
# Usage: ./scripts/download_things.sh [destination]
set -euo pipefail

DEST=${1:-data/things}
FMRI_URL=https://ndownloader.figshare.com/files/43635873
MEG_URL=s3://openneuro.org/ds004212/derivatives/preprocessed/
RATINGS_URL=https://osf.io/download/670d5918125f07cd45015270

for cmd in curl unzip aws; do
  command -v "$cmd" >/dev/null || { echo "$cmd is required" >&2; exit 1; }
done

ratings="$DEST/annotations/property-ratings.tsv"
if [[ ! -s $ratings ]]; then
  mkdir -p "$(dirname "$ratings")"
  curl -fL --retry 8 --progress-bar "$RATINGS_URL" -o "$ratings.tmp"
  mv "$ratings.tmp" "$ratings"
fi

archive="$DEST/fmri/betas_csv.zip"
complete="$DEST/fmri/betas_csv/.download-complete"
if [[ ! -e $complete ]]; then
  mkdir -p "$DEST/fmri"
  unzip -Z1 "$archive" >/dev/null 2>&1 || \
    curl -fL --retry 8 --progress-bar -C - "$FMRI_URL" -o "$archive"
  unzip -o "$archive" 'betas_csv/*' -d "$DEST/fmri"
  touch "$complete"
fi
rm -f "$archive"

mkdir -p "$DEST/meg"
aws --endpoint-url https://s3.amazonaws.com \
    --region us-east-1 \
    --no-sign-request \
    s3 sync \
    --exclude '*' \
    --include 'preprocessed_P*-epo*.fif' \
  "$MEG_URL" \
  "$DEST/meg/"
