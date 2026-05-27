#!/usr/bin/env bash
set -euo pipefail

# Generate 2500 synthetic BCC and BKL images from existing StyleGAN snapshots
# and package them into a downloadable archive.
#
# Typical usage on the server:
#   bash generate_bcc_bkl_2500_archive.sh
#
# If auto-detection does not find a snapshot, pass it explicitly:
#   NETWORK_BKL=/path/to/network-snapshot-XXXXXX.pkl bash generate_bcc_bkl_2500_archive.sh

COUNT="${COUNT:-2500}"
TRUNC="${TRUNC:-0.7}"
SEED_START="${SEED_START:-0}"
OUT_ROOT="${OUT_ROOT:-synthetic_generation_runs/bcc_bkl_${COUNT}}"
ARCHIVE="${ARCHIVE:-synthetic_bcc_bkl_${COUNT}.tar.gz}"
GEN_SCRIPT="${GEN_SCRIPT:-stylegan3-brecahad/stylegan3/gen_images.py}"

NETWORK_BCC="${NETWORK_BCC:-}"
NETWORK_BKL="${NETWORK_BKL:-}"

SEARCH_ROOTS=(
  "gan_training_runs_transfer"
  "gan_training_runs_quality"
  "gan_training_runs_batch32"
  "gan_training_runs"
  "gan_training_runs_raw"
)

find_snapshot() {
  local cls="$1"
  local best=""

  for root in "${SEARCH_ROOTS[@]}"; do
    [[ -d "$root" ]] || continue
    best="$(find "$root" -type f -name 'network-snapshot-*.pkl' \
      | grep -Ei "/|${cls}" \
      | grep -Ei "(${cls}|${cls}_raw|raw_${cls})" \
      | sort -V \
      | tail -n 1 || true)"
    if [[ -n "$best" ]]; then
      echo "$best"
      return 0
    fi
  done

  return 1
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] $label not found: $path" >&2
    exit 1
  fi
}

count_images() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l
}

generate_class() {
  local cls="$1"
  local network="$2"
  local out_dir="$OUT_ROOT/$cls"
  local seed_end=$((SEED_START + COUNT - 1))

  mkdir -p "$out_dir"

  local existing
  existing="$(count_images "$out_dir")"
  if [[ "$existing" -ge "$COUNT" ]]; then
    echo "[SKIP] $cls already has $existing images in $out_dir"
    return 0
  fi

  echo "============================================================"
  echo "[GENERATE] class=$cls count=$COUNT trunc=$TRUNC seeds=${SEED_START}-${seed_end}"
  echo "[NETWORK] $network"
  echo "[OUT] $out_dir"
  echo "============================================================"

  python "$GEN_SCRIPT" \
    --outdir="$out_dir" \
    --trunc="$TRUNC" \
    --seeds="${SEED_START}-${seed_end}" \
    --network="$network"

  existing="$(count_images "$out_dir")"
  if [[ "$existing" -lt "$COUNT" ]]; then
    echo "[ERROR] $cls generation incomplete: expected $COUNT, got $existing" >&2
    exit 1
  fi
}

echo "============================================================"
echo "[1/4] Resolve StyleGAN snapshots"
echo "============================================================"
require_file "$GEN_SCRIPT" "StyleGAN generation script"

if [[ -z "$NETWORK_BCC" ]]; then
  NETWORK_BCC="$(find_snapshot bcc || true)"
fi
if [[ -z "$NETWORK_BKL" ]]; then
  NETWORK_BKL="$(find_snapshot bkl || true)"
fi

if [[ -z "$NETWORK_BCC" ]]; then
  echo "[ERROR] Could not auto-detect BCC snapshot. Set NETWORK_BCC=/path/to/network-snapshot.pkl" >&2
  exit 1
fi
if [[ -z "$NETWORK_BKL" ]]; then
  echo "[ERROR] Could not auto-detect BKL snapshot. Set NETWORK_BKL=/path/to/network-snapshot.pkl" >&2
  echo "[HINT] Check available snapshots with:" >&2
  echo "  find gan_training_runs* -type f -name 'network-snapshot-*.pkl' | grep -i bkl" >&2
  exit 1
fi

require_file "$NETWORK_BCC" "BCC snapshot"
require_file "$NETWORK_BKL" "BKL snapshot"
echo "[OK] BCC: $NETWORK_BCC"
echo "[OK] BKL: $NETWORK_BKL"

echo
echo "============================================================"
echo "[2/4] Generate images"
echo "============================================================"
generate_class bcc "$NETWORK_BCC"
generate_class bkl "$NETWORK_BKL"

echo
echo "============================================================"
echo "[3/4] Write manifest"
echo "============================================================"
MANIFEST="$OUT_ROOT/manifest.csv"
{
  echo "image_path,label,source,network,truncation"
  find "$OUT_ROOT/bcc" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort | sed "s#^#/#" | while read -r p; do
    rel="${p#/}"
    echo "$rel,bcc,synthetic,$NETWORK_BCC,$TRUNC"
  done
  find "$OUT_ROOT/bkl" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort | sed "s#^#/#" | while read -r p; do
    rel="${p#/}"
    echo "$rel,bkl,synthetic,$NETWORK_BKL,$TRUNC"
  done
} > "$MANIFEST"

echo "[OK] Manifest: $MANIFEST"
echo "[OK] bcc images: $(count_images "$OUT_ROOT/bcc")"
echo "[OK] bkl images: $(count_images "$OUT_ROOT/bkl")"

echo
echo "============================================================"
echo "[4/4] Package archive"
echo "============================================================"
if [[ -e "$ARCHIVE" ]]; then
  BACKUP="${ARCHIVE}.bak.$(date +%Y%m%d_%H%M%S)"
  echo "[WARN] Archive already exists: $ARCHIVE"
  echo "[WARN] Moving old archive to: $BACKUP"
  mv "$ARCHIVE" "$BACKUP"
fi

tar -czf "$ARCHIVE" -C "$OUT_ROOT" bcc bkl manifest.csv

echo "[SUCCESS] Archive ready: $(pwd)/$ARCHIVE"
echo "[SUCCESS] Source folder: $(pwd)/$OUT_ROOT"
