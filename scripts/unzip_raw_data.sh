#!/usr/bin/env bash
set -euo pipefail

# Unzip any `data/raw/*.zip` files in-place.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT/data/raw"

if [[ ! -d "$RAW_DIR" ]]; then
  echo "ERROR: raw data directory not found: $RAW_DIR" >&2
  exit 1
fi

shopt -s nullglob
zips=("$RAW_DIR"/*.zip)

if [[ ${#zips[@]} -eq 0 ]]; then
  echo "No zip files found in $RAW_DIR (nothing to do)."
  exit 0
fi

echo "Unzipping ${#zips[@]} file(s) into $RAW_DIR ..."
for z in "${zips[@]}"; do
  echo "  - $(basename "$z")"
  unzip -o -q "$z" -d "$RAW_DIR"
done

echo "Done."



