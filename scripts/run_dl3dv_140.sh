#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
    echo "Usage: bash scripts/run_dl3dv_140.sh [output-directory]" >&2
    exit 2
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_dir"

output_dir="${1:-results/dl3dv_a05_140_refine16k}"
exec python scripts/run_dl3dv_a05.py "$output_dir" \
    --all --steps 48000 --refine-stop-iter 16000 --growth-headroom-mib 0
