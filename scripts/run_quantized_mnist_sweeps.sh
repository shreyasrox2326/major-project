#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root inside the CUDA-enabled container.
# Each precision gets its own archive, preserving the previous run_N folders.
for entry in 'q4 run_6_q4_mlp_centered_minus1to1' \
             'q8 run_7_q8_mlp_centered_minus1to1' \
             'q16 run_8_q16_mlp_centered_minus1to1'; do
  read -r model output_dir <<< "$entry"
  if [ -e "$output_dir" ]; then
    if [ "${RESUME:-0}" != 1 ]; then
      echo "$output_dir already exists; set RESUME=1 only to continue this sweep." >&2
      exit 1
    fi
    if [ -f "$output_dir/checkpoints/init_summary.json" ]; then
      python - "$output_dir/checkpoints/init_summary.json" "$model" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    actual = json.load(handle)["model"]
if actual != sys.argv[2]:
    raise SystemExit(f"Existing run model {actual} does not match {sys.argv[2]}")
PY
    fi
  fi
  MODEL="$model" OUTPUT_DIR="$output_dir" bash scripts/run_ternary_mnist_sweep.sh
done
