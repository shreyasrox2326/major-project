#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
ARCHIVE_NAME="${ARCHIVE_NAME:-run_5_plateau_best_stress}"
STRESS_LR="${STRESS_LR:-1e-4}"
STRESS_EPOCHS="${STRESS_EPOCHS:-100}"

if [ -e "$OUTPUT_ROOT" ]; then
  echo "$OUTPUT_ROOT already exists; move or remove it before starting a fresh stress run." >&2
  exit 1
fi

if [ -e "$ARCHIVE_NAME" ]; then
  echo "$ARCHIVE_NAME already exists; choose a different ARCHIVE_NAME." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

log_step() {
  printf '%s | SCRIPT | %s\n' "$(TZ=Asia/Kolkata date '+%Y-%m-%d %I:%M:%S %p IST')" "$*" | tee -a "$OUTPUT_ROOT/experiment.log"
}

run_stress() {
  local model="$1"
  local output_dir="$OUTPUT_ROOT/$model"

  log_step "stress setup model=$model output=$output_dir"
  python experiments/mnist_subliminal.py --stage init \
    --model "$model" \
    --seed 0 \
    --noise-seed 1000 \
    --batch-size 256 \
    --teacher-lr 1e-3 \
    --student-lr "$STRESS_LR" \
    --scheduler none \
    --epochs-teacher 5 \
    --epochs-student 30 \
    --hidden-size 256 \
    --val-fraction 0.2 \
    --num-workers 2 \
    --data-dir data \
    --output-dir "$output_dir"

  log_step "stress teacher model=$model"
  python experiments/mnist_subliminal.py --stage train-teacher \
    --model "$model" \
    --seed 0 \
    --noise-seed 1000 \
    --batch-size 256 \
    --teacher-lr 1e-3 \
    --student-lr "$STRESS_LR" \
    --scheduler none \
    --epochs-teacher 5 \
    --epochs-student 30 \
    --hidden-size 256 \
    --val-fraction 0.2 \
    --num-workers 2 \
    --data-dir data \
    --output-dir "$output_dir" \
    --resume

  log_step "stress logits model=$model"
  python experiments/mnist_subliminal.py --stage generate-logits \
    --model "$model" \
    --seed 0 \
    --noise-seed 1000 \
    --batch-size 256 \
    --teacher-lr 1e-3 \
    --student-lr "$STRESS_LR" \
    --scheduler none \
    --epochs-teacher 5 \
    --epochs-student 30 \
    --hidden-size 256 \
    --val-fraction 0.2 \
    --num-workers 2 \
    --data-dir data \
    --output-dir "$output_dir" \
    --max-noise-size 400000

  log_step "stress student model=$model lr=$STRESS_LR scheduler=plateau_best epochs=$STRESS_EPOCHS"
  python experiments/mnist_subliminal.py --stage train-student \
    --model "$model" \
    --seed 0 \
    --noise-seed 1000 \
    --batch-size 256 \
    --teacher-lr 1e-3 \
    --student-lr "$STRESS_LR" \
    --scheduler plateau_best \
    --scheduler-monitor mnist_test_accuracy \
    --epochs-teacher 5 \
    --epochs-student "$STRESS_EPOCHS" \
    --hidden-size 256 \
    --val-fraction 0.2 \
    --num-workers 2 \
    --data-dir data \
    --output-dir "$output_dir" \
    --noise-size 400000 \
    --distill aux \
    --student-init same \
    --run-tag "plateau_best_lr${STRESS_LR}_${STRESS_EPOCHS}epoch" \
    --resume
}

run_stress ternary
run_stress fp32

log_step "stress run complete; archiving $OUTPUT_ROOT to $ARCHIVE_NAME"
mv "$OUTPUT_ROOT" "$ARCHIVE_NAME"
