#!/usr/bin/env bash
set -euo pipefail

COMMON_ARGS=(
  --model "${MODEL:-ternary}"
  --seed 0
  --noise-seed 1000
  --batch-size 256
  --teacher-lr 1e-3
  --student-lr 1e-4
  --scheduler none
  --epochs-teacher 5
  --epochs-student 30
  --hidden-size 256
  --val-fraction 0.2
  --num-workers 2
  --data-dir data
  --output-dir runs
)

log_step() {
  mkdir -p runs
  printf '%s | SCRIPT | %s\n' "$(TZ=Asia/Kolkata date '+%Y-%m-%d %I:%M:%S %p IST')" "$*" | tee -a runs/experiment.log
}

log_step "init fixed checkpoints for model=${MODEL:-ternary}"
python experiments/mnist_subliminal.py --stage init "${COMMON_ARGS[@]}"
log_step "train teacher for model=${MODEL:-ternary}"
python experiments/mnist_subliminal.py --stage train-teacher "${COMMON_ARGS[@]}" --resume
log_step "generate 400k teacher logits for model=${MODEL:-ternary}"
python experiments/mnist_subliminal.py --stage generate-logits "${COMMON_ARGS[@]}" --max-noise-size 400000

log_step "student 10k aux same"
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 10000 --distill aux --student-init same --resume
log_step "student 40k aux same"
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 40000 --distill aux --student-init same --resume
log_step "student 100k aux same"
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 100000 --distill aux --student-init same --resume
log_step "student 400k aux same"
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 400000 --distill aux --student-init same --resume

log_step "student 200k aux same"
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 200000 --distill aux --student-init same --resume
log_step "student 200k main same"
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 200000 --distill main --student-init same --resume
log_step "student 200k all same"
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 200000 --distill all --student-init same --resume
log_step "student 200k aux different"
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 200000 --distill aux --student-init different --resume
log_step "student 200k main different"
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 200000 --distill main --student-init different --resume
log_step "student 200k all different"
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 200000 --distill all --student-init different --resume

log_step "diagnostic student 400k aux same plateau-best lr1e-4 100epoch"
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 400000 --distill aux --student-init same --student-lr 1e-4 --scheduler plateau_best --scheduler-monitor mnist_test_accuracy --epochs-student 100 --run-tag plateau_best_lr1e-4_100epoch --resume
log_step "sweep complete"
