#!/usr/bin/env bash
set -euo pipefail

COMMON_ARGS=(
  --model ternary
  --seed 0
  --noise-seed 1000
  --batch-size 256
  --teacher-lr 1e-3
  --student-lr 1e-3
  --epochs-teacher 5
  --epochs-student 30
  --hidden-size 256
  --val-fraction 0.2
  --num-workers 2
  --data-dir data
  --output-dir runs
)

python experiments/mnist_subliminal.py --stage init "${COMMON_ARGS[@]}"
python experiments/mnist_subliminal.py --stage train-teacher "${COMMON_ARGS[@]}" --resume
python experiments/mnist_subliminal.py --stage generate-logits "${COMMON_ARGS[@]}" --max-noise-size 400000

python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 10000 --distill aux --student-init same --resume
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 40000 --distill aux --student-init same --resume
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 100000 --distill aux --student-init same --resume
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 400000 --distill aux --student-init same --resume

python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 200000 --distill aux --student-init same --resume
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 200000 --distill main --student-init same --resume
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 200000 --distill all --student-init same --resume
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 200000 --distill aux --student-init different --resume
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 200000 --distill main --student-init different --resume
python experiments/mnist_subliminal.py --stage train-student "${COMMON_ARGS[@]}" --noise-size 200000 --distill all --student-init different --resume
