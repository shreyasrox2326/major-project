# BitNet Subliminal Learning Experiments

This repository extends the earlier subliminal learning project from regular neural networks to BitNet-style low-bit networks.

The first target is the MNIST logit-distillation experiment:

1. Train a teacher MLP on MNIST digit labels.
2. Feed random noise images into the trained teacher.
3. Save the noise images and all 13 teacher logits once.
4. Train students on slices of that fixed noise/logit dataset.
5. Evaluate whether the student's primary digit logits classify real MNIST digits.
6. Compare ternary runs across noise size, logit target, and initialization mode.

## Implementation Choice

The MNIST BitNet experiments do not depend on external BitNet libraries.

`experiments/mnist_subliminal.py` includes a small PyTorch implementation of a BitNet-style `BitLinear` layer:

- latent full-precision trainable weights,
- ternary forward weights in `{-1, 0, +1}`,
- abs-mean scaling,
- straight-through estimator for training.

This is enough for the first research question: whether ternary weight discretization changes subliminal transfer in the old MNIST setup.

## Setup

On the laptop:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On the Lightning server:

```bash
cd ~/repos/major-project
python3 -m venv ~/venvs/mnist-bitnet
source ~/venvs/mnist-bitnet/bin/activate
pip install -r requirements.txt
```

The intended workflow is:

```bash
# Laptop: develop and commit locally
git add .
git commit -m "Update MNIST BitNet experiment pipeline"
git push

# Lightning: pull and run experiments
cd ~/repos/major-project
git pull
source ~/venvs/mnist-bitnet/bin/activate
```

## First Experiments

Run a smoke test:

```bash
python experiments/mnist_subliminal.py --stage all --model ternary --max-noise-size 1000 --epochs-teacher 1 --epochs-student 1 --quick --num-workers 0 --output-dir runs_smoke
```

## Controlled Ternary Sweep

The main experiment uses one fixed ternary teacher initialization, one fixed different-student initialization, one trained ternary teacher, and one cached 400k noise/logit dataset.

Run everything:

```bash
bash scripts/run_ternary_mnist_sweep.sh
```

Or run stages manually:

```bash
python experiments/mnist_subliminal.py --stage init --model ternary
python experiments/mnist_subliminal.py --stage train-teacher --model ternary --epochs-teacher 5 --resume
python experiments/mnist_subliminal.py --stage generate-logits --model ternary --max-noise-size 400000
python experiments/mnist_subliminal.py --stage train-student --model ternary --noise-size 200000 --distill aux --student-init same --epochs-student 30 --resume
```

Results are written under `runs/`.

Each run directory contains:

- `config.json`: exact run parameters.
- `checkpoints/teacher_init.pt`: fixed teacher initialization used by same-init students.
- `checkpoints/student_different_init.pt`: fixed different initialization used by different-init students.
- `checkpoints/teacher_latest.ckpt`: resumable teacher checkpoint with model and optimizer state.
- `checkpoints/teacher_trained.pt`: final trained teacher weights.
- `checkpoints/teacher_history.csv`: teacher train loss, validation loss, validation accuracy, test loss, test accuracy, and learning rate per epoch.
- `logit_cache/noise.pt`: fixed random noise inputs.
- `logit_cache/teacher_logits_all.pt`: all 13 teacher logits for the fixed 400k noise inputs.
- `students/<run>/student_history.csv`: student train distillation loss, validation distillation loss, MNIST test loss, MNIST test accuracy, learning rate, target type, noise size, and init mode per epoch.
- `students/<run>/student_latest.ckpt`: resumable student checkpoint with model and optimizer state.
- `students/<run>/student_trained.pt`: final student weights.
- `summary.json`: final metrics and output file list.
- `teacher_loss.png`, `teacher_accuracy.png`, `student_distill_loss.png`, `student_mnist_accuracy.png`: plots for the report.

To extend a student run later, increase `--epochs-student` and pass `--resume` with the same run parameters. The script loads `student_latest.ckpt`, resumes Adam's optimizer state, and appends to the same history CSV.
