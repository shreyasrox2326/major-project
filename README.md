# BitNet Subliminal Learning Experiments

This repository extends the earlier subliminal learning project from regular neural networks to BitNet-style low-bit networks.

The first target is the MNIST auxiliary-logit experiment:

1. Train a teacher MLP on MNIST digit labels.
2. Feed random noise images into the trained teacher.
3. Save only the noise images and teacher auxiliary logits.
4. Train a student only on that noise/logit dataset.
5. Evaluate whether the student's primary digit logits classify real MNIST digits.
6. Compare full-precision MLPs against BitNet-style ternary MLPs.

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
cd major-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The intended workflow is:

```bash
# Laptop: develop and commit locally
git add .
git commit -m "Add MNIST BitNet subliminal experiment scaffold"
git push

# Laptop: sync code to Lightning, since the server does not have git
./scripts/sync_to_lightning.sh

# Lightning: run experiments
cd ~/major-project
python experiments/mnist_subliminal.py --model ternary --noise-size 10000 --epochs-student 1 --quick
```

If `rsync` is unavailable, use `scp`:

```bash
tar --exclude .git --exclude .venv --exclude data --exclude runs -czf /tmp/major-project.tar.gz .
scp /tmp/major-project.tar.gz s_01m2fw60p95y4zweav7yree60q@ssh.lightning.ai:~/
ssh s_01m2fw60p95y4zweav7yree60q@ssh.lightning.ai 'mkdir -p ~/major-project && tar -xzf ~/major-project.tar.gz -C ~/major-project'
```

## First Experiments

Run a smoke test:

```bash
python experiments/mnist_subliminal.py --model ternary --noise-size 10000 --epochs-teacher 1 --epochs-student 1 --quick
```

Run the regular full-precision baseline:

```bash
python experiments/mnist_subliminal.py --model fp32 --noise-size 200000 --epochs-teacher 5 --epochs-student 30 --student-init same --distill aux
```

Run the BitNet-style ternary version:

```bash
python experiments/mnist_subliminal.py --model ternary --noise-size 200000 --epochs-teacher 5 --epochs-student 30 --student-init same --distill aux
```

Useful controls:

```bash
python experiments/mnist_subliminal.py --model ternary --noise-size 200000 --student-init different --distill aux
python experiments/mnist_subliminal.py --model ternary --noise-size 200000 --student-init same --distill all
python experiments/mnist_subliminal.py --model fp32 --noise-size 200000 --student-init different --distill aux
```

Results are written under `runs/`.

Each run directory contains:

- `config.json`: exact run parameters.
- `teacher_history.csv`: teacher train loss, validation loss, validation accuracy, test loss, and test accuracy per epoch.
- `student_history.csv`: student train KL loss, validation KL loss, MNIST test loss, and MNIST test accuracy per epoch.
- `summary.json`: final metrics and output file list.
- `teacher_loss.png`, `teacher_accuracy.png`, `student_kl_loss.png`, `student_mnist_accuracy.png`: plots for the report.
- `teacher.pt`, `student.pt`: saved model weights.
