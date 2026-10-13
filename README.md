# Major Project: Subliminal Learning in BitNet-Style Models

This repository contains experiments for a B.Tech major project on subliminal learning in low-bit neural networks.

The project extends prior subliminal-learning experiments from full-precision neural networks to BitNet-style models. The current work focuses on controlled MNIST MLP experiments. A later part of the project is intended to study subliminal preference transfer in BitNet-style language models.

## Project Scope

The repository is organized around two related questions:

1. Does subliminal learning still occur when a full-precision MLP is replaced by a BitNet-style ternary MLP?
2. If it does, is the signal weaker, less stable, or more sensitive to optimization and preprocessing?

The MNIST experiment follows the logit-distillation setup:

1. Train a teacher MLP on MNIST digit labels.
2. Feed random noise images into the trained teacher.
3. Cache the noise images and all 13 teacher logits.
4. Train student models on selected teacher logits from the cached noise dataset.
5. Evaluate whether the student's primary digit logits classify real MNIST digits.
6. Compare full-precision and BitNet-style MLPs across noise size, logit target, and initialization mode.

## Repository Layout

```text
experiments/
  mnist_subliminal.py        # MNIST teacher/student experiment runner
scripts/
  run_ternary_mnist_sweep.sh # Controlled MNIST sweep; supports ternary and fp32 models
docs/
  bitnet_mlp_results.md     # Current MNIST result summary
  images/                   # Figures used by the result summary
requirements.txt
```

Generated outputs such as `runs/`, archived `run_*/` folders, checkpoints, cached logits, PDFs, and local agent notes are intentionally not tracked.

## MNIST MLP Experiment

Both model variants use the same outer architecture:

```text
784 -> 256 -> 256 -> 13
```

The first 10 outputs are MNIST class logits. The final 3 outputs are auxiliary logits.

| Model | Hidden layer type |
|---|---|
| Full-precision MLP | `torch.nn.Linear` |
| BitNet-style MLP | local `BitLinear` implementation |
| 4/8/16-bit MLP | local `QuantizedLinear` implementation |

The local `BitLinear` layer uses full-precision latent trainable weights, ternary forward weights in `{-1, 0, +1}`, abs-mean scaling, and a straight-through estimator for backpropagation. This implementation is intended for controlled learning-behavior experiments, not optimized BitNet inference.

The `q4`, `q8`, and `q16` variants use full-precision latent trainable weights and symmetric per-tensor uniform quantization in each forward pass, with a straight-through gradient. Their biases and activations remain full precision. The signed quantizer uses `2^(bits-1)-1` as its maximum integer magnitude, giving 15, 255, and 65,535 weight levels respectively. These are training-time weight quantization experiments, not compressed inference benchmarks; checkpoint files retain full-precision latent weights and the same layout as previous runs.

## Setup

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a small smoke test:

```bash
python experiments/mnist_subliminal.py \
  --stage all \
  --model ternary \
  --max-noise-size 1000 \
  --epochs-teacher 1 \
  --epochs-student 1 \
  --quick \
  --num-workers 0 \
  --output-dir runs_smoke
```

## Running the Controlled Sweep

The sweep trains one teacher, caches one 400k noise/logit dataset, and reuses it across student runs.

Main sweep settings:

| Setting | Value |
|---|---:|
| MNIST/noise range | `[-1, 1]` |
| Teacher epochs | `5` |
| Student epochs | `30` |
| Teacher LR | `1e-3` |
| Student LR | `1e-4` |
| Batch size | `256` |
| Optimizer | Adam |
| Validation fraction | `0.2` |

Run the BitNet-style ternary sweep:

```bash
bash scripts/run_ternary_mnist_sweep.sh
```

Run the full-precision MLP sweep with the same settings:

```bash
MODEL=fp32 bash scripts/run_ternary_mnist_sweep.sh
```

From the repository root in the CUDA-enabled Docker container (mounted at `/workspaces/code/major-project`), run the continuation sweeps:

```bash
bash scripts/run_quantized_mnist_sweeps.sh
```

This writes the same checkpoints, histories, summaries, and plots as the previous sweeps under separate `run_6_q4_mlp_centered_minus1to1`, `run_7_q8_mlp_centered_minus1to1`, and `run_8_q16_mlp_centered_minus1to1` directories. Run one precision at a time with `MODEL=q4 OUTPUT_DIR=run_6_q4_mlp_centered_minus1to1 bash scripts/run_ternary_mnist_sweep.sh` (replace `q4` and the output folder for the other variants). Keep completed run folders intact for the combined comparison figures.
If the container stops mid-sweep, rerun the wrapper with `RESUME=1 bash scripts/run_quantized_mnist_sweeps.sh`.

After restoring the archived FP32 and ternary run folders and completing the new sweeps, rebuild the seven comparison figures and their source metrics with `scripts/plot_precision_comparison.py`. Pass one `--run LABEL=PATH` per model. If the older plateau stress runs are stored separately, use `--stress-run LABEL=PATH` for those models. The script writes to `docs/images/continuation` by default, preserving the already published figures until the combined results are reviewed.

The script also includes one diagnostic run at the end: 400k aux same-init, `student_lr=1e-4`, 100 epochs, and a plateau-best strategy that restores the run's own best checkpoint whenever the monitored accuracy plateaus before lowering the learning rate.

## Outputs

Experiment outputs are written under the selected output directory, usually `runs/`.

Important files:

| Path | Meaning |
|---|---|
| `experiment.log` | Timestamped stage and epoch log |
| `checkpoints/teacher_init.pt` | Fixed teacher initialization |
| `checkpoints/student_different_init.pt` | Fixed different student initialization |
| `checkpoints/teacher_history.csv` | Teacher loss and accuracy per epoch |
| `checkpoints/teacher_trained.pt` | Final trained teacher weights |
| `logit_cache/noise.pt` | Cached random noise inputs |
| `logit_cache/teacher_logits_all.pt` | Cached teacher logits for all 13 outputs |
| `students/<run>/student_history.csv` | Student loss, accuracy, LR, target, size, and init per epoch |
| `students/<run>/student_latest.ckpt` | Resumable student checkpoint |
| `students/<run>/student_best.ckpt` | Best resumable student checkpoint |
| `students/<run>/student_best.pt` | Best student weights only |
| `students/<run>/student_final.pt` | Final student weights only |
| `students/<run>/summary.json` | Best/final metrics for the run |

To continue a student run, use the same run parameters, increase `--epochs-student`, and pass `--resume`.

## Current Results

The current MNIST results are summarized in:

[docs/bitnet_mlp_results.md](docs/bitnet_mlp_results.md)

Short version: subliminal learning survives in the BitNet-style MLP, but the aux-only transfer is weaker and less stable than in the full-precision MLP.
