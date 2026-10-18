#!/usr/bin/env python3
"""Rebuild the MNIST comparison figures from archived run folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIZES = (10_000, 40_000, 100_000, 400_000)
TARGETS = ("aux", "main", "all")
INITS = ("same", "different")


def parse_mapping(value: str) -> tuple[str, Path]:
    label, separator, location = value.partition("=")
    if not separator or not label or not location:
        raise argparse.ArgumentTypeError("expected LABEL=RUN_DIRECTORY")
    path = Path(location)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"run directory does not exist: {path}")
    return label, path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def student_dir(root: Path, size: int, target: str = "aux", init: str = "same") -> Path:
    matches = sorted((root / "students").glob(f"*_noise{size}_{target}_{init}_seed0"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one student run for {size} {target} {init} in {root}")
    return matches[0]


def stress_dir(root: Path) -> Path:
    matches = sorted((root / "students").glob("*plateau_best*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one plateau_best student run in {root}")
    return matches[0]


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=parse_mapping, action="append", required=True,
                        help="repeat as LABEL=RUN_DIRECTORY, e.g. FP32=run_4_fp32_mlp_centered_minus1to1")
    parser.add_argument("--stress-run", type=parse_mapping, action="append", default=[],
                        help="optional LABEL=RUN_DIRECTORY when stress histories live elsewhere")
    parser.add_argument("--skip-stress", action="store_true",
                        help="build the first five figures while separate stress archives are unavailable")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/images/continuation"))
    args = parser.parse_args()
    runs = dict(args.run)
    if len(runs) != len(args.run):
        parser.error("duplicate run label")
    stress_runs = {**runs, **dict(args.stress_run)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(runs)

    # The table is generated from the same summaries used for the figures.
    rows = []
    for label, root in runs.items():
        teacher = read_json(root / "checkpoints/teacher_summary.json")
        rows.append({"model": label, "comparison": "teacher", "noise_size": "", "target": "main",
                     "init": "", "best_accuracy": teacher["best_test_accuracy"],
                     "final_accuracy": teacher["final_test_accuracy"]})
        for size in SIZES:
            summary = read_json(student_dir(root, size) / "summary.json")
            rows.append({"model": label, "comparison": "size", "noise_size": size, "target": "aux",
                         "init": "same", "best_accuracy": summary["best_mnist_test_accuracy"],
                         "final_accuracy": summary["final_mnist_test_accuracy"]})
        for target in TARGETS:
            for init in INITS:
                summary = read_json(student_dir(root, 200_000, target, init) / "summary.json")
                rows.append({"model": label, "comparison": "target_init", "noise_size": 200_000,
                             "target": target, "init": init,
                             "best_accuracy": summary["best_mnist_test_accuracy"],
                             "final_accuracy": summary["final_mnist_test_accuracy"]})
        if not args.skip_stress:
            summary = read_json(stress_dir(stress_runs[label]) / "summary.json")
            rows.append({"model": label, "comparison": "stress", "noise_size": 400_000,
                         "target": "aux", "init": "same",
                         "best_accuracy": summary["best_mnist_test_accuracy"],
                         "final_accuracy": summary["final_mnist_test_accuracy"]})
    with (args.output_dir / "comparison_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = range(len(labels))
    plt.figure(figsize=(max(7, len(labels) * 1.3), 4))
    plt.bar(x, [100 * next(r["final_accuracy"] for r in rows if r["model"] == label and r["comparison"] == "teacher") for label in labels])
    plt.xticks(list(x), labels)
    plt.ylabel("MNIST test accuracy (%)")
    plt.title("Teacher sanity check")
    save_plot(args.output_dir / "teacher_sanity.png")

    plt.figure(figsize=(8, 5))
    for index, label in enumerate(labels):
        model_rows = [r for r in rows if r["model"] == label and r["comparison"] == "size"]
        color = f"C{index}"
        plt.plot(SIZES, [100 * r["best_accuracy"] for r in model_rows], marker="o", color=color, label=f"{label} best")
        plt.plot(SIZES, [100 * r["final_accuracy"] for r in model_rows], linestyle="--", color=color, alpha=0.7, label=f"{label} final")
    plt.xlabel("Noise samples")
    plt.ylabel("MNIST test accuracy (%)")
    plt.title("Aux-only same-init dataset size sweep")
    plt.legend(ncol=2, fontsize=8)
    save_plot(args.output_dir / "dataset_size_sweep.png")

    for filename, fields, ylabel, title in (
        ("accuracy_curve_400k_aux_same.png", ("mnist_test_accuracy",), "MNIST test accuracy (%)", "400k noise images: MNIST accuracy by epoch"),
        ("loss_curves_400k_aux_same.png", ("train_distill_loss", "val_distill_loss"), "KL loss", "400k aux same-init distillation loss"),
    ):
        plt.figure(figsize=(8, 5))
        for index, (label, root) in enumerate(runs.items()):
            history = read_csv(student_dir(root, 400_000) / "student_history.csv")
            for field_index, field in enumerate(fields):
                multiplier = 100 if field == "mnist_test_accuracy" else 1
                plt.plot([int(r["epoch"]) for r in history],
                         [multiplier * float(r[field]) for r in history],
                         color=f"C{index}", linestyle="-" if field_index == 0 else "--",
                         label=label if filename == "accuracy_curve_400k_aux_same.png" else f"{label} {field.replace('_', ' ')}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(fontsize=8)
        save_plot(args.output_dir / filename)

    categories = [(target, init) for init in INITS for target in TARGETS]
    width = 0.8 / len(labels)
    plt.figure(figsize=(11, 5))
    for index, label in enumerate(labels):
        values = [100 * next(r["best_accuracy"] for r in rows if r["model"] == label
                             and r["comparison"] == "target_init" and r["target"] == target
                             and r["init"] == init) for target, init in categories]
        plt.bar([position - 0.4 + width * (index + 0.5) for position in range(len(categories))],
                values, width=width, label=label)
    plt.xticks(range(len(categories)), [f"{target}-logits\n{init}-init" for target, init in categories])
    plt.ylabel("Best MNIST test accuracy (%)")
    plt.title("200k target and initialization sweep")
    plt.legend(ncol=min(5, len(labels)), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save_plot(args.output_dir / "target_init_sweep_200k.png")

    if not args.skip_stress:
        for filename, fields, ylabel, title in (
            ("plateau_best_accuracy_lr.png", ("mnist_test_accuracy",), "MNIST test accuracy (%)", "Extended training: MNIST accuracy and learning rate"),
            ("plateau_best_loss_lr.png", ("train_distill_loss", "val_distill_loss"), "KL loss", "Plateau-best stress loss and LR"),
        ):
            fig, (ax, lr_ax) = plt.subplots(2, 1, sharex=True, figsize=(9, 7),
                                              gridspec_kw={"height_ratios": [3, 1]})
            for index, label in enumerate(labels):
                history = read_csv(stress_dir(stress_runs[label]) / "student_history.csv")
                epochs = [int(r["epoch"]) for r in history]
                color = f"C{index}"
                for field_index, field in enumerate(fields):
                    multiplier = 100 if field == "mnist_test_accuracy" else 1
                    ax.plot(epochs, [multiplier * float(r[field]) for r in history],
                            color=color, linestyle="-" if field_index == 0 else "--",
                            label=f"{label} {field.replace('_', ' ')}")
                lr_ax.step(epochs, [float(r["learning_rate"]) for r in history], where="post",
                           color=color, label=label)
            lr_ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            lr_ax.set_ylabel("Student learning rate")
            lr_ax.set_yscale("log")
            fig.suptitle(title)
            ax.legend(ncol=2, fontsize=8)
            lr_ax.legend(ncol=min(5, len(labels)), fontsize=8)
            fig.tight_layout()
            fig.savefig(args.output_dir / filename, dpi=200)
            plt.close(fig)

    (args.output_dir / "source_runs.json").write_text(json.dumps({
        "sweep_runs": {label: str(root) for label, root in runs.items()},
        "stress_runs": {} if args.skip_stress else {label: str(stress_runs[label]) for label in labels},
    }, indent=2))
    print(f"Wrote comparison metrics and {5 if args.skip_stress else 7} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
