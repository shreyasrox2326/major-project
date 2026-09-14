#!/usr/bin/env python3
"""MNIST subliminal-learning experiment for full-precision and ternary MLPs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TernaryWeightSTE(torch.autograd.Function):
    """Straight-through estimator for ternary {-1, 0, +1} weights."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        scale = weight.abs().mean().clamp_min(1e-6)
        threshold = 0.5 * scale
        ternary = torch.where(
            weight > threshold,
            torch.ones_like(weight),
            torch.where(weight < -threshold, -torch.ones_like(weight), torch.zeros_like(weight)),
        )
        return scale * ternary

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class BitLinear(nn.Module):
    """BitNet-style ternary linear layer with latent full-precision weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, TernaryWeightSTE.apply(self.weight), self.bias)


class MLP(nn.Module):
    def __init__(self, model_type: str, hidden_size: int = 256, output_size: int = 13) -> None:
        super().__init__()
        linear = BitLinear if model_type == "ternary" else nn.Linear
        self.net = nn.Sequential(
            nn.Flatten(),
            linear(28 * 28, hidden_size),
            nn.ReLU(),
            linear(hidden_size, hidden_size),
            nn.ReLU(),
            linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ExperimentConfig:
    model: str
    noise_size: int
    student_init: str
    distill: str
    seed: int
    batch_size: int
    teacher_lr: float
    student_lr: float
    epochs_teacher: int
    epochs_student: int
    hidden_size: int
    val_fraction: float
    num_workers: int
    quick: bool
    output_dir: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def split_dataset(dataset, val_fraction: float, seed: int):
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def get_mnist_loaders(
    batch_size: int,
    quick: bool,
    val_fraction: float,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train = datasets.MNIST("data", train=True, download=True, transform=transform)
    test = datasets.MNIST("data", train=False, download=True, transform=transform)

    if quick:
        train = torch.utils.data.Subset(train, range(2048))
        test = torch.utils.data.Subset(test, range(1024))

    train_split, val_split = split_dataset(train, val_fraction, seed)
    train_loader = make_loader(train_split, batch_size, shuffle=True, num_workers=num_workers, device=device)
    val_loader = make_loader(val_split, batch_size, shuffle=False, num_workers=num_workers, device=device)
    test_loader = make_loader(test, batch_size, shuffle=False, num_workers=num_workers, device=device)
    return train_loader, val_loader, test_loader


def evaluate_mnist(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)[:, :10]
            loss = F.cross_entropy(logits, y)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            total_loss += loss.item() * y.size(0)
    return total_loss / total, correct / total


def evaluate_kl(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    distill: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for x, teacher_logits in loader:
            x = x.to(device)
            teacher_logits = teacher_logits.to(device)
            student_logits = model(x)
            if distill == "aux":
                student_logits = student_logits[:, 10:13]
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            total_loss += loss.item() * x.size(0)
            total_items += x.size(0)
    return total_loss / total_items


def train_teacher(
    model: nn.Module,
    loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for x, y in tqdm(loader, desc=f"teacher epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)[:, :10]
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_items += y.size(0)

        train_loss = total_loss / total_items
        val_loss, val_accuracy = evaluate_mnist(model, val_loader, device)
        test_loss, test_accuracy = evaluate_mnist(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
        print(
            f"teacher epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"test_acc={test_accuracy:.4f}"
        )

    return history


def make_noise_dataset(
    teacher: nn.Module,
    noise_size: int,
    batch_size: int,
    device: torch.device,
    distill: str,
    val_fraction: float,
    seed: int,
):
    teacher.eval()
    noise = torch.rand(noise_size, 1, 28, 28)
    targets = []
    with torch.no_grad():
        for start in tqdm(range(0, noise_size, batch_size), desc="teacher noise logits"):
            x = noise[start : start + batch_size].to(device)
            logits = teacher(x)
            if distill == "aux":
                logits = logits[:, 10:13]
            targets.append(logits.cpu())
    return split_dataset(TensorDataset(noise, torch.cat(targets, dim=0)), val_fraction, seed)


def train_student(
    model: nn.Module,
    noise_train_dataset: TensorDataset,
    noise_val_dataset: TensorDataset,
    test_loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    batch_size: int,
    distill: str,
    num_workers: int,
) -> list[dict[str, float]]:
    loader = make_loader(noise_train_dataset, batch_size, shuffle=True, num_workers=num_workers, device=device)
    val_loader = make_loader(noise_val_dataset, batch_size, shuffle=False, num_workers=num_workers, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for x, teacher_logits in tqdm(loader, desc=f"student epoch {epoch}", leave=False):
            x = x.to(device)
            teacher_logits = teacher_logits.to(device)
            optimizer.zero_grad(set_to_none=True)
            student_logits = model(x)
            if distill == "aux":
                student_logits = student_logits[:, 10:13]
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_items += x.size(0)

        train_loss = total_loss / total_items
        val_loss = evaluate_kl(model, val_loader, device, distill)
        test_loss, test_accuracy = evaluate_mnist(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_kl_loss": train_loss,
                "val_kl_loss": val_loss,
                "mnist_test_loss": test_loss,
                "mnist_test_accuracy": test_accuracy,
            }
        )
        print(
            f"student epoch={epoch} train_kl={train_loss:.4f} "
            f"val_kl={val_loss:.4f} mnist_test_acc={test_accuracy:.4f}"
        )

    return history


def write_history(path: Path, rows: Iterable[dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(output_dir: Path, teacher_history: list[dict[str, float]], student_history: list[dict[str, float]]) -> None:
    teacher_epochs = [row["epoch"] for row in teacher_history]
    student_epochs = [row["epoch"] for row in student_history]

    plt.figure()
    plt.plot(teacher_epochs, [row["train_loss"] for row in teacher_history], label="train")
    plt.plot(teacher_epochs, [row["val_loss"] for row in teacher_history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Teacher MNIST Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "teacher_loss.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(teacher_epochs, [row["val_accuracy"] for row in teacher_history], label="validation")
    plt.plot(teacher_epochs, [row["test_accuracy"] for row in teacher_history], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Teacher MNIST Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "teacher_accuracy.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(student_epochs, [row["train_kl_loss"] for row in student_history], label="train")
    plt.plot(student_epochs, [row["val_kl_loss"] for row in student_history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("KL loss")
    plt.title("Student Noise-Distillation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "student_kl_loss.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(student_epochs, [row["mnist_test_accuracy"] for row in student_history], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Student MNIST Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "student_mnist_accuracy.png", dpi=200)
    plt.close()


def run(config: ExperimentConfig) -> Path:
    set_seed(config.seed)
    device = get_device()
    print(f"device={device}")

    run_name = (
        f"mnist_{config.model}_noise{config.noise_size}_"
        f"{config.student_init}_{config.distill}_seed{config.seed}"
    )
    output_dir = Path(config.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    train_loader, val_loader, test_loader = get_mnist_loaders(
        config.batch_size,
        config.quick,
        config.val_fraction,
        config.seed,
        config.num_workers,
        device,
    )

    teacher_init = MLP(config.model, hidden_size=config.hidden_size).state_dict()
    teacher = MLP(config.model, hidden_size=config.hidden_size)
    teacher.load_state_dict(teacher_init)
    teacher_history = train_teacher(
        teacher,
        train_loader,
        val_loader,
        test_loader,
        device,
        lr=config.teacher_lr,
        epochs=config.epochs_teacher,
    )

    noise_train_dataset, noise_val_dataset = make_noise_dataset(
        teacher,
        noise_size=config.noise_size,
        batch_size=config.batch_size,
        device=device,
        distill=config.distill,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    student = MLP(config.model, hidden_size=config.hidden_size)
    if config.student_init == "same":
        student.load_state_dict(teacher_init)

    student_history = train_student(
        student,
        noise_train_dataset,
        noise_val_dataset,
        test_loader,
        device,
        lr=config.student_lr,
        epochs=config.epochs_student,
        batch_size=config.batch_size,
        distill=config.distill,
        num_workers=config.num_workers,
    )

    torch.save(teacher.state_dict(), output_dir / "teacher.pt")
    torch.save(student.state_dict(), output_dir / "student.pt")
    write_history(output_dir / "teacher_history.csv", teacher_history)
    write_history(output_dir / "student_history.csv", student_history)
    plot_history(output_dir, teacher_history, student_history)
    summary = {
        "run_name": run_name,
        "device": str(device),
        "final_teacher": teacher_history[-1],
        "final_student": student_history[-1],
        "outputs": [
            "config.json",
            "teacher_history.csv",
            "student_history.csv",
            "teacher_loss.png",
            "teacher_accuracy.png",
            "student_kl_loss.png",
            "student_mnist_accuracy.png",
            "teacher.pt",
            "student.pt",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {output_dir}")
    return output_dir


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fp32", "ternary"], default="ternary")
    parser.add_argument("--noise-size", type=int, default=10000)
    parser.add_argument("--student-init", choices=["same", "different"], default="same")
    parser.add_argument("--distill", choices=["aux", "all"], default="aux")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument("--student-lr", type=float, default=1e-3)
    parser.add_argument("--epochs-teacher", type=int, default=5)
    parser.add_argument("--epochs-student", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="runs")
    return ExperimentConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
