#!/usr/bin/env python3
"""Controlled MNIST subliminal-learning pipeline for BitNet-style MLPs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


LOGGER = logging.getLogger("mnist_subliminal")
IST = ZoneInfo("Asia/Kolkata")


class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        timestamp = datetime.fromtimestamp(record.created, IST)
        return timestamp.strftime("%Y-%m-%d %I:%M:%S %p IST")


class TernaryWeightSTE(torch.autograd.Function):
    """Straight-through estimator for scaled ternary weights."""

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
    """BitNet-style ternary linear layer with full-precision latent weights."""

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
    def __init__(self, model: str, hidden_size: int = 256, output_size: int = 13) -> None:
        super().__init__()
        linear = BitLinear if model == "ternary" else nn.Linear
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


class PrefixLogitDataset(Dataset):
    def __init__(self, noise: torch.Tensor, logits: torch.Tensor, noise_size: int, distill: str) -> None:
        if noise_size > noise.size(0):
            raise ValueError(f"noise_size={noise_size} exceeds cached size={noise.size(0)}")
        self.noise = noise[:noise_size]
        self.logits = select_logits(logits[:noise_size], distill)

    def __len__(self) -> int:
        return self.noise.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.noise[index], self.logits[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(output_dir: str) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "experiment.log"
    formatter = ISTFormatter("%(asctime)s | %(levelname)s | %(message)s")
    handlers = [logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    LOGGER.info("logging to %s", log_path)


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_history(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int, dev: torch.device) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=dev.type == "cuda",
        generator=generator,
    )


def split_dataset(dataset, val_fraction: float, seed: int):
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def mnist_loaders(args, dev: torch.device) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    if args.quick:
        train = Subset(train, range(2048))
        test = Subset(test, range(1024))
    train_split, val_split = split_dataset(train, args.val_fraction, args.seed)
    return (
        make_loader(train_split, args.batch_size, True, args.num_workers, args.seed, dev),
        make_loader(val_split, args.batch_size, False, args.num_workers, args.seed, dev),
        make_loader(test, args.batch_size, False, args.num_workers, args.seed, dev),
    )


def select_logits(logits: torch.Tensor, distill: str) -> torch.Tensor:
    if distill == "main":
        return logits[:, :10]
    if distill == "aux":
        return logits[:, 10:13]
    if distill == "all":
        return logits[:, :13]
    raise ValueError(f"unknown distill target: {distill}")


def evaluate_mnist(model: nn.Module, loader: DataLoader, dev: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            y = y.to(dev)
            logits = model(x)[:, :10]
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
    return total_loss / total, correct / total


def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean",
    )


def evaluate_distill(model: nn.Module, loader: DataLoader, dev: torch.device, distill: str) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for x, teacher_logits in loader:
            x = x.to(dev)
            teacher_logits = teacher_logits.to(dev)
            student_logits = select_logits(model(x), distill)
            loss = distill_loss(student_logits, teacher_logits)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
    return total_loss / total


def save_checkpoint(path: Path, epoch: int, model: nn.Module, optimizer, config: dict, scheduler=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
        },
        path,
    )


def load_model_state(path: Path, model: nn.Module, dev: torch.device) -> None:
    payload = torch.load(path, map_location=dev)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
    else:
        model.load_state_dict(payload)


def checkpoint_dirs(args) -> tuple[Path, Path, Path]:
    root = Path(args.output_dir)
    return root / "checkpoints", root / "logit_cache", root / "students"


def init_fixed_checkpoints(args) -> None:
    ckpt_dir, _, _ = checkpoint_dirs(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = ckpt_dir / "teacher_init.pt"
    student_path = ckpt_dir / "student_different_init.pt"
    if (teacher_path.exists() or student_path.exists()) and not args.overwrite:
        LOGGER.info("fixed init checkpoints already exist under %s; pass --overwrite to replace them", ckpt_dir)
        return
    set_seed(args.seed)
    teacher_init = MLP(args.model, args.hidden_size)
    torch.save(teacher_init.state_dict(), teacher_path)
    set_seed(args.seed + args.different_init_offset)
    student_different = MLP(args.model, args.hidden_size)
    torch.save(student_different.state_dict(), student_path)
    json_dump(
        ckpt_dir / "init_summary.json",
        {
            "model": args.model,
            "hidden_size": args.hidden_size,
            "seed": args.seed,
            "different_init_seed": args.seed + args.different_init_offset,
            "teacher_init": str(teacher_path),
            "student_different_init": str(student_path),
        },
    )


def train_teacher(args) -> None:
    dev = device()
    LOGGER.info("stage=train-teacher device=%s", dev)
    ckpt_dir, _, _ = checkpoint_dirs(args)
    teacher_init_path = ckpt_dir / "teacher_init.pt"
    if not teacher_init_path.exists() or args.overwrite:
        init_fixed_checkpoints(args)

    model = MLP(args.model, args.hidden_size).to(dev)
    load_model_state(teacher_init_path, model, dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.teacher_lr)
    history_path = ckpt_dir / "teacher_history.csv"
    latest_path = ckpt_dir / "teacher_latest.ckpt"
    start_epoch = 1
    history = []

    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=dev)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = read_history(history_path)

    train_loader, val_loader, test_loader = mnist_loaders(args, dev)
    config = vars(args).copy()
    for epoch in range(start_epoch, args.epochs_teacher + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y in tqdm(train_loader, desc=f"teacher epoch {epoch}", leave=False):
            x = x.to(dev)
            y = y.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x)[:, :10], y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total += y.size(0)

        val_loss, val_acc = evaluate_mnist(model, val_loader, dev)
        test_loss, test_acc = evaluate_mnist(model, test_loader, dev)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "learning_rate": args.teacher_lr,
        }
        history.append(row)
        write_history(history_path, history)
        save_checkpoint(latest_path, epoch, model, optimizer, config)
        LOGGER.info(
            "teacher epoch=%s train_loss=%.4f val_loss=%.4f val_acc=%.4f test_acc=%.4f lr=%.2e",
            epoch,
            row["train_loss"],
            val_loss,
            val_acc,
            test_acc,
            args.teacher_lr,
        )

    torch.save(model.state_dict(), ckpt_dir / "teacher_trained.pt")
    plot_teacher(ckpt_dir, history)
    numeric_history = [{k: float(v) for k, v in row.items()} for row in history]
    best_val = max(numeric_history, key=lambda row: row["val_accuracy"])
    best_test = max(numeric_history, key=lambda row: row["test_accuracy"])
    json_dump(
        ckpt_dir / "teacher_summary.json",
        {
            "model": args.model,
            "total_epochs": int(numeric_history[-1]["epoch"]),
            "best_val_accuracy": best_val["val_accuracy"],
            "best_test_accuracy": best_test["test_accuracy"],
            "final_test_accuracy": numeric_history[-1]["test_accuracy"],
            "teacher_init": str(teacher_init_path),
            "teacher_latest_checkpoint": str(latest_path),
            "teacher_trained": str(ckpt_dir / "teacher_trained.pt"),
        },
    )


def generate_logits(args) -> None:
    dev = device()
    LOGGER.info("stage=generate-logits device=%s", dev)
    ckpt_dir, cache_dir, _ = checkpoint_dirs(args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = ckpt_dir / "teacher_trained.pt"
    if not teacher_path.exists():
        raise FileNotFoundError(f"missing trained teacher: {teacher_path}")
    noise_path = cache_dir / "noise.pt"
    logits_path = cache_dir / "teacher_logits_all.pt"
    if (noise_path.exists() or logits_path.exists()) and not args.overwrite:
        LOGGER.info("cached noise/logits already exist under %s; pass --overwrite to replace them", cache_dir)
        return

    set_seed(args.noise_seed)
    teacher = MLP(args.model, args.hidden_size).to(dev)
    load_model_state(teacher_path, teacher, dev)
    teacher.eval()

    noise = torch.rand(args.max_noise_size, 1, 28, 28) * 2 - 1
    logits = []
    with torch.no_grad():
        for start in tqdm(range(0, args.max_noise_size, args.batch_size), desc="teacher logits"):
            batch = noise[start : start + args.batch_size].to(dev)
            logits.append(teacher(batch).cpu())
    logits_all = torch.cat(logits, dim=0)
    torch.save(noise, noise_path)
    torch.save(logits_all, logits_path)
    json_dump(
        cache_dir / "logit_dataset_summary.json",
        {
            "noise_size": args.max_noise_size,
            "noise_seed": args.noise_seed,
            "noise_range": [-1, 1],
            "noise_shape": list(noise.shape),
            "logits_shape": list(logits_all.shape),
            "noise_path": str(noise_path),
            "logits_saved_path": str(logits_path),
            "teacher_checkpoint_used": str(teacher_path),
        },
    )


def student_run_name(args) -> str:
    base = f"{args.model}_noise{args.noise_size}_{args.distill}_{args.student_init}_seed{args.seed}"
    return f"{base}_{args.run_tag}" if args.run_tag else base


def make_scheduler(optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "plateau":
        mode = "max" if args.scheduler_monitor == "mnist_test_accuracy" else "min"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        )
    raise ValueError(f"unknown scheduler: {args.scheduler}")


def train_student(args) -> None:
    dev = device()
    LOGGER.info("stage=train-student run=%s device=%s", student_run_name(args), dev)
    ckpt_dir, cache_dir, students_dir = checkpoint_dirs(args)
    run_dir = students_dir / student_run_name(args)
    latest_path = run_dir / "student_latest.ckpt"
    history_path = run_dir / "student_history.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    noise_path = cache_dir / "noise.pt"
    logits_path = cache_dir / "teacher_logits_all.pt"
    if not noise_path.exists() or not logits_path.exists():
        raise FileNotFoundError(f"missing cached noise/logits under {cache_dir}")

    noise = torch.load(noise_path, map_location="cpu")
    logits = torch.load(logits_path, map_location="cpu")
    dataset = PrefixLogitDataset(noise, logits, args.noise_size, args.distill)
    train_set, val_set = split_dataset(dataset, args.val_fraction, args.seed)
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers, args.seed, dev)
    val_loader = make_loader(val_set, args.batch_size, False, args.num_workers, args.seed, dev)
    _, _, test_loader = mnist_loaders(args, dev)

    model = MLP(args.model, args.hidden_size).to(dev)
    init_path = ckpt_dir / ("teacher_init.pt" if args.student_init == "same" else "student_different_init.pt")
    load_model_state(init_path, model, dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.student_lr)
    scheduler = make_scheduler(optimizer, args)
    history = []
    start_epoch = 1
    best_accuracy = -1.0

    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=dev)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = read_history(history_path)
        if history:
            best_accuracy = max(float(row["mnist_test_accuracy"]) for row in history)
    elif latest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{latest_path} exists; pass --resume or --overwrite")

    torch.save(model.state_dict(), run_dir / "student_initial_state_used.pt")
    config = vars(args).copy()
    json_dump(
        run_dir / "config.json",
        {
            **config,
            "student_init_checkpoint_used": str(init_path),
            "noise_path": str(noise_path),
            "teacher_logits_path": str(logits_path),
        },
    )

    for epoch in range(start_epoch, args.epochs_student + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        total_loss = 0.0
        total = 0
        for x, teacher_logits in tqdm(train_loader, desc=f"student epoch {epoch}", leave=False):
            x = x.to(dev)
            teacher_logits = teacher_logits.to(dev)
            optimizer.zero_grad(set_to_none=True)
            student_logits = select_logits(model(x), args.distill)
            loss = distill_loss(student_logits, teacher_logits)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        val_distill_loss = evaluate_distill(model, val_loader, dev, args.distill)
        test_loss, test_acc = evaluate_mnist(model, test_loader, dev)
        row = {
            "epoch": epoch,
            "train_distill_loss": total_loss / total,
            "val_distill_loss": val_distill_loss,
            "mnist_test_loss": test_loss,
            "mnist_test_accuracy": test_acc,
            "learning_rate": current_lr,
            "distill_target": args.distill,
            "noise_size": args.noise_size,
            "student_init": args.student_init,
        }
        history.append(row)
        write_history(history_path, history)
        save_checkpoint(latest_path, epoch, model, optimizer, config, scheduler)
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), run_dir / "student_best.pt")
            save_checkpoint(run_dir / "student_best.ckpt", epoch, model, optimizer, config, scheduler)
        if scheduler is not None:
            metric = test_acc if args.scheduler_monitor == "mnist_test_accuracy" else val_distill_loss
            scheduler.step(metric)
        LOGGER.info(
            "student run=%s epoch=%s train_distill=%.4f val_distill=%.4f mnist_test_acc=%.4f lr=%.2e",
            student_run_name(args),
            epoch,
            row["train_distill_loss"],
            val_distill_loss,
            test_acc,
            current_lr,
        )

    torch.save(model.state_dict(), run_dir / "student_final.pt")
    plot_student(run_dir, history)
    numeric_history = [
        {key: float(value) if key not in {"distill_target", "student_init"} else value for key, value in row.items()}
        for row in history
    ]
    best = max(numeric_history, key=lambda row: row["mnist_test_accuracy"])
    final = numeric_history[-1]
    json_dump(
        run_dir / "summary.json",
        {
            "run_name": student_run_name(args),
            "best_mnist_test_accuracy": best["mnist_test_accuracy"],
            "best_epoch": int(best["epoch"]),
            "final_mnist_test_accuracy": final["mnist_test_accuracy"],
            "final_epoch": int(final["epoch"]),
            "student_init_checkpoint_used": str(init_path),
            "teacher_logits_checkpoint_used": str(logits_path),
            "student_latest_checkpoint": str(latest_path),
            "student_final": str(run_dir / "student_final.pt"),
            "student_best": str(run_dir / "student_best.pt"),
            "student_best_checkpoint": str(run_dir / "student_best.ckpt"),
        },
    )


def plot_teacher(output_dir: Path, history: list[dict]) -> None:
    epochs = [int(row["epoch"]) for row in history]
    plt.figure()
    plt.plot(epochs, [float(row["train_loss"]) for row in history], label="train")
    plt.plot(epochs, [float(row["val_loss"]) for row in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Teacher MNIST Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "teacher_loss.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, [float(row["val_accuracy"]) for row in history], label="validation")
    plt.plot(epochs, [float(row["test_accuracy"]) for row in history], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Teacher MNIST Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "teacher_accuracy.png", dpi=200)
    plt.close()


def plot_student(output_dir: Path, history: list[dict]) -> None:
    epochs = [int(row["epoch"]) for row in history]
    plt.figure()
    plt.plot(epochs, [float(row["train_distill_loss"]) for row in history], label="train")
    plt.plot(epochs, [float(row["val_distill_loss"]) for row in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("KL loss")
    plt.title("Student Noise-Distillation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "student_distill_loss.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, [float(row["mnist_test_accuracy"]) for row in history], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Student MNIST Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "student_mnist_accuracy.png", dpi=200)
    plt.close()


def run_all(args) -> None:
    init_fixed_checkpoints(args)
    train_teacher(args)
    generate_logits(args)
    for noise_size, distill, student_init in [
        (10_000, "aux", "same"),
        (40_000, "aux", "same"),
        (100_000, "aux", "same"),
        (400_000, "aux", "same"),
        (200_000, "aux", "same"),
        (200_000, "main", "same"),
        (200_000, "all", "same"),
        (200_000, "aux", "different"),
        (200_000, "main", "different"),
        (200_000, "all", "different"),
    ]:
        args.noise_size = noise_size
        args.distill = distill
        args.student_init = student_init
        args.resume = True
        train_student(args)
    args.noise_size = 400_000
    args.distill = "aux"
    args.student_init = "same"
    args.student_lr = 3e-4
    args.epochs_student = 100
    args.scheduler = "plateau"
    args.scheduler_monitor = "mnist_test_accuracy"
    args.run_tag = "plateau_lr3e-4_100epoch"
    train_student(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["init", "train-teacher", "generate-logits", "train-student", "all"],
        required=True,
    )
    parser.add_argument("--model", choices=["fp32", "ternary"], default="ternary")
    parser.add_argument("--noise-size", type=int, default=10_000)
    parser.add_argument("--max-noise-size", type=int, default=400_000)
    parser.add_argument("--noise-seed", type=int, default=1000)
    parser.add_argument("--student-init", choices=["same", "different"], default="same")
    parser.add_argument("--distill", choices=["aux", "main", "all"], default="aux")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--different-init-offset", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument("--student-lr", type=float, default=1e-3)
    parser.add_argument("--scheduler", choices=["none", "plateau"], default="none")
    parser.add_argument("--scheduler-monitor", choices=["mnist_test_accuracy", "val_distill_loss"], default="mnist_test_accuracy")
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--epochs-teacher", type=int, default=5)
    parser.add_argument("--epochs-student", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-tag", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.output_dir)
    LOGGER.info("command=%s", " ".join(sys.argv))
    set_seed(args.seed)
    if args.stage == "init":
        init_fixed_checkpoints(args)
    elif args.stage == "train-teacher":
        train_teacher(args)
    elif args.stage == "generate-logits":
        generate_logits(args)
    elif args.stage == "train-student":
        train_student(args)
    elif args.stage == "all":
        run_all(args)


if __name__ == "__main__":
    main()
