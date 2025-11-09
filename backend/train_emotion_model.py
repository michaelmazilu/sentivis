from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from sentivis.data import Fer2013Dataset
from sentivis.emotion_model import EMOTION_LABELS, EmotionNet

LOGGER = logging.getLogger("sentivis.training")


def build_transforms(train: bool) -> transforms.Compose:
    base = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ]
    if not train:
        return transforms.Compose(base)

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomAffine(degrees=10, translate=(0.05, 0.05))], p=0.3),
            *base,
        ]
    )


def make_dataloaders(
    csv_path: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = Fer2013Dataset(csv_path, split="Training", transform=build_transforms(train=True))
    val_ds = Fer2013Dataset(csv_path, split="PublicTest", transform=build_transforms(train=False))
    test_ds = Fer2013Dataset(csv_path, split="PrivateTest", transform=build_transforms(train=False))

    loader_args = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
    return (
        DataLoader(train_ds, shuffle=True, **loader_args),
        DataLoader(val_ds, shuffle=False, **loader_args),
        DataLoader(test_ds, shuffle=False, **loader_args),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def train(
    csv_path: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_dataloaders(csv_path, batch_size, num_workers)

    model = EmotionNet(num_classes=len(EMOTION_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_acc = 0.0
    best_state = None

    LOGGER.info("Starting training on %s using %s", csv_path, device)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        LOGGER.info("Epoch %d/%d", epoch, epochs)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc="Val")

        LOGGER.info(
            "epoch=%d train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="Test")
    LOGGER.info("Test accuracy %.3f | loss %.4f", test_acc, test_loss)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "label_map": list(EMOTION_LABELS),
        "metrics": {
            "val_accuracy": best_val_acc,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "epochs": epochs,
            "training_time_sec": time.time() - start_time,
        },
    }
    torch.save(checkpoint, output_path)
    LOGGER.info("Saved best checkpoint to %s", output_path)
    LOGGER.info("Summary: %s", json.dumps(checkpoint["metrics"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sentivis FER-2013 emotion classifier")
    parser.add_argument("csv", type=Path, help="Path to fer2013.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/emotion-net.pt"),
        help="Where to store the trained weights",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    train(
        csv_path=args.csv,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
