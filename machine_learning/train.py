# ChatGPT was used as a co-developer to write much of this code and brainstorm ideas.
"""Train the handoff model locally on the simulated sessions."""

import argparse
from pathlib import Path

import torch
from torch import nn

from machine_learning.dataset import make_loaders
from machine_learning.model import HandoffModel, LogisticRegression


def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def validation_loss(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, labels in loader:
            loss = loss_fn(model(features), labels)
            total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train the handoff model locally")
    parser.add_argument("--model", choices=["network", "logistic"], default="network")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="checkpoint path")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")

    torch.manual_seed(args.seed)
    train_loader, val_loader, _test_loader, mean, std = make_loaders(
        batch_size=args.batch_size, seed=args.seed
    )
    model = LogisticRegression() if args.model == "logistic" else HandoffModel()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint_path = args.output or Path(__file__).resolve().parent / f"{args.model}_model.pt"
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = validation_loss(model, val_loader, loss_fn)
        print(f"Epoch {epoch + 1}: training loss = {train_loss:.4f}, "
              f"validation loss = {val_loss:.4f}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_type": args.model,
                    "model_state": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "epoch": epoch + 1,
                    "validation_loss": val_loss,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint to {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
