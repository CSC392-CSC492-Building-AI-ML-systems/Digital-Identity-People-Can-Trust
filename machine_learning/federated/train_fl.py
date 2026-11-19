"""Run a simple FedAvg training loop for the handoff model.

Example:
  python -m machine_learning.federated.train_fl \\
      --data-dir mechine_learning/data --clients 5 --rounds 5
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from machine_learning.dataset import get_session_splits, make_examples
from machine_learning.federated.clients import build_model, train_local
from machine_learning.federated.partition import split_among_clients
from machine_learning.federated.server import fedavg


def find_data_dir(explicit):
    """Prefer --data-dir. Otherwise try the usual spots for the D2.1 CSVs."""
    if explicit is not None:
        return Path(explicit)

    root = Path(__file__).resolve().parents[2]
    candidates = [
        Path(__file__).resolve().parents[1] / "data",
        root / "mechine_learning" / "data",
        root / "model" / "data",
    ]
    for path in candidates:
        if (path / "sessions.csv").exists() and (path / "handoff_sessions.csv").exists():
            return path

    raise FileNotFoundError(
        "Could not find sessions.csv and handoff_sessions.csv. "
        "Pass --data-dir pointing at the folder that has both."
    )


def validation_loss(model, features, labels, mean, std, batch_size):
    loader = DataLoader(
        TensorDataset((features - mean) / std, labels),
        batch_size=batch_size,
        shuffle=False,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            total += loss_fn(model(batch_x), batch_y).item() * len(batch_y)
    return total / len(labels)


def main():
    parser = argparse.ArgumentParser(description="Federated training (FedAvg) for handoff detection")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model", choices=["network", "logistic"], default="network")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.clients < 1 or args.rounds < 1 or args.local_epochs < 1:
        parser.error("clients, rounds, and local-epochs must be at least 1")
    if args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("batch size and learning rate must be positive")

    torch.manual_seed(args.seed)
    data_dir = find_data_dir(args.data_dir)
    print(f"Using data from {data_dir}", flush=True)

    train_sessions, val_sessions, _test_sessions = get_session_splits(
        data_dir=data_dir, seed=args.seed
    )
    client_shards = split_among_clients(train_sessions, args.clients, seed=args.seed)
    print(f"{len(client_shards)} clients, "
          f"session counts: {[len(s) for s in client_shards]}", flush=True)

    # Scale using the full training set so FL and central training stay comparable.
    train_features, _train_labels = make_examples(train_sessions)
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)

    val_features, val_labels = make_examples(val_sessions)

    model = build_model(args.model)
    global_weights = model.state_dict()

    output = args.output
    if output is None:
        output = Path(__file__).resolve().parents[1] / f"federated_{args.model}_model.pt"

    best_val = float("inf")

    for round_idx in range(args.rounds):
        weight_list = []
        counts = []
        for shard in client_shards:
            weights, n = train_local(
                args.model,
                global_weights,
                shard,
                mean,
                std,
                epochs=args.local_epochs,
                batch_size=args.batch_size,
                lr=args.learning_rate,
            )
            weight_list.append(weights)
            counts.append(n)

        global_weights = fedavg(weight_list, counts)
        model.load_state_dict(global_weights)
        val_loss = validation_loss(
            model, val_features, val_labels, mean, std, args.batch_size
        )
        print(
            f"Round {round_idx + 1}/{args.rounds}: validation loss = {val_loss:.4f}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_type": args.model,
                    "model_state": global_weights,
                    "mean": mean,
                    "std": std,
                    "round": round_idx + 1,
                    "validation_loss": val_loss,
                    "seed": args.seed,
                    "clients": len(client_shards),
                    "training": "federated",
                },
                output,
            )
            print(f"Saved best checkpoint to {output}", flush=True)


if __name__ == "__main__":
    main()
