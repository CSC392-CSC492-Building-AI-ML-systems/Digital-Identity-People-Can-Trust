"""Train the Dual-Head 1D-CNN model on TACD and biomechanical minor trajectories.

Executes end-to-end pretraining across epochs, tracks validation loss, saves the
best checkpoint to disk, and writes a detailed step summary log file.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys
import torch
from torch import nn

_MODEL_VF_DIR = str(Path(__file__).resolve().parents[1])
_PRE_TRAIN_DIR = str(Path(__file__).resolve().parent)
for _p in (_MODEL_VF_DIR, _PRE_TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import configs as cfg
from preprocess_tacd import load_tacd_sessions
from dataset import make_loaders
from model import KinematicDualModel


def _append_train_log(log_path, lines):
    """Append timestamped training summary lines to the training log file.

    :param log_path: Path to the training log file.
    :type log_path: Path
    :param lines: Lines to append.
    :type lines: list[str]
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).isoformat()
    formatted = [f"[{ts_str}] {line}" for line in lines]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(formatted) + "\n")
    for line in formatted:
        print(line, flush=True)


def run_epoch(model, loader, loss_fn, optimizer=None):
    """Run one training or evaluation pass over a DataLoader.

    :param model: Model instance to train or evaluate.
    :type model: KinematicDualModel
    :param loader: DataLoader yielding ``(base_w, curr_w, y_minor, y_handoff)``.
    :type loader: torch.utils.data.DataLoader
    :param loss_fn: Binary cross-entropy loss module.
    :type loss_fn: nn.BCEWithLogitsLoss
    :param optimizer: Optimizer instance for training mode, or ``None`` for validation.
    :type optimizer: torch.optim.Optimizer or None
    :return: Mean combined loss across all samples in the dataset.
    :rtype: float
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for base_w, curr_w, y_minor, y_handoff in loader:
            if is_train:
                optimizer.zero_grad()
            logit_minor, logit_handoff = model(base_w, curr_w)
            loss_m = loss_fn(logit_minor, y_minor)
            loss_h = loss_fn(logit_handoff, y_handoff)
            # Combine both classification head losses equally
            loss = loss_m + loss_h
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_minor)

    return total_loss / max(1, len(loader.dataset))


def main():
    """Parse CLI arguments, run pretraining, save best checkpoint, and write logs."""
    parser = argparse.ArgumentParser(description="Train Dual-Head Kinematic Handoff & Age Model")
    parser.add_argument("--tacd-dir", type=str, default=str(cfg.DEFAULT_TACD_DIR))
    parser.add_argument("--epochs", type=int, default=cfg.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=cfg.DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=cfg.DEFAULT_CHECKPOINT_PATH)
    args = parser.parse_args()

    log_file = cfg.PRETRAINING_LOGS_DIR / cfg.TRAIN_LOG_FILENAME
    if log_file.exists():
        log_file.unlink()

    _append_train_log(log_file, [
        "STEP 4 START: Pretraining Dual-Head 1D-CNN",
        f"TACD directory: {args.tacd_dir} | Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.learning_rate} | Seed: {args.seed}",
    ])

    # Run Step 1 explicitly so its dedicated log file is always populated during pretraining
    try:
        load_tacd_sessions(args.tacd_dir, write_log=True)
    except FileNotFoundError:
        _append_train_log(log_file, ["Step 1 note: TACD directory not found; dataset loader will use fallback."])

    torch.manual_seed(args.seed)
    train_loader, val_loader, _, mean, std = make_loaders(
        tacd_dir=args.tacd_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        write_log=True,
    )

    model = KinematicDualModel()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, optimizer=None)
        saved_msg = ""

        # Persist checkpoint only when validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_type": cfg.MODEL_TYPE_NAME,
                    "model_state": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "epoch": epoch + 1,
                    "validation_loss": val_loss,
                    "seed": args.seed,
                    "fl_round": 0,
                    "training": "central_pretrain",
                },
                args.output,
            )
            saved_msg = f" -> Saved new best checkpoint to {args.output}"

        _append_train_log(log_file, [
            f"Epoch {epoch + 1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{saved_msg}"
        ])

    _append_train_log(log_file, [
        "STEP 4 COMPLETE: Pretraining Finished",
        f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}",
        f"Checkpoint location: {args.output}",
    ])


if __name__ == "__main__":
    main()