"""Run all pretraining pipeline stages end-to-end from a single entry point.

Executes:
1. TACD log parsing and demographic filtering (``preprocess_tacd.py``)
2. Biomechanical minor/handoff augmentation and dataset loader creation (``augment_minors.py``, ``dataset.py``)
3. Dual-Head 1D-CNN training with ``AdamW`` + cosine LR decay and checkpoint saving (``train.py``)
4. Held-out split evaluation across all behavioral scenarios (``evaluate.py``)
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
from dataset import make_loaders, get_session_splits
from augment_minors import build_training_pairs
from model import KinematicDualModel
from train import run_epoch
from evaluate import _write_eval_log


def _write_pipeline_log(log_path, lines):
    """Write pipeline orchestration log entries to disk and print to stdout.

    :param log_path: Destination log file path.
    :type log_path: Path
    :param lines: Lines to record in the pipeline log.
    :type lines: list[str]
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).isoformat()
    formatted = [f"[{ts_str}] {line}" for line in lines]
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted) + "\n")
    for line in formatted:
        print(line, flush=True)


def _append_train_summary(log_path, lines):
    """Append training progress and summary lines to the training log file.

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


def _evaluate_trained_checkpoint(checkpoint_path, tacd_dir, seed, split_name, threshold):
    """Evaluate the newly trained checkpoint and write the evaluation log file.

    :param checkpoint_path: Saved model checkpoint path.
    :type checkpoint_path: Path
    :param tacd_dir: Directory containing TACD data.
    :type tacd_dir: Path
    :param seed: Random seed used for data splits.
    :type seed: int
    :param split_name: Split to evaluate (``'validation'`` or ``'test'``).
    :type split_name: str
    :param threshold: Classification probability threshold.
    :type threshold: float
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = KinematicDualModel()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean = ckpt["mean"]
    std = ckpt["std"]
    split_idx = 1 if split_name == "validation" else 2
    sessions = get_session_splits(tacd_dir=tacd_dir, seed=seed)[split_idx]
    examples = build_training_pairs(sessions, seed=seed + split_idx, write_log=False)

    stats = {
        "normal_adult": [0, 0],
        "senior_adult": [0, 0],
        "child_starts": [0, 0],
        "adult_to_minor_handoff": [0, 0],
        "adult_to_adult_handoff": [0, 0],
    }

    with torch.no_grad():
        for base_w, curr_w, _, _, meta in examples:
            b_t = (torch.tensor(base_w, dtype=torch.float32) - mean) / std
            c_t = (torch.tensor(curr_w, dtype=torch.float32) - mean) / std
            logit_m, logit_h = model(b_t.unsqueeze(0), c_t.unsqueeze(0))
            prob_m = torch.sigmoid(logit_m).item()
            prob_h = torch.sigmoid(logit_h).item()
            flagged = (prob_m >= threshold) or (prob_h >= threshold)

            scen = meta["scenario"]
            stats[scen][0] += 1
            stats[scen][1] += int(flagged)

            if scen == "normal_adult" and meta["is_senior"]:
                stats["senior_adult"][0] += 1
                stats["senior_adult"][1] += int(flagged)

    summary_lines = [
        "STEP 5 SUMMARY: Checkpoint Evaluation",
        f"Checkpoint: {checkpoint_path} | Split: {split_name} | Threshold: {threshold}",
    ]
    for key, (total, alerts) in stats.items():
        pct = (alerts / total * 100.0) if total else 0.0
        summary_lines.append(f"  {key:26s}: {alerts:4d}/{total:4d} ({pct:5.1f}%)")

    _write_eval_log(summary_lines)


def main():
    """Execute all pretraining steps sequentially and write all summary logs."""
    parser = argparse.ArgumentParser(description="Run full pretraining pipeline for modelVF")
    parser.add_argument("--tacd-dir", type=Path, default=cfg.DEFAULT_TACD_DIR)
    parser.add_argument("--epochs", type=int, default=cfg.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=cfg.DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=cfg.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=cfg.DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--eval-split",
        choices=["validation", "test"],
        default=cfg.DEFAULT_EVAL_SPLIT,
    )
    parser.add_argument("--threshold", type=float, default=cfg.DEFAULT_EVAL_THRESHOLD)
    args = parser.parse_args()

    _write_pipeline_log(
        cfg.PIPELINE_LOG_PATH,
        [
            "FULL PRETRAINING PIPELINE STARTED",
            f"TACD directory: {args.tacd_dir}",
            f"Checkpoint output: {args.output}",
            f"Logs directory: {cfg.PRETRAINING_LOGS_DIR}",
        ],
    )

    # Stage 1: Parse TACD sessions and write step1 log
    try:
        load_tacd_sessions(tacd_dir=args.tacd_dir, write_log=True)
    except FileNotFoundError as exc:
        _write_pipeline_log(cfg.PIPELINE_LOG_PATH, [f"Stage 1 warning: {exc}"])

    # Stage 2 & 3: Augment minor/handoff windows, split by user, and write step2 & step3 logs
    torch.manual_seed(args.seed)
    train_loader, val_loader, _, mean, std = make_loaders(
        tacd_dir=args.tacd_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        write_log=True,
    )

    # Stage 4: Train Dual-Head 1D-CNN with AdamW + CosineAnnealingLR and write step4 log
    if cfg.TRAIN_LOG_PATH.exists():
        cfg.TRAIN_LOG_PATH.unlink()

    _append_train_summary(
        cfg.TRAIN_LOG_PATH,
        [
            "STEP 4 START: Training Dual-Head 1D-CNN",
            f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.learning_rate} | Weight decay: {args.weight_decay} | Seed: {args.seed}",
        ],
    )

    model = KinematicDualModel()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, optimizer=None)
        scheduler.step()
        saved_note = ""

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
            saved_note = f" -> Saved best checkpoint to {args.output}"

        _append_train_summary(
            cfg.TRAIN_LOG_PATH,
            [
                f"Epoch {epoch + 1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{saved_note}"
            ],
        )

    _append_train_summary(
        cfg.TRAIN_LOG_PATH,
        [
            "STEP 4 COMPLETE: Training Finished",
            f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}",
            f"Checkpoint saved at: {args.output}",
        ],
    )

    # Stage 5: Evaluate saved checkpoint and write step5 log
    _evaluate_trained_checkpoint(
        checkpoint_path=args.output,
        tacd_dir=args.tacd_dir,
        seed=args.seed,
        split_name=args.eval_split,
        threshold=args.threshold,
    )

    _write_pipeline_log(
        cfg.PIPELINE_LOG_PATH,
        [
            "FULL PRETRAINING PIPELINE COMPLETED SUCCESSFULLY",
            f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})",
            f"Checkpoint ready at: {args.output}",
        ],
    )


if __name__ == "__main__":
    main()