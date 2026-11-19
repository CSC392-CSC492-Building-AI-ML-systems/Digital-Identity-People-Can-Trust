"""Evaluate a trained Dual-Head 1D-CNN checkpoint and log scenario metrics.

Evaluates a saved checkpoint on the validation or test split across normal adults,
seniors (age bins >= 8), simulated child-starts, and handoff scenarios, writing
a summary log file to disk.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys
import torch

_MODEL_VF_DIR = str(Path(__file__).resolve().parents[1])
_PRE_TRAIN_DIR = str(Path(__file__).resolve().parent)
for _p in (_MODEL_VF_DIR, _PRE_TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import configs as cfg
from dataset import get_session_splits
from augment_minors import build_training_pairs
from model import KinematicDualModel


def _write_eval_log(lines):
    """Write evaluation summary lines to disk and print to stdout.

    :param lines: Summary lines to write.
    :type lines: list[str]
    :return: Path to the written log file.
    :rtype: Path
    """
    log_path = cfg.PRETRAINING_LOGS_DIR / cfg.EVALUATE_LOG_FILENAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).isoformat()
    formatted = [f"[{ts_str}] {line}" for line in lines]
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted) + "\n")
    for line in formatted:
        print(line, flush=True)
    return log_path


def main():
    """Load checkpoint, evaluate scenarios on held-out split, and write log."""
    parser = argparse.ArgumentParser(description="Evaluate saved Dual-Head Kinematic model")
    parser.add_argument("checkpoint", type=Path, default=cfg.DEFAULT_CHECKPOINT_PATH, nargs="?")
    parser.add_argument("--tacd-dir", type=str, default=str(cfg.DEFAULT_TACD_DIR))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--threshold", type=float, default=cfg.DEFAULT_EVAL_THRESHOLD)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = KinematicDualModel()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean = ckpt["mean"]
    std = ckpt["std"]
    seed = ckpt.get("seed", cfg.DEFAULT_SEED)
    split_idx = 1 if args.split == "validation" else 2
    sessions = get_session_splits(tacd_dir=args.tacd_dir, seed=seed)[split_idx]
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
            flagged = (prob_m >= args.threshold) or (prob_h >= args.threshold)

            scen = meta["scenario"]
            stats[scen][0] += 1
            stats[scen][1] += int(flagged)

            if scen == "normal_adult" and meta["is_senior"]:
                stats["senior_adult"][0] += 1
                stats["senior_adult"][1] += int(flagged)

    summary_lines = [
        "STEP 5 SUMMARY: Checkpoint Evaluation",
        f"Checkpoint: {args.checkpoint} | Split: {args.split} | Threshold: {args.threshold}",
    ]
    for key, (total, alerts) in stats.items():
        pct = (alerts / total * 100.0) if total else 0.0
        summary_lines.append(f"  {key:26s}: {alerts:4d}/{total:4d} ({pct:5.1f}%)")

    _write_eval_log(summary_lines)


if __name__ == "__main__":
    main()