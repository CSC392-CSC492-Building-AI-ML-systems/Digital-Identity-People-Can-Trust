"""Prepare participant-split kinematic tensors and DataLoaders for the Dual-Head 1D-CNN.

Sessions are split by participant into training, validation, and test subsets before
generating window pairs so windows from the same user do not leak across splits.
Feature normalization (mean and standard deviation) is computed strictly from the
training split and writes a step summary log file.
"""

from datetime import datetime, timezone
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_MODEL_VF_DIR = str(Path(__file__).resolve().parents[1])
_PRE_TRAIN_DIR = str(Path(__file__).resolve().parent)
for _p in (_MODEL_VF_DIR, _PRE_TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import configs as cfg
from preprocess_tacd import load_tacd_sessions
from augment_minors import build_training_pairs
from features import extract_windows_from_stream


def _write_dataset_log(lines):
    """Write dataset split and normalization summary lines to disk.

    :param lines: Summary lines to record.
    :type lines: list[str]
    :return: Path to the log file.
    :rtype: Path
    """
    log_path = cfg.PRETRAINING_LOGS_DIR / cfg.DATASET_LOG_FILENAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).isoformat()
    formatted = [f"[{ts_str}] {line}" for line in lines]
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted) + "\n")
    for line in formatted:
        print(line, flush=True)
    return log_path


def _fallback_synthetic_adult_sessions(
    n_sessions=cfg.FALLBACK_SESSION_COUNT, seed=cfg.DEFAULT_SEED
):
    """Generate synthetic adult sessions when TACD logs are unavailable.

    :param n_sessions: Number of fallback sessions to synthesize.
    :type n_sessions: int
    :param seed: Random seed for reproducibility.
    :type seed: int
    :return: List of synthetic session dictionaries.
    :rtype: list[dict]
    """
    rng = np.random.default_rng(seed)
    sessions = []
    for idx in range(n_sessions):
        age_bin = int(rng.choice(list(cfg.FALLBACK_AGE_BINS)))
        is_senior = age_bin >= cfg.SENIOR_AGE_BIN_THRESHOLD
        speed = cfg.FALLBACK_SENIOR_SPEED_SCALE if is_senior else cfg.FALLBACK_ADULT_SPEED_SCALE
        events = []
        x_pos = cfg.FALLBACK_START_X
        y_pos = cfg.FALLBACK_START_Y
        ts_val = cfg.FALLBACK_START_TS
        for _ in range(cfg.FALLBACK_EVENT_COUNT):
            dt_val = float(rng.uniform(8.0, 22.0)) / speed
            ts_val += dt_val
            x_pos += float(rng.normal(4.0 * speed, 2.0))
            y_pos += float(rng.normal(3.0 * speed, 2.0))
            events.append((x_pos, y_pos, ts_val))
        windows = extract_windows_from_stream(
            events,
            width=int(cfg.CANVAS_NORM_WIDTH),
            height=int(cfg.CANVAS_NORM_HEIGHT),
        )
        sessions.append({
            "user_id": f"syn_{idx}",
            "log_id": f"syn_{idx}",
            "age_bin": age_bin,
            "is_senior": is_senior,
            "width": int(cfg.CANVAS_NORM_WIDTH),
            "height": int(cfg.CANVAS_NORM_HEIGHT),
            "events": events,
            "windows": windows,
        })
    return sessions


def get_session_splits(tacd_dir=cfg.DEFAULT_TACD_DIR, seed=cfg.DEFAULT_SEED):
    """Split sessions into train, validation, and test lists by participant.

    :param tacd_dir: Directory containing TACD ``participants.tsv`` and ``logs/``.
    :type tacd_dir: Path or str
    :param seed: Shuffle seed for splitting.
    :type seed: int
    :return: Tuple of ``(train_sessions, val_sessions, test_sessions)``.
    :rtype: tuple[list[dict], list[dict], list[dict]]
    """
    try:
        sessions = load_tacd_sessions(tacd_dir, write_log=False)
    except FileNotFoundError:
        sessions = []

    if len(sessions) < cfg.MIN_REAL_SESSIONS_BEFORE_FALLBACK:
        print("Warning: Few or no TACD logs found; using fallback kinematic streams.")
        sessions = _fallback_synthetic_adult_sessions(seed=seed)

    rng = random.Random(seed)
    sessions = list(sessions)
    rng.shuffle(sessions)

    total = len(sessions)
    train_end = int(total * cfg.TRAIN_SPLIT_RATIO)
    val_end = int(total * cfg.VAL_SPLIT_RATIO)
    return sessions[:train_end], sessions[train_end:val_end], sessions[val_end:]


def _examples_to_tensors(examples):
    """Convert raw training pair tuples into float32 PyTorch tensors.

    :param examples: List of ``(base_win, curr_win, y_minor, y_handoff, meta)`` items.
    :type examples: list[tuple]
    :return: Tuple of ``(base_tensor, curr_tensor, y_minor_tensor, y_handoff_tensor)``.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    """
    base_arr = np.array([ex[0] for ex in examples], dtype=np.float32)
    curr_arr = np.array([ex[1] for ex in examples], dtype=np.float32)
    y_minor = np.array([ex[2] for ex in examples], dtype=np.float32)
    y_handoff = np.array([ex[3] for ex in examples], dtype=np.float32)
    return (
        torch.from_numpy(base_arr),
        torch.from_numpy(curr_arr),
        torch.from_numpy(y_minor),
        torch.from_numpy(y_handoff),
    )


def make_loaders(
    tacd_dir=cfg.DEFAULT_TACD_DIR,
    batch_size=cfg.DEFAULT_BATCH_SIZE,
    seed=cfg.DEFAULT_SEED,
    write_log=True,
):
    """Build normalized PyTorch ``DataLoader`` instances and write a step summary log.

    :param tacd_dir: Directory containing TACD files.
    :type tacd_dir: Path or str
    :param batch_size: Batch size for the data loaders.
    :type batch_size: int
    :param seed: Seed used for splitting and augmentation.
    :type seed: int
    :param write_log: Whether to write the dataset step summary log.
    :type write_log: bool
    :return: Tuple of ``(train_loader, val_loader, test_loader, mean, std)``.
    :rtype: tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor]
    """
    train_s, val_s, test_s = get_session_splits(tacd_dir=tacd_dir, seed=seed)
    splits = [train_s, val_s, test_s]

    tensor_splits = [
        _examples_to_tensors(
            build_training_pairs(
                split_sessions, seed=seed + idx, write_log=(write_log and idx == 0)
            )
        )
        for idx, split_sessions in enumerate(splits)
    ]

    # Fit feature standardization statistics strictly on the training split
    train_base, train_curr, _, _ = tensor_splits[0]
    combined_train = torch.cat([train_base, train_curr], dim=0)
    mean = combined_train.mean(dim=(0, 1))
    std = combined_train.std(dim=(0, 1), unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)

    loaders = []
    for idx, (base_t, curr_t, ym_t, yh_t) in enumerate(tensor_splits):
        base_norm = (base_t - mean) / std
        curr_norm = (curr_t - mean) / std
        dataset = TensorDataset(base_norm, curr_norm, ym_t, yh_t)
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=(idx == 0)))

    if write_log:
        _write_dataset_log([
            "STEP 3 SUMMARY: Dataset Splits & Standardization",
            f"Session split counts -> train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}",
            f"Sample counts -> train: {len(tensor_splits[0][2])}, val: {len(tensor_splits[1][2])}, test: {len(tensor_splits[2][2])}",
            f"Batch size: {batch_size} | Seed: {seed}",
            f"Training channel mean: {[round(float(v), 6) for v in mean.tolist()]}",
            f"Training channel std:  {[round(float(v), 6) for v in std.tolist()]}",
        ])

    return (*loaders, mean, std)