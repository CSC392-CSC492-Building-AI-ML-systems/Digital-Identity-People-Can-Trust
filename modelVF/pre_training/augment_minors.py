"""Physics-guided biomechanical synthesizer for minor trajectories and handoff sessions.

Because crowdsourced telemetry datasets such as TACD contain adult participants,
this module perturbs real coordinate streams using motor-control heuristics (pacing
shifts, angular jitter, and target overshoot oscillations) to synthesize minor
trajectories, handoff sequences, and negative anchors for training and updates.
"""

from datetime import datetime, timezone
from pathlib import Path
import math
import sys
import numpy as np
import torch

_MODEL_VF_DIR = str(Path(__file__).resolve().parents[1])
_PRE_TRAIN_DIR = str(Path(__file__).resolve().parent)
for _p in (_MODEL_VF_DIR, _PRE_TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import configs as cfg
from features import extract_windows_from_stream


def _write_augment_log(lines):
    """Persist step summary lines for minor trajectory augmentation to disk.

    :param lines: Summary lines to write into the augmentation log file.
    :type lines: list[str]
    :return: Path to the log file.
    :rtype: Path
    """
    log_path = cfg.PRETRAINING_LOGS_DIR / cfg.AUGMENT_LOG_FILENAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).isoformat()
    formatted = [f"[{ts_str}] {line}" for line in lines]
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted) + "\n")
    for line in formatted:
        print(line, flush=True)
    return log_path


def perturb_trajectory_to_minor(events, rng, profile=cfg.MINOR_PROFILE_YOUNG):
    """Apply developmental motor perturbations to a real ``(x, y, timestamp)`` stream.

    :param events: Sequence of ``(x, y, timestamp)`` cursor samples.
    :type events: list[tuple[float, float, float | int]]
    :param rng: NumPy random generator instance.
    :type rng: numpy.random.Generator
    :param profile: Perturbation profile identifier (younger or older minor).
    :type profile: str
    :return: Perturbed ``(x, y, timestamp)`` trajectory.
    :rtype: list[tuple[float, float, float]]
    """
    if len(events) < cfg.MIN_TRAJECTORY_EVENTS:
        return events

    # Select profile-specific speed, spatial jitter, overshoot, and timing variability
    if profile == cfg.MINOR_PROFILE_YOUNG:
        speed_scale = rng.uniform(*cfg.MINOR_YOUNG_SPEED_RANGE)
        jitter_px = rng.uniform(*cfg.MINOR_YOUNG_JITTER_RANGE)
        overshoot_gain = rng.uniform(*cfg.MINOR_YOUNG_OVERSHOOT_RANGE)
        pause_noise = cfg.MINOR_YOUNG_PAUSE_NOISE
    else:
        speed_scale = rng.uniform(*cfg.MINOR_OLDER_SPEED_RANGE)
        jitter_px = rng.uniform(*cfg.MINOR_OLDER_JITTER_RANGE)
        overshoot_gain = rng.uniform(*cfg.MINOR_OLDER_OVERSHOOT_RANGE)
        pause_noise = cfg.MINOR_OLDER_PAUSE_NOISE

    perturbed = []
    virtual_ts = float(events[0][2])
    perturbed.append((float(events[0][0]), float(events[0][1]), virtual_ts))

    for idx in range(1, len(events)):
        prev_x, prev_y, prev_ts = events[i_prev := idx - 1]
        cur_x, cur_y, cur_ts = events[idx]

        raw_dt = max(1.0, float(cur_ts) - float(prev_ts))
        scaled_dt = (raw_dt / speed_scale) * float(np.exp(rng.normal(0, pause_noise)))
        scaled_dt = max(cfg.MIN_SYNTH_DT_MS, scaled_dt)
        virtual_ts += scaled_dt

        dx = float(cur_x) - float(prev_x)
        dy = float(cur_y) - float(prev_y)
        dist = math.hypot(dx, dy)

        # Add perpendicular tremor/jitter plus sinusoidal target overshoot along path
        if dist > 1.0:
            perp_x = -dy / dist
            perp_y = dx / dist
            lateral_err = float(rng.normal(0, jitter_px))
            overshoot = math.sin(i_prev * cfg.MINOR_OVERSHOOT_PHASE_GAIN) * dist * overshoot_gain
            new_x = float(cur_x) + perp_x * lateral_err + (dx / dist) * overshoot
            new_y = float(cur_y) + perp_y * lateral_err + (dy / dist) * overshoot
        else:
            new_x = float(cur_x) + float(rng.normal(0, jitter_px * 0.5))
            new_y = float(cur_y) + float(rng.normal(0, jitter_px * 0.5))

        perturbed.append((
            max(cfg.MIN_SYNTH_COORD_PX, new_x),
            max(cfg.MIN_SYNTH_COORD_PX, new_y),
            virtual_ts,
        ))

    return perturbed


def make_synthetic_minor_window_tensor(adult_window_tensor: torch.Tensor) -> torch.Tensor:
    """Create a synthetic minor-like normalized window tensor as a negative anchor.

    :param adult_window_tensor: Normalized window tensor of shape ``(T, 5)``.
    :type adult_window_tensor: torch.Tensor
    :return: Perturbed minor-anchor tensor of shape ``(T, 5)``.
    :rtype: torch.Tensor
    """
    noise = torch.randn_like(adult_window_tensor) * cfg.SYNTH_MINOR_NOISE_STD
    perturbed = adult_window_tensor.clone() + noise
    perturbed[:, 0] *= cfg.SYNTH_MINOR_VEL_MULT
    perturbed[:, 1] *= cfg.SYNTH_MINOR_ACC_MULT
    perturbed[:, 2] *= cfg.SYNTH_MINOR_JERK_MULT
    return perturbed


def build_training_pairs(adult_sessions, seed=cfg.DEFAULT_SEED, write_log=False):
    """Build labeled ``(base_win, curr_win, y_minor, y_handoff, meta)`` training pairs.

    Produces four scenarios:
    - ``normal_adult``: Same adult across windows (``y_minor=0, y_handoff=0``).
    - ``child_starts``: Synthesized minor from window 1 (``y_minor=1, y_handoff=0``).
    - ``adult_to_minor_handoff``: Adult baseline to minor current (``y_minor=1, y_handoff=1``).
    - ``adult_to_adult_handoff``: Adult baseline to different adult (``y_minor=0, y_handoff=1``).

    :param adult_sessions: List of preprocessed adult session dictionaries.
    :type adult_sessions: list[dict]
    :param seed: Random seed for reproducible augmentation.
    :type seed: int
    :param write_log: Whether to write a summary log file for this augmentation run.
    :type write_log: bool
    :return: List of training pair tuples.
    :rtype: list[tuple]
    """
    rng = np.random.default_rng(seed)
    examples = []
    counts = {
        "normal_adult": 0,
        "child_starts": 0,
        "adult_to_minor_handoff": 0,
        "adult_to_adult_handoff": 0,
    }

    if not adult_sessions:
        return examples

    for idx, session in enumerate(adult_sessions):
        w_list = session["windows"]
        base_win = w_list[0]

        for cur_win in w_list[1:]:
            examples.append((
                base_win,
                cur_win,
                cfg.DP_VERIFIED_MINOR_LABEL,
                cfg.DP_VERIFIED_HANDOFF_LABEL,
                {"scenario": "normal_adult", "is_senior": session["is_senior"]},
            ))
            counts["normal_adult"] += 1

        profile = cfg.MINOR_PROFILE_YOUNG if idx % 2 == 0 else cfg.MINOR_PROFILE_OLDER
        minor_events = perturb_trajectory_to_minor(session["events"], rng, profile=profile)
        minor_windows = extract_windows_from_stream(
            minor_events, width=session["width"], height=session["height"]
        )

        if len(minor_windows) >= cfg.MIN_SESSION_WINDOWS:
            minor_base = minor_windows[0]
            for m_win in minor_windows[1:]:
                examples.append((
                    minor_base,
                    m_win,
                    cfg.DP_ANCHOR_MINOR_LABEL,
                    cfg.DP_VERIFIED_HANDOFF_LABEL,
                    {"scenario": "child_starts", "is_senior": False},
                ))
                counts["child_starts"] += 1

            for m_win in minor_windows[1:]:
                examples.append((
                    base_win,
                    m_win,
                    cfg.DP_ANCHOR_MINOR_LABEL,
                    cfg.DP_ANCHOR_HANDOFF_LABEL,
                    {"scenario": "adult_to_minor_handoff", "is_senior": False},
                ))
                counts["adult_to_minor_handoff"] += 1

        other_session = adult_sessions[(idx + 1) % len(adult_sessions)]
        for other_win in other_session["windows"][cfg.ADULT_HANDOFF_WINDOW_SLICE]:
            examples.append((
                base_win,
                other_win,
                cfg.DP_VERIFIED_MINOR_LABEL,
                cfg.DP_ANCHOR_HANDOFF_LABEL,
                {"scenario": "adult_to_adult_handoff", "is_senior": other_session["is_senior"]},
            ))
            counts["adult_to_adult_handoff"] += 1

    if write_log:
        _write_augment_log([
            "STEP 2 SUMMARY: Biomechanical Minor & Handoff Pair Generation",
            f"Input adult sessions: {len(adult_sessions)} | Seed: {seed}",
            f"normal_adult pairs: {counts['normal_adult']}",
            f"child_starts pairs: {counts['child_starts']}",
            f"adult_to_minor_handoff pairs: {counts['adult_to_minor_handoff']}",
            f"adult_to_adult_handoff pairs: {counts['adult_to_adult_handoff']}",
            f"Total training pairs generated: {len(examples)}",
        ])

    return examples