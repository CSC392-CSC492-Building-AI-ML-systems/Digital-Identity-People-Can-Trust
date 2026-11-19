"""Shared kinematic feature extraction for offline training and live inference.

This module computes five per-step kinematic features from consecutive cursor
coordinates: velocity, acceleration, jerk, movement angle, and log-compressed
inter-event time delta. Using the same ``compute_kinematic_step`` function for
both offline dataset preparation and live session scoring eliminates
training-serving skew.
"""

from pathlib import Path
import math
import sys

_MODEL_VF_DIR = str(Path(__file__).resolve().parents[1])
if _MODEL_VF_DIR not in sys.path:
    sys.path.insert(0, _MODEL_VF_DIR)

import configs as cfg


def compute_kinematic_step(
    x,
    y,
    timestamp,
    last_x,
    last_y,
    last_ts,
    last_v=0.0,
    last_a=0.0,
    width=None,
    height=None,
):
    """Compute a single 5-element kinematic feature tuple from mouse coordinates.

    If viewport ``width`` and ``height`` are provided, raw coordinates are
    normalized onto a standard reference canvas (``1920x1080`` by default) so
    that cursor speeds remain comparable across different display sizes.

    :param x: Horizontal cursor coordinate in pixels.
    :type x: float or int
    :param y: Vertical cursor coordinate in pixels.
    :type y: float or int
    :param timestamp: Event timestamp in milliseconds.
    :type timestamp: float or int
    :param last_x: Previous normalized horizontal coordinate, or ``None``.
    :type last_x: float or None
    :param last_y: Previous normalized vertical coordinate, or ``None``.
    :type last_y: float or None
    :param last_ts: Previous event timestamp in milliseconds, or ``None``.
    :type last_ts: float or int or None
    :param last_v: Previous velocity in pixels per millisecond.
    :type last_v: float
    :param last_a: Previous acceleration in pixels per millisecond squared.
    :type last_a: float
    :param width: Optional viewport width in pixels.
    :type width: float or int or None
    :param height: Optional viewport height in pixels.
    :type height: float or int or None
    :return: Tuple of ``(feature_vector, new_state_dict, status_str)`` where
        ``status_str`` is ``'ok'``, ``'first_point'``, ``'non_positive_dt'``,
        or ``'superhuman_velocity'``.
    :rtype: tuple[tuple[float, ...] | None, dict[str, float] | None, str]
    """
    # Normalize coordinates to a common canvas resolution when viewport size is known
    if width and height and float(width) > 0 and float(height) > 0:
        norm_x = (float(x) / float(width)) * cfg.CANVAS_NORM_WIDTH
        norm_y = (float(y) / float(height)) * cfg.CANVAS_NORM_HEIGHT
    else:
        norm_x = float(x)
        norm_y = float(y)
    ts_val = float(timestamp)

    # Seed initial state on the very first point of a stream because velocity requires two points
    if last_ts is None:
        return (
            None,
            {
                "last_x": norm_x,
                "last_y": norm_y,
                "last_ts": ts_val,
                "last_v": 0.0,
                "last_a": 0.0,
            },
            "first_point",
        )

    delta_t = ts_val - float(last_ts)
    if delta_t <= 0:
        return None, None, "non_positive_dt"

    delta_x = norm_x - float(last_x)
    delta_y = norm_y - float(last_y)

    # Compute first, second, and third spatial-temporal derivatives plus movement angle
    velocity = math.sqrt(delta_x ** 2 + delta_y ** 2) / delta_t
    if velocity > cfg.MAX_VELOCITY_PX_MS:
        return None, None, "superhuman_velocity"

    acceleration = (velocity - float(last_v)) / delta_t
    jerk = (acceleration - float(last_a)) / delta_t
    angle = math.atan2(delta_y, delta_x)

    # Log-transform capped time deltas so occasional user pauses do not dominate features
    log_dt = math.log1p(min(delta_t, cfg.MAX_DELTA_T_MS))

    feature_vector = (velocity, acceleration, jerk, angle, log_dt)
    new_state = {
        "last_x": norm_x,
        "last_y": norm_y,
        "last_ts": ts_val,
        "last_v": velocity,
        "last_a": acceleration,
    }
    return feature_vector, new_state, "ok"


def extract_windows_from_stream(
    events,
    window_size=cfg.WINDOW_SIZE,
    stride=cfg.STRIDE,
    width=None,
    height=None,
):
    """Convert an ordered sequence of cursor events into fixed-size kinematic windows.

    :param events: Iterable of ``(x, y, timestamp)`` tuples in chronological order.
    :type events: list[tuple[float, float, float | int]]
    :param window_size: Number of kinematic steps per window.
    :type window_size: int
    :param stride: Step offset between consecutive overlapping windows.
    :type stride: int
    :param width: Optional viewport width for coordinate normalization.
    :type width: float or int or None
    :param height: Optional viewport height for coordinate normalization.
    :type height: float or int or None
    :return: List of windows, each shaped ``(window_size, 5)``.
    :rtype: list[list[tuple[float, float, float, float, float]]]
    """
    steps = []
    state = {"last_x": None, "last_y": None, "last_ts": None, "last_v": 0.0, "last_a": 0.0}

    for x, y, ts in events:
        feat, next_state, status = compute_kinematic_step(
            x,
            y,
            ts,
            state["last_x"],
            state["last_y"],
            state["last_ts"],
            state["last_v"],
            state["last_a"],
            width=width,
            height=height,
        )
        if status == "first_point":
            state = next_state
        elif status == "ok":
            state = next_state
            steps.append(feat)

    # Slice consecutive steps with the configured stride to form overlapping windows
    windows = []
    for start in range(0, len(steps) - window_size + 1, stride):
        windows.append(steps[start : start + window_size])
    return windows