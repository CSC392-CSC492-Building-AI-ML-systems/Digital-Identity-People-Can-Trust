"""Parse The Attentive Cursor Dataset (TACD) logs, XML metadata, and demographics.

This module reads participant metadata from ``participants.tsv``, extracts browser
viewport dimensions from XML log files, filters space-delimited CSV mouse event
logs for valid pointer interactions, converts trajectories into kinematic windows,
and writes a step summary log file.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pandas as pd

_MODEL_VF_DIR = str(Path(__file__).resolve().parents[1])
_PRE_TRAIN_DIR = str(Path(__file__).resolve().parent)
for _p in (_MODEL_VF_DIR, _PRE_TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import configs as cfg
from features import extract_windows_from_stream


def _write_step_log(log_path, lines):
    """Write step summary lines to a log file and echo them to standard output.

    :param log_path: Destination log file path.
    :type log_path: Path or str
    :param lines: Sequence of human-readable log lines to record.
    :type lines: list[str]
    :return: Path to the written log file.
    :rtype: Path
    """
    out_path = Path(log_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now(timezone.utc).isoformat()
    formatted = [f"[{timestamp_str}] {line}" for line in lines]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted) + "\n")
    for line in formatted:
        print(line, flush=True)
    return out_path


def parse_xml_viewport(xml_path):
    """Extract browser viewport dimensions ``(width, height)`` from a TACD XML file.

    :param xml_path: Path to the session ``.xml`` metadata file.
    :type xml_path: Path or str
    :return: Tuple of ``(width, height)`` in pixels.
    :rtype: tuple[int, int]
    """
    path = Path(xml_path)
    if not path.exists():
        return cfg.DEFAULT_VIEWPORT_WIDTH, cfg.DEFAULT_VIEWPORT_HEIGHT
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        # Prefer active browser window size; fall back to screen dimensions if absent
        win_text = root.findtext("window") or root.findtext("screen")
        if not win_text:
            return cfg.DEFAULT_VIEWPORT_WIDTH, cfg.DEFAULT_VIEWPORT_HEIGHT
        w_str, h_str = win_text.strip().lower().split("x", 1)
        return int(w_str), int(h_str)
    except Exception:
        return cfg.DEFAULT_VIEWPORT_WIDTH, cfg.DEFAULT_VIEWPORT_HEIGHT


def parse_csv_events(csv_path):
    """Read a space-delimited TACD CSV log and return valid mouse events.

    Non-mouse events and zero-coordinate ``(0, 0)`` non-positional events are
    excluded so kinematic derivatives reflect genuine cursor motion.

    :param csv_path: Path to the session ``.csv`` event log.
    :type csv_path: Path or str
    :return: Chronological list of ``(xpos, ypos, timestamp)`` tuples.
    :rtype: list[tuple[float, float, int]]
    """
    events = []
    path = Path(csv_path)
    if not path.exists():
        return events

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        _header = f.readline()
        for line in f:
            parts = line.strip().split(cfg.CSV_DELIMITER, 5)
            if len(parts) < 5:
                continue
            _, ts_str, x_str, y_str, event_name = parts[:5]
            if event_name not in cfg.VALID_MOUSE_EVENTS:
                continue
            try:
                x_val = float(x_str)
                y_val = float(y_str)
                ts_val = int(ts_str)
            except ValueError:
                continue
            # Filter out (0, 0) sentinel coordinates emitted for non-cursor events
            if x_val == 0.0 and y_val == 0.0:
                continue
            events.append((x_val, y_val, ts_val))
    return events


def load_tacd_sessions(tacd_dir=cfg.DEFAULT_TACD_DIR, write_log=True):
    """Load TACD sessions joined with participant age bins and generate a summary log.

    :param tacd_dir: Root directory containing ``participants.tsv`` and ``logs/``.
    :type tacd_dir: Path or str
    :param write_log: Whether to write the step summary log file to disk.
    :type write_log: bool
    :return: List of parsed session dictionaries with events and kinematic windows.
    :rtype: list[dict]
    :raises FileNotFoundError: If ``participants.tsv`` cannot be found.
    """
    tacd_path = Path(tacd_dir)
    participants_file = tacd_path / cfg.TACD_PARTICIPANTS_FILENAME
    logs_dir = tacd_path / cfg.TACD_LOGS_SUBDIR

    if not participants_file.exists():
        raise FileNotFoundError(f"Cannot find {participants_file}")

    df_part = pd.read_csv(participants_file, sep=cfg.TSV_DELIMITER)
    total_participant_rows = len(df_part)

    # Drop records with missing ('na') age bins before integer casting
    df_valid = df_part[df_part["age"].astype(str) != cfg.NA_AGE_VALUE].copy()
    df_valid["age_bin"] = df_valid["age"].astype(int)

    sessions = []
    missing_csv_count = 0
    short_session_count = 0
    total_events_parsed = 0
    total_windows_created = 0
    senior_session_count = 0

    for _, row in df_valid.iterrows():
        log_id = str(row["log_id"]).strip()
        csv_path = logs_dir / f"{log_id}{cfg.TACD_CSV_EXTENSION}"
        xml_path = logs_dir / f"{log_id}{cfg.TACD_XML_EXTENSION}"

        if not csv_path.exists():
            missing_csv_count += 1
            continue

        width, height = parse_xml_viewport(xml_path)
        events = parse_csv_events(csv_path)
        windows = extract_windows_from_stream(events, width=width, height=height)

        if len(windows) < cfg.MIN_SESSION_WINDOWS:
            short_session_count += 1
            continue

        age_bin = int(row["age_bin"])
        is_senior = age_bin >= cfg.SENIOR_AGE_BIN_THRESHOLD
        if is_senior:
            senior_session_count += 1

        total_events_parsed += len(events)
        total_windows_created += len(windows)

        sessions.append({
            "user_id": str(row["user_id"]),
            "log_id": log_id,
            "age_bin": age_bin,
            "is_senior": is_senior,
            "width": width,
            "height": height,
            "events": events,
            "windows": windows,
        })

    if write_log:
        log_file = cfg.PRETRAINING_LOGS_DIR / cfg.PREPROCESS_LOG_FILENAME
        _write_step_log(
            log_file,
            [
                "STEP 1 SUMMARY: TACD Log & Demographic Preprocessing",
                f"TACD directory: {tacd_path}",
                f"Total rows in participants.tsv: {total_participant_rows}",
                f"Rows with valid age bin (excluding '{cfg.NA_AGE_VALUE}'): {len(df_valid)}",
                f"Missing CSV log files skipped: {missing_csv_count}",
                f"Sessions with fewer than {cfg.MIN_SESSION_WINDOWS} windows skipped: {short_session_count}",
                f"Usable adult sessions retained: {len(sessions)} (seniors bin >= {cfg.SENIOR_AGE_BIN_THRESHOLD}: {senior_session_count})",
                f"Total valid mouse events extracted: {total_events_parsed}",
                f"Total ({cfg.WINDOW_SIZE}, {cfg.IN_CHANNELS}) kinematic windows generated: {total_windows_created}",
            ],
        )

    return sessions


if __name__ == "__main__":
    load_tacd_sessions(write_log=True)