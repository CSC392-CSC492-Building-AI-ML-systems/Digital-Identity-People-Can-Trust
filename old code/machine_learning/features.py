# ChatGPT was used as a co-developer to write much of this code and brainstorm ideas.
"""Build behavioural features from a single session's bet records."""

import math


def summarize(rows):
    """Return the six behavioural averages for a nonempty list of bets."""
    count = len(rows)
    return [
        sum(math.log(float(row["gap_s"])) for row in rows) / count,
        sum(float(row["bet"]) / float(row["balance_before"]) for row in rows) / count,
        sum(float(row["mouse_speed"]) for row in rows) / count,
        sum(float(row["straightness"]) for row in rows) / count,
        sum(float(row["click_ms"]) for row in rows) / count,
        sum(int(row["misclick"]) for row in rows) / count,
    ]


def make_features(rows, baseline_size=16, window_size=8):
    """Return (step, features) for each complete recent window in a session.

    Rows must be from one session and in step order. Each window ends at the
    returned step and includes only actions up to that step.
    """
    if baseline_size <= 0 or window_size <= 0:
        raise ValueError("Baseline and window sizes must be positive")
    if len(rows) < baseline_size + window_size:
        return []

    baseline = summarize(rows[:baseline_size])
    result = []
    for end in range(baseline_size + window_size, len(rows) + 1):
        recent = summarize(rows[end - window_size:end])
        differences = [current - original for original, current in zip(baseline, recent)]
        result.append((int(rows[end - 1]["step"]), baseline + recent + differences))
    return result
