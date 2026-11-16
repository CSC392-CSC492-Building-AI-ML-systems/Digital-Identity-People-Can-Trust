# ChatGPT was used as a co-developer to write much of this code and brainstorm ideas.
"""Evaluate checkpoint alerts on held-out synthetic sessions."""

import argparse

import torch

from machine_learning.dataset import get_session_splits
from machine_learning.features import make_features
from machine_learning.model import HandoffModel, LogisticRegression


def score_sessions(model, sessions, mean, std):
    scored = []
    model.eval()
    with torch.no_grad():
        for rows in sessions:
            windows = make_features(rows)
            if not windows:
                raise ValueError(f"Session {rows[0]['session_id']} has no complete windows")
            steps, values = zip(*windows)
            features = torch.tensor(values, dtype=torch.float32)
            scores = torch.sigmoid(model((features - mean) / std)).tolist()
            switch = rows[0].get("switch_step")
            scored.append((steps, scores, None if switch is None else int(switch)))
    return scored


def evaluate(scored, threshold):
    normal_count = normal_alerts = handoff_count = pre_alerts = detected = 0
    delays = []
    for steps, scores, switch in scored:
        alerts = [step for step, score in zip(steps, scores) if score >= threshold]
        if switch is None:
            normal_count += 1
            normal_alerts += bool(alerts)
        else:
            handoff_count += 1
            pre_alerts += any(step < switch for step in alerts)
            post_alerts = [step for step in alerts if step >= switch]
            if post_alerts:
                detected += 1
                delays.append(post_alerts[0] - switch)
    return normal_count, normal_alerts, handoff_count, pre_alerts, detected, delays


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved handoff model")
    parser.add_argument("checkpoint", help="path to a trained checkpoint")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true", help="compare thresholds on validation")
    parser.add_argument("--seed", type=int, help="split seed (required for older checkpoints)")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    if args.sweep and args.split != "validation":
        parser.error("threshold sweeps are only available on validation data")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved_seed = checkpoint.get("seed")
    if saved_seed is None and args.seed is None:
        parser.error("older checkpoint has no split seed; provide its training seed with --seed")
    if saved_seed is not None and args.seed is not None and saved_seed != args.seed:
        parser.error("seed does not match the checkpoint's training split")
    seed = saved_seed if saved_seed is not None else args.seed
    models = {"network": HandoffModel, "logistic": LogisticRegression}
    model = models[checkpoint["model_type"]]()
    model.load_state_dict(checkpoint["model_state"])
    sessions = get_session_splits(seed=seed)[1 if args.split == "validation" else 2]
    scored = score_sessions(model, sessions, checkpoint["mean"], checkpoint["std"])
    if args.sweep:
        print("Threshold | Normal alerts | Before switch | Detected | Mean delay (detected)")
        for i in range(1, 10):
            threshold = i / 10
            normal, false_normal, handoffs, false_before, detected, delays = evaluate(scored, threshold)
            delay = f"{sum(delays) / len(delays):.2f}" if delays else "n/a"
            print(f"{threshold:.1f}       | {false_normal}/{normal}        | {false_before}/{handoffs}          | {detected}/{handoffs}    | {delay}")
        return
    normal, false_normal, handoffs, false_before, detected, delays = evaluate(scored, args.threshold)
    print(f"Split: {args.split}; threshold: {args.threshold}")
    print(f"Normal sessions with an alert: {false_normal}/{normal}")
    print(f"Handoff sessions with an alert before the switch: {false_before}/{handoffs}")
    print(f"Handoffs with an alert after the switch: {detected}/{handoffs}")
    print(f"Mean delay among detected handoffs: {sum(delays) / len(delays):.2f} actions" if delays
          else "Mean delay: unavailable (no handoffs detected)")


if __name__ == "__main__":
    main()
