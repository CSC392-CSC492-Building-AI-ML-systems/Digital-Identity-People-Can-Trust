"""D2.1: makes the fake gambling data.

Two kinds of sessions:
  * normal: one player plays from start to finish (used to train and test the model)
  * handoff: an adult plays, then a kid takes over partway (used to test the pause rules)

How to run:
    python3 src/generate_data.py --seed 0 --sessions 3000 --handoffs 400 --out data
"""
import argparse
import os

import numpy as np
import pandas as pd

import personas


def apply_multipliers(settings, multipliers):
    # e.g. speed 0.6 means the mouse moves at 60% of normal speed
    for name, multiplier in multipliers.items():
        settings[name] *= multiplier


def sample_player(random_gen, age_group=None, motor=None, new=None):
    """Makes one fake player and figures out how they'll behave."""

    # pick an age group if we weren't given one
    if age_group is None:
        groups = list(personas.AGE_MIX.keys())
        chances = list(personas.AGE_MIX.values())
        age_group = random_gen.choice(groups, p=np.array(chances))

    is_minor = age_group in personas.MINOR_GROUPS

    # only adults can have a motor difference in our data
    if motor is None:
        if is_minor:
            motor = False
        else:
            chance = personas.MOTOR_P.get(age_group, 0.0)
            roll = random_gen.random()
            motor = roll < chance

    # is this player new to the site?
    if new is None:
        chance = personas.NEW_P[age_group]
        roll = random_gen.random()
        new = roll < chance

    # start with the normal settings for their age group
    settings = dict(personas.BASE[age_group])

    if motor:
        apply_multipliers(settings, personas.MOTOR_MOD)
    if new:
        apply_multipliers(settings, personas.NEW_MOD)

    # nudge each setting a bit so every player is a little different
    for name, spread in personas.SESSION_SPREAD.items():
        settings[name] *= float(np.exp(random_gen.normal(0, spread)))

    # keep these from going too high
    settings["straight"] = min(settings["straight"], 0.98)
    settings["chase_p"] = min(settings["chase_p"], 0.95)
    settings["allin_p"] = min(settings["allin_p"], 0.95)
    settings["misclick_p"] = min(settings["misclick_p"], 0.95)

    return {
        "age_group": str(age_group),
        "is_minor": int(is_minor),
        "motor": int(motor),
        "new": int(new),
        "params": settings,
    }


def choose_bet_share(random_gen, settings, state):
    # usual bet size for this player, plus a bit of randomness
    share = settings["bet_frac"] * float(np.exp(random_gen.normal(0, 0.35)))

    # chasing: after a loss, they might raise their last bet by 1.3x to 2.2x
    if state["last_loss"]:
        if random_gen.random() < settings["chase_p"]:
            raised_share = state["last_frac"] * random_gen.uniform(1.3, 2.2)
            share = max(share, raised_share)

    # going all in: bet 50% to 100% of the balance
    if random_gen.random() < settings["allin_p"]:
        share = random_gen.uniform(0.5, 1.0)

    # never less than 0.5% or more than 100% of the balance
    share = min(max(share, 0.005), 1.0)
    return float(share)


def play(random_gen, player, n_bets, state):
    """Has one player place n_bets bets and returns one row per bet."""
    settings = player["params"]
    rows = []

    for _ in range(n_bets):
        balance = state["balance"]

        # if they're almost broke, top them back up so the session keeps going
        if balance < 10:
            balance = personas.START_BALANCE
            state["balance"] = balance

        share = choose_bet_share(random_gen, settings, state)
        bet = round(max(1.0, share * balance), 2)
        bet = min(bet, balance)

        # flip the coin. a win pays back double the bet
        won = random_gen.random() < personas.WIN_P
        if won:
            state["balance"] = balance - bet + bet * personas.PAYOUT
        else:
            state["balance"] = balance - bet

        # how long they waited since the last bet
        gap = float(settings["gap_s"] * np.exp(random_gen.normal(0, 0.45)))

        # mouse behaviour for this bet
        speed = float(max(50, random_gen.normal(settings["speed"], settings["speed"] * 0.18)))
        straightness = float(min(max(random_gen.normal(settings["straight"], 0.06), 0.2), 1.0))
        click_ms = float(max(30, random_gen.normal(settings["click_ms"], settings["click_ms"] * 0.2)))
        misclick = random_gen.random() < settings["misclick_p"]

        rows.append({
            "gap_s": gap,
            "bet": bet,
            "balance_before": round(balance, 2),
            "won": int(won),
            "mouse_speed": speed,
            "straightness": straightness,
            "click_ms": click_ms,
            "misclick": int(misclick),
        })

        # remember this bet for the next one (needed for chasing)
        state["last_loss"] = not won
        state["last_frac"] = bet / balance

    return rows


def new_state():
    # every session starts with $500, no loss yet, and a small "last bet"
    return {"balance": personas.START_BALANCE, "last_loss": False, "last_frac": 0.02}


def make_normal_sessions(random_gen, n_sessions, start_id=0):
    """Makes sessions where one player plays from start to finish."""
    all_rows = []

    for i in range(n_sessions):
        player = sample_player(random_gen)
        n_bets = int(random_gen.integers(40, 121))  # somewhere from 40 to 120 bets
        rows = play(random_gen, player, n_bets, new_state())

        # label every bet. these labels are only used to grade the model later
        for step, row in enumerate(rows):
            row["session_id"] = start_id + i
            row["step"] = step
            row["kind"] = "normal"
            row["age_group"] = player["age_group"]
            row["is_minor"] = player["is_minor"]
            row["motor"] = player["motor"]
            row["new"] = player["new"]
            row["actor_minor"] = player["is_minor"]  # same person the whole session

        all_rows.extend(rows)

    return all_rows


def make_handoff_sessions(random_gen, n_sessions, start_id):
    """Makes sessions where an adult plays, then hands the device to a kid."""
    adult_groups = ["adult_19_21", "adult_22_59", "adult_60_plus"]
    minor_groups = ["minor_13_15", "minor_16_18"]
    all_rows = []

    for i in range(n_sessions):
        # make one adult and one kid
        adult_group = random_gen.choice(adult_groups)
        adult = sample_player(random_gen, age_group=adult_group)
        minor_group = random_gen.choice(minor_groups)
        minor = sample_player(random_gen, age_group=minor_group)

        # how many bets each of them places
        n_adult_bets = int(random_gen.integers(20, 41))  # 20 to 40
        n_minor_bets = int(random_gen.integers(30, 61))  # 30 to 60

        # the adult plays first, then the kid takes over.
        # they share the same state, so the kid starts with the adult's balance
        state = new_state()
        adult_rows = play(random_gen, adult, n_adult_bets, state)
        minor_rows = play(random_gen, minor, n_minor_bets, state)

        # mark who actually placed each bet
        for row in adult_rows:
            row["actor_minor"] = 0
        for row in minor_rows:
            row["actor_minor"] = 1

        rows = adult_rows + minor_rows

        # label every bet. the account is the adult's, so these labels use the adult's info
        for step, row in enumerate(rows):
            row["session_id"] = start_id + i
            row["step"] = step
            row["kind"] = "handoff"
            row["age_group"] = adult["age_group"]
            row["is_minor"] = 0
            row["motor"] = adult["motor"]
            row["new"] = adult["new"]
            row["minor_group"] = minor["age_group"]
            row["switch_step"] = n_adult_bets  # the bet number where the kid took over

        all_rows.extend(rows)

    return all_rows


def generate(seed=0, sessions=3000, handoffs=400):
    """Makes all the fake data. Returns the normal sessions and the handoff sessions."""

    # one random machine for the whole run. same seed = same data every time
    random_gen = np.random.default_rng(seed)

    # normal sessions get ids 0 to 2999
    normal_rows = make_normal_sessions(random_gen, sessions)

    # handoff ids start right after, at 3000, so they never clash
    handoff_rows = make_handoff_sessions(random_gen, handoffs, start_id=sessions)

    # turn the lists of rows into tables
    normal = pd.DataFrame(normal_rows)
    handoff = pd.DataFrame(handoff_rows)

    return normal, handoff


def main():
    # python3 src/generate_data.py --seed 0 --sessions 3000 --handoffs 400 --out data
    parser = argparse.ArgumentParser(description="Make the fake gambling data.")
    parser.add_argument("--seed", type=int, default=0, help="same seed = same data")
    parser.add_argument("--sessions", type=int, default=3000, help="number of normal sessions")
    parser.add_argument("--handoffs", type=int, default=400, help="number of handoff sessions")
    parser.add_argument("--out", default="data", help="folder to save the CSVs in")
    args = parser.parse_args()

    # make the data
    normal, handoff = generate(args.seed, args.sessions, args.handoffs)

    # save it. makedirs creates the folder if it doesn't exist yet
    os.makedirs(args.out, exist_ok=True)
    normal.to_csv(os.path.join(args.out, "sessions.csv"), index=False)
    handoff.to_csv(os.path.join(args.out, "handoff_sessions.csv"), index=False)

    # quick summary so we can see it worked
    n_normal = normal["session_id"].nunique()
    n_handoff = handoff["session_id"].nunique()
    minor_share = normal.groupby("session_id")["is_minor"].first().mean()
    print(f"normal: {n_normal} sessions, {len(normal)} bets, {minor_share:.0%} minors")
    print(f"handoff: {n_handoff} sessions, {len(handoff)} bets")


if __name__ == "__main__":
    main()
