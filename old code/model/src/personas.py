"""Persona settings for the synthetic data.

Every number here is an assumption informed by docs/research_notes.md.
Tune them in one place and rerun the pipeline.
"""

AGE_GROUPS = ["minor_13_15", "minor_16_18", "adult_19_21", "adult_22_59", "adult_60_plus"]
MINOR_GROUPS = {"minor_13_15", "minor_16_18"}

# Share of each age group when we sample normal (non handoff) sessions.
AGE_MIX = {
    "minor_13_15": 0.17,
    "minor_16_18": 0.18,
    "adult_19_21": 0.15,
    "adult_22_59": 0.35,
    "adult_60_plus": 0.15,
}

# Base behaviour per age group.
# gap_s: mean seconds between bets
# bet_frac: mean bet as a share of balance
# chase_p: chance of raising the bet right after a loss
# allin_p: chance of betting more than half the balance
# speed: mouse speed in px/s
# straight: path straightness, 1 is a perfect line
# click_ms: click hold time
# misclick_p: chance a click misses
BASE = {
    "minor_13_15": dict(gap_s=3.2, bet_frac=0.12, chase_p=0.55, allin_p=0.10, speed=1100, straight=0.76, click_ms=95, misclick_p=0.05),
    "minor_16_18": dict(gap_s=3.8, bet_frac=0.10, chase_p=0.47, allin_p=0.08, speed=1050, straight=0.79, click_ms=100, misclick_p=0.04),
    "adult_19_21": dict(gap_s=4.6, bet_frac=0.075, chase_p=0.36, allin_p=0.05, speed=1000, straight=0.82, click_ms=105, misclick_p=0.03),
    "adult_22_59": dict(gap_s=6.0, bet_frac=0.045, chase_p=0.22, allin_p=0.02, speed=900, straight=0.85, click_ms=110, misclick_p=0.02),
    "adult_60_plus": dict(gap_s=8.0, bet_frac=0.035, chase_p=0.18, allin_p=0.015, speed=620, straight=0.77, click_ms=140, misclick_p=0.05),
}

# Chance an adult in each group has a motor difference (for example a tremor).
MOTOR_P = {"adult_19_21": 0.05, "adult_22_59": 0.06, "adult_60_plus": 0.18}

# Chance the player is new to the site.
NEW_P = {"minor_13_15": 0.7, "minor_16_18": 0.7, "adult_19_21": 0.45, "adult_22_59": 0.25, "adult_60_plus": 0.3}

# How a motor difference changes mouse behaviour (multipliers).
MOTOR_MOD = dict(speed=0.6, straight=0.78, click_ms=1.45, misclick_p=2.8)

# How being new changes behaviour (multipliers).
NEW_MOD = dict(gap_s=1.15, bet_frac=1.2, chase_p=1.15, misclick_p=1.4)

# How much each session wanders from its group's mean (log normal spread).
SESSION_SPREAD = dict(gap_s=0.25, bet_frac=0.30, chase_p=0.25, allin_p=0.35, speed=0.15, straight=0.05, click_ms=0.15, misclick_p=0.35)

START_BALANCE = 500.0
WIN_P = 0.47  # house edge
PAYOUT = 2.0
