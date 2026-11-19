"""Split training sessions across simulated clients."""

import random


def split_among_clients(sessions, num_clients, seed=0):
    """Deal sessions out round-robin so each client gets its own shard.

    We keep whole sessions together. That way one client's raw rows never
    show up on another client.
    """
    if num_clients < 1:
        raise ValueError("need at least one client")

    sessions = list(sessions)
    rng = random.Random(seed)
    rng.shuffle(sessions)

    buckets = [[] for _ in range(num_clients)]
    for i, session in enumerate(sessions):
        buckets[i % num_clients].append(session)

    # ignore empty buckets (only happens if we ask for more clients than sessions)
    return [bucket for bucket in buckets if bucket]
