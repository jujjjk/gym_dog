"""Absolute deadlines and a bounded latest-target slot, without device I/O."""
import math


def next_deadline(deadline, finished, period):
    # Keep the phase, skip missed slots, never emit a catch-up burst.
    return deadline + max(1, math.floor((finished-deadline)/period)+1)*period


class LatestTarget:
    def __init__(self, max_age=.05):
        self.max_age = max_age
        self.target = None
        self.stamp = None

    def put(self, target, now):
        self.target = target.copy()
        self.stamp = now

    def get(self, now):
        if self.stamp is None or not 0 <= now-self.stamp <= self.max_age:
            return None
        return self.target.copy()
