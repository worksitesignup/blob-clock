"""What the clock's eyes do. Pure Python, no GTK: the window steps them on a timer.

Every eye runs the same little mind on its own clock, so they never move as one:

  * it watches the cursor while the mouse moves, unless it's currently not
    interested (interest comes and goes at random);
  * now and then it glances at the cursor anyway, even when the mouse is still;
  * otherwise it wanders like a resting eye: small saccades, now and then back
    to straight ahead;
  * it blinks at random, sometimes twice.

Gaze reacts after a short delay and glides on a critically damped spring.
"""

import math
import random

import numpy as np

BLINK_DUR = 0.16             # close + open, s
BLINK_EVERY = (2.0, 7.0)
DOUBLE_BLINK = 0.2           # chance the next blink comes right after
GAZE_REACH = 260.0           # px from an eye at which it looks fully sideways
LOOK_ON_MOVE = 1.5           # s an eye keeps watching after the mouse last moved
GLANCE_EVERY = (4.0, 12.0)   # random glances at the cursor while it sits still
GLANCE_HOLD = (0.6, 1.4)
WANDER_EVERY = (0.6, 2.8)    # idle saccades
WANDER_REACH = 0.7           # how far idle looks stray (gaze units)
REACT = (0.12, 0.32)         # each eye's own reaction delay, s
GLIDE = 0.09                 # eye-movement time constant, s
INTEREST = 0.75              # chance an eye cares about the mouse moving...
INTEREST_EVERY = (6.0, 25.0)  # ...re-decided this often


def _soon(now, span):
    return now + random.uniform(*span)


class Eye:
    def __init__(self, offset, now):
        self.offset = np.array(offset, float)        # px from the body centre
        self.react = random.uniform(*REACT)
        self.blink = 0.0
        self.gaze = (0.0, 0.0)                        # rounded, what gets drawn
        self.p, self.v = np.zeros(2), np.zeros(2)     # smooth gaze + velocity
        self.tgt = np.zeros(2)
        self.queue = []                               # (time, target) waiting for the reaction delay
        self.look_until = 0.0
        self.wander = (0.0, 0.0)
        self.interested = random.random() < INTEREST
        self.blink_t0 = None
        self.next_blink = _soon(now, BLINK_EVERY)
        self.next_glance = _soon(now, GLANCE_EVERY)
        self.next_wander = _soon(now, (0.0, WANDER_EVERY[1]))
        self.next_mood = _soon(now, INTEREST_EVERY)

    def mouse_moved(self, now):
        if self.interested:
            self.look_until = max(self.look_until, now + LOOK_ON_MOVE * random.uniform(0.8, 1.3))

    def _events(self, now):
        if now >= self.next_mood:
            self.interested = random.random() < INTEREST
            self.next_mood = _soon(now, INTEREST_EVERY)
        if now >= self.next_glance:
            self.look_until = max(self.look_until, _soon(now, GLANCE_HOLD))
            self.next_glance = _soon(now, GLANCE_EVERY)
        if now >= self.next_wander:
            a = random.uniform(0, 2 * math.pi)
            r = 0.0 if random.random() < 0.3 else WANDER_REACH * random.random() ** 1.5
            self.wander = (round(r * math.cos(a), 1), round(r * math.sin(a) * 0.7, 1))
            self.next_wander = _soon(now, WANDER_EVERY)
        if now >= self.next_blink and self.blink_t0 is None:
            self.blink_t0 = now
            self.next_blink = now + (0.35 if random.random() < DOUBLE_BLINK else 0) + random.uniform(*BLINK_EVERY)

    def _target(self, now, to_cursor):
        if to_cursor is None or now >= self.look_until:
            return self.wander
        d = to_cursor - self.offset
        dist = float(np.hypot(*d))
        k = min(1.0, dist / GAZE_REACH) / max(dist, 1e-6)
        return (round(float(d[0] * k), 1), round(float(d[1] * k), 1))    # 0.1 steps

    def step(self, now, dt, to_cursor):
        """Advance; to_cursor: cursor position relative to the body centre (or None).
        Returns True while the eye is still moving or blinking."""
        self._events(now)
        tgt = self._target(now, to_cursor)
        last = self.queue[-1][1] if self.queue else tuple(self.tgt)
        if tgt != last:
            self.queue.append((now, tgt))
        while self.queue and now - self.queue[0][0] >= self.react:
            self.tgt = np.array(self.queue.pop(0)[1], float)
        w = 1.0 / GLIDE
        self.v += (w * w * (self.tgt - self.p) - 2 * w * self.v) * dt
        self.p += self.v * dt
        moving = bool(self.queue) or np.abs(self.tgt - self.p).max() > 0.005 or np.abs(self.v).max() > 0.05
        if not moving:
            self.p[:] = self.tgt
            self.v[:] = 0
        self.gaze = (round(float(self.p[0]), 2), round(float(self.p[1]), 2))
        if self.blink_t0 is not None:
            q = (now - self.blink_t0) / BLINK_DUR
            if q >= 1:
                self.blink, self.blink_t0 = 0.0, None
            else:
                self.blink = math.sin(math.pi * q) ** 0.7
        return moving or self.blink_t0 is not None

    def next_event(self, now):
        """When this eye next needs stepping (it may change its mind then)."""
        t = min(self.next_blink, self.next_glance, self.next_wander, self.next_mood)
        return min(t, self.look_until) if self.look_until > now else t

    def state(self):
        return (self.blink, self.gaze[0], self.gaze[1])
