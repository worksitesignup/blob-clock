"""Water-balloon soft body: a tensioned stretchy membrane around incompressible fill.

Pure numpy, no GTK. Positions are in output (screen) pixels; every node has mass 1.

    body = SoftBody(R, centre)
    body.grab(pt) / body.drag_to(pt, vel) / body.release()
    body.kick(f)                 radial velocity kick, f(angle) -> px/s
    awake = body.step(dt, bounds)

After each step the body exposes what the renderer and the sound need:
    centre, hit (momentum pushed into walls), patch (widest wall contact, fraction of the skin), jiggle (RMS skin
    speed about the centre) and stretch (how far from round, fraction of R).

Integration is XPBD: forces -> predicted positions -> membrane / area / wall
constraints -> velocities from the position change. The body never rotates:
its net turn is undone every substep, so the face can squash but not tilt.
"""

import math

import numpy as np

N = 48              # nodes on the membrane
SHAPE = 15.0        # faint rubber memory of the round shape
TENSION = 0.985     # membrane rest length vs. round: slightly pre-stretched, like a filled balloon
EDGE_C = 5e-5       # membrane compliance (XPBD, higher = stretchier)
BEND = 80.0         # membrane bending stiffness
DAMP = 1.4          # internal damping: low, so it keeps sloshing
AIR = 0.35          # momentum drop-off of the whole balloon, 1/s
RESTITUTION = 0.12  # nodes barely bounce off walls by themselves; the membrane does
FRICTION = 6.0      # tangential wall friction, 1/s
GRAB_K = 500.0      # pointer spring
GRAB_C = 18.0
ITERS = 4           # constraint iterations per substep
H = 1 / 480         # physics substep
VMAX = 2200.0       # throw speed cap
STOP = 40.0         # below this speed the ball settles faster (rolling resistance)
SKIN = 0.965        # membrane radius as a fraction of R (the glass sits just inside the gel)


# neighbour indices: a[NXT] == np.roll(a, -1), a[PRV] == np.roll(a, 1), but far cheaper
# (np.roll's per-call overhead dominated the whole simulation at this size)
NXT = np.roll(np.arange(N), -1)
PRV = np.roll(np.arange(N), 1)


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * y[NXT] - x[NXT] * y))


class SoftBody:
    def __init__(self, R, centre):
        self.R = float(R)
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        self.angles = a
        self.rest = np.stack([np.cos(a), np.sin(a)], 1) * (self.R * SKIN)
        self.rest_len = np.linalg.norm(self.rest - np.roll(self.rest, -1, 0), axis=1)
        self.rest_area = polygon_area(self.rest)
        self.p = self.rest + np.asarray(centre, float)
        self.v = np.zeros_like(self.p)
        self.drag = None
        self.acc = 0.0
        self.idle_t = 0.0
        self.hit = 0.0
        self.patch = 0.0
        self.jiggle = 0.0
        self.stretch = 0.0

    @property
    def centre(self):
        return self.p.mean(0)

    # ---- interaction ------------------------------------------------------------

    def kick(self, f):
        """Radial velocity kick: f(node angle) gives px/s outward."""
        self.v += (f(self.angles) / (self.R * SKIN))[:, None] * self.rest

    def grab(self, pt):
        """Tie the skin near `pt` to the pointer with a damped spring."""
        d = np.linalg.norm(self.p - pt, axis=1)
        w = np.exp(-(d / (0.75 * self.R)) ** 2)
        self.drag = {"w": w[:, None] / w.max(), "off": self.p - pt,
                     "tgt": np.array(pt, float), "pv": np.zeros(2)}

    def drag_to(self, pt, vel):
        if self.drag is not None:
            self.drag["tgt"], self.drag["pv"] = np.array(pt, float), np.asarray(vel, float)

    def release(self):
        self.drag = None

    # ---- simulation -------------------------------------------------------------

    def _unrotate(self, p, old):
        """Undo the skin's best-fit turn about its centre (positions and previous positions)."""
        c = p.mean(0)
        r = p - c
        rest = self.rest
        th = math.atan2(float(np.sum(rest[:, 0] * r[:, 1] - rest[:, 1] * r[:, 0])),
                        float(np.sum(rest[:, 0] * r[:, 0] + rest[:, 1] * r[:, 1])))
        if abs(th) <= 1e-6:
            return
        cs, sn = math.cos(-th), math.sin(-th)
        p[:, 0], p[:, 1] = c[0] + cs * r[:, 0] - sn * r[:, 1], c[1] + sn * r[:, 0] + cs * r[:, 1]
        o = old - c                                          # keep velocities in the same frame
        old[:, 0], old[:, 1] = c[0] + cs * o[:, 0] - sn * o[:, 1], c[1] + sn * o[:, 0] + cs * o[:, 1]

    def _substep(self, h, bounds):
        p, v = self.p, self.v
        c = p.mean(0)
        dr = self.drag
        # soft forces
        F = SHAPE * (c + self.rest - p)
        F += BEND * (0.5 * (p[PRV] + p[NXT]) - p)
        F -= DAMP * (v - v.mean(0))
        if dr is not None:
            F += dr["w"] * (GRAB_K * (dr["tgt"] + dr["off"] - p) - GRAB_C * (v - dr["pv"]))
        v += F * h
        # whole-body speed: cap, air drag, settle when slow
        vm = v.mean(0)
        sp = math.hypot(*vm)
        if sp > VMAX:
            v -= vm * (1 - VMAX / sp)
            sp = VMAX
        drop = AIR + (3.0 * (1 - sp / STOP) if sp < STOP and dr is None else 0.0)
        v -= vm * (1 - math.exp(-drop * h))
        # predict, then project constraints
        old = p.copy()
        p += v * h
        self._unrotate(p, old)                               # before walls: contact stays exact
        x0, y0, x1, y1 = bounds
        L0 = self.rest_len * TENSION
        alpha = EDGE_C / (h * h)
        for _ in range(ITERS):
            # membrane: compliant distance constraints
            e = p[NXT] - p
            L = np.linalg.norm(e, axis=1) + 1e-9
            corr = ((L - L0) / (2 + alpha) / L)[:, None] * e
            p += corr
            p -= corr[PRV]
            # fill: hard area (volume) constraint
            x, y = p[:, 0], p[:, 1]
            grad = 0.5 * np.stack([y[NXT] - y[PRV], x[PRV] - x[NXT]], 1)
            lam = (self.rest_area - polygon_area(p)) / (np.sum(grad * grad) + 1e-9)
            p += lam * grad
            # monitor edges
            np.clip(p[:, 0], x0, x1, out=p[:, 0])
            np.clip(p[:, 1], y0, y1, out=p[:, 1])
        vnew = (p - old) / h
        # contact response: slight bounce + friction for nodes on a wall
        for ax, lo, hi in ((0, x0, x1), (1, y0, y1)):
            for side, lim in ((-1, lo), (1, hi)):
                m = np.abs(p[:, ax] - lim) < 1e-6
                if not m.any():
                    continue
                vin = v[m, ax]
                into = vin * side > 0
                vn = vnew[m, ax]
                if into.any():
                    self.hit += float(np.abs(vin[into]).sum())   # momentum into the wall
                    self._patch_n = max(self._patch_n, int(m.sum()))
                vn[into] = -vin[into] * RESTITUTION
                vnew[m, ax] = vn
                vnew[m, 1 - ax] *= math.exp(-FRICTION * h)
        # no spin: strip the rigid-rotation part of the velocities
        r = p - p.mean(0)
        u = vnew - vnew.mean(0)
        w = np.sum(r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0]) / (np.sum(r * r) + 1e-9)
        vnew[:, 0] += w * r[:, 1]
        vnew[:, 1] -= w * r[:, 0]
        v[:] = vnew

    def step(self, dt, bounds):
        """Advance by dt (fixed substeps). Returns False once it has settled (sleep)."""
        self.acc += dt
        self.hit, self._patch_n = 0.0, 0
        n = 0
        while self.acc >= H and n < 40:
            self._substep(H, bounds)
            self.acc -= H
            n += 1
        c = self.p.mean(0)
        q = self.p - c
        self.patch = self._patch_n / N
        u = self.v - self.v.mean(0)
        self.jiggle = float(np.sqrt((u * u).sum(1).mean()))
        self.stretch = float(np.abs(np.linalg.norm(q, axis=1) - self.R * SKIN).max() / self.R)
        # asleep once everything has settled for a moment
        if self.drag is None and np.abs(self.v).max() < 3.0 and np.abs(q - self.rest).max() < 0.3:
            self.idle_t += dt
        else:
            self.idle_t = 0.0
        if self.idle_t > 0.25:
            self.p = c + self.rest
            self.v[:] = 0
            self.acc = 0.0
            return False
        return True
