"""Physics-driven rubber-ball audio.

One continuous mono stream into `pw-play`. The simulation feeds it two numbers:

    contact(impulse)  momentum the skin pushed into a wall during a frame
    jiggle(speed)     RMS speed of the skin relative to its centre (px/s)

Both only *excite* one small bank of damped resonators tuned like a soft rubber
body: wall contact strikes it (a muted "bok"), and the skin's own jiggle rubs
it very faintly. Nothing is sampled or scripted; still = silent.

The writer thread sleeps whenever everything is silent, so an idle clock costs
nothing.
"""

import math
import shutil
import subprocess
import threading
import time

import numpy as np

try:
    from scipy import signal
except ImportError:                                     # sound is optional
    signal = None

SR = 48000
BLOCK = 256                     # ~5 ms
LEAD = int(0.025 * SR)          # how far ahead of real time we stay (latency)

# resonators: (Hz, Q, gain) — a soft rubber body: low, short, slightly inharmonic
MODES = ((330, 14, 1.0), (690, 16, 0.4), (1180, 18, 0.18))   # lighter, higher body
CONTACT_FLOOR = 150.0     # resting/sliding contact below this is silent
CONTACT_FULL = 15000.0    # impulse of a hard throw
CONTACT_TAU = 0.006       # strike release, s
JIGGLE_GAIN = 0.0         # in-air wobble tone (off: only bounces make sound)
WUB_LO, WUB_HI = 190.0, 420.0   # wobble tone pitch: round -> fully squashed
BLOOP_GAIN = 0.55         # the bubbly "bloop" a contact starts
# Bloops borrow the dock's voice: a clean sine (+ a little 2nd harmonic) that
# sweeps upward with a soft liquid vibrato, landing on a major-pentatonic note.
# Which note is pure physics: harder hits climb the scale, broad splats sit lower.
PENTA = (0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24, 26)   # same scale as the dock
BLOOP_ROOT = 196.0        # G3: the dock's notes, an octave down
SWEEP = 2.4               # pitch rises this much over a bloop (dock pop: ~3.6x)
WOBBLE = 0.06             # vibrato depth on a full splat (28 Hz, like the dock)
SNAP_HZ = 3200.0          # tiny bright snap on hard hits
PATCH_FULL = 0.30         # contact patch (fraction of the skin) that counts as a broad splat
STRIKE_GAIN = 0.5         # body strike from a rise in contact force
OUT_GAIN = 3.2


class _Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.impulse = 0.0       # accumulated since last block
        self.jig = 0.0           # latest jiggle speed
        self.jig_s = 0.0         # smoothed jiggle level
        self.cl = 0.0            # contact strike level
        self.stretch = 0.0       # how far the skin is from round (fraction of R)
        self.wub_f, self.wub_ph = WUB_LO, 0.0
        self.gate = 0.0          # slow contact memory: one bloop per hit, not per frame
        self.voices = []         # active bloops: [age_samples, phase, f0, amp, dur, width]
        self.proc = None
        self.patch = 0.0         # widest wall-contact patch this block (fraction of skin)
        self.res = []
        for f, q, g in MODES:
            b, a = signal.iirpeak(f, q, fs=SR)
            self.res.append((b, a, signal.lfilter_zi(b, a) * 0.0, g))
        self.low = signal.butter(2, 400, fs=SR, output="sos")
        self.low_z = np.zeros((self.low.shape[0], 2))
        self.tail = 0.0          # resonator energy estimate, to know when to sleep

    def start(self):
        exe = shutil.which("pw-play")
        if not exe:
            return False
        self.proc = subprocess.Popen(
            [exe, "--raw", "--format=s16", f"--rate={SR}", "--channels=1", "--latency=20ms", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._run, daemon=True).start()
        return True

    # ---- fed from the physics (GTK thread) ----
    def contact(self, imp, patch=0.0):
        with self.lock:
            self.impulse += imp
            self.patch = max(self.patch, patch)
        self.wake.set()

    def jiggle(self, speed, stretch=0.0):
        self.jig = speed
        self.stretch = stretch
        if speed > 15 and JIGGLE_GAIN > 0:
            self.wake.set()

    # ---- audio thread ----
    def _block(self):
        with self.lock:
            imp, self.impulse = self.impulse, 0.0
            patch, self.patch = self.patch, 0.0
        n = BLOCK
        t = np.arange(1, n + 1)
        # wall contact -> one continuous strike envelope: instant attack, smooth release.
        # Physics reports contact once per frame; holding the level and letting it fade
        # sample by sample stops that frame rate turning into a stutter.
        exc = np.zeros(n)
        if imp > CONTACT_FLOOR:
            amp = min(1.0, (imp - CONTACT_FLOOR) / CONTACT_FULL) ** 0.7
            width = min(1.0, patch / PATCH_FULL)
            if amp > 2.0 * self.gate + 0.01:             # a new hit, not the same contact
                # note on the pentatonic scale: impact climbs it, a broad splat pulls down
                i = int(round(np.clip(2 + 9 * amp - 3 * width, 0, len(PENTA) - 1)))
                f0 = BLOOP_ROOT * 2 ** (PENTA[i] / 12)
                self.voices.append([0, 0.0, f0, amp, 0.09 + 0.08 * amp, width])
            elif self.voices:                            # same squash still spreading
                v = self.voices[-1]
                v[3] = max(v[3], amp)
                v[5] = max(v[5], width)
            # strike the body by how much the contact force just rose (no noise)
            rise = amp - self.cl
            if rise > 0:
                k = 96
                exc[:k] = rise * np.hanning(2 * k)[k:] * STRIKE_GAIN
            self.cl = max(self.cl, amp)
            self.gate = max(self.gate, amp)
        self.gate *= math.exp(-n / (0.09 * SR))
        self.cl *= math.exp(-n / (CONTACT_TAU * SR))
        # deformation: the skin's own motion rubs the body; louder the harder it flexes
        target = np.clip((self.jig - 30.0) / 600.0, 0, 1) ** 1.2
        prev, fprev = self.jig_s, self.wub_f
        self.jig_s += (target - self.jig_s) * 0.3
        # pitch follows how stretched the skin is: rounder = lower, squashed = higher
        self.wub_f += (WUB_LO + (WUB_HI - WUB_LO) * min(1.0, self.stretch / 0.35) - self.wub_f) * 0.35
        wub = None
        if JIGGLE_GAIN > 0 and max(prev, self.jig_s) > 1e-4:
            ramp = np.linspace(prev, self.jig_s, n)          # no steps between blocks
            f = np.linspace(fprev, self.wub_f, n)
            ph = self.wub_ph + 2 * np.pi * np.cumsum(f) / SR
            self.wub_ph = float(ph[-1] % (2 * np.pi))
            wub = JIGGLE_GAIN * ramp * (np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph))
        out = np.zeros(n)
        for i, (b, a, z, g) in enumerate(self.res):
            y, z = signal.lfilter(b, a, exc, zi=z)
            self.res[i] = (b, a, z, g)
            out += g * y
        out *= OUT_GAIN
        out += BLOOP_GAIN * self._bloops(n)
        if wub is not None:
            out += wub
        self.tail = max(self.tail * 0.9, float(np.abs(out).max()))
        return out

    def _bloops(self, n):
        """Dock-style bloop per hit: upward sweep, liquid vibrato, sine + 2nd harmonic.

        Note from impact (see contact handling), length from impact, vibrato
        depth from how broad the splat was; hard hits add a tiny high snap.
        """
        out = np.zeros(n)
        keep = []
        for v in self.voices:
            age, ph, f0, amp, dur, width = v
            t = (age + np.arange(n)) / SR
            u = t / dur
            f = f0 * SWEEP ** np.minimum(u * 1.4, 1.0)
            f *= 1 + WOBBLE * (0.3 + 0.7 * width) * np.sin(2 * np.pi * 28 * t) * np.clip(1 - u, 0, 1)
            p = ph + 2 * np.pi * np.cumsum(f) / SR
            e = np.minimum(1.0, t / 0.004) * np.exp(-t / (dur * 0.32))
            sig = np.sin(p) * e + 0.25 * np.sin(2 * p + 0.6) * e * np.clip(1 - u, 0, 1)
            snap = amp ** 2 * np.exp(-t / 0.0018) * np.sin(2 * np.pi * SNAP_HZ * t)
            out += amp * sig + 0.3 * snap
            v[0], v[1] = age + n, float(p[-1] % (2 * np.pi))
            if t[-1] < dur * 2.5:
                keep.append(v)
        self.voices = keep
        return out

    def _run(self):
        t0, sent = time.monotonic(), 0
        quiet = 0
        while True:
            if quiet > SR // 2:                          # half a second of silence: sleep
                self.wake.clear()
                self.wake.wait()
                t0, sent, quiet = time.monotonic(), 0, 0
            ahead = sent - (time.monotonic() - t0) * SR
            if ahead > LEAD:
                time.sleep((ahead - LEAD) / SR)
                continue
            x = self._block()
            quiet = quiet + BLOCK if (self.tail < 1e-4 and (JIGGLE_GAIN == 0 or self.jig_s < 1e-4) and not self.voices) else 0
            pcm = (np.clip(np.tanh(x * 1.4), -1, 1) * 32000).astype("<i2")
            try:
                self.proc.stdin.write(pcm.tobytes())
                self.proc.stdin.flush()
            except (OSError, ValueError):
                return
            sent += BLOCK


_eng = None


def init():
    global _eng
    if signal is None:
        print("sfx unavailable: scipy not installed", flush=True)
        return
    try:
        e = _Engine()
        if e.start():
            _eng = e
    except Exception as ex:                              # sound is optional
        print("sfx unavailable:", ex, flush=True)


def contact(impulse, patch=0.0):
    """impulse: momentum into walls this frame; patch: widest contact, fraction of skin."""
    if _eng is not None and impulse > 0:
        _eng.contact(impulse, patch)


def jiggle(speed, stretch=0.0):
    if _eng is not None:
        _eng.jiggle(speed, stretch)
