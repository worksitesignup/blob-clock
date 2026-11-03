"""Floating blob clock widget.

A round, water-balloon-soft glass bubble on the desktop holding a gel clock face
(hour, minute and second hands, 12h) full of eyes: one at the hub and one at
each hour tip, each with a mind of its own (see eyes.py). Drag it and
throw it: it keeps its momentum, squashes against the monitor edges and wobbles.
Click to jiggle it, right-click to poke it. Position is saved in config.json.

Modules:
    softbody.py   the physics (membrane + incompressible fill), no GTK
    warp.py       draws the flat clock texture deformed by the skin
    gel.py        the metaball hands / rim / eyes, shaded per pixel
    eyes.py       what each eye does: follow the cursor, glance, wander, blink
    face.py       shadow and frosted glass disc
    sfx.py        physics-driven sound
    blurbehind.py KWin blur behind the glass
    kwin-gaze/    KWin script: reports the cursor so the eyes can follow it
                  (Wayland apps can't see the pointer outside their own window)

This file is the window: placement, input, timers and drawing.
"""

import datetime
import json
import math
import os
import random
import time

import cairo
import gi
import numpy as np

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GtkLayerShell", "0.1")
from gi.repository import Gdk, Gio, GLib, Gtk, GtkLayerShell  # noqa: E402

import blurbehind  # noqa: E402
import eyes  # noqa: E402
import face  # noqa: E402
import gel  # noqa: E402
import sfx  # noqa: E402
import softbody  # noqa: E402
from warp import Warp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config.json")
DEFAULTS = {"x": 60, "y": 80, "radius": 78, "layer": "top"}

PAD = 64            # room around the body for shadow + squash
TICK_DUR = 0.24     # seconds a second-hand tick takes to land
DBUS_XML = """<node><interface name="org.tiger.Clock">
  <method name="Look"><arg type="i" name="x" direction="in"/><arg type="i" name="y" direction="in"/></method>
</interface></node>"""


# ---- config ---------------------------------------------------------------------

def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG) as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass
    return {k: cfg[k] for k in DEFAULTS}


def save_config(cfg):
    try:
        with open(CONFIG, "w") as f:
            json.dump(cfg, f, indent=2)
    except OSError:
        pass


# ---- small animation helpers ------------------------------------------------------

def tick_bounce(p):
    """Ease-out with one soft overshoot: 0 -> ~1.08 -> 1."""
    c = 1.9
    q = p - 1
    return 1 + (c + 1) * q ** 3 + c * q ** 2


class Spring:
    __slots__ = ("x", "v")

    def __init__(self, x=0.0):
        self.x, self.v = x, 0.0

    def step(self, target, dt, k, c):
        n = max(1, int(math.ceil(dt / 0.004)))
        h = dt / n
        for _ in range(n):
            self.v += (-k * (self.x - target) - c * self.v) * h
            self.x += self.v * h

    def near(self, target, eps=0.01, veps=0.05):
        return abs(self.x - target) < eps and abs(self.v) < veps


# ---- the window -------------------------------------------------------------------

class Clock(Gtk.Window):
    def __init__(self):
        super().__init__(title="Blob Clock")
        self.cfg = load_config()
        self.R = float(self.cfg["radius"])
        self.W = self.H = int(2 * (self.R + PAD))
        self.cx = self.cy = PAD + self.R                     # body centre, window-local

        # state
        self.ox, self.oy = int(self.cfg["x"]), int(self.cfg["y"])   # window origin on the output
        self.body = softbody.SoftBody(self.R, (self.ox + self.cx, self.oy + self.cy))
        self.warp = Warp(self.body.rest, self.W, (self.cx, self.cy), self.R + PAD - 2)
        self.appear = Spring(0.0)      # pop-in scale
        self.sec = 0.0                 # second-hand angle, degrees
        self.cursor = None             # global pointer position from KWin
        now = time.monotonic()
        self.eye_layout = gel.eye_layout(self.R)
        self.eyes = [eyes.Eye((x, y), now) for x, y, _th, _r in self.eye_layout]
        self.eye_states = tuple(e.state() for e in self.eyes)
        self._eyes_id = None
        self.awake = True              # physics running
        self.bounds = None
        self.press = None              # pointer press bookkeeping
        self.tick_id = None
        self.last_t = None
        self._tex = self._tex_key = None
        self._bg = self._bg_key = None
        self._glass = None             # shadow + glass disc never change: drawn once

        self._setup_window()
        self.show_all()

        sfx.init()
        self.body.kick(lambda a: 60.0 * np.sin(3 * a))      # pop in like the dock's bubbles
        self.wake()
        self._last_sec = None
        GLib.timeout_add(100, self._poll)
        self._sweep_fps = None
        self._sweep_id = GLib.timeout_add(30, self._sweep)
        self._own_dbus()
        self._eyes_t = time.monotonic()
        self._run_eyes(0)

    def _setup_window(self):
        GtkLayerShell.init_for_window(self)
        GtkLayerShell.set_namespace(self, "tiger-clock")
        layer = GtkLayerShell.Layer.BOTTOM if self.cfg["layer"] == "bottom" else GtkLayerShell.Layer.OVERLAY
        GtkLayerShell.set_layer(self, layer)
        GtkLayerShell.set_anchor(self, GtkLayerShell.Edge.TOP, True)
        GtkLayerShell.set_anchor(self, GtkLayerShell.Edge.LEFT, True)
        GtkLayerShell.set_exclusive_zone(self, -1)
        GtkLayerShell.set_keyboard_mode(self, GtkLayerShell.KeyboardMode.NONE)
        self._place()

        visual = self.get_screen().get_rgba_visual()
        if visual is not None:
            self.set_visual(visual)
        self.set_app_paintable(True)

        self.area = Gtk.DrawingArea()
        self.area.set_size_request(self.W, self.H)
        self.area.add_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK
                             | Gdk.EventMask.POINTER_MOTION_MASK)
        self.area.connect("draw", self.on_draw)
        self.area.connect("button-press-event", self.on_press)
        self.area.connect("button-release-event", self.on_release)
        self.area.connect("motion-notify-event", self.on_motion)
        self.add(self.area)
        self.connect("realize", self._on_realize)
        self.connect("map-event", self._on_map)
        self.connect("destroy", Gtk.main_quit)

    # ---- placement ------------------------------------------------------------------

    def _place(self):
        GtkLayerShell.set_margin(self, GtkLayerShell.Edge.LEFT, int(self.ox))
        GtkLayerShell.set_margin(self, GtkLayerShell.Edge.TOP, int(self.oy))

    def _follow_body(self):
        """Keep the window centred on the ball."""
        c = self.body.centre
        nx, ny = round(c[0] - self.cx), round(c[1] - self.cy)
        if (nx, ny) != (self.ox, self.oy):
            self.ox, self.oy = nx, ny
            self.cfg["x"], self.cfg["y"] = nx, ny
            self._place()

    def _on_map(self, *_):
        # real frosted glass: KWin blurs whatever is behind the face
        def blur():
            print("blur-behind:", blurbehind.blur_circle(self, self.cx, self.cy, self.R), flush=True)
            return False
        GLib.idle_add(blur)
        return False

    def _on_realize(self, *_):
        # only the glass is clickable; everything else passes through
        reg = cairo.Region(cairo.RectangleInt(PAD, PAD, int(2 * self.R), int(2 * self.R)))
        self.get_window().input_shape_combine_region(reg, 0, 0)

    def _get_bounds(self):
        if self.bounds is None:
            disp = self.get_display()
            gw = self.get_window()
            mon = (disp.get_monitor_at_window(gw) if gw else None) or disp.get_monitor(0)
            g = mon.get_geometry()
            self.bounds = (0.0, 0.0, float(g.width), float(g.height))
        return self.bounds

    # ---- input ------------------------------------------------------------------------

    def _screen(self, ev):
        """Pointer position on the output (window-local + where we put the window)."""
        return np.array([self.ox + ev.x, self.oy + ev.y])

    def on_press(self, _w, ev):
        if ev.button == 1:
            pt = self._screen(ev)
            self.body.grab(pt)
            self.press = {"pt": pt, "last": pt, "t": time.monotonic(), "pv": np.zeros(2), "moved": False}
            self.wake()
        elif ev.button == 3:                                  # a poke from the top
            self.body.kick(lambda a: 90.0 * np.exp(-((np.angle(np.exp(1j * (a + np.pi / 2)))) / 0.7) ** 2))
            self.wake()
        return True

    def on_motion(self, _w, ev):
        pr = self.press
        if pr is None:
            return False
        pt = self._screen(ev)
        if not pr["moved"] and np.abs(pt - pr["pt"]).sum() < 3:
            return True
        pr["moved"] = True
        now = time.monotonic()
        dt = max(1e-3, now - pr["t"])
        a = min(1.0, dt / 0.03)
        pr["pv"] = pr["pv"] * (1 - a) + (pt - pr["last"]) / dt * a   # smoothed hand velocity
        pr["last"], pr["t"] = pt, now
        self.body.drag_to(pt, pr["pv"])
        self.wake()
        return True

    def on_release(self, _w, ev):
        if ev.button != 1 or self.press is None:
            return False
        moved, self.press = self.press["moved"], None
        self.body.release()
        if not moved:                                         # a click: jiggle
            self.body.kick(lambda a: 70.0 * np.sin(4 * a))
        self.wake()
        return True

    # ---- gaze (cursor from the kwin-gaze script) -------------------------------------

    def _own_dbus(self):
        node = Gio.DBusNodeInfo.new_for_xml(DBUS_XML)

        def on_bus(conn, _name):
            conn.register_object("/org/tiger/Clock", node.interfaces[0], self._on_dbus_call, None, None)
        Gio.bus_own_name(Gio.BusType.SESSION, "org.tiger.Clock", Gio.BusNameOwnerFlags.REPLACE,
                         on_bus, None, None)

    def _on_dbus_call(self, _conn, _sender, _path, _iface, method, params, invocation):
        if method == "Look":                                # KWin only sends while it moves
            self.cursor = params.unpack()
            now = time.monotonic()
            for e in self.eyes:
                e.mouse_moved(now)
            self._run_eyes(0)
        invocation.return_value(None)

    # ---- eyes -------------------------------------------------------------------------

    def _to_cursor(self):
        """Cursor relative to the body centre, in output px (None until KWin reports one)."""
        if self.cursor is None:
            return None
        disp = self.get_display()
        gw = self.get_window()
        mon = (disp.get_monitor_at_window(gw) if gw else None) or disp.get_monitor(0)
        g = mon.get_geometry()
        return np.array([self.cursor[0] - g.x, self.cursor[1] - g.y], float) - self.body.centre

    def _run_eyes(self, delay_ms):
        """(Re)arm the single eye timer."""
        if self._eyes_id is not None:
            GLib.source_remove(self._eyes_id)
        self._eyes_id = GLib.timeout_add(max(1, int(delay_ms)), self._eyes_tick)

    def _eyes_tick(self):
        self._eyes_id = None
        now = time.monotonic()
        dt, self._eyes_t = min(0.05, now - self._eyes_t), now
        to_cursor = self._to_cursor()
        busy = [e.step(now, dt, to_cursor) for e in self.eyes]
        states = tuple(e.state() for e in self.eyes)
        if states != self.eye_states:
            old, self.eye_states = self.eye_states, states
            if self.tick_id is None:                          # still: repaint only the eyes that moved
                for (x, y, _th, re), a, b in zip(self.eye_layout, old, states):
                    if a != b:
                        m = int(re) + 4
                        self.area.queue_draw_area(int(self.cx + x) - m, int(self.cy + y) - m, 2 * m, 2 * m)
        if any(e.blink_t0 is not None for e in self.eyes):
            self._run_eyes(16)                                # blinks are quick
        elif any(busy):
            self._run_eyes(33)                                # 30 fps is plenty for gazes
        else:                                                 # all still: sleep till the next event
            nxt = min(e.next_event(now) for e in self.eyes)
            self._run_eyes((nxt - now) * 1000 + 5)
        return False

    # ---- physics frames -------------------------------------------------------------

    def wake(self):
        if self.tick_id is None:
            self.last_t = None
            self.tick_id = self.area.add_tick_callback(self.on_tick)

    def on_tick(self, _w, _clock):
        now = time.monotonic()
        dt = 0.016 if self.last_t is None else min(now - self.last_t, 0.05)
        self.last_t = now
        self.appear.step(1.0, dt, 300, 17)
        b = self.body
        self.awake = b.step(dt, self._get_bounds())
        sfx.contact(b.hit, b.patch)
        sfx.jiggle(b.jiggle, b.stretch)
        self._follow_body()
        if self._eyes_id is not None and self.cursor is not None:
            self._run_eyes(0)                                 # eyes keep tracking while it flies
        if not self.awake:
            save_config(self.cfg)
        self.area.queue_draw()
        if self.awake or not self.appear.near(1.0, 0.002, 0.01):
            return GLib.SOURCE_CONTINUE
        self.tick_id = None                                   # at rest: sleep until poked
        return GLib.SOURCE_REMOVE

    # ---- clock timers -----------------------------------------------------------------

    def _poll(self):
        now = datetime.datetime.now()
        if now.second != self._last_sec:
            self._last_sec = now.second
            if now.second == 0 and now.minute == 0:           # hourly jiggle
                self.body.kick(lambda a: 40.0 * np.sin(3 * a))
                self.wake()
        return True

    def _sweep(self):
        """Second hand: a bouncy tick each second, on its own timer (moving or not)."""
        now = time.monotonic()
        sec = datetime.datetime.now().second
        if sec != getattr(self, "_tick_sec", None):
            self._tick_sec, self._tick_t0, self._tick_to = sec, now, sec * 6.0
        p = (now - self._tick_t0) / TICK_DUR
        if p < 1.0 or self.sec != self._tick_to:
            self.sec = self._tick_to - 6.0 * (1.0 - tick_bounce(min(p, 1.0)))
            if self.tick_id is None:                          # otherwise frames are drawing anyway
                self.area.queue_draw()
        self._set_sweep_rate(24 if p < 1.0 else 4)
        return True

    def _set_sweep_rate(self, fps):
        """Run the timer fast only while a tick is animating."""
        if self._sweep_fps == fps:
            return
        self._sweep_fps = fps
        GLib.source_remove(self._sweep_id)
        self._sweep_id = GLib.timeout_add(1000 // fps, self._sweep)

    # ---- drawing ------------------------------------------------------------------------

    def on_draw(self, _widget, cr):
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.set_source_rgba(0, 0, 0, 0)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)
        ap = max(0.0, self.appear.x)
        if ap < 0.01:
            return
        if ap > 0.999 and not self.awake:
            self._draw_at_rest(cr)
        else:
            cr.save()
            cr.translate(self.cx, self.cy)
            cr.scale(ap, ap)
            cr.translate(-self.cx, -self.cy)
            self.warp.draw(cr, self._texture(), self.body.p - self.body.centre)
            cr.restore()

    def _draw_at_rest(self, cr):
        """Still: cached glass + shadow, live hands on top."""
        t = datetime.datetime.now()
        key = t.strftime("%H:%M")
        if self._bg_key != key:
            self._bg_key = key
            self._bg = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.W, self.H)
            bc = cairo.Context(self._bg)
            bc.set_source_surface(self._glass_layer(), 0, 0)
            bc.paint()
        cr.set_source_surface(self._bg, 0, 0)
        cr.paint()
        self._draw_hands(cr)

    def _texture(self):
        """The whole clock, undeformed, in an offscreen surface (the frame the warp maps)."""
        t = datetime.datetime.now()
        key = (t.hour, t.minute, round(self.sec, 1), self.eye_states)
        if self._tex_key == key:
            return self._tex                                  # hands haven't moved: reuse
        self._tex_key = key
        if self._tex is None:
            self._tex = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.W, self.H)
        tc = cairo.Context(self._tex)
        tc.set_operator(cairo.OPERATOR_SOURCE)
        tc.set_source_surface(self._glass_layer(), 0, 0)
        tc.paint()
        tc.set_operator(cairo.OPERATOR_OVER)
        self._draw_hands(tc)
        return self._tex

    def _glass_layer(self):
        if self._glass is None:
            self._glass = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.W, self.H)
            self._draw_glass(cairo.Context(self._glass))
        return self._glass

    def _draw_glass(self, cr):
        cx, cy, R = self.cx, self.cy, self.R
        face.soft_shadow(cr, cx, cy, R)
        face.glass_disc(cr, [(cx + x, cy + y) for x, y in self.body.rest], cy - R - 8, cy + R + 8)

    def _draw_hands(self, cr):
        t = datetime.datetime.now()
        hh = (t.hour % 12 + t.minute / 60 + t.second / 3600) * 30
        mm = (t.minute + t.second / 60) * 6
        cr.save()
        cr.translate(self.cx, self.cy)
        cr.arc(0, 0, self.R - 0.5, 0, 2 * math.pi)           # gel fills right to the glass edge
        cr.clip()
        gel.hands_blob(cr, self.R, hh, mm, self.sec, self.eye_states)
        cr.restore()


if __name__ == "__main__":
    Clock()
    Gtk.main()
