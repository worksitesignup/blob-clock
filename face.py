"""Static drawing pieces of the clock body: the shadow and the frosted glass disc."""

import math

import cairo


def spline(cr, pts):
    """Closed Catmull-Rom spline through the points."""
    n = len(pts)
    cr.new_path()
    cr.move_to(*pts[0])
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        cr.curve_to(p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6,
                    p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6,
                    p2[0], p2[1])
    cr.close_path()


def glass_disc(cr, pts, top, bot, opacity=0.40):
    """The translucent frosted face, filled along the skin outline (KWin blurs behind it)."""
    spline(cr, pts)
    grad = cairo.LinearGradient(0, top, 0, bot)
    grad.add_color_stop_rgba(0.0, 1, 1, 1, 0.97 * opacity)
    grad.add_color_stop_rgba(0.55, 0.97, 0.98, 1.00, 0.94 * opacity)
    grad.add_color_stop_rgba(1.0, 0.93, 0.95, 0.98, 0.94 * opacity)
    cr.set_source(grad)
    cr.fill()


def soft_shadow(cr, cx, cy, R):
    """Shadow for a round object: a wide soft penumbra below it plus a tight contact shadow."""
    cr.save()
    # never paint under the (see-through) face itself
    cr.rectangle(cx - R * 2, cy - R * 2, R * 4, R * 4)
    cr.arc_negative(cx, cy, R * 0.97, 0, -2 * math.pi)
    cr.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    cr.clip()
    # ambient / drop shadow
    g = cairo.RadialGradient(cx, cy + R * 0.09, R * 0.80, cx, cy + R * 0.09, R * 1.24)
    for t in range(11):
        u = t / 10
        g.add_color_stop_rgba(u, 0.02, 0.06, 0.16, 0.34 * math.exp(-4.2 * u * u))
    cr.arc(cx, cy + R * 0.09, R * 1.24, 0, 2 * math.pi)
    cr.set_source(g)
    cr.fill()
    # contact shadow hugging the edge
    c = cairo.RadialGradient(cx, cy + R * 0.025, R * 0.95, cx, cy + R * 0.025, R * 1.06)
    c.add_color_stop_rgba(0.0, 0.0, 0.03, 0.10, 0.30)
    c.add_color_stop_rgba(0.5, 0.0, 0.03, 0.10, 0.10)
    c.add_color_stop_rgba(1.0, 0.0, 0.03, 0.10, 0.0)
    cr.arc(cx, cy + R * 0.025, R * 1.06, 0, 2 * math.pi)
    cr.set_source(c)
    cr.fill()
    cr.restore()
