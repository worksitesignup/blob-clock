"""The clock's gel: rim, hands and eye as one metaball surface, shaded per pixel.

Everything here is pure numpy + cairo and independent of GTK:

    hands_blob(cr, R, hour, minute, second, eyes)   draw the gel centred on (0, 0)
    eye_layout(R)                                   where the eyes sit

`eyes` is one (blink, gaze_x, gaze_y) per eye: the hub eye first, then one at
each hour tip from 12 o'clock clockwise. Changing only the eyes repaints just
their small patches, so a face full of busy eyes stays cheap.

The field is a sum of metaball terms (rim ring, hub bulb, one lobe per hand),
turned into a smooth alpha via a log-field signed distance. Lighting comes from
a height map built from that alpha (plus sculpted eyelid relief); colour is
picked per pixel by which part's field dominates (rim blue, hub/hour/minute
black, second hand red), and the eyes are painted into the hub bulb and the
blue rim's hour tips; the gel around each one is its eyelid.
"""

import math

import cairo
import numpy as np

def _up2(a, W):
    """2x upsample (bilinear-ish) of an (h, w, c) array to (W, W, c)."""
    u = a.repeat(2, 0).repeat(2, 1)
    u = (u + np.roll(u, 1, 0)) * 0.5
    u = (u + np.roll(u, 1, 1)) * 0.5
    if u.shape[0] < W:
        u = np.pad(u, [(0, W - u.shape[0]), (0, W - u.shape[1]), (0, 0)], mode="edge")
    return u


def _box(a, r):
    """Separable box blur of radius r (edge-clamped)."""
    for axis in (0, 1):
        p = np.pad(a, [(r + 1, r) if ax == axis else (0, 0) for ax in (0, 1)], mode="edge")
        c = np.cumsum(p, axis=axis)
        hi = np.take(c, np.arange(2 * r + 1, c.shape[axis]), axis=axis)
        lo = np.take(c, np.arange(0, c.shape[axis] - 2 * r - 1), axis=axis)
        a = (hi - lo) / (2 * r + 1)
    return a


def _seg_dist2(px, py, x1, y1, with_t=False):
    """Squared distance from pixel grid to the segment (0,0)-(x1,y1)."""
    L2 = x1 * x1 + y1 * y1
    t = np.clip((px * x1 + py * y1) / L2, 0.0, 1.0)
    dx, dy = px - t * x1, py - t * y1
    if with_t:
        return dx * dx + dy * dy, t
    return dx * dx + dy * dy


_light = np.array([-0.45, -0.65, 0.62])
_light /= np.linalg.norm(_light)
_half = _light + np.array([0, 0, 1.0])
_half /= np.linalg.norm(_half)


_blob_cache = {}
_eye_light = {}     # eye lid lighting, kept across renders: (eye, blink, gel under it) -> terms
_rim_cache = {}


def _rim_field(R, half, px, py, term):
    """Static part of the gel: a thin rim ring with a bump at every hour."""
    key = (R, half)
    if key not in _rim_cache:
        rad = np.sqrt(px * px + py * py)
        th = np.arctan2(px, -py)                        # 0 at 12 o'clock, clockwise
        # inner edge: one band that swells into soft rounded points at each hour
        bump = np.zeros_like(rad)
        for i in range(12):
            d = np.angle(np.exp(1j * (th - i * math.pi / 6)))
            depth = R * (0.25 if i % 3 == 0 else 0.15)
            bump = np.maximum(bump, depth * np.exp(-(d / 0.15) ** 2))
        r_in = R * 0.86 - bump
        r_out = R * 1.0
        # signed distance to the band (negative inside), turned into a field
        sd = np.maximum(r_in - rad, rad - r_out)
        k = R * 0.17                                    # gooeyness of the blend with the hands
        f = ((k / np.maximum(k + sd, k * 0.08)) ** 2).astype(np.float32)
        # quadrant points end in a small round bulb instead of a point
        for i in (0, 3, 6, 9):
            a = i * math.pi / 6
            rt = R * (0.86 - 0.25) + R * 0.035
            f = f + term((px - rt * math.sin(a)) ** 2 + (py + rt * math.cos(a)) ** 2, R * 0.065)
        _rim_cache.clear()
        _rim_cache[key] = f
    return _rim_cache[key]


def hands_blob(cr, R, hour, minute, second, eyes):
    # the hub eye moves in fine steps; the tiny tip eyes can't show finer than 0.1
    eyes = tuple((round(b, 2), round(gx, 2 if k == 0 else 1), round(gy, 2 if k == 0 else 1))
                 for k, (b, gx, gy) in enumerate(eyes))
    key = (R, round(hour, 1), round(minute, 1), None if second is None else round(second, 1))
    c = _blob_cache
    if c.get("k") != key:
        c["k"] = key
        c["v"] = _render_blob(R, hour, minute, second, eyes)
        c["eyes"] = eyes
    elif c["eyes"] != eyes:                            # only eyes moved: repaint their patches
        for k, (old, new) in enumerate(zip(c["eyes"], eyes)):
            if old != new:
                _paint_eye(k, new)
        c["eyes"] = eyes
        c["v"][2].mark_dirty()
    half, ssurf, surf = c["v"]
    cr.set_source_surface(ssurf, -half + 1.5, -half + 3.5)
    cr.paint()
    cr.set_source_surface(surf, -half, -half)
    cr.paint()


#                         rim (blue)          hour/min (black)      second (red)
_PAL_DEEP = np.array([[0.05, 0.28, 0.78], [0.02, 0.02, 0.03], [0.72, 0.03, 0.03]], np.float32)
_PAL_LITE = np.array([[0.45, 0.80, 1.00], [0.30, 0.31, 0.34], [1.00, 0.42, 0.38]], np.float32)
_PAL_RIM = np.array([[0.06, 0.21, 0.33], [0.08, 0.08, 0.09], [0.25, 0.05, 0.04]], np.float32)


EYE_R = 0.17          # hub bulb radius (fraction of R); the eye lives in it
TIP_R = 0.93          # hour eyes: centred in the rim band (0.86R..1.0R)
TIP_EYE = 0.068       # hour-eye radius (fraction of R): fills most of the rim band
EYE_OPEN = 0.62       # lid opening height vs. eye width (lower = more almond)
#                    sclera               iris                  pupil
_EYE_DEEP = np.array([[0.78, 0.79, 0.84], [0.12, 0.30, 0.55], [0.01, 0.01, 0.02]], np.float32)
_EYE_LITE = np.array([[1.00, 1.00, 1.00], [0.36, 0.62, 0.90], [0.14, 0.15, 0.18]], np.float32)


def eye_layout(R):
    """(x, y, theta, radius) of every eye, px from the centre. theta: the almond's long
    axis points along (sin theta, -cos theta), so the tip eyes radiate like a sunburst."""
    out = [(0.0, 0.0, math.pi / 2, R * EYE_R * 0.86)]       # hub: lying flat
    for i in range(12):
        th = i * math.pi / 6
        out.append((R * TIP_R * math.sin(th), -R * TIP_R * math.cos(th), th, R * TIP_EYE))
    return out


def _local(dx, dy, th):
    """Rotate into an eye's frame: x along its long axis, y across it."""
    s, c = math.sin(th), math.cos(th)
    return dx * s - dy * c, dx * c + dy * s


def _eye_height(px, py, re, blink):
    """Height (px) the eye adds to the gel: recessed eyeball dome, rolled lid rims, a crease.
    px, py are in the eye's own frame."""
    rr2 = px * px + py * py
    inside = np.clip(1 - (px / re) ** 2, 0, 1)
    h = re * EYE_OPEN * (1 - blink) * inside ** 0.6
    ay = np.abs(py)
    in_eye = np.clip((re - np.sqrt(rr2)) / 1.5 + 0.5, 0, 1)
    opening = np.clip((h - ay) / 1.0 + 0.5, 0, 1) * in_eye
    dome = 0.28 * np.sqrt(np.clip(re * re - rr2, 0, None))      # the eyeball under the lids
    ball = opening * (dome - 1.4)                               # sits a little behind the lids
    lid_edge = np.exp(-((ay - h) / 1.4) ** 2)                   # rolled rim along each lid
    lids = (1 - opening) * (1.8 * lid_edge + 0.6) * in_eye ** 0.5
    seam = -1.0 * blink * np.exp(-(py / 0.9) ** 2) * inside      # where the lids meet
    return ball + lids + seam


GAZE_X, GAZE_Y = 0.42, 0.22   # how far the iris can travel (fraction of eye radius)


def _eye_paint(deep, lite, px, py, re, blink, gaze=(0.0, 0.0)):
    """Paint an eye into the black hub gel. The surrounding gel is the eyelid.

    Same surface, same lighting: only the colour underneath changes, so the eye
    bulges and shines with the hands it is part of. px, py, gaze are in the eye's
    own frame. Returns the new (deep, lite) and the opening mask.
    """
    rr = np.sqrt(px * px + py * py)
    inside = np.clip(1 - (px / re) ** 2, 0, 1)
    h = re * EYE_OPEN * (1 - blink) * inside ** 0.6          # lid opening: almond with rounded corners
    open_ = np.clip(h - np.abs(py) + 0.5, 0, 1) * np.clip((re - rr) + 0.5, 0, 1)
    open_ = open_[..., None]                               # geometry only, hands don't trim it
    # the iris + pupil slide with the gaze; the lids and sclera stay put
    qx, qy = px - gaze[0] * GAZE_X * re, py - gaze[1] * GAZE_Y * re
    rr = np.sqrt(qx * qx + qy * qy)
    ri, rp = re * 0.60, re * 0.27
    iris = np.clip(ri - rr + 0.5, 0, 1)[..., None]
    pupil = np.clip((rp - rr) / 0.8 + 0.5, 0, 1)[..., None]   # soft pupil edge
    # iris texture: radial fibres (layered, so they look organic, not striped),
    # a lighter collarette ring around the pupil and a dark limbal ring at the edge
    ang = np.arctan2(qy, qx)
    t = np.clip((rr - rp) / (ri - rp), 0, 1)                  # 0 at pupil -> 1 at limbus
    # few, wide strands: at real size the iris is only ~6 px, finer ones vanish
    fib = (0.65 * np.cos(ang * 14 + 1.5 * t) + 0.35 * np.cos(ang * 23 - 1.0 * t + 1.3))
    fib = fib * (0.5 + 0.5 * t)                               # fibres fan out toward the edge
    collar = np.exp(-((t - 0.10) / 0.18) ** 2)                # darkening hugging the pupil
    limbus = np.clip((t - 0.72) / 0.28, 0, 1) ** 1.3
    # real blue irises: pale grey-blue stroma fibres over a mid-blue base,
    # gaps only a little darker; never black
    pale = (np.clip(fib, 0, 1) * 0.55)[..., None]             # fibres lift toward pale
    gap = np.clip(1.0 + 0.15 * np.clip(fib, -1, 0), 0, 1)     # between fibres: slightly deeper
    shade_i = (gap * (1 - 0.30 * collar) * (1 - 0.45 * limbus))[..., None]
    warm = (collar * 0.30)[..., None]                         # faint amber near the pupil
    amber_d = np.array([0.30, 0.24, 0.12], np.float32)
    amber_l = np.array([0.62, 0.52, 0.30], np.float32)
    pale_d = np.array([0.42, 0.52, 0.62], np.float32)
    pale_l = np.array([0.78, 0.88, 0.96], np.float32)
    base_d = _EYE_DEEP[1] * (1 - warm) + amber_d * warm
    base_l = _EYE_LITE[1] * (1 - warm) + amber_l * warm
    limb = np.array([0.10, 0.18, 0.30], np.float32)          # limbal ring: dark slate, not black
    lb = (limbus * 0.6)[..., None]
    iris_d = ((base_d * (1 - pale) + pale_d * pale) * shade_i) * (1 - lb) + limb * 0.6 * lb
    iris_l = ((base_l * (1 - pale) + pale_l * pale) * shade_i) * (1 - lb) + limb * lb
    # pupil: near black, a touch deeper in the middle
    pdepth = np.clip(rr / rp, 0, 1)[..., None]
    pup_d = _EYE_DEEP[2] * (0.6 + 0.4 * pdepth)
    pup_l = _EYE_LITE[2] * (0.5 + 0.5 * pdepth)
    def mix(scl, ir, pu):
        return scl * (1 - iris) + (ir * (1 - pupil) + pu * pupil) * iris
    # sclera: a little shadow near the lids so the ball reads as round
    scl_sh = np.clip(1 - 0.25 * (np.abs(py) / (re * EYE_OPEN + 1e-6)) ** 2, 0.7, 1)[..., None]
    eye_d = mix(_EYE_DEEP[0] * scl_sh, iris_d, pup_d)
    eye_l = mix(_EYE_LITE[0] * scl_sh, iris_l, pup_l)
    return deep * (1 - open_) + eye_d * open_, lite * (1 - open_) + eye_l * open_, open_


def _light_terms(al, R, rb, extra=None):
    """Shading terms (diffuse, specular, rim) from the gel's alpha (+ sculpted detail)."""
    z = al
    for _ in range(3):
        z = _box(z, rb)
    z = np.sqrt(np.clip(z, 0, 1)) * al
    z = z * R * 0.20
    if extra is not None:                              # sculpted detail (eyelids) on the gel
        z = z + extra * al
    zy, zx = np.gradient(z)
    nz = 1.0 / np.sqrt(zx * zx + zy * zy + 1)
    nx, ny = -zx * nz, -zy * nz
    diff = np.clip(nx * _light[0] + ny * _light[1] + nz * _light[2], 0, 1)
    spec = np.clip(nx * _half[0] + ny * _half[1] + nz * _half[2], 0, 1) ** 60
    rim = (1 - nz) ** 2
    return np.stack([diff, spec, rim], -1).astype(np.float32)


_TINT = np.array([0.0, 0.03, 0.06], dtype=np.float32)


def _compose(shade, deep, lite, rimc, zth, a8, img):
    """Final colour from shading + palette + zebra stripes, written into BGRA `img`."""
    diff, spec, rim = shade[..., 0:1], shade[..., 1:2], shade[..., 2:3]
    col = np.clip(deep + (lite - deep) * (0.30 + 0.70 * diff) + rim * rimc + spec, 0, 1)
    col = col * (1 - 0.22 * zth) + _TINT * zth
    img[..., 0] = (col[..., 2] * a8 * 255).astype(np.uint8)
    img[..., 1] = (col[..., 1] * a8 * 255).astype(np.uint8)
    img[..., 2] = (col[..., 0] * a8 * 255).astype(np.uint8)
    img[..., 3] = (a8 * 255).astype(np.uint8)


def _paint_eye(k, state):
    """(Re)paint eye k into the cached gel image: relight its lids, paint its eyeball."""
    c = _blob_cache
    R, half, W, M, rb = c["R"], c["half"], c["W"], c["M"], c["rb"]
    ex, ey, th, re = c["layout"][k]
    blink, gx, gy = state
    X, Y = half + ex, half + ey
    o = re + M + 3                                     # lighting needs a margin around the eye
    x0, x1 = max(0, int(X - o)), min(W, int(X + o) + 1)
    y0, y1 = max(0, int(Y - o)), min(W, int(Y + o) + 1)
    px, py = c["px"][y0:y1, x0:x1], c["py"][y0:y1, x0:x1]
    lx, ly = _local(px - ex, py - ey, th)
    # lighting depends on the lids and the gel under the eye, not the gaze: keyed on
    # exactly those, so it survives second-hand ticks and stays exact
    al = c["alpha"][y0:y1, x0:x1]
    lk = (k, blink, R, al.tobytes())
    sub = _eye_light.get(lk)
    if sub is None:
        sub = _light_terms(al, R, rb, _eye_height(lx, ly, re, blink))
        if len(_eye_light) > 600:
            _eye_light.clear()
        _eye_light[lk] = sub
    # only the eye's inner box is written, so neighbouring eyes never overwrite each other
    i = re + 3
    # drop the lighting margin, except where the box hit the image edge (none to drop)
    X0 = max(x0 + (M if x0 > 0 else 0), int(X - i))
    X1 = min(x1 - (M if x1 < W else 0), int(X + i) + 1)
    Y0 = max(y0 + (M if y0 > 0 else 0), int(Y - i))
    Y1 = min(y1 - (M if y1 < W else 0), int(Y + i) + 1)
    ys, xs = slice(Y0, Y1), slice(X0, X1)
    sy, sx = slice(Y0 - y0, Y1 - y0), slice(X0 - x0, X1 - x0)
    deep, lite, open_ = _eye_paint(c["deep"][ys, xs], c["lite"][ys, xs], lx[sy, sx], ly[sy, sx],
                                   re, blink, _local(gx, gy, th))
    zth = c["zth"][ys, xs] * (1 - open_)               # no stripes on the eyeball
    _compose(sub[sy, sx], deep, lite, c["rimc"][ys, xs], zth, c["alpha"][ys, xs], c["img"][ys, xs])


def _render_blob(R, hour, minute, second, eyes):
    """Hour, minute and second hands as one gooey metaball gel, shaded per pixel.

    The first render (or a new hour/minute) does the whole image. After that a
    second-hand tick only recomputes the region the hand left or entered, plus
    the eyes touching it: everything else is reused, pixel for pixel.
    """
    c = _blob_cache
    half = int(R * 1.08)
    n = W = 2 * half
    cc = np.arange(n, dtype=np.float32) + 0.5 - half
    px, py = np.meshgrid(cc, cc)

    def term(d2, r):
        q = np.float32(r * r) / np.maximum(d2, np.float32(0.25))
        return q ** np.float32(1.15)                   # ~1/d^2.3 falloff: long reach, gooey bridges

    def add_lobe(f, ang, d, b, w, px, py):
        sx, sy = math.sin(ang), -math.cos(ang)
        f += term((px - d * sx) ** 2 + (py - d * sy) ** 2, b)
        # tapered gooey neck: fat where it leaves the core, easing into the bubble
        d2, tt = _seg_dist2(px, py, (d - b * 0.4) * sx, (d - b * 0.4) * sy, with_t=True)
        ease = tt * tt * (3 - 2 * tt)
        f += term(d2, w * (1.5 - 0.6 * ease))

    # rim + core + hour + minute change slowly: cache that part of the field
    skey = (R, round(hour, 1), round(minute, 1))
    if c.get("sk") != skey:
        rimf = _rim_field(R, half, px, py, term).astype(np.float32)
        hands = term(px * px + py * py, R * EYE_R)        # the hub bulb: the eye
        add_lobe(hands, math.radians(hour), R * 0.40, R * 0.040, R * 0.018, px, py)
        add_lobe(hands, math.radians(minute), R * 0.55, R * 0.040, R * 0.018, px, py)
        hands = hands.astype(np.float32)
        c["sk"], c["sf"] = skey, rimf + hands
        c["rimf"], c["handf"] = rimf, hands
        c.pop("img", None)                              # static part changed: full render
    sf = c["sf"]
    rb = max(1, int(R * 0.05))
    M = 3 * rb + 3                                     # lighting reach (blur + gradient)

    # the second hand: its lobe only matters inside a box around it (fades within ~R/3)
    box = sec_sub = sec_f = None
    if second is not None:
        ang = math.radians(second)
        tx, ty = R * 0.71 * math.sin(ang) + half, -R * 0.71 * math.cos(ang) + half
        m = int(R * 0.32)
        x0, x1 = max(0, int(min(half, tx)) - m), min(n, int(max(half, tx)) + m)
        y0, y1 = max(0, int(min(half, ty)) - m), min(n, int(max(half, ty)) + m)
        box = (x0, x1, y0, y1)
        sub = sf[y0:y1, x0:x1].copy()
        before = sub.copy()
        add_lobe(sub, ang, R * 0.71, R * 0.040, R * 0.018, px[y0:y1, x0:x1], py[y0:y1, x0:x1])
        sec_sub = sub - before                          # the hand's own field (for colour)
        sec_f = sub                                     # the total field inside its box

    partial = c.get("img") is not None and c.get("W") == W and c.get("R") == R
    if partial:
        changed = [b for b in (c.get("sbox"), box) if b is not None]
        if not changed:
            return c["v"]
        g = M + 3                                      # lighting spreads up to M px from a change
        U = (max(0, min(b[0] for b in changed) - g), min(W, max(b[1] for b in changed) + g),
             max(0, min(b[2] for b in changed) - g), min(W, max(b[3] for b in changed) + g))
    else:
        U = (0, W, 0, W)
    ux0, ux1, uy0, uy1 = U
    us = (slice(uy0, uy1), slice(ux0, ux1))

    def put(out, r, src):
        """Copy the second-hand box array `src` into `out`, which covers region r."""
        if box is not None:
            ix0, ix1 = max(r[0], box[0]), min(r[1], box[1])
            iy0, iy1 = max(r[2], box[2]), min(r[3], box[3])
            if ix0 < ix1 and iy0 < iy1:
                out[iy0 - r[2]:iy1 - r[2], ix0 - r[0]:ix1 - r[0]] = \
                    src[iy0 - box[2]:iy1 - box[2], ix0 - box[0]:ix1 - box[0]]
        return out

    def sec_in(r):
        """The second hand's own field over region r = (x0, x1, y0, y1)."""
        return put(np.zeros((r[3] - r[2], r[1] - r[0]), np.float32), r, sec_sub)

    # signed distance (px, positive inside) from the log of the field: smooth everywhere,
    # so no false edges inside the gel. Gradient needs one pixel of context around U.
    G = (max(0, ux0 - 1), min(W, ux1 + 1), max(0, uy0 - 1), min(W, uy1 + 1))
    f = put(sf[G[2]:G[3], G[0]:G[1]].copy(), G, sec_f)
    lf = np.log(np.clip(np.nan_to_num(f, nan=1e6, posinf=1e6), 1e-6, 1e6))
    gy, gx = np.gradient(lf)
    gm = np.sqrt(gx * gx + gy * gy) + 1e-6
    sd = lf / gm
    a_new = np.clip(sd * 0.5 + 0.5, 0, 1).astype(np.float32)
    a_new = a_new[uy0 - G[2]:uy1 - G[2], ux0 - G[0]:ux1 - G[0]]
    if partial:
        alpha = c["alpha"]
        alpha[us] = a_new
    else:
        alpha = a_new

    def light(al):
        return _light_terms(al, R, rb)

    # lighting: whole gel, or just where the second hand left or entered
    if not partial:
        col = light(alpha)
    else:
        col = c["col"]
        # relight U, with M px of context around it (dropped again below)
        X0, X1, Y0, Y1 = max(0, ux0 - M), min(W, ux1 + M), max(0, uy0 - M), min(W, uy1 + M)
        sub = light(alpha[Y0:Y1, X0:X1])
        mm = M - 1
        ix0, iy0 = (X0 + mm if X0 > 0 else 0), (Y0 + mm if Y0 > 0 else 0)
        ix1, iy1 = (X1 - mm if X1 < W else W), (Y1 - mm if Y1 < W else W)
        col[iy0:iy1, ix0:ix1] = sub[iy0 - Y0:iy1 - Y0, ix0 - X0:ix1 - X0]

    # paint: rim blue, hub + hour/minute black, second hand red (by whose field dominates)
    fr, fh = c["rimf"][us], c["handf"][us]
    fs = sec_in(U)
    # the second hand starts at the hub eye's edge: no red inside the eyeball
    rr_c = np.sqrt(px[us] ** 2 + py[us] ** 2)
    fs = fs * np.clip((rr_c - R * EYE_R * 1.25) / (R * EYE_R * 0.6), 0, 1)
    fr, fh, fs = fr * fr, fh * fh, fs * fs
    tot = fr + fh + fs + 1e-12
    wts = np.stack([fr / tot, fh / tot, fs / tot], -1)
    # zebra sunburst: stripes radiating from the hub, deep/light blue alternating
    zth = c.get("zth")
    if zth is None or zth.shape != (W, W, 1):
        ang = np.arctan2(px, -py)
        band = 0.5 + 0.5 * np.cos(ang * 96)
        band = np.clip((band - 0.5) * 1.6 + 0.5, 0, 1)            # soft, anti-aliased zebra edges
        rr = np.sqrt(px ** 2 + py ** 2) / R
        rim_only = np.clip((rr - 0.60) / 0.10, 0, 1)             # stripes on the rim, not the hands
        zth = (band * rim_only).astype(np.float32)[..., None]
        c["zth"] = zth
    if partial:
        deep, lite, rimc, img = c["deep"], c["lite"], c["rimc"], c["img"]
        deep[us], lite[us], rimc[us] = wts @ _PAL_DEEP, wts @ _PAL_LITE, wts @ _PAL_RIM
    else:
        deep, lite, rimc = wts @ _PAL_DEEP, wts @ _PAL_LITE, wts @ _PAL_RIM
        img = np.zeros((W, W, 4), dtype=np.uint8)        # cairo ARGB32 = BGRA, premultiplied
    # the face without eyes, then the eyes on top (later updates repaint single eyes)
    _compose(col[us], deep[us], lite[us], rimc[us], zth[us], alpha[us], img[us])
    c.update(R=R, half=half, W=W, M=M, rb=rb, layout=eye_layout(R), px=px, py=py,
             alpha=alpha, col=col, deep=deep, lite=lite, rimc=rimc, img=img, sbox=box, ck=skey)
    old = c.get("eyes") if partial else None
    for k, state in enumerate(eyes):
        ex, ey, _th, re = c["layout"][k]
        o = re + M + 3
        touched = not (half + ex + o < ux0 or half + ex - o > ux1 or half + ey + o < uy0 or half + ey - o > uy1)
        if not partial or touched or old is None or old[k] != state:
            _paint_eye(k, state)

    # soft shadow: the alpha, blurred and offset (cheap at half resolution)
    a2 = alpha[:W - W % 2, :W - W % 2].reshape(W // 2, 2, W // 2, 2).mean(axis=(1, 3)).astype(np.float32)
    sh = _up2(_box(_box(a2, 1), 1)[..., None], W)[..., 0]
    if partial:
        simg = c["simg"]
        simg[..., 3] = (sh * 0.28 * 255).astype(np.uint8)
        half_, ssurf, surf = c["v"]
        surf.mark_dirty()
        ssurf.mark_dirty()
        return c["v"]
    simg = np.zeros((W, W, 4), dtype=np.uint8)
    simg[..., 3] = (sh * 0.28 * 255).astype(np.uint8)
    surf = cairo.ImageSurface.create_for_data(memoryview(img), cairo.FORMAT_ARGB32, W, W, W * 4)
    ssurf = cairo.ImageSurface.create_for_data(memoryview(simg), cairo.FORMAT_ARGB32, W, W, W * 4)
    c["simg"] = simg
    c["buf"] = (img, simg)                    # keep the numpy buffers alive
    return half, ssurf, surf
