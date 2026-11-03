"""Draw a flat texture deformed by the soft body's skin.

A coarse grid covers the texture. Each grid point is tied to the rest-shape skin
by mean-value coordinates (computed once), so moving the skin moves the grid
smoothly, interior included. Each grid triangle is then painted with the affine
map from its rest triangle to its deformed one. The grid is only a deformation
cage; it is never visible.
"""

import cairo
import numpy as np


class Warp:
    def __init__(self, rest, size, centre, reach, cells=5):
        """rest: (N, 2) skin at rest relative to its centre; size: texture W (=H);
        centre: skin centre in texture px; reach: radius beyond which cells are skipped."""
        cx, cy = centre
        self.centre = np.array(centre, float)
        poly = rest + self.centre
        xs = np.linspace(0, size, cells + 1)
        gx, gy = np.meshgrid(xs, xs)
        G = np.stack([gx.ravel(), gy.ravel()], 1)
        # mean-value coordinates of every grid point w.r.t. the rest polygon
        d = poly[None, :, :] - G[:, None, :]                 # (g, N, 2)
        r = np.linalg.norm(d, axis=2) + 1e-9
        d1 = np.roll(d, -1, 1)
        cross = d[..., 0] * d1[..., 1] - d[..., 1] * d1[..., 0]
        dot = (d * d1).sum(2)
        t = np.tan(np.arctan2(cross, dot) / 2)               # half signed angle v_i -> v_i+1
        w = (np.roll(t, 1, 1) + t) / r
        self.weights = w / w.sum(1, keepdims=True)
        # triangles that touch anything drawn
        tris = []
        lim = reach ** 2
        n = cells
        for j in range(n):
            for i in range(n):
                a, b = j * (n + 1) + i, j * (n + 1) + i + 1
                c, e = a + n + 1, b + n + 1
                for tri in ((a, b, e), (a, e, c)):
                    if any(((G[k] - (cx, cy)) ** 2).sum() < lim for k in tri):
                        tris.append(tri)
        self.tris = np.array(tris)
        S = np.concatenate([G[self.tris].transpose(0, 2, 1), np.ones((len(tris), 1, 3))], 1)
        self.src_inv = np.linalg.inv(S)                      # per-triangle inverse, done once

    def draw(self, cr, tex, skin):
        """Paint `tex` deformed so its rest skin lands on `skin` (relative to its centre)."""
        dst = self.weights @ (skin + self.centre)
        T = self.tris
        M = dst[T].transpose(0, 2, 1) @ self.src_inv         # (t, 2, 3) affine per triangle
        cr.save()
        cr.set_antialias(cairo.ANTIALIAS_NONE)               # exact tiling: no seams, no overlaps
        pat = cairo.SurfacePattern(tex)
        pat.set_filter(cairo.FILTER_BILINEAR)
        for k in range(len(T)):
            (a, c, e), (b, d, f) = M[k]
            t0, t1, t2 = dst[T[k]]
            cr.save()
            cr.move_to(*t0)
            cr.line_to(*t1)
            cr.line_to(*t2)
            cr.close_path()
            cr.clip()
            m = cairo.Matrix(a, b, c, d, e, f)
            m.invert()
            pat.set_matrix(m)
            cr.set_source(pat)
            cr.paint()
            cr.restore()
        cr.restore()
