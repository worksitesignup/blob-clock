"""Ask KWin to blur what's behind part of a GTK3 Wayland window.

Uses the org_kde_kwin_blur protocol (bindings generated into ./wl/kwinproto)
over GTK's own Wayland connection. Fails quietly anywhere that isn't KWin.
"""

import ctypes
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "wl"))

_keep = []          # keep pywayland wrappers alive for the life of the app


def _gpointer(obj):
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return get(obj.__gpointer__, None)


class _Ref:
    """Stand-in proxy so pywayland can pass GTK's wl_surface as an argument."""
    def __init__(self, ptr):
        self._ptr = ptr


def blur_circle(window, cx, cy, r, step=2):
    """Blur behind a circle (window-local coordinates). Returns True on success."""
    try:
        from pywayland import ffi
        from pywayland.client import Display
        from pywayland.protocol.wayland import WlCompositor
        from kwinproto.blur import OrgKdeKwinBlurManager
        from kwinproto.ext_background_effect_v1 import ExtBackgroundEffectManagerV1

        gdk = ctypes.CDLL("libgdk-3.so.0")
        for fn in ("gdk_wayland_display_get_wl_display", "gdk_wayland_window_get_wl_surface"):
            getattr(gdk, fn).restype = ctypes.c_void_p
            getattr(gdk, fn).argtypes = [ctypes.c_void_p]
        gwin = window.get_window()
        disp_ptr = gdk.gdk_wayland_display_get_wl_display(_gpointer(gwin.get_display()))
        surf_ptr = gdk.gdk_wayland_window_get_wl_surface(_gpointer(gwin))
        if not disp_ptr or not surf_ptr:
            print("blur-behind: no wayland pointers", disp_ptr, surf_ptr, file=sys.stderr)
            return False

        display = Display()
        display._ptr = ffi.cast("struct wl_display *", disp_ptr)
        display.disconnect = lambda: None              # GTK owns the connection

        found = {}
        registry = display.get_registry()

        def on_global(reg, name, iface, version):
            if iface in ("ext_background_effect_manager_v1", "org_kde_kwin_blur_manager", "wl_compositor"):
                found[iface] = (name, version)

        registry.dispatcher["global"] = on_global
        display.roundtrip()
        if "ext_background_effect_manager_v1" not in found and "org_kde_kwin_blur_manager" not in found:
            print("blur-behind: compositor offers no blur protocol", file=sys.stderr)
            return False
        comp = registry.bind(found["wl_compositor"][0], WlCompositor,
                             min(4, found["wl_compositor"][1]))
        surface = _Ref(ffi.cast("struct wl_proxy *", surf_ptr))
        region = comp.create_region()
        y = -r
        while y < r:                                   # circle as thin horizontal bands
            yc = y + step / 2
            half = int((r * r - yc * yc) ** 0.5) if abs(yc) < r else 0
            if half > 0:
                region.add(int(cx - half), int(cy + y), 2 * half, step)
            y += step
        if "ext_background_effect_manager_v1" in found:      # standard protocol (KWin 6.6+)
            mgr = registry.bind(found["ext_background_effect_manager_v1"][0], ExtBackgroundEffectManagerV1, 1)
            blur = mgr.get_background_effect(surface)
            blur.set_blur_region(region)
        else:                                                # older KWin
            mgr = registry.bind(found["org_kde_kwin_blur_manager"][0], OrgKdeKwinBlurManager, 1)
            blur = mgr.create(surface)
            blur.set_region(region)
            blur.commit()
        region.destroy()
        display.flush()
        _keep.extend([display, registry, mgr, comp, blur])
        window.queue_draw()                            # next surface commit applies it
        return True
    except Exception as e:                             # never take the clock down
        print("blur-behind unavailable:", e, file=sys.stderr)
        return False
