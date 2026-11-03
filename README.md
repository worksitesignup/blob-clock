# Blob Clock

A floating, water-balloon-soft glass clock for Wayland desktops (layer-shell), drawn with cairo/GTK3.

The face is a gel full of eyes — one at the hub and one at each hour tip — that glance around, blink, and follow your mouse. Drag it and throw it: it keeps its momentum, squashes against the screen edges and wobbles. Click to jiggle it, right-click to poke it. Bounces make soft rubbery sounds.

Built and tested on **KDE Plasma 6 (Wayland)**. Other layer-shell compositors (Hyprland, Sway) should show the clock, but the frosted-glass blur and the mouse-following eyes use KWin and won't work elsewhere.

## Install (Arch)
```sh
sudo pacman -S --needed python-gobject python-cairo python-numpy python-scipy python-pywayland gtk3 gtk-layer-shell pipewire kwin
```
`python-scipy` + `pipewire` give it sound and `python-pywayland` + `kwin` give the frosted-glass blur and mouse-following eyes. The clock still runs without them.

## Run
```sh
./run
```

## Eyes that follow the mouse (KDE only)
Wayland apps can't see the pointer outside their own window, so a tiny KWin script reports it:
```sh
./install-kwin-script
```
If the eyes still don't track, enable **Blob Clock Gaze** in *System Settings → Window Management → KWin Scripts*.

## Settings
Position is remembered in `config.json` (created on first move). You can also set:
- `radius` — clock size in px (default 78)
- `layer` — `"top"` (above windows) or `"bottom"` (on the desktop, below windows)

## Autostart
Add `run` (full path) as a login script in *System Settings → Autostart*.
