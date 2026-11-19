# SPDX-License-Identifier: LGPL-2.1-or-later
"""Procedural leaf textures: a leaf outline description -> the RGBA PNG a Helios leaf prototype uses.

``GenericLeafPrototype`` maps the image onto the blade with ``u`` along the leaf (u = 0 at the base, 1 at the tip)
and ``v`` across it; triangles whose texels are all transparent are dropped, and a texel counts as solid when its
alpha is at least 250. So the image is the blade *mask* (alpha 255 inside the outline, 0 outside, no
anti-aliasing) plus a colour, and its proportions are those of the blade: ``length_px`` columns by
``round(aspect * length_px)`` rows, with ``aspect`` the blade's width / length (set the leaf prototype's
``leaf_aspect_ratio`` to the same value).

Outline (all lengths as fractions of the blade length; ``t`` runs from the blade base, 0, to the tip, 1)::

    half-width(t) = W/2 * p(t),   W = aspect
    p(t) = sin(pi/2 * t/m) ** base                         t <= m            (m = widest_at)
         = 1                                               m < t <= m + plateau
         = cos(pi/2 * (t - m - plateau)/(1 - m - plateau)) ** tip      beyond

``base`` < 1 gives a rounded / truncate base, > 1 a wedge (cuneate) one; ``tip`` > 1 an acuminate tip, < 1 an
obtuse one; ``plateau`` keeps the blade at full width (grass straps). ``lobes`` (``count``, ``depth``, ``sharpness``)
multiplies p by ``1 - depth + depth * |sin(pi * count * t)| ** sharpness`` (sinuses at t = k / count);
``serration`` (``count``, ``depth``) by a forward-pointing saw tooth ``1 - depth * frac(count * t)``;
``asymmetry`` narrows one half. A ``profile`` of (t, p) points, linearly interpolated, replaces the formula.

Colour: ``color`` (RGB in [0, 1]) with a lighter ``midrib_color`` midrib, ``veins`` pairs of lateral veins at
``vein_angle`` degrees to the midrib (``vein_style: pinnate``) or ``veins`` parallel veins (``parallel``, grasses) in
``vein_color``, and the margin darkened by ``margin_shade``. Everything is evaluated in float64 numpy on the pixel
centres, so the image depends only on the description. The PNG is written by :func:`write_png` (8-bit RGBA, no
filtering, zlib level 9, no metadata chunks).

Command line: ``python -m heliades.species.leaf_texture [species ...]`` regenerates the leaf textures of the
species files (``heliades/species/assets/<species>/<name>.png``); ``--check`` only compares.
"""
from __future__ import annotations

import os
import struct
import zlib

import numpy as np

#: key -> (type, default, min, max)
OUTLINE_KEYS = {
    "length_px": (int, 512, 16, 4096),
    "aspect": (float, None, 0.005, 4.0),
    "widest_at": (float, 0.4, 0.01, 0.99),
    "plateau": (float, 0.0, 0.0, 0.98),
    "base": (float, 1.0, 0.05, 10.0),
    "tip": (float, 1.0, 0.05, 10.0),
    "asymmetry": (float, 0.0, 0.0, 0.9),
    "lobes": (dict, None, None, None),
    "serration": (dict, None, None, None),
    "sinus": (dict, None, None, None),
    "profile": (list, None, None, None),
    "color": (list, [0.22, 0.38, 0.13], None, None),
    "midrib_color": (list, None, None, None),
    "midrib_width": (float, 0.012, 0.0, 0.2),
    "vein_color": (list, None, None, None),
    "veins": (int, 0, 0, 100),
    "vein_style": (str, "pinnate", None, None),
    "vein_angle": (float, 50.0, 5.0, 89.0),
    "vein_width": (float, 0.006, 0.0, 0.1),
    "margin_shade": (float, 0.15, 0.0, 1.0),
}
LOBE_KEYS = {"count": (int, None, 1, 50), "depth": (float, None, 0.0, 0.95), "sharpness": (float, 1.0, 0.05, 10.0)}
SERRATION_KEYS = {"count": (int, None, 1, 400), "depth": (float, None, 0.0, 0.5)}
#: basal sinus of a cordate blade (S1): an elliptical notch ``depth`` (fraction of the length) deep and ``width``
#: (fraction of the blade width) wide at the base; the midrib stays solid across it (the petiole's continuation).
SINUS_KEYS = {"depth": (float, None, 0.0, 0.5), "width": (float, 0.3, 0.01, 1.0)}


class LeafOutlineError(ValueError):
    pass


def _check_block(d, keys, where):
    if not isinstance(d, dict):
        raise LeafOutlineError(f"{where}: expected a mapping, got {d!r}")
    out = {}
    for k in d:
        if k not in keys:
            raise LeafOutlineError(f"{where}: unknown key '{k}' (allowed: {', '.join(keys)})")
    for k, (t, default, lo, hi) in keys.items():
        if k not in d or (d[k] is None and default is None):
            if default is None and t in (int, float) and k in ("aspect", "count", "depth"):
                raise LeafOutlineError(f"{where}: '{k}' is required")
            out[k] = default
            continue
        v = d[k]
        if t is float and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        if isinstance(v, bool) or not isinstance(v, t):
            raise LeafOutlineError(f"{where}.{k}: expected {t.__name__}, got {v!r}")
        if lo is not None and not (lo <= v <= hi):
            raise LeafOutlineError(f"{where}.{k}: {v} is outside [{lo}, {hi}]")
        out[k] = v
    return out


def _rgb(v, where):
    if v is None:
        return None
    if not (isinstance(v, list) and len(v) == 3 and all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in v)):
        raise LeafOutlineError(f"{where}: expected three numbers in [0, 1], got {v!r}")
    return np.array(v, np.float64)


def validate_outline(desc: dict, where: str = "leaf_texture") -> dict:
    """Check a description and fill in the defaults."""
    d = _check_block(desc, OUTLINE_KEYS, where)
    if d["lobes"] is not None:
        d["lobes"] = _check_block(dict(d["lobes"]), LOBE_KEYS, where + ".lobes")
    if d["serration"] is not None:
        d["serration"] = _check_block(d["serration"], SERRATION_KEYS, where + ".serration")
    if d["sinus"] is not None:
        d["sinus"] = _check_block(d["sinus"], SINUS_KEYS, where + ".sinus")
    if d["vein_style"] not in ("pinnate", "parallel"):
        raise LeafOutlineError(f"{where}.vein_style: must be pinnate or parallel")
    if d["widest_at"] + d["plateau"] >= 1.0:
        raise LeafOutlineError(f"{where}: widest_at + plateau must be < 1")
    if d["profile"] is not None:
        pts = d["profile"]
        ok = all(isinstance(p, list) and len(p) == 2 and all(isinstance(x, (int, float)) for x in p) for p in pts)
        if not ok or len(pts) < 3:
            raise LeafOutlineError(f"{where}.profile: expected at least three [t, p] pairs")
        ts = [p[0] for p in pts]
        if ts[0] != 0 or ts[-1] != 1 or any(b <= a for a, b in zip(ts, ts[1:])):
            raise LeafOutlineError(f"{where}.profile: t must rise strictly from 0 to 1")
        if any(not 0 <= p[1] <= 1 for p in pts):
            raise LeafOutlineError(f"{where}.profile: p must be in [0, 1]")
    for k in ("color", "midrib_color", "vein_color"):
        _rgb(d[k], f"{where}.{k}")
    return d


def profile(d: dict, t: np.ndarray) -> np.ndarray:
    """p(t), the half-width as a fraction of the maximum, for t in [0, 1] (0 outside)."""
    t = np.asarray(t, np.float64)
    inside = (t >= 0) & (t <= 1)
    tc = np.clip(t, 0.0, 1.0)
    if d["profile"] is not None:
        pts = np.array(d["profile"], np.float64)
        p = np.interp(tc, pts[:, 0], pts[:, 1])
    else:
        m, pl = d["widest_at"], d["plateau"]
        rise = np.sin(0.5 * np.pi * np.clip(tc / m, 0, 1)) ** d["base"]
        fall_t = np.clip((tc - m - pl) / (1.0 - m - pl), 0, 1)
        fall = np.cos(0.5 * np.pi * fall_t) ** d["tip"]
        p = np.where(tc <= m, rise, np.where(tc <= m + pl, 1.0, fall))
    if d["lobes"] is not None:
        lb = d["lobes"]
        p = p * (1.0 - lb["depth"] + lb["depth"] * np.abs(np.sin(np.pi * lb["count"] * tc)) ** lb["sharpness"])
    if d["serration"] is not None:
        sr = d["serration"]
        p = p * (1.0 - sr["depth"] * np.mod(sr["count"] * tc, 1.0))
    return np.where(inside, np.maximum(p, 0.0), 0.0)


def render(desc: dict) -> np.ndarray:
    """The (H, W, 4) uint8 RGBA image (row 0 is the top row) of a validated or raw description."""
    d = validate_outline(desc)
    W = d["length_px"]
    H = max(2, int(round(d["aspect"] * W)))
    aspect = d["aspect"]
    x = (np.arange(W, dtype=np.float64) + 0.5) / W                  # along the blade (u)
    v = (np.arange(H, dtype=np.float64)[::-1] + 0.5) / H              # row 0 = top = v near 1
    X, Vv = np.meshgrid(x, v)
    Y = (Vv - 0.5) * aspect                                          # across, in blade lengths
    half = 0.5 * aspect * profile(d, X)
    half = np.where(Y < 0, half * (1.0 - d["asymmetry"]), half)
    inside = np.abs(Y) <= half

    base = _rgb(d["color"], "color")
    col = np.broadcast_to(base, (H, W, 3)).copy()
    # margin shade: darker towards the edge
    rel = np.where(half > 0, np.abs(Y) / np.maximum(half, 1e-12), 1.0)
    col *= (1.0 - d["margin_shade"] * rel ** 2)[..., None]
    vein_c = _rgb(d["vein_color"], "vein_color")
    if vein_c is None:
        vein_c = np.clip(base * 1.35 + 0.04, 0, 1)
    if d["veins"] > 0:
        if d["vein_style"] == "pinnate":
            s = X - np.abs(Y) / np.tan(np.radians(d["vein_angle"]))
            phase = np.mod(s * d["veins"], 1.0)
            vein = (phase < d["vein_width"] * d["veins"]) & (s > 0)
        else:
            phase = np.mod(np.where(half > 0, np.abs(Y) / np.maximum(half, 1e-12), 0) * d["veins"], 1.0)
            vein = phase < d["vein_width"] * d["veins"] * 2
        col = np.where(vein[..., None], 0.5 * col + 0.5 * vein_c, col)
    mid_c = _rgb(d["midrib_color"], "midrib_color")
    if mid_c is None:
        mid_c = np.clip(base * 1.6 + 0.08, 0, 1)
    midrib = np.abs(Y) <= d["midrib_width"] * aspect * (1.0 - 0.7 * X) + 0.5 * aspect / H
    col = np.where(midrib[..., None], mid_c, col)
    if d["sinus"] is not None:
        hw = 0.5 * aspect * d["sinus"]["width"]
        notch = X < d["sinus"]["depth"] * np.sqrt(np.clip(1.0 - (Y / hw) ** 2, 0.0, 1.0))
        inside = inside & (~notch | midrib)

    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., :3] = np.clip(np.round(col * 255.0), 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(inside, 255, 0).astype(np.uint8)
    rgba[~inside, :3] = rgba[inside, :3].mean(0).round().astype(np.uint8) if inside.any() else 0
    return rgba


def write_png(path: str, rgba: np.ndarray) -> str:
    """A minimal deterministic PNG writer (8-bit RGBA, filter 0, zlib level 9, IHDR/IDAT/IEND only)."""
    rgba = np.ascontiguousarray(rgba, np.uint8)
    H, W, C = rgba.shape
    assert C == 4
    raw = b"".join(b"\x00" + rgba[j].tobytes() for j in range(H))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)
    return path


def read_png_rgba(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGBA"))


def blade_fraction(rgba: np.ndarray) -> float:
    """Solid fraction of the image (alpha >= 250), what Helios's ``solid_fraction`` sees."""
    return float((rgba[..., 3] >= 250).mean())


def _main(argv):
    from .loader import available_species, load_spec
    from .assets import leaf_texture_path
    check = "--check" in argv
    names = [a for a in argv if not a.startswith("--")] or available_species()
    bad = 0
    for name in names:
        spec = load_spec(name)
        for tex, desc in spec.leaf_textures.items():
            path = leaf_texture_path(spec.name, tex)
            img = render(desc)
            if check:
                same = os.path.exists(path) and np.array_equal(read_png_rgba(path), img)
                bad += not same
                print(f"{'ok  ' if same else 'DIFF'} {path}")
            else:
                write_png(path, img)
                print(f"wrote {path} ({img.shape[1]}x{img.shape[0]}, solid {blade_fraction(img):.3f})")
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
