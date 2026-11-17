# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for textured primitives and objects (W1-A).

Real Helios textures (plantarchitecture leaves: RGBA, palette+tRNS, gray without alpha, JPEG),
libpng's own test images (every color type and bit depth), and a few synthetic PNGs written here
(alpha at the 250 threshold in 8 and 16 bit, gray/RGB tRNS keys, low bit depths).
"""
from __future__ import annotations

import math
import os
import struct
import tempfile
import zlib

import numpy as np

from ...cases import _stem, case

HELIOS = "/home/yun/Desktop/plants/third_party/pyhelios_src/helios-core"
_PA = f"{HELIOS}/plugins/plantarchitecture/assets/textures/"
_CORE = f"{HELIOS}/core/lib/images/"
_LIBPNG = f"{HELIOS}/core/lib/libpng/contrib/testpngs/"

TEXTURES = [_PA + f for f in ("TomatoLeaf.png", "CherryTomatoLeaf.png", "AlmondLeaf.png", "GrapeLeaf.png",
                              "BeanLeaf_tip.png", "RiceLeaf_A_0.png", "StrawberryLeaf.png", "AppleBark_classes.png",
                              "AlmondBark.jpg", "SorghumSeed.jpeg")] + [_CORE + "diamond_texture.png",
                                                                        _CORE + "disk_texture.png"]
ALPHA_TEXTURES = [t for t in TEXTURES if t.endswith(".png") and "classes" not in t]


def _pick(r, pool=TEXTURES):
    return pool[int(r.integers(len(pool)))]


# ---- synthetic PNGs ------------------------------------------------------------------------

def write_png(path, samples: np.ndarray, bit_depth: int, color_type: int, trns: bytes | None = None,
              palette: bytes | None = None):
    """Minimal PNG writer (filter 0, no interlace). ``samples`` (H,W[,C]) holds raw sample values."""
    s = np.asarray(samples)
    h, w = s.shape[:2]
    s = s.reshape(h, w, -1)
    if bit_depth == 16:
        rows = [s[y].astype(">u2").tobytes() for y in range(h)]
    elif bit_depth == 8:
        rows = [s[y].astype(np.uint8).tobytes() for y in range(h)]
    else:
        per = 8 // bit_depth
        rows = []
        for y in range(h):
            v = s[y, :, 0].astype(np.uint8)
            v = np.r_[v, np.zeros((-len(v)) % per, np.uint8)].reshape(-1, per)
            sh = (8 - bit_depth * (np.arange(per) + 1)).astype(np.uint8)
            rows.append((v << sh).sum(1).astype(np.uint8).tobytes())
    raw = b"".join(b"\0" + r for r in rows)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bit_depth, color_type, 0, 0, 0))
    if palette is not None:
        out += chunk(b"PLTE", palette)
    if trns is not None:
        out += chunk(b"tRNS", trns)
    out += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(out)


def synthetic_pngs() -> list[str]:
    """Deterministic edge-case PNGs in a temp dir (rewritten each call; content depends only on code)."""
    d = os.path.join(tempfile.gettempdir(), "heliades_w1a_pngs")
    os.makedirs(d, exist_ok=True)
    r = np.random.default_rng(1234)
    H, W = 37, 53
    files = []

    def add(name, *a, **k):
        p = os.path.join(d, name)
        write_png(p, *a, **k)
        files.append(p)

    a8 = r.choice(np.array([0, 128, 248, 249, 250, 251, 255]), (H, W))
    add("rgba8_threshold.png", np.dstack([r.integers(0, 256, (H, W, 3)), a8]), 8, 6)
    a16 = r.choice(np.array([0, 63999, 64000, 64255, 64256, 65535, 63744, 63743]), (H, W))
    add("rgba16_threshold.png", np.dstack([r.integers(0, 65536, (H, W, 3)), a16]), 16, 6)
    add("ga8.png", np.dstack([r.integers(0, 256, (H, W)), a8]), 8, 4)
    add("ga16.png", np.dstack([r.integers(0, 65536, (H, W)), a16]), 16, 4)
    g8 = r.integers(0, 4, (H, W)) * 60
    add("gray8_trns.png", g8, 8, 0, trns=struct.pack(">H", 120))
    g16 = r.choice(np.array([300, 301, 556, 1000]), (H, W))
    add("gray16_trns.png", g16, 16, 0, trns=struct.pack(">H", 300))   # 300 and 301 share a high byte
    add("gray2_trns.png", r.integers(0, 4, (H, W)), 2, 0, trns=struct.pack(">H", 2))
    add("gray4_trns.png", r.integers(0, 16, (H, W)), 4, 0, trns=struct.pack(">H", 7))
    add("gray1_trns.png", r.integers(0, 2, (H, W)), 1, 0, trns=struct.pack(">H", 1))
    rgb = r.integers(0, 3, (H, W, 3)) * 100
    add("rgb8_trns.png", rgb, 8, 2, trns=struct.pack(">HHH", 100, 0, 200))
    add("rgb16_trns_unambiguous.png", rgb * 256 + 7, 16, 2, trns=struct.pack(">HHH", 0x1234, 0x0001, 0x00FF))
    pal = bytes(r.integers(0, 256, 16 * 3).astype(np.uint8))
    add("pal4_trns.png", r.integers(0, 16, (H, W)), 4, 3, palette=pal,
        trns=bytes([0, 249, 250, 255, 10, 251]))
    add("pal8_bad_trns.png", r.integers(0, 16, (H, W)), 8, 3, palette=pal, trns=bytes([0]) * 17)  # > PLTE: ignored
    add("gray8_bad_trns.png", g8, 8, 0, trns=b"\0")                                              # wrong size: ignored
    add("rgb8_noalpha.png", rgb, 8, 2)
    # one-pixel-wide/tall images: 1/(W-1) or 1/(H-1) is inf, so edge tests see inf and NaN
    add("col1_rgba.png", np.dstack([np.zeros((H, 1, 3)), a8[:, :1]]), 8, 6)
    add("row1_rgba.png", np.dstack([np.zeros((1, W, 3)), a8[:1]]), 8, 6)
    add("px1_rgba.png", np.array([[[9, 9, 9, 255]]]), 8, 6)
    return files


def libpng_testpngs() -> list[str]:
    return sorted(_LIBPNG + f for f in os.listdir(_LIBPNG) if f.endswith(".png"))


# ---- cases ---------------------------------------------------------------------------------

def _rot(r):
    return (0.0, 0.0) if r.random() < 0.25 else (r.uniform(-math.pi, math.pi), r.uniform(0, 2 * math.pi))


def _uv_window(r):
    s = r.uniform(0.02, 1.0, 2)
    c = s / 2 + r.random(2) * (1 - s)
    return c, s


@case("texture.patch", fuzz=6)
def tex_patch(B, seed):
    r = np.random.default_rng(seed)
    for _ in range(30):
        tex = _pick(r)
        if r.random() < 0.5:
            B.add_patch_textured(r.uniform(-3, 3, 3), r.uniform(0.01, 2, 2), _rot(r), tex)
        else:
            c, s = _uv_window(r)
            B.add_patch_textured(r.uniform(-3, 3, 3), r.uniform(0.01, 2, 2), _rot(r), tex, c, s)
    B.add_patch((0, 0, 0), (1, 1), (0.1, 0.2), (0.3, 0.4, 0.5))     # untextured alongside
    return B.snapshot()


@case("texture.triangle", fuzz=6)
def tex_triangle(B, seed):
    r = np.random.default_rng(seed)
    for k in range(40):
        tex = _pick(r)
        c = r.uniform(-3, 3, 3)
        v = [c + r.normal(0, 0.5, 3) for _ in range(3)]
        mode = r.random()
        if mode < 0.5:
            uv = r.random((3, 2))                                    # either winding
        elif mode < 0.75:
            uv = r.uniform(-0.3, 1.4, (3, 2))                        # out of range: upper end clamped
        else:
            uv = r.random(2) + r.normal(0, 0.004, (3, 2))            # a few pixels: edge tests dominate
        B.add_triangle_textured(*v, tex, *uv)
    B.add_triangle((0, 0, 0), (1, 0, 0), (0, 1, 0), (0.2, 0.2, 0.2))
    return B.snapshot()


@case("texture.solid_fraction_cache", fuzz=4)
def tex_cache(B, seed):
    """Polygons whose UVs round to the same pixels reuse the first one's cached solid fraction."""
    r = np.random.default_rng(seed)
    for tex in ALPHA_TEXTURES[:6]:
        uv = r.uniform(0.1, 0.9, (3, 2))
        for _ in range(5):
            jit = uv + r.uniform(-0.3, 0.3, (3, 2)) * 1e-3          # mostly the same rounded pixel key
            B.add_triangle_textured((0, 0, 0), (1, 0, 0), (0, 1, 0), tex, *jit)
        c, s = _uv_window(r)
        for _ in range(3):
            B.add_patch_textured((0, 0, 1), (1, 1), (0, 0), tex, c + r.uniform(-1, 1, 2) * 1e-4, s)
    return B.snapshot()


@case("texture.tile", fuzz=6)
def tex_tile(B, seed):
    r = np.random.default_rng(seed)
    for _ in range(5):
        sub = (int(r.integers(1, 7)), int(r.integers(1, 7)))
        rep = None if r.random() < 0.3 else (int(r.integers(1, 5)), int(r.integers(1, 5)))
        B.add_tile_textured(r.uniform(-3, 3, 3), r.uniform(0.05, 2, 2), _rot(r), sub, _pick(r, ALPHA_TEXTURES), rep)
    return B.snapshot()


@case("texture.tube", fuzz=6)
def tex_tube(B, seed):
    """Includes fully transparent triangles, which the textured overload deletes (UUID gaps)."""
    r = np.random.default_rng(seed)
    for _ in range(4):
        n = int(r.integers(2, 12))
        nodes = _stem(r, n)
        rad = r.uniform(0.0005, 0.02, n)
        rad[r.random(n) < 0.1] = 1e-7
        uf = None if r.random() < 0.5 else np.sort(r.random(n))
        B.add_tube_textured(int(r.integers(3, 12)), nodes, rad, _pick(r, ALPHA_TEXTURES), uf)
    return B.snapshot()


@case("texture.png_formats", fuzz=3)
def tex_png_formats(B, seed):
    """Every libpng test image plus the synthetic edge cases: full patch, a UV window and two triangles."""
    r = np.random.default_rng(99 + seed)
    for tex in libpng_testpngs() + synthetic_pngs():
        B.add_patch_textured((0, 0, 0), (1, 1), (0, 0), tex)
        B.add_patch_textured((0, 0, 0), (1, 1), (0, 0), tex, (0.5, 0.5), (0.6, 0.8))
        B.add_triangle_textured((0, 0, 0), (1, 0, 0), (0, 1, 0), tex, (0, 0), (1, 0), (0, 1))
        B.add_triangle_textured((0, 0, 0), (1, 0, 0), (0, 1, 0), tex, *r.random((3, 2)))
        u, v = r.random(2)                                            # zero-width UV triangles (A or B == 0)
        B.add_triangle_textured((0, 0, 0), (1, 0, 0), (0, 1, 0), tex, (u, 0.1), (u, 0.5), (u, 0.9))
        B.add_triangle_textured((0, 0, 0), (1, 0, 0), (0, 1, 0), tex, (0.1, v), (0.9, v), (0.5, v))
    return B.snapshot()


@case("texture.queries", tier="exact", fuzz=1)
def tex_queries(B, seed):
    """primitiveTextureHasTransparencyChannel and getPrimitiveTextureSize for each texture kind."""
    for tex in TEXTURES + libpng_testpngs()[::7] + synthetic_pngs():
        B.add_patch_textured((0, 0, 0), (1, 1), (0, 0), tex)
    B.add_patch((0, 0, 0), (1, 1), (0, 0), (1, 0, 0))
    return B.texture_queries()
