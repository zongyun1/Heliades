# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core (Context.cpp: Texture, Context::addTexture; global.cpp: readPNGAlpha, PNGHasAlpha,
# resolveFilePath, getFileExtension; Copyright (C) 2016-2026 Brian Bailey).
"""Texture registry entries: file resolution, PNG alpha mask, and per-UV-polygon solid fraction.

PNG pixels are decoded by Pillow (lossless, so the samples are identical to libpng's). What libpng's
read transforms would do to them in ``helios::readPNGAlpha`` is reproduced here from the raw IHDR and
tRNS chunks, which are parsed directly rather than taken from Pillow's ``info`` (Pillow drops or
rescales some tRNS forms):

  * 16-bit samples go through ``png_set_strip_16``: the high byte is kept (Pillow does the same);
  * palette alpha comes from the tRNS table (entries past its end are opaque);
  * gray / RGB tRNS gives alpha 0 where the *raw* sample equals the tRNS value, else 255;
  * a pixel is solid iff its 8-bit alpha is >= 250.

Images without an alpha channel or tRNS chunk give an all-solid mask and a solid fraction of 1.
"""
from __future__ import annotations

import math
import os
import struct
from functools import lru_cache

import numpy as np

from ..migration_registry import ports

f32 = np.float32

_TEXTURE_EXTS = (".png", ".PNG", ".jpg", ".jpeg", ".JPG", ".JPEG")


# ---- paths -----------------------------------------------------------------------------------

@ports("helios::getFileExtension")
def get_file_extension(path: str) -> str:
    """std::filesystem::path::extension(): '' for no dot, for a leading-dot name, or for '.'/'..'."""
    name = path.rsplit("/", 1)[-1]
    if name in (".", ".."):
        return ""
    i = name.rfind(".")
    return "" if i <= 0 else name[i:]


def _build_directory() -> str:
    """getBuildDirectory() (global.cpp): $HELIOS_BUILD, else cwd or its parent if it holds plugins/."""
    env = os.environ.get("HELIOS_BUILD")
    if env is not None:
        return env
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, "plugins")):
        return cwd
    parent = os.path.dirname(cwd)
    if os.path.exists(os.path.join(parent, "plugins")):
        return parent
    return cwd


@ports("helios::tryResolveFilePath")
def try_resolve_file_path(filename: str) -> str:
    """Absolute path if it exists; else cwd/filename; else build_dir/filename (canonicalised). '' if none."""
    if not filename:
        return ""
    if os.path.isabs(filename):
        return os.path.realpath(filename) if os.path.exists(filename) else ""
    for base in (os.getcwd(), _build_directory()):
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return os.path.realpath(p)
    return ""


@ports("helios::resolveFilePath")
def resolve_file_path(filename: str) -> str:
    if not filename:
        return os.getcwd()
    r = try_resolve_file_path(filename)
    if not r:
        raise FileNotFoundError(f"ERROR (helios::resolveFilePath): Could not locate asset file: {filename}")
    return r


@ports("Context::validateTextureFileExtenstion")
def validate_texture_file_extension(texture_file: str) -> bool:
    return get_file_extension(texture_file) in _TEXTURE_EXTS


@ports("Context::doesTextureFileExist")
def does_texture_file_exist(texture_file: str) -> bool:
    try:
        return os.path.exists(resolve_file_path(texture_file))
    except FileNotFoundError:
        return False


# ---- PNG alpha -------------------------------------------------------------------------------

def _png_header(path: str):
    """(width, height, bit_depth, color_type, n_palette, tRNS bytes or None), chunks before the first IDAT."""
    with open(path, "rb") as fh:
        if fh.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"ERROR (readPNGAlpha): File {path} is not a valid PNG.")
        ihdr = trns = None
        n_plte = 0
        while True:
            head = fh.read(8)
            if len(head) < 8:
                break
            n, t = struct.unpack(">I4s", head)
            if t == b"IDAT" or t == b"IEND":
                break
            data = fh.read(n)
            fh.read(4)  # CRC
            if t == b"IHDR":
                ihdr = struct.unpack(">IIBBBBB", data)
            elif t == b"PLTE":
                n_plte = n // 3
            elif t == b"tRNS":
                trns = data
    if ihdr is None:
        raise ValueError(f"ERROR (readPNGAlpha): File {path} has no IHDR chunk.")
    w, h, depth, ctype = ihdr[:4]
    # libpng png_handle_tRNS ignores ("invalid") a tRNS chunk of the wrong size or on a type with alpha
    if trns is not None:
        if ctype == 0 and len(trns) != 2:
            trns = None
        elif ctype == 2 and len(trns) != 6:
            trns = None
        elif ctype == 3 and (len(trns) == 0 or len(trns) > n_plte or len(trns) > 256):
            trns = None
        elif ctype in (4, 6):
            trns = None
    return w, h, depth, ctype, trns


@ports("helios::PNGHasAlpha")
def png_has_alpha(path: str) -> bool:
    _, _, _, ctype, trns = _png_header(path)
    return bool(ctype & 4) or trns is not None


def _alpha8(path: str) -> np.ndarray:
    """(H, W) uint8 alpha after libpng's strip_16 / palette / tRNS transforms; row 0 is the top row."""
    from PIL import Image
    w, h, depth, ctype, trns = _png_header(path)
    with Image.open(path) as im:
        im.load()
        a = np.asarray(im)
    if ctype in (4, 6):                                 # GA / RGBA: Pillow keeps the high byte of 16-bit
        return a[..., -1].astype(np.uint8)
    if ctype == 3:
        table = np.full(256, 255, np.uint8)
        table[: len(trns)] = np.frombuffer(trns, np.uint8)
        return table[a.astype(np.intp)]
    if ctype == 0:
        key = struct.unpack(">H", trns)[0]
        if depth == 1:
            raw = (a != 0).astype(np.uint16)
        elif depth in (2, 4):                           # Pillow scales L;2 / L;4 by 85 / 17 exactly
            raw = (a.astype(np.uint16) // (85 if depth == 2 else 17))
        else:
            raw = a.astype(np.uint16)                   # 8-bit L, or 16-bit I;16 (raw)
        return np.where(raw == key, 0, 255).astype(np.uint8)
    if ctype == 2:
        key = np.array(struct.unpack(">HHH", trns), np.uint16)
        if depth == 8:
            return np.where((a == key.astype(np.uint8)).all(-1), 0, 255).astype(np.uint8)
        # 16-bit RGB: Pillow only exposes the high bytes. A pixel whose high bytes differ from the key's
        # is certainly opaque; if any pixel's high bytes match, its low bytes decide and we cannot tell.
        if (a == (key >> 8).astype(np.uint8)).all(-1).any():
            raise NotImplementedError(f"16-bit RGB PNG with tRNS whose key is ambiguous after strip_16: {path}")
        return np.full((h, w), 255, np.uint8)
    raise ValueError(f"unsupported PNG color type {ctype}: {path}")


@lru_cache(maxsize=64)
def _read_png_alpha_cached(path: str, mtime_ns: int, size: int) -> np.ndarray:
    w, h, _, ctype, trns = _png_header(path)
    if not (ctype & 4) and trns is None:
        m = np.ones((h, w), bool)
    else:
        m = _alpha8(path) >= 250
    m.setflags(write=False)
    return m


@ports("helios::readPNGAlpha")
def read_png_alpha(filename: str) -> np.ndarray:
    """(H, W) bool mask, True where solid (alpha >= 250). Row 0 is the image's top row, as libpng reads it."""
    if get_file_extension(filename).lower() != ".png":
        raise ValueError(f"ERROR (readPNGAlpha): File {filename} is not PNG format.")
    p = resolve_file_path(filename)
    st = os.stat(p)
    return _read_png_alpha_cached(p, st.st_mtime_ns, st.st_size)


@ports("helios::getImageResolutionJPEG")
def image_resolution_jpeg(filename: str) -> tuple[int, int]:
    from PIL import Image
    with Image.open(resolve_file_path(filename)) as im:
        return int(im.size[0]), int(im.size[1])


# ---- Texture ---------------------------------------------------------------------------------

def _round_half_away(x) -> int:
    """std::round on a float, then int(). float32 + 0.5 is exact in double, so this is exact."""
    x = float(x)
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


class Texture:
    """helios::Texture: a texture file's alpha mask and a solid-fraction cache keyed by UV pixel coords."""

    @ports("Texture::Texture")
    def __init__(self, texture_file: str):
        self.filename = texture_file
        ext = get_file_extension(texture_file).lower()
        self.has_transparency_channel = png_has_alpha(texture_file) if ext == ".png" else False
        if ext == ".png":
            self.transparency_data = read_png_alpha(texture_file)
            self.image_resolution = (int(self.transparency_data.shape[1]), int(self.transparency_data.shape[0]))
        else:
            self.transparency_data = None
            self.image_resolution = image_resolution_jpeg(texture_file)
        self._cache: dict[tuple, np.float32] = {}
        self._mask_flipped = None

    @ports("Texture::getTextureFile")
    def get_texture_file(self) -> str:
        return self.filename

    @ports("Texture::getImageResolution")
    def get_image_resolution(self) -> tuple[int, int]:
        return self.image_resolution

    @ports("Texture::hasTransparencyChannel")
    def has_transparency(self) -> bool:
        return self.has_transparency_channel

    @ports("Texture::getSolidFraction")
    def get_solid_fraction(self, uvs) -> np.float32:
        """Cached by the UVs rounded to pixel indices, so later polygons that round to the same pixels
        get the first one's value even if their exact UVs differ (Helios behaviour, kept on purpose)."""
        uvs = np.asarray(uvs, f32).reshape(-1, 2)
        wm1, hm1 = f32(self.image_resolution[0] - 1), f32(self.image_resolution[1] - 1)
        key = []
        for u, v in uvs:
            key += [_round_half_away(u * wm1), _round_half_away(v * hm1)]
        key = tuple(key)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        sf = self.compute_solid_fraction(uvs)
        self._cache[key] = sf
        return sf

    @ports("Texture::computeSolidFraction")
    def compute_solid_fraction(self, uvs) -> np.float32:
        uvs = np.asarray(uvs, f32).reshape(-1, 2)
        if not self.has_transparency_channel or len(uvs) < 3:
            return f32(1.0)
        W, H = self.image_resolution
        if self._mask_flipped is None:
            self._mask_flipped = self.transparency_data[::-1]   # mask[y*W+x] = alpha2D[H-1-y][x]
        mask = self._mask_flipped

        minU = maxU = uvs[0, 0]
        minV = maxV = uvs[0, 1]
        for p in uvs:                     # std::min(a,b) = b<a ? b : a;  std::max(a,b) = a<b ? b : a
            minU = p[0] if p[0] < minU else minU
            maxU = p[0] if maxU < p[0] else maxU
            minV = p[1] if p[1] < minV else minV
            maxV = p[1] if maxV < p[1] else maxV
        fw, fh = f32(W - 1), f32(H - 1)
        xmin = min(max(int(math.floor(float(minU * fw))), 0), W - 1)
        xmax = min(max(int(math.ceil(float(maxU * fw))), 0), W - 1)
        ymin = min(max(int(math.floor(float(minV * fh))), 0), H - 1)
        ymax = min(max(int(math.ceil(float(maxV * fh))), 0), H - 1)
        if xmin > xmax or ymin > ymax:
            return f32(0.0)

        N = len(uvs)
        A, B, C = np.empty(N, f32), np.empty(N, f32), np.empty(N, f32)
        for i in range(N):
            a, b = uvs[i], uvs[(i + 1) % N]
            A[i] = b[0] - a[0]
            B[i] = -(b[1] - a[1])
            C[i] = f32(a[0] * b[1]) - f32(a[1] * b[0])
        signed_area = f32(0.0)
        for i in range(N):
            j = (i + 1) % N
            signed_area = f32(signed_area + f32(f32(uvs[i, 0] * uvs[j, 1]) - f32(uvs[j, 0] * uvs[i, 1])))
        if signed_area < f32(0.0):
            A, B, C = -A, -B, -C

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            inv_w = f32(1.0) / f32(W - 1)
            inv_h = f32(1.0) / f32(H - 1)
            yuv = ((np.arange(ymin, ymax + 1).astype(f32) + f32(0.5)) * inv_h)[:, None]
            xuv = ((np.arange(xmin, xmax + 1).astype(f32) + f32(0.5)) * inv_w)[None, :]
            inside = np.ones((len(yuv), xuv.shape[1]), bool)
            for k in range(N):
                L = (A[k] * yuv + B[k] * xuv) + C[k]
                inside &= ~(L < f32(0.0))                        # NaN passes, as `if (L < 0)` does
        total = int(inside.sum())
        if total == 0:
            return f32(0.0)
        opaque = int(mask[ymin:ymax + 1, xmin:xmax + 1][inside].sum())
        return f32(f32(opaque) / f32(total))
