# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core (Context_fileIO.cpp: Context::loadOBJ, Context::loadMTL; global.cpp: string2vec2,
# string2vec3, string2RGBcolor, parse_float, parse_int, trim_whitespace, getFilePath, rotatePoint,
# calculateTriangleArea; Copyright (C) 2016-2026 Brian Bailey).
"""Wavefront OBJ/MTL parsing exactly as Helios does it, plus the float32 helpers loadOBJ needs.

Helios reads the files with ``std::ifstream >> token`` and ``std::getline``. Their corner cases decide
what a file means (a token at EOF without a newline, a face line with trailing whitespace or a CR,
which re-reads the last vertex reference), so ``CStream`` reproduces libstdc++'s formatted-input
semantics rather than splitting lines. Numbers go through libc ``strtod`` / ``strtol`` (``std::stod`` /
``std::stoi`` call them), then ``(float)``: the double rounding of ``helios::parse_float`` is kept.
"""
from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass

import numpy as np

from . import vec as V
from ..migration_registry import ports

f32 = np.float32

_WS = " \t\n\v\f\r"   # isspace() in the C locale (std::istream's whitespace)


class CStream:
    """A std::istringstream / std::ifstream over a decoded string: operator>>(string) and getline.

    The libstdc++ details that matter: when the sentry fails (stream not good, or only whitespace is
    left) operator>> sets failbit and leaves the string *unchanged*; getline with a good stream erases
    the string first, and sets failbit if it extracted nothing. Reaching the end while extracting sets
    eofbit, so good() turns false after the last token or line."""

    def __init__(self, s: str):
        self.s, self.i, self.n = s, 0, len(s)
        self.eof = self.fail = False

    def good(self) -> bool:
        return not (self.eof or self.fail)

    def token(self, cur: str) -> str:
        if not self.good():
            self.fail = True
            return cur
        s, i, n = self.s, self.i, self.n
        while i < n and s[i] in _WS:
            i += 1
        if i == n:
            self.i, self.eof, self.fail = n, True, True
            return cur
        j = i
        while j < n and s[j] not in _WS:
            j += 1
        if j == n:
            self.eof = True
        self.i = j
        return s[i:j]

    def getline(self, cur: str) -> str:
        if not self.good():
            self.fail = True
            return cur
        if self.i == self.n:
            self.eof = self.fail = True
            return ""
        k = self.s.find("\n", self.i)
        if k < 0:
            out, self.i, self.eof = self.s[self.i:], self.n, True
            return out
        out, self.i = self.s[self.i:k], k + 1
        return out


def _read_text(path: str) -> str:
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8", "surrogateescape")


def _cbytes(s: str) -> bytes:
    return s.encode("utf-8", "surrogateescape")


def cpp_key(s: str) -> bytes:
    """std::map<std::string, ...> order: bytewise (char_traits<char>::compare is memcmp)."""
    return _cbytes(s)


# ---- string conversions (global.cpp) -------------------------------------------------------

_libc = ctypes.CDLL(None, use_errno=True)
_strtod = _libc.strtod
_strtod.restype, _strtod.argtypes = ctypes.c_double, [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
_strtol = _libc.strtol
_strtol.restype, _strtol.argtypes = ctypes.c_long, [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
_ERANGE = 34


def _c_convert(fn, s: str, *extra):
    """(value, n_chars_read, errno) of strtod/strtol on s; s must hold no NUL."""
    b = _cbytes(s)
    buf = ctypes.create_string_buffer(b)
    end = ctypes.c_char_p()
    ctypes.set_errno(0)
    val = fn(buf, ctypes.byref(end), *extra)
    err = ctypes.get_errno()
    read = ctypes.cast(end, ctypes.c_void_p).value - ctypes.addressof(buf)
    return val, read, len(b), err


@ports("helios::trim_whitespace")
def trim_whitespace(s: str) -> str:
    a = 0
    n = len(s)
    while a < n and s[a] in _WS:
        a += 1
    if a == n:
        return ""
    b = n - 1
    while s[b] in _WS:
        b -= 1
    return s[a:b + 1]


@ports("helios::parse_float")
def parse_float(s: str):
    """(ok, float32). std::stod then (float); invalid_argument / out_of_range (ERANGE) -> not ok."""
    s = trim_whitespace(s)
    if "\0" in s:            # strtod stops at the NUL, so the full-length check fails
        return False, f32(0)
    val, read, size, err = _c_convert(_strtod, s)
    if read == 0 or err == _ERANGE:
        return False, f32(0)
    return read == size, f32(val)


@ports("helios::parse_int")
def parse_int(s: str):
    """(ok, int). std::stoi: strtol base 10, out_of_range on ERANGE or outside int, full-string check."""
    s = trim_whitespace(s)
    if "\0" in s:
        return False, 0
    val, read, size, err = _c_convert(_strtol, s, 10)
    if read == 0 or err == _ERANGE or not (-2**31 <= val < 2**31):
        return False, 0
    return read == size, int(val)


def _string2floats(s: str, n: int, who: str):
    st = CStream(s)
    out = []
    tmp = ""
    while True:
        tmp = st.token(tmp)
        if st.fail or len(out) >= n:   # while (stream >> tmp && c < n)
            break
        ok, v = parse_float(tmp)
        if not ok:
            raise ValueError(f"ERROR ({who}): Invalid float value '{tmp}' in input string '{s}'")
        out.append(v)
    if len(out) < n:
        raise ValueError(f"ERROR ({who}): Insufficient values in input string '{s}'. Expected {n} values, got {len(out)}")
    return np.array(out, f32)


@ports("helios::string2vec3")
def string2vec3(s: str) -> np.ndarray:
    return _string2floats(s, 3, "string2vec3")


@ports("helios::string2vec2")
def string2vec2(s: str) -> np.ndarray:
    return _string2floats(s, 2, "string2vec2")


@ports("helios::string2RGBcolor")
def string2rgbcolor(s: str) -> np.ndarray:
    """RGBA, alpha defaults to 1; 3 or 4 values."""
    o = [f32(0), f32(0), f32(0), f32(1)]
    st = CStream(s)
    c = 0
    tmp = ""
    while True:
        tmp = st.token(tmp)
        if st.fail:
            break
        if c >= 4:
            raise ValueError(f"ERROR (string2RGBcolor): Too many values in input string '{s}'. Expected at most 4 values "
                             f"(RGBA), but found additional value '{tmp}'")
        ok, v = parse_float(tmp)
        if not ok:
            raise ValueError(f"ERROR (string2RGBcolor): Invalid float value '{tmp}' in input string '{s}'")
        o[c] = v
        c += 1
    if c < 3:
        raise ValueError(f"ERROR (string2RGBcolor): Insufficient values in input string '{s}'. Expected at least 3 values (RGB), got {c}")
    return np.array(o, f32)


@ports("helios::getFilePath")
def get_file_path(filepath: str, trailingslash: bool = True) -> str:
    """std::filesystem::path::parent_path, plus a trailing '/' when non-empty."""
    p = filepath
    # parent_path: everything before the last filename element, with trailing separators removed
    # (but a root "/" kept). Our inputs are canonical absolute paths, so this suffices.
    k = p.rfind("/")
    if k < 0:
        out = ""
    else:
        out = p[:k]
        while len(out) > 1 and out.endswith("/"):
            out = out[:-1]
        if out == "" and p.startswith("/"):
            out = "/"
    if trailingslash and out and out[-1] not in "/\\":
        out += "/"
    return out


def path_join(base: str, rel: str) -> str:
    """std::filesystem::path(base) / rel for POSIX paths."""
    if rel.startswith("/"):
        return rel
    if not base:
        return rel
    return base + rel if base.endswith("/") else base + "/" + rel


# ---- float32 geometry helpers (global.cpp) -------------------------------------------------

@ports("helios::rotatePoint")
def rotate_point(P: np.ndarray, theta, phi) -> np.ndarray:
    """rotatePoint(position, theta=elevation, phi=azimuth) on (3, N) float32 columns. sin/cos are the
    double versions on the float arguments, as in global.cpp."""
    theta, phi = f32(theta), f32(phi)
    if theta == f32(0) and phi == f32(0):
        return P
    st, ct = V.sin_d(theta), V.cos_d(theta)
    sp, cp = V.sin_d(phi), V.cos_d(phi)
    z = f32(0)
    Ry = [[ct, z, st], [z, f32(1), z], [-st, z, ct]]
    Rz = [[cp, -sp, z], [sp, cp, z], [z, z, f32(1)]]
    R = [[f32(0)] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                R[i][j] = f32(R[i][j] + f32(Rz[i][k] * Ry[k][j]))
    x, y, zz = P[0], P[1], P[2]
    return np.array([(R[r][0] * x + R[r][1] * y) + R[r][2] * zz for r in range(3)], f32)


def _mag(X: np.ndarray) -> np.ndarray:
    """vec3::magnitude on (3, N) columns."""
    return np.sqrt((X[0] * X[0] + X[1] * X[1]) + X[2] * X[2])


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], f32)


@ports("helios::calculateTriangleArea")
def triangle_area(v0, v1, v2) -> np.ndarray:
    """0.5f * cross(v1 - v0, v2 - v0).magnitude() on (3, N) columns."""
    return f32(0.5) * _mag(_cross(v1 - v0, v2 - v0))


# ---- MTL -----------------------------------------------------------------------------------

@dataclass
class OBJMaterial:
    """Context::OBJmaterial."""
    color: np.ndarray                     # RGB float32
    texture: str = ""
    texture_has_transparency: bool = False
    texture_color_overridden: bool = False
    normal_map: str = ""
    normal_map_kind: str = "none"         # "none" | "tangent" | "bump" (bump images are not classified here)
    normal_map_strength: np.float32 = f32(1)


def _exists(p: str) -> bool:
    return os.path.exists(p)


@ports("Context::loadMTL")
def load_mtl(filebase: str, material_file: str, default_color) -> dict[str, OBJMaterial]:
    from .texture import get_file_extension, resolve_file_path
    if material_file.startswith("/"):
        resolved = resolve_file_path(material_file)
    else:
        resolved = resolve_file_path(path_join(filebase, material_file))
    try:
        text = _read_text(resolved)
    except OSError:
        raise FileNotFoundError(f"ERROR (Context::loadMTL): Could not open material file {resolved} after successful path resolution.")
    dc = np.asarray(default_color, f32)[:3].copy()
    materials: dict[str, OBJMaterial] = {}
    st = CStream(text)
    line = st.token("")
    while st.good():
        if line == "#":
            line = st.getline(line)
            line = st.token(line)
        elif line == "newmtl":
            line = st.getline(line)
            name = trim_whitespace(line)
            materials.setdefault(name, OBJMaterial(dc.copy()))     # std::map::emplace keeps an existing entry
            map_Kd = map_d = norm_file = bump_file = ""
            norm_strength = bump_strength = f32(1)
            while line != "newmtl" and st.good():
                line = st.token(line)
                if line == "newmtl":
                    break
                elif line in ("norm", "map_Bump", "map_bump", "bump"):
                    is_norm = line == "norm"
                    line = st.getline(line)
                    ts = CStream(trim_whitespace(line))
                    tokens, tok = [], ""
                    while True:
                        tok = ts.token(tok)
                        if ts.fail:
                            break
                        tokens.append(tok)
                    map_file, strength = "", f32(1)
                    t = 0
                    while t < len(tokens):
                        if tokens[t] == "-bm" and t + 1 < len(tokens):
                            ok, strength = parse_float(tokens[t + 1])
                            if not ok:
                                raise ValueError(f"ERROR (Context::loadOBJ): The -bm option of the bump line of material '{name}' in {resolved} must be followed by a number.")
                            t += 2
                            continue
                        if get_file_extension(tokens[t]).lower() in (".png", ".jpg", ".jpeg"):
                            map_file = tokens[t]
                            if not _exists(map_file):
                                map_file = path_join(filebase, tokens[t])
                                if not _exists(map_file):
                                    raise FileNotFoundError(f"ERROR (Context::loadOBJ): Normal or bump map '{tokens[t]}' referenced by material '{name}' in {resolved} cannot be found. "
                                                            f"Searched in current directory and OBJ file directory ({filebase}).")
                        t += 1
                    if not map_file:
                        raise ValueError(f"ERROR (Context::loadOBJ): The {'norm' if is_norm else 'bump'} line of material '{name}' in {resolved} names no PNG or JPEG image.")
                    if is_norm:
                        norm_file, norm_strength = map_file, strength
                    else:
                        bump_file, bump_strength = map_file, strength
                elif line in ("map_a", "map_Ka", "Ks", "Ka", "map_Ks"):
                    line = st.getline(line)
                elif line in ("map_Kd", "map_d"):
                    maptype = line
                    line = trim_whitespace(st.getline(line))
                    ts = CStream(line)
                    tmp = ""
                    while ts.good():
                        tmp = ts.token(tmp)
                        if get_file_extension(tmp) in (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"):
                            texturefile = tmp
                            if not _exists(texturefile):
                                texturefile = path_join(filebase, tmp)
                                if not _exists(texturefile):
                                    raise FileNotFoundError(f"ERROR (Context::loadOBJ): Texture file '{tmp}' referenced in .mtl file cannot be found. "
                                                            f"Searched in current directory and OBJ file directory ({filebase}). "
                                                            "Ensure texture file exists or remove texture reference from material.")
                            if maptype == "map_d":
                                map_d = texturefile
                            else:
                                map_Kd = texturefile
                elif line == "Kd":
                    line = st.getline(line)
                    c = string2rgbcolor(trim_whitespace(line))
                    materials[name].color = c[:3].copy()
                else:
                    line = st.getline(line)
            m = materials[name]
            if norm_file:
                m.normal_map, m.normal_map_kind, m.normal_map_strength = norm_file, "tangent", norm_strength
            elif bump_file:
                # classifyBumpImage() decides height vs tangent-space normal map from the image (not ported)
                m.normal_map, m.normal_map_kind, m.normal_map_strength = bump_file, "bump", bump_strength
            if map_Kd:
                m.texture = map_Kd
                if map_d and map_d != map_Kd:
                    m.texture_has_transparency = True
            elif map_d:
                m.texture = map_d
                m.texture_color_overridden = True
        else:
            line = st.getline(line)
            line = st.token(line)
    return materials


# ---- OBJ -----------------------------------------------------------------------------------

@dataclass
class OBJFile:
    vertices: np.ndarray          # (n,3) float32
    objects: list                 # per vertex: the current 'o' name when it was declared
    texture_uv: np.ndarray        # (k,2) float32
    vertex_normals: np.ndarray    # (m,3) float32
    face_inds: dict               # material -> list of 1-based vertex index lists
    texture_inds: dict
    normal_inds: dict
    materials: dict               # name -> OBJMaterial (from the last mtllib)
    boxmin: np.float32
    boxmax: np.float32
    filebase: str


def _face_refs(line: str, nv: int, nt: int, nn: int, lineno: int):
    st = CStream(line)
    f, u, n = [], [], []
    tmp = ""
    while st.good():
        tmp = st.token(tmp)          # a failed read keeps tmp: trailing whitespace repeats the last reference
        if not tmp:
            continue
        fields = ["", "", ""]
        k = 0
        for ch in tmp:
            if ch == "/":
                k += 1
                if k > 2:
                    raise ValueError(f"ERROR (Context::loadOBJ): Face vertex reference '{tmp}' on line {lineno} has more than three fields. "
                                     "Valid forms are v, v/vt, v//vn and v/vt/vn.")
            else:
                fields[k] += ch
        if fields[0]:
            ok, face = parse_int(fields[0])
            if not ok:
                raise ValueError(f"ERROR (Context::loadOBJ): Face index on line {lineno} must be an integer value.")
            if face < 0:
                face = nv + face + 1
            if face <= 0 or face > nv:
                raise ValueError(f"ERROR (Context::loadOBJ): Face vertex index {face} on line {lineno} is out of range. Valid range is 1-{nv}.")
            f.append(face)
        if fields[1]:
            ok, uv = parse_int(fields[1])
            if not ok:
                raise ValueError(f"ERROR (Context::loadOBJ): u,v index on line {lineno} must be an integer value.")
            if uv < 0:
                uv = nt + uv + 1
            if uv <= 0 or uv > nt:
                raise ValueError(f"ERROR (Context::loadOBJ): Texture coordinate index {uv} on line {lineno} is out of range. Valid range is 1-{nt}.")
            u.append(uv)
        if fields[2]:
            ok, nm = parse_int(fields[2])
            if not ok:
                raise ValueError(f"ERROR (Context::loadOBJ): Vertex normal index on line {lineno} must be an integer value.")
            if nm < 0:
                nm = nn + nm + 1
            if nm <= 0 or nm > nn:
                raise ValueError(f"ERROR (Context::loadOBJ): Vertex normal index {nm} on line {lineno} is out of range. Valid range is 1-{nn}.")
            n.append(nm)
    return f, u, n


def parse_obj(resolved_filename: str, default_color) -> OBJFile:
    """The reading loop of Context::loadOBJ (everything before the scale is worked out)."""
    text = _read_text(resolved_filename)
    filebase = get_file_path(resolved_filename)
    verts, objects, uvs, normals = [], [], [], []
    face_inds, texture_inds, normal_inds = {}, {}, {}
    materials: dict = {}
    boxmin, boxmax = f32(100000), f32(-100000)
    current_material = current_object = "none"
    st = CStream(text)
    line = ""
    lineno = 0
    while st.good():
        lineno += 1
        line = st.token(line)
        if line == "#":
            line = st.getline(line)
        elif line == "mtllib":
            line = st.getline(line)
            materials = load_mtl(filebase, trim_whitespace(line), default_color)
        elif line == "o":
            line = st.getline(line)
            current_object = trim_whitespace(line)
        elif line == "v":
            line = st.getline(line)
            v = string2vec3(line)
            verts.append(v)
            objects.append(current_object)
            if v[2] < boxmin:
                boxmin = v[2]
            if v[2] > boxmax:
                boxmax = v[2]
        elif line == "vt":
            line = trim_whitespace(st.getline(line))
            uvs.append(string2vec2(line))
        elif line == "vn":
            line = trim_whitespace(st.getline(line))
            normals.append(string2vec3(line))
        elif line == "usemtl":
            line = st.getline(line)
            current_material = trim_whitespace(line)
        elif line == "f":
            line = st.getline(line)
            f, u, n = _face_refs(line, len(verts), len(uvs), len(normals), lineno)
            face_inds.setdefault(current_material, []).append(f)
            texture_inds.setdefault(current_material, []).append(u)
            normal_inds.setdefault(current_material, []).append(n)
        else:
            line = st.getline(line)
    return OBJFile(np.array(verts, f32).reshape(-1, 3), objects, np.array(uvs, f32).reshape(-1, 2),
                   np.array(normals, f32).reshape(-1, 3), face_inds, texture_inds, normal_inds, materials,
                   f32(boxmin), f32(boxmax), filebase)


def obj_scale(scale, boxmin, boxmax) -> np.ndarray:
    """The scl vector of loadOBJ: (0,0,h>0) means 'height h', zeros otherwise become 1."""
    s = np.asarray(scale, f32).copy()
    if s[0] == 0 and s[1] == 0 and s[2] > 0:
        d = f32(boxmax - boxmin)
        if d > f32(1e-6):
            k = f32(s[2] / d)
            return np.array([k, k, k], f32)
        return np.array([s[2], s[2], s[2]], f32)
    for i, ax in enumerate("xyz"):
        if s[i] == 0 and (s[(i + 1) % 3] != 0 or s[(i + 2) % 3] != 0):
            print(f"WARNING (Context::loadOBJ): Scaling factor given for {ax}-direction is zero. Setting scaling factor to 1")
    s[s == 0] = 1
    return s


def orient(P: np.ndarray, upaxis: str, rotation) -> np.ndarray:
    """YUP -> ZUP (rotatePointAboutLine about +x by 0.5*M_PI as float), then rotatePoint(rotation)."""
    if upaxis == "YUP":
        P = V.rotate_point_about_line(P, V.v3(0, 0, 0), V.v3(1, 0, 0), f32(0.5 * math.pi))
    return rotate_point(P, rotation[0], rotation[1])
