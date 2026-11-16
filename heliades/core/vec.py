# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core/include/helios_vector_types.h and core/src/global.cpp
# (Copyright (C) 2016-2026 Brian Bailey).
"""float32 vec3 arithmetic with Helios's exact operation order.

Every function takes and returns float32 numpy arrays of shape (3,) or float32 scalars. numpy >= 2
keeps float32 when mixed with Python scalars (NEP 50), and elementwise ops round after each step,
so each line reproduces the C++ expression it is named after.
"""
from __future__ import annotations

import math

import numpy as np

from ..migration_registry import ports

f32 = np.float32
PI_F = f32(3.14159265358979)


def v3(x, y=None, z=None) -> np.ndarray:
    if y is None:
        return np.asarray(x, f32).reshape(3).copy()
    return np.array([x, y, z], f32)


def dot(a, b) -> np.float32:
    """vec3::operator*(vec3): a.x*x + a.y*y + a.z*z."""
    return f32(f32(a[0] * b[0] + a[1] * b[1]) + a[2] * b[2])


@ports("helios::cross")
def cross(a, b) -> np.ndarray:
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], f32)


def magnitude(v) -> np.float32:
    """vec3::magnitude: sqrtf(x*x + y*y + z*z)."""
    return f32(np.sqrt(f32(f32(v[0] * v[0] + v[1] * v[1]) + v[2] * v[2])))


def normalize(v) -> np.ndarray:
    """vec3::normalize / helios::normalize: multiply by 1/sqrtf(m2); unchanged if m2 <= 0."""
    m2 = f32(f32(v[0] * v[0] + v[1] * v[1]) + v[2] * v[2])
    if m2 <= 0:
        return np.asarray(v, f32).copy()
    inv = f32(f32(1.0) / f32(np.sqrt(m2)))
    return np.asarray(v, f32) * inv


def div(v, a) -> np.ndarray:
    """vec3::operator/(float): multiply by 1/a (inf with a's sign if a == 0)."""
    a = f32(a)
    inv = f32(1.0) / a if a != 0 else f32(math.copysign(math.inf, a))
    return np.asarray(v, f32) * f32(inv)


def smul(a, v) -> np.ndarray:
    return f32(a) * np.asarray(v, f32)


def _load_libm():
    """glibc's float sinf/cosf/acosf are not correctly rounded (about 1 in 40 arguments of the form
    2*PI_F*j/N differ by 1 ulp from the rounded double result). Helios calls them, so we do too."""
    import ctypes
    import ctypes.util
    try:
        m = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
        fns = (m.sinf, m.cosf, m.acosf)
        for fn in fns:
            fn.restype, fn.argtypes = ctypes.c_float, [ctypes.c_float]
        return fns
    except (OSError, AttributeError):
        return None, None, None


_SINF, _COSF, _ACOSF = _load_libm()


def sinf(x) -> np.float32:
    """C sinf (float in, float out) -- use where the C++ calls sinf or std::sin on a float."""
    return f32(_SINF(float(f32(x)))) if _SINF else f32(math.sin(float(f32(x))))


def cosf(x) -> np.float32:
    return f32(_COSF(float(f32(x)))) if _COSF else f32(math.cos(float(f32(x))))


def acosf(x) -> np.float32:
    return f32(_ACOSF(float(f32(x)))) if _ACOSF else f32(math.acos(float(f32(x))))


def sin_d(x) -> np.float32:
    """C sin(double) of a float argument, rounded to float on assignment."""
    return f32(math.sin(float(f32(x))))


def cos_d(x) -> np.float32:
    return f32(math.cos(float(f32(x))))


def acos_d(x) -> np.float32:
    """C acos(double) of a float argument (Helios's tube frame calls acos, not acosf)."""
    return f32(math.acos(float(f32(x))))


@ports("helios::rotatePointAboutLine")
def rotate_point_about_line(point, line_base, line_direction, theta) -> np.ndarray:
    theta = f32(theta)
    if theta == 0:
        return np.asarray(point, f32).copy()
    mag = magnitude(line_direction)
    if mag < f32(1e-6):
        return np.asarray(point, f32).copy()
    u, v, w = div(line_direction, mag)
    a, b, c = np.asarray(line_base, f32)
    x, y, z = np.asarray(point, f32)
    st, ct = sin_d(theta), cos_d(theta)  # global.cpp calls sin/cos(double); verified by parity probe
    one_ct = f32(1) - ct
    px = f32(f32(f32(f32(a * f32(v * v + w * w)) - f32(u * f32(f32(f32(f32(b * v + c * w) - u * x) - v * y) - w * z))) * one_ct
                 + x * ct) + f32(f32(f32(-c * v + b * w) - w * y) + v * z) * st)
    py = f32(f32(f32(f32(b * f32(u * u + w * w)) - f32(v * f32(f32(f32(f32(a * u + c * w) - u * x) - v * y) - w * z))) * one_ct
                 + y * ct) + f32(f32(f32(c * u - a * w) + w * x) - u * z) * st)
    pz = f32(f32(f32(f32(c * f32(u * u + v * v)) - f32(w * f32(f32(f32(f32(a * u + b * v) - u * x) - v * y) - w * z))) * one_ct
                 + z * ct) + f32(f32(f32(-b * u + a * v) - v * x) + u * y) * st)
    return np.array([px, py, pz], f32)


# ---- helios:: scalar helpers (global.cpp / global.h) ------------------------------------------

def _load_libm_extra():
    import ctypes
    import ctypes.util
    try:
        m = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
        fn = m.asinf
        fn.restype, fn.argtypes = ctypes.c_float, [ctypes.c_float]
        return fn
    except (OSError, AttributeError):
        return None


_ASINF = _load_libm_extra()
_M_PI_F = f32(math.pi)          # float(M_PI)
_180_F = f32(180.0)


def asinf(x) -> np.float32:
    """C asinf (glibc float version, not necessarily correctly rounded)."""
    return f32(_ASINF(float(f32(x)))) if _ASINF else f32(math.asin(float(f32(x))))


@ports("helios::deg2rad")
def deg2rad(deg) -> np.float32:
    """deg * float(M_PI) / 180.f, left to right in float."""
    return f32(f32(f32(deg) * _M_PI_F) / _180_F)


@ports("helios::rad2deg")
def rad2deg(rad) -> np.float32:
    """rad * 180.f / float(M_PI), left to right in float."""
    return f32(f32(f32(rad) * _180_F) / _M_PI_F)


@ports("helios::acos_safe")
def acos_safe(x) -> np.float32:
    """Clamp to [-1, 1] (compared in double, which is exact for a float), then acosf."""
    x = f32(x)
    if x < -1.0:
        x = f32(-1.0)
    elif x > 1.0:
        x = f32(1.0)
    return acosf(x)


@ports("helios::asin_safe")
def asin_safe(x) -> np.float32:
    x = f32(x)
    if x < -1.0:
        x = f32(-1.0)
    elif x > 1.0:
        x = f32(1.0)
    return asinf(x)


@ports("helios::clamp")
def clamp(value, lo, hi):
    """helios::clamp: ``value < min`` first, then ``value > max``; NaN passes through unchanged.
    The result has the type of ``value`` (callers pass numpy scalars of the C++ type)."""
    if value < lo:
        return type(value)(lo) if isinstance(value, np.generic) else lo
    if value > hi:
        return type(value)(hi) if isinstance(value, np.generic) else hi
    return value


@ports("helios::sum(const std::vector<float>&)")
def sum_f(values) -> np.float32:
    """Running float sum from 0 in order; raises on an empty vector as Helios does."""
    if len(values) == 0:
        raise ValueError("ERROR (sum): Vector is empty.")
    m = f32(0)
    for v in values:
        m = f32(m + f32(v))
    return m
