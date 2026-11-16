# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantArchitecture.cpp:29-40, 266-378, 1307-1327,
# 4597-4617, 4844-4853, 12879-12938 (Copyright (C) 2016-2026 Brian Bailey).
"""Small numeric helpers of the plant-architecture plugin, float32 op for op.

* ``interpolate_tube`` (``PlantArchitecture::interpolateTube``, both overloads; ``fplus = 1.f + 1e-3`` is a
  double literal narrowed to float, and ``fabs(frac - fplus) < 0.0001`` compares in double),
* ``get_axis_vector`` (``Phytomer::getAxisVector``),
* ``make_petiole_tube`` (free function ``makePetioleTube``),
* ``clamp_offset``, ``leaflet_size_max`` (``powf`` from libm), ``petiole_radius_fraction``,
  ``compound_leaf_rotation`` (``0.5 * PI_F`` is evaluated in double), ``distribution_mean``.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np

from ..core import vec as V
from ..migration_registry import ports

f32 = np.float32
PI_F = f32(math.pi)                           # constexpr float PI_F (global.h:28)

MIN_TUBE_RADIUS_FOR_GEOMETRY = f32(1e-5)      # PA:29
PETIOLE_MIN_RADIUS_FRACTION = f32(0.3)        # PA:34
MIN_TUBE_LENGTH_FOR_GEOMETRY = f32(1e-4)      # PA:40
NO_PETIOLE_OBJID = 4294967294                 # Phytomer::no_petiole_objID (H:2524)
NO_INTERNODE_TUBE_OBJID = 4294967294          # Shoot::no_internode_tube_objID (H:3306)


def _load_powf():
    try:
        m = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
        fn = m.powf
        fn.restype, fn.argtypes = ctypes.c_float, [ctypes.c_float, ctypes.c_float]
        return fn
    except (OSError, AttributeError):
        return None


_POWF = _load_powf()


def powf(x, y) -> np.float32:
    """glibc ``powf`` (float in, float out)."""
    if _POWF is not None:
        return f32(_POWF(float(f32(x)), float(f32(y))))
    return f32(math.pow(float(f32(x)), float(f32(y))))


def _vsub(a, b):
    return np.asarray(a, f32) - np.asarray(b, f32)


@ports("petioleRadiusFraction")
def petiole_radius_fraction(petiole_scale_factor_fraction) -> np.float32:
    """``std::max(fraction, PETIOLE_MIN_RADIUS_FRACTION)`` (PA:37)."""
    x = f32(petiole_scale_factor_fraction)
    return x if x > PETIOLE_MIN_RADIUS_FRACTION else PETIOLE_MIN_RADIUS_FRACTION


@ports("clampOffset")
def clamp_offset(count_per_axis: int, offset) -> np.float32:
    """PA:266-274."""
    offset = f32(offset)
    if count_per_axis > 2:
        denom = f32(f32(f32(0.5) * f32(count_per_axis)) - f32(1))
        if f32(offset * denom) > f32(1):
            offset = f32(f32(1) / denom)
    return offset


@ports("leafletSizeMax")
def leaflet_size_max(ind_from_tip, leaflet_scale, intercalary_leaflet_scale, prototype_scale) -> np.float32:
    """PA:309-323 (``std::fabs`` of a float is float; ``powf`` from libm)."""
    slot = f32(abs(f32(ind_from_tip)))
    leaflet_scale, intercalary_leaflet_scale = f32(leaflet_scale), f32(intercalary_leaflet_scale)
    prototype_scale = f32(prototype_scale)
    if slot == 0 or leaflet_scale == 1:
        return prototype_scale
    if intercalary_leaflet_scale <= 0:
        return f32(powf(leaflet_scale, slot) * prototype_scale)
    pair_index = int(math.ceil(float(slot)))
    size = f32(powf(leaflet_scale, f32((pair_index + 1) // 2)) * prototype_scale)
    return f32(intercalary_leaflet_scale * size) if pair_index % 2 == 0 else size


@ports("PlantArchitecture::interpolateTube(const std::vector<float>&,float)")
def interpolate_tube_float(P, frac) -> np.float32:
    frac = f32(frac)
    n = len(P)
    assert n > 0
    dl = f32(f32(1) / f32(n - 1))
    f = f32(0)
    for i in range(n - 1):
        fplus = f32(f + dl)
        if fplus >= 1:
            fplus = f32(1.0 + 1e-3)
        if frac >= f and (frac <= fplus or abs(float(f32(frac - fplus))) < 0.0001):
            t = f32(f32(frac - f) / f32(fplus - f))
            return f32(f32(P[i]) + f32(t * f32(f32(P[i + 1]) - f32(P[i]))))
        f = fplus
    return f32(P[0])


@ports("PlantArchitecture::interpolateTube(const std::vector<vec3>&,float)")
def interpolate_tube(P, frac) -> np.ndarray:
    frac = f32(frac)
    n = len(P)
    assert n > 0
    dl = f32(0)
    for i in range(n - 1):
        dl = f32(dl + V.magnitude(_vsub(P[i + 1], P[i])))
    f = f32(0)
    for i in range(n - 1):
        dseg = V.magnitude(_vsub(P[i + 1], P[i]))
        fplus = f32(f + f32(dseg / dl))
        if fplus >= 1:
            fplus = f32(1.0 + 1e-3)
        if frac >= f and (frac <= fplus or abs(float(f32(frac - fplus))) < 0.0001):
            t = f32(f32(frac - f) / f32(fplus - f))
            return np.asarray(P[i], f32) + t * _vsub(P[i + 1], P[i])
        f = fplus
    return np.asarray(P[0], f32).copy()


@ports("Phytomer::getAxisVector")
def get_axis_vector(stem_fraction, axis_vertices) -> np.ndarray:
    """PA:1307-1327: forward difference over df = 0.1 of the arc-length fraction, normalised."""
    stem_fraction = f32(stem_fraction)
    df = f32(0.1)
    if f32(stem_fraction + df) <= 1:
        frac_minus, frac_plus = stem_fraction, f32(stem_fraction + df)
    else:
        frac_minus, frac_plus = f32(stem_fraction - df), stem_fraction
    node_minus = interpolate_tube(axis_vertices, frac_minus)
    node_plus = interpolate_tube(axis_vertices, frac_plus)
    return V.normalize(node_plus - node_minus)


@ports("makePetioleTube")
def make_petiole_tube(radial_subdivisions: int, vertices, radii, colors, context) -> int:
    """PA:12879-12938: drop nodes closer than 1e-6 to the last kept one, clamp radii to
    MIN_TUBE_RADIUS_FOR_GEOMETRY, then ``addTubeObject``. Returns NO_PETIOLE_OBJID when nothing is built."""
    n = len(vertices)
    if len(radii) != n or len(colors) != n:
        raise ValueError("ERROR (makePetioleTube): Length of vertex vectors is not consistent.")
    if n < 2:
        return NO_PETIOLE_OBJID
    all_radii_too_small = not any(f32(r) >= MIN_TUBE_RADIUS_FOR_GEOMETRY for r in radii)
    total_length = f32(0)
    for v in range(n - 1):
        total_length = f32(total_length + V.magnitude(_vsub(vertices[v + 1], vertices[v])))
    if all_radii_too_small or total_length < MIN_TUBE_LENGTH_FOR_GEOMETRY:
        return NO_PETIOLE_OBJID

    def rmax(r):
        r = f32(r)
        return r if r > MIN_TUBE_RADIUS_FOR_GEOMETRY else MIN_TUBE_RADIUS_FOR_GEOMETRY

    tv, tr, tc = [np.asarray(vertices[0], f32)], [rmax(radii[0])], [colors[0]]
    for v in range(1, n):
        if V.magnitude(_vsub(vertices[v], tv[-1])) < f32(1e-6):
            continue
        tv.append(np.asarray(vertices[v], f32))
        tr.append(rmax(radii[v]))
        tc.append(colors[v])
    if len(tv) < 2:
        return NO_PETIOLE_OBJID
    return context.add_tube_object(radial_subdivisions, tv, tr, tc)


@ports("Phytomer::compoundLeafRotation")
def compound_leaf_rotation(leaves_per_petiole: int, leaf_index: int, leaflet_offset_val) -> np.float32:
    """PA:4597-4617."""
    compound_rotation = f32(0)
    if leaves_per_petiole > 1:
        if f32(leaflet_offset_val) == 0:
            # float dphi = PI_F / (floor(0.5 * float(n - 1)) + 1): float / double -> double -> float
            dphi = f32(float(PI_F) / (math.floor(0.5 * float(f32(leaves_per_petiole - 1))) + 1))
            compound_rotation = f32(-PI_F + f32(dphi * f32(f32(leaf_index) + f32(0.5))))
        else:
            half = f32(f32(leaves_per_petiole - 1) / f32(2))
            if f32(leaf_index) == half:
                compound_rotation = f32(0)
            elif f32(leaf_index) < half:
                compound_rotation = f32(-0.5 * float(PI_F))
            else:
                compound_rotation = f32(0.5 * float(PI_F))
    return compound_rotation


@ports("distributionMean")
def distribution_mean(parameter) -> np.float32:
    """PA:4844-4853 (no draw for a distributed parameter; ``val()`` for a constant)."""
    p = parameter.distribution_parameters
    if parameter.distribution == "uniform":
        return f32(f32(0.5) * f32(f32(p[0]) + f32(p[1])))
    if parameter.distribution == "normal":
        return f32(p[0])
    if parameter.distribution == "weibull":
        return f32(f32(p[1]) * f32(math.gamma(float(f32(f32(1) + f32(f32(1) / f32(p[0])))))))   # std::tgamma(float)
    return parameter.val()


__all__ = ["interpolate_tube", "interpolate_tube_float", "get_axis_vector", "make_petiole_tube", "clamp_offset",
           "leaflet_size_max", "petiole_radius_fraction", "compound_leaf_rotation", "distribution_mean", "powf",
           "MIN_TUBE_RADIUS_FOR_GEOMETRY", "MIN_TUBE_LENGTH_FOR_GEOMETRY", "NO_PETIOLE_OBJID", "PI_F"]
