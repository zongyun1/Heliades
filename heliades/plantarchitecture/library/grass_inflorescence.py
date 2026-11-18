# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/Assets.cpp:2458-2540, 2616-2625, 3196-3300 and core/src/Context.cpp:2114-2206
# (Copyright (C) 2016-2026 Brian Bailey).
"""Pieces shared by the procedural grass inflorescences (maize tassel, rice panicle, sorghum panicle).

* the organ build-parameter key tables (``OrganFloatKey`` / ``OrganIntegerKey`` / ``OrganColorKey``) with
  ``blendOrganParameters``, ``applyOrganBuildParameters`` and ``appendOrganBuildParameterDeclarations``;
* ``getBuildParameterIntegerValue`` (the raw overload, PL:203-209);
* ``perpendicularTo``, ``quantizedBrightness``, ``GrassInflorescenceMesh`` and ``addGrassAnther``;
* ``Context::addTube`` (the colored, object-less overload, Context.cpp:2114-2206), which the core does not have.

Float order: every vec3 expression is written left to right as the C++ evaluates it (no FMA in the oracle build);
``std::pow(float, float)`` is glibc ``powf``, ``std::round(float)`` is ``roundf``, ``cosf``/``sinf`` are glibc's.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np

from ...core import vec as V
from ...migration_registry import ports
from ..random_parameter import PlantArchitectureError
from .build_parameters import BuildParameterDeclaration, getBuildParameterValueRaw, _std_to_string

f32 = np.float32
PI_F = f32(3.14159265358979323846)
TWO_PI_F = f32(f32(2) * PI_F)

_m = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
_powf = _m.powf
_powf.restype, _powf.argtypes = ctypes.c_float, [ctypes.c_float, ctypes.c_float]
_sinf, _cosf = V.sinf, V.cosf


def powf(x, y) -> np.float32:
    return f32(_powf(float(f32(x)), float(f32(y))))


def roundf(x) -> np.float32:
    """``std::round(float)``: half away from zero."""
    x = float(x)
    return f32(math.copysign(math.floor(abs(x) + 0.5), x))


def vec(x, y, z) -> np.ndarray:
    return np.array([x, y, z], f32)


def normalize(v) -> np.ndarray:
    """free ``helios::normalize(const vec3&)``: ``v * (1/sqrtf(m2))`` with no zero check."""
    m2 = f32(f32(v[0] * v[0] + v[1] * v[1]) + v[2] * v[2])
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = f32(f32(1) / f32(np.sqrt(m2)))
        return v * inv


cross = V.cross
dot = V.dot


def rgb(r, g, b) -> np.ndarray:
    return np.array([r, g, b], f32)


def rgb_scale(color, factor) -> np.ndarray:
    """``RGBcolor::scale``: each channel times the factor (a negative factor throws)."""
    factor = f32(factor)
    if factor < 0:
        raise PlantArchitectureError("ERROR (RGBcolor::scale): cannot scale RGBcolor by negative factor.")
    return color * factor


def add_triangle(ctx, v0, v1, v2, color) -> int:
    """``Context::addTriangle(v0, v1, v2, const RGBcolor&)``: the color goes through ``make_RGBAcolor(color, 1)``, whose
    constructor clamps every channel to [0, 1] (``RGBcolor::scale`` does not clamp, so a brightened color can exceed 1).
    The Heliades core clamps primitive colors the same way since P1 (``scene._color_rgba``)."""
    return ctx.add_triangle(v0, v1, v2, np.asarray(color, f32))


# ---- build parameters --------------------------------------------------------------------------------------------

@ports("getBuildParameterIntegerValue(raw)")
def getBuildParameterIntegerValueRaw(build_parameters, parameter_name, default_value, min_value, max_value, description) -> int:
    """PL:203-209."""
    value = getBuildParameterValueRaw(build_parameters, parameter_name, f32(int(default_value)), f32(int(min_value)),
                                      f32(int(max_value)), description)
    if value != f32(round(float(value))):
        raise PlantArchitectureError(f"ERROR (getBuildParameterIntegerValue): build parameter '{parameter_name}' "
                                     f"({description}) has value {_std_to_string(value)}, but it must be a whole number.")
    return int(value)


class OrganKeyTables:
    """``OrganKeyTables<T>``: float keys ``(name, field, min, max, description)``, integer keys ``(name, field, min, max,
    description)`` and color keys ``(name, field, channel, description)``; fields are attribute names, colors are float32
    arrays indexed by channel (0 r, 1 g, 2 b)."""

    def __init__(self, float_keys, integer_keys, color_keys):
        self.float_keys = [(n, f, f32(lo), f32(hi), d) for n, f, lo, hi, d in float_keys]
        self.integer_keys = [(n, f, int(lo), int(hi), d) for n, f, lo, hi, d in integer_keys]
        self.color_keys = list(color_keys)


def _copy(params):
    out = type(params).__new__(type(params))
    out.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in params.__dict__.items()}
    return out


@ports("blendOrganParameters")
def blendOrganParameters(tables, from_, to, fraction):
    """AS:2499-2513."""
    fraction = f32(fraction)
    blended = _copy(from_)
    one_minus = f32(f32(1) - fraction)
    for _n, field, _lo, _hi, _d in tables.float_keys:
        setattr(blended, field, f32(f32(one_minus * getattr(from_, field)) + f32(fraction * getattr(to, field))))
    for _n, field, _lo, _hi, _d in tables.integer_keys:
        v = f32(f32(one_minus * f32(getattr(from_, field))) + f32(fraction * f32(getattr(to, field))))
        setattr(blended, field, int(roundf(v)))
    for _n, field, ch, _d in tables.color_keys:
        c = getattr(blended, field)
        c[ch] = f32(f32(one_minus * getattr(from_, field)[ch]) + f32(fraction * getattr(to, field)[ch]))
    return blended


@ports("applyOrganBuildParameters")
def applyOrganBuildParameters(tables, build_parameters, parameters):
    """AS:2515-2527."""
    for name, field, lo, hi, desc in tables.float_keys:
        setattr(parameters, field, getBuildParameterValueRaw(build_parameters, name, getattr(parameters, field), lo, hi, desc))
    for name, field, lo, hi, desc in tables.integer_keys:
        setattr(parameters, field, getBuildParameterIntegerValueRaw(build_parameters, name, getattr(parameters, field), lo, hi,
                                                                    desc))
    for name, field, ch, desc in tables.color_keys:
        c = getattr(parameters, field)
        c[ch] = getBuildParameterValueRaw(build_parameters, name, c[ch], f32(0), f32(1), desc)


@ports("appendOrganBuildParameterDeclarations")
def appendOrganBuildParameterDeclarations(tables, declarations):
    """AS:2529-2539."""
    for name, _f, lo, hi, desc in tables.float_keys:
        declarations.append(BuildParameterDeclaration(name, lo, hi, False, desc))
    for name, _f, lo, hi, desc in tables.integer_keys:
        declarations.append(BuildParameterDeclaration(name, f32(lo), f32(hi), True, desc))
    for name, _f, _ch, desc in tables.color_keys:
        declarations.append(BuildParameterDeclaration(name, 0.0, 1.0, False, desc))


def fromBuildParameters(cls, tables, type_declaration, subdivisions_declaration, subdivisions_field, low_preset,
                        high_preset, build_parameters):
    """The shared body of ``SorghumPanicleParameters`` / ``MaizeTasselParameters`` / ``RicePanicleParameters``
    ``::fromBuildParameters``: preset blend first (``<type> <= 0.5``: low preset -> defaults; above: defaults -> high
    preset), then each key, then the subdivisions key."""
    from .build_parameters import getBuildParameterValue, getBuildParameterIntegerValue
    build_parameters = build_parameters or {}
    parameters = cls()
    if type_declaration.name in build_parameters:
        t = getBuildParameterValue(build_parameters, type_declaration, f32(0.5))
        if t <= f32(0.5):
            parameters = blendOrganParameters(tables, low_preset(), cls(), f32(f32(2) * t))
        else:
            parameters = blendOrganParameters(tables, cls(), high_preset(), f32(f32(f32(2) * t) - f32(1)))
    applyOrganBuildParameters(tables, build_parameters, parameters)
    setattr(parameters, subdivisions_field, int(getBuildParameterIntegerValue(build_parameters, subdivisions_declaration,
                                                                              int(getattr(parameters, subdivisions_field)))))
    return parameters


# ---- geometry helpers --------------------------------------------------------------------------------------------

@ports("perpendicularTo")
def perpendicularTo(direction) -> np.ndarray:
    """AS:2621-2625."""
    reference = vec(1, 0, 0) if abs(direction[0]) < f32(0.9) else vec(0, 1, 0)
    return normalize(cross(direction, reference))


@ports("quantizedBrightness")
def quantizedBrightness(ctx, coefficient_of_variation) -> np.float32:
    """AS:3199-3204: ``round(clamp(1 + cv * randn(), 0.6, 1.4) / 0.04) * 0.04``, all float."""
    step = f32(0.04)
    v = V.clamp(f32(f32(1) + f32(f32(coefficient_of_variation) * ctx.randn())), f32(0.6), f32(1.4))
    return f32(roundf(f32(v / step)) * step)


class GrassInflorescenceMesh:
    """AS:3206-3238."""

    def __init__(self):
        self.vertices: list = []
        self.faces: list = []
        self.UUIDs: list = []

    def addVertex(self, vertex) -> int:
        self.vertices.append(vertex)
        return len(self.vertices) - 1

    def addTriangle(self, ctx, v0, v1, v2, color):
        self.UUIDs.append(add_triangle(ctx, self.vertices[v0], self.vertices[v1], self.vertices[v2], color))
        self.faces.append((v0, v1, v2))

    def addIsolatedTriangle(self, ctx, p0, p1, p2, color):
        first = self.addVertex(p0)
        self.addVertex(p1)
        self.addVertex(p2)
        self.addTriangle(ctx, first, first + 1, first + 2, color)

    def addTube(self, ctx, tube_UUIDs):
        for u in tube_UUIDs:
            first = len(self.vertices)
            self.vertices.extend(np.asarray(c, f32) for c in ctx.primitive_vertices(u))
            self.faces.append((first, first + 1, first + 2))
            self.UUIDs.append(u)

    def finish(self, ctx):
        """``setPrimitiveData(UUIDs, "object_label", "fruit")``, ``addPolymeshObject`` and
        ``setPolymeshObjectTopology(objID, vertices, faces, UUIDs, {}, {}, NORMAL_SOURCE_NONE)``."""
        ctx.set_primitive_data(self.UUIDs, "object_label", "fruit")
        objID = ctx.add_polymesh_object(self.UUIDs)
        ctx.set_polymesh_object_topology(objID, np.asarray(self.vertices, f32).reshape(-1, 3),
                                         np.asarray(self.faces, np.int64).reshape(-1, 3), self.UUIDs)
        return objID


GRASS_ANTHER_FACETS = 3


@ports("addGrassAnther")
def addGrassAnther(ctx, filament_base, hang_direction, anther_axis, filament_length, anther_length, anther_width, color,
                   mesh):
    """AS:3273-3300."""
    filament_width = f32(0.00025)
    anther_top = filament_base + hang_direction * f32(filament_length)
    anther_bottom = anther_top + anther_axis * f32(anther_length)
    side_a = perpendicularTo(anther_axis)
    side_b = cross(anther_axis, side_a)
    if filament_length > 0:
        filament_color = rgb_scale(color, 1.1)
        filament_side = perpendicularTo(hang_direction) * f32(f32(0.5) * filament_width)
        mesh.addIsolatedTriangle(ctx, filament_base - filament_side, filament_base + filament_side, anther_top, filament_color)
    half_width = f32(f32(0.5) * f32(anther_width))
    top_ring, bottom_ring = [], []
    for facet in range(GRASS_ANTHER_FACETS):
        angle = f32(f32(TWO_PI_F * f32(facet)) / f32(GRASS_ANTHER_FACETS))
        radial = side_a * f32(half_width * _cosf(angle)) + side_b * f32(half_width * _sinf(angle))
        top_ring.append(mesh.addVertex(anther_top + radial * f32(0.8)))
        bottom_ring.append(mesh.addVertex(anther_bottom + radial))
    for facet in range(GRASS_ANTHER_FACETS):
        nxt = (facet + 1) % GRASS_ANTHER_FACETS
        mesh.addTriangle(ctx, top_ring[facet], bottom_ring[facet], bottom_ring[nxt], color)
        mesh.addTriangle(ctx, top_ring[facet], bottom_ring[nxt], top_ring[nxt], color)


@ports("Context::addTube(color)")
def context_add_tube(ctx, radial_subdivisions, nodes, radius, color) -> list:
    """Context.cpp:2114-2206: the colored tube as loose triangles (no object). ``color`` is one RGB color per node."""
    n = len(nodes)
    if n == 0:
        raise PlantArchitectureError("ERROR (Context::addTube): Node and radius arrays are empty.")
    if n != len(radius):
        raise PlantArchitectureError("ERROR (Context::addTube): Size of `nodes' and `radius' arguments must agree.")
    if n != len(color):
        raise PlantArchitectureError("ERROR (Context::addTube): Size of `nodes' and `color' arguments must agree.")
    nr = int(radial_subdivisions)
    cfact = [_cosf(f32(f32(TWO_PI_F * f32(j)) / f32(nr))) for j in range(nr + 1)]
    sfact = [_sinf(f32(f32(TWO_PI_F * f32(j)) / f32(nr))) for j in range(nr + 1)]
    seed_vec = vec(0.1817, 0.6198, 0.7634)
    nvec = seed_vec.copy()
    nodes = [np.asarray(p, f32) for p in nodes]
    xyz = [[None] * n for _ in range(nr + 1)]
    half = f32(0.5)
    for i in range(n):
        r = f32(radius[i])
        if r < 0:
            raise PlantArchitectureError("ERROR (Context::addTube): Radius of tube must be positive.")
        if i == 0:
            v = nodes[1] - nodes[0]
        elif i == n - 1:
            v = nodes[i] - nodes[i - 1]
        else:
            v = half * ((nodes[i] - nodes[i - 1]) + (nodes[i + 1] - nodes[i]))
        v = V.normalize(v)                                        # vec3::normalize (member: zero-checked)
        if abs(float(dot(nvec, v))) > float(f32(0.95)):
            nvec = seed_vec.copy()
            if abs(float(dot(nvec, v))) > float(f32(0.95)):
                nvec = vec(1, 0, 0)
        if abs(float(v[2])) > float(f32(0.95)):
            nvec = vec(1, 0, 0)
        convec = V.normalize(cross(nvec, v))
        nvec = V.normalize(cross(v, convec))
        for j in range(nr + 1):
            cr, sr = f32(cfact[j] * r), f32(sfact[j] * r)
            xyz[j][i] = nodes[i] + (nvec * cr + convec * sr)
    UUIDs = []
    for i in range(n - 1):
        c = color[i]
        for j in range(nr):
            UUIDs.append(add_triangle(ctx, xyz[j][i], xyz[j + 1][i + 1], xyz[j + 1][i], c))
            UUIDs.append(add_triangle(ctx, xyz[j][i], xyz[j][i + 1], xyz[j + 1][i + 1], c))
    return UUIDs
