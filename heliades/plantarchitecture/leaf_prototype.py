# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/Assets.cpp:24-282, 517-924 and
# src/PlantArchitecture.cpp:181-276, 3449-3497, 4532-4596, 4844-5014 (Copyright (C) 2016-2026 Brian Bailey).
"""Leaf blade prototypes (W3-A): ``GenericLeafPrototype`` and the per-shoot-type prototype cache.

Every float expression follows its C++ type, operation by operation:

* ``f``-suffixed libm calls (``sinf``, ``cosf``, ``atanf``, ``powf``, ``atan2f``, ``remainderf``, ``hypotf``)
  go through glibc with ctypes, because glibc's float functions are not correctly rounded;
* unsuffixed calls on a float argument (``atan`` in the longitudinal curvature, ``exp``/``fmin``/``fabs`` in the
  petiole roll and the wave) are the C ``double`` functions (no ``using namespace std`` in Helios, and the
  translation unit only sees ``::atan(double)`` etc.), so they are evaluated in double and narrowed on
  assignment. Each choice is pinned by a parity probe (``plantarch.leafproto.*``, notes/W3-A.md);
* ``float * M_PI`` is a double product.

The Phytomer members ported here (``buildLeafPrototype``, ``labelLeafPrototype``, ``assignPetioluleMaterial`` and
the cache block of the constructor) live in :class:`PhytomerLeafMixin`, which needs these attributes on the
phytomer (the same names as the C++ members):

``context_ptr`` (Scene), ``plantarchitecture_ptr`` (PlantArchitecture), ``phytomer_parameters``
(PhytomerParameters), ``parent_shoot_ptr`` (anything with ``shoot_type_label``), ``petiole_objIDs`` (list of
object IDs), ``plantID``; and on the PlantArchitecture ``plant_instances[plantID].plant_name``.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math
from dataclasses import dataclass, field

import numpy as np

from ..core import vec as V
from ..core.random import MinstdRand0
from ..core.texture import try_resolve_file_path
from ..migration_registry import ports
from . import hooks
from .params import LeafShapeValues, vec3
from .random_parameter import PlantArchitectureError

f32 = np.float32

M_PI = math.pi
PI_F = f32(3.14159265358979)          # helios PI_F
FLOAT_MAX = f32(np.finfo(np.float32).max)

# ---- glibc float libm ------------------------------------------------------------------------------


def _libm():
    m = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    out = {}
    for name, nargs in (("sinf", 1), ("cosf", 1), ("atanf", 1), ("powf", 2), ("atan2f", 2), ("remainderf", 2),
                        ("hypotf", 2), ("sqrtf", 1)):
        fn = getattr(m, name)
        fn.restype = ctypes.c_float
        fn.argtypes = [ctypes.c_float] * nargs
        out[name] = fn
    return out


_M = _libm()


def sinf(x) -> np.float32:
    return f32(_M["sinf"](float(f32(x))))


def cosf(x) -> np.float32:
    return f32(_M["cosf"](float(f32(x))))


def atanf(x) -> np.float32:
    return f32(_M["atanf"](float(f32(x))))


def powf(x, y) -> np.float32:
    return f32(_M["powf"](float(f32(x)), float(f32(y))))


def atan2f(y, x) -> np.float32:
    return f32(_M["atan2f"](float(f32(y)), float(f32(x))))


def remainderf(x, y) -> np.float32:
    return f32(_M["remainderf"](float(f32(x)), float(f32(y))))


def hypotf(x, y) -> np.float32:
    return f32(_M["hypotf"](float(f32(x)), float(f32(y))))


def sqrtf(x) -> np.float32:
    return f32(np.sqrt(f32(x)))          # IEEE sqrt is correctly rounded


def _v3(x, y, z) -> np.ndarray:
    return np.array([x, y, z], f32)


def _vadd(a, b) -> np.ndarray:
    return (np.asarray(a, f32) + np.asarray(b, f32)).astype(f32)


def _vsub(a, b) -> np.ndarray:
    return (np.asarray(a, f32) - np.asarray(b, f32)).astype(f32)


def _vmul(s, v) -> np.ndarray:
    """float * vec3 (each component one float product)."""
    return (f32(s) * np.asarray(v, f32)).astype(f32)


# ---- petiolule constants (AS:24-35) ------------------------------------------------------------------

PETIOLULE_RADIAL_SUBDIVISIONS = 8
PETIOLULE_JOIN_RINGS = 6


@ports("petioluleRadialSubdivisions")
def petioluleRadialSubdivisions() -> int:
    return PETIOLULE_RADIAL_SUBDIVISIONS


@ports("petioluleVertexCount")
def petioluleVertexCount() -> int:
    return (2 + PETIOLULE_JOIN_RINGS - 1) * PETIOLULE_RADIAL_SUBDIVISIONS + 1


# ---- deformLeafLattice (AS:37-282) -----------------------------------------------------------------------

@ports("deformLeafLattice")
def deformLeafLattice(rest_vertices, Nx: int, Ny: int, scale, mature_scale, flexibility, taper=1.0,
                      gravity=(0.0, 0.0, -1.0)) -> np.ndarray:
    """Self-weight deflection of a blade lattice. Rigid leaves (``flexibility == 0`` or ``scale == 0``, every
    tomato leaf) return the rest vertices untouched (AS:56)."""
    rest = np.asarray(rest_vertices, f32).reshape(-1, 3)
    Nx, Ny = int(Nx), int(Ny)
    scale, mature_scale, flexibility, taper = f32(scale), f32(mature_scale), f32(flexibility), f32(taper)
    gravity = np.asarray(gravity, f32).reshape(3)
    vertex_count = (Nx + 1) * (Ny + 1)
    if len(rest) != vertex_count:
        raise PlantArchitectureError(f"ERROR (deformLeafLattice): Expected {vertex_count} vertices for a {Nx}x{Ny} leaf "
                                     f"lattice, but {len(rest)} were given.")
    elif scale < f32(0):
        raise PlantArchitectureError(f"ERROR (deformLeafLattice): Leaf scale must be non-negative, but {scale:f} was given.")
    elif flexibility < f32(0):
        raise PlantArchitectureError(f"ERROR (deformLeafLattice): Leaf flexibility must be non-negative, but {flexibility:f} was given.")
    elif mature_scale <= f32(0) and flexibility > f32(0):
        raise PlantArchitectureError("ERROR (deformLeafLattice): Mature leaf scale must be positive when the leaf is flexible, "
                                     f"but {mature_scale:f} was given. The mature size sets the length that the dimensionless "
                                     "flexibility is measured against.")
    elif not np.all(np.isfinite(gravity)) or V.magnitude(gravity) == f32(0):
        raise PlantArchitectureError(f"ERROR (deformLeafLattice): The gravity direction must be finite and non-zero, but "
                                     f"({gravity[0]:f}, {gravity[1]:f}, {gravity[2]:f}) was given.")

    if flexibility == f32(0) or scale == f32(0):
        return rest.copy()

    compliance = f32(flexibility / powf(max(mature_scale, f32(1e-6)), f32(4)))

    def idx(i, j):
        return j * (Nx + 1) + i

    j_mid = Ny // 2
    midrib = [rest[idx(i, j_mid)].copy() for i in range(Nx + 1)]

    ds = [f32(0)] * (Nx + 1)
    width = [f32(0)] * (Nx + 1)
    for i in range(Nx + 1):
        if i > 0:
            ds[i] = V.magnitude(_vsub(midrib[i], midrib[i - 1]))
        width[i] = V.magnitude(_vsub(rest[idx(i, Ny)], rest[idx(i, 0)]))

    theta_rest = [f32(0)] * (Nx + 1)
    for i in range(1, Nx + 1):
        seg = _vsub(midrib[i], midrib[i - 1])
        horizontal = sqrtf(f32(seg[0] * seg[0]) + f32(seg[1] * seg[1]))
        theta_rest[i] = atan2f(seg[2], horizontal)
    theta_rest[0] = theta_rest[1]

    gravity_direction = V.div(gravity, V.magnitude(gravity))
    station_heading = [None] * (Nx + 1)
    station_axis = [None] * (Nx + 1)
    turn_floor = [f32(f32(f32(-0.5) * PI_F) + f32(1e-3))] * (Nx + 1)
    turn_ceiling = [f32(f32(0.5) * PI_F)] * (Nx + 1)
    for i in range(Nx + 1):
        rs = _vsub(midrib[i], midrib[i - 1]) if i > 0 else _vsub(midrib[1], midrib[0])
        rh = sqrtf(f32(rs[0] * rs[0]) + f32(rs[1] * rs[1]))
        station_heading[i] = _v3(rs[0] / rh, rs[1] / rh, 0) if rh > f32(1e-9) else _v3(1, 0, 0)
        station_axis[i] = _v3(-station_heading[i][1], station_heading[i][0], 0)
        gravity_along = V.dot(gravity_direction, station_heading[i])
        gravity_vertical = gravity_direction[2]
        if hypotf(gravity_along, gravity_vertical) < f32(1e-6):
            turn_floor[i] = -FLOAT_MAX
            turn_ceiling[i] = FLOAT_MAX
            continue
        hanging_angle = atan2f(gravity_vertical, gravity_along)
        rest_from_hanging = remainderf(f32(theta_rest[i] - hanging_angle), f32(f32(2) * PI_F))
        hanging_branch = f32(theta_rest[i] - rest_from_hanging)
        if rest_from_hanging >= f32(0):
            turn_floor[i] = f32(hanging_branch + f32(1e-3))
            turn_ceiling[i] = f32(hanging_branch + PI_F)
        else:
            turn_floor[i] = f32(hanging_branch - PI_F)
            turn_ceiling[i] = f32(hanging_branch - f32(1e-3))

    taper_floor = max(taper, f32(1e-3))
    load = f32(0)
    straight_moment = f32(0)
    distal_mass = f32(0)
    for i in range(Nx - 1, 0, -1):
        io = i + 1
        segment_mass = f32(f32(f32(width[io] * ds[io]) * scale) * scale)
        lever_arm = f32(ds[io] * scale)
        straight_moment = f32(straight_moment + f32(f32(distal_mass * lever_arm) + f32(f32(segment_mass * f32(0.5)) * lever_arm)))
        distal_mass = f32(distal_mass + segment_mass)
        load = f32(load + f32(f32(f32(f32(compliance * powf(taper_floor, f32(f32(i) / f32(Nx)))) * straight_moment) * ds[i]) * scale))
    step = min(f32(0.5), f32(f32(1) / max(load, f32(1e-12))))
    maximum_moment_iterations = 5000
    converged_fraction = f32(1e-4)
    unconverged_limit = f32(0.01)

    theta = list(theta_rest)
    deflected = [np.zeros(3, f32) for _ in range(Nx + 1)]
    moment = [f32(0)] * (Nx + 1)
    theta_balanced = list(theta_rest)
    unbalanced = f32(0)

    def integrate():
        deflected[0] = _vmul(scale, midrib[0])                  # vec3 * float
        for i in range(1, Nx + 1):
            d = _vadd(_vmul(cosf(theta[i]), station_heading[i]), _v3(0, 0, sinf(theta[i])))
            deflected[i] = _vadd(deflected[i - 1], _vmul(f32(ds[i] * scale), d))

    for _iteration in range(maximum_moment_iterations):
        integrate()
        moment = [f32(0)] * (Nx + 1)
        distal_mass = f32(0)
        for i in range(Nx - 1, -1, -1):
            io = i + 1
            segment_mass = f32(f32(f32(width[io] * ds[io]) * scale) * scale)
            lever = _vsub(deflected[io], deflected[i])
            lever_arm = V.dot(V.cross(lever, gravity_direction), station_axis[i])
            moment[i] = f32(f32(moment[io] + f32(distal_mass * lever_arm)) + f32(f32(segment_mass * f32(0.5)) * lever_arm))
            distal_mass = f32(distal_mass + segment_mass)
        accumulated_bend = f32(0)
        residual = f32(0)
        largest_bend = f32(0)
        for i in range(1, Nx + 1):
            station = f32(f32(i) / f32(Nx))
            local_compliance = f32(compliance * powf(taper_floor, station))
            curvature = f32(-local_compliance * moment[i])
            accumulated_bend = f32(accumulated_bend + f32(f32(curvature * ds[i]) * scale))
            b = f32(theta_rest[i] + accumulated_bend)
            lo, hi = turn_floor[i], turn_ceiling[i]
            balanced = lo if b < lo else (hi if hi < b else b)          # std::clamp
            accumulated_bend = f32(balanced - theta_rest[i])
            residual = max(residual, abs(f32(balanced - theta[i])))
            largest_bend = max(largest_bend, abs(accumulated_bend))
            theta_balanced[i] = balanced
        unbalanced = residual
        if residual <= f32(converged_fraction * largest_bend):
            break
        for i in range(1, Nx + 1):
            theta[i] = f32(theta[i] + f32(step * f32(theta_balanced[i] - theta[i])))
        theta[0] = theta_rest[0]
    if unbalanced > unconverged_limit:
        raise PlantArchitectureError(f"ERROR (deformLeafLattice): The blade did not settle under its own weight within "
                                     f"{maximum_moment_iterations} passes; it was still {unbalanced:f} rad from balance at "
                                     f"flexibility {flexibility:f} and taper {taper:f}. This is far outside the flexibility a "
                                     "real leaf has.")
    integrate()

    deformed = np.zeros_like(rest)
    for i in range(Nx + 1):
        turn = f32(theta[i] - theta_rest[i])
        cos_turn, sin_turn = cosf(turn), sinf(turn)
        heading = station_heading[i]
        for j in range(Ny + 1):
            k = idx(i, j)
            offset = (_vsub(rest[k], midrib[i]) * scale).astype(f32)
            along = f32(f32(offset[0] * heading[0]) + f32(offset[1] * heading[1]))
            lateral = _vsub(_vsub(offset, _vmul(along, heading)), _v3(0, 0, offset[2]))
            along_rotated = f32(f32(along * cos_turn) - f32(offset[2] * sin_turn))
            vertical_rotated = f32(f32(along * sin_turn) + f32(offset[2] * cos_turn))
            deformed[k] = _vadd(_vadd(_vadd(deflected[i], _vmul(along_rotated, heading)), lateral),
                                _v3(0, 0, vertical_rotated))
    return deformed


# ---- buildLeafBladeLattice (AS:517-638) -------------------------------------------------------------------

@ports("buildLeafBladeLattice")
def buildLeafBladeLattice(shape: LeafShapeValues) -> np.ndarray:
    """(Nx+1)*(Ny+1) vertices, row j major: ``lattice[j*(Nx+1) + i]``."""
    Nx, Ny = int(shape.subdivisions_x), int(shape.subdivisions_y)
    if Nx == 0 or Ny < 2 or Ny % 2 != 0:
        raise PlantArchitectureError("ERROR (buildLeafBladeLattice): A leaf blade lattice needs at least one subdivision "
                                     "along the blade and an even number of at least two across it, but "
                                     f"{Nx} x {Ny} was given.")
    if not np.isfinite(shape.aspect_ratio) or shape.aspect_ratio <= f32(0):
        raise PlantArchitectureError("ERROR (buildLeafBladeLattice): The blade aspect ratio must be greater than zero, but "
                                     f"{shape.aspect_ratio:f} was given.")
    if not np.isfinite(shape.roll_fraction) or shape.roll_fraction < f32(0) or shape.roll_fraction > f32(1):
        raise PlantArchitectureError("ERROR (buildLeafBladeLattice): The roll fraction must be between 0 (flat) and 1 "
                                     f"(rolled into a closed tube), but {shape.roll_fraction:f} was given.")

    ar = f32(shape.aspect_ratio)
    mf = f32(shape.midrib_fold_fraction)
    lc = f32(shape.lateral_curvature)
    rf = f32(shape.roll_fraction)
    dx = f32(f32(1) / f32(Nx))
    dy = f32(ar / f32(Ny))
    cells_per_side = Ny // 2
    half_width = f32(f32(0.5) * ar)
    pure_fold = lc == f32(0) and rf == f32(0)
    roll_curvature = f32(f32(rf * PI_F) / half_width)
    substeps = 16
    ar4 = f32(f32(f32(ar * ar) * ar) * ar)

    cross_section = [[(f32(0), f32(0)) for _ in range(cells_per_side + 1)] for _ in range(Nx + 1)]
    if not pure_fold:
        for i in range(Nx + 1):
            x = f32(f32(i) * dx)
            fold_angle = f32(f32(f32(f32(f32(0.5) * mf) * f32(1.5)) * f32(f32(1) - f32(x * x))) * PI_F)
            for k in range(1, cells_per_side + 1):
                sum_cos = f32(0)
                sum_sin = f32(0)
                for m in range(substeps):
                    s = f32(f32(f32(k - 1) + f32(f32(f32(m) + f32(0.5)) / f32(substeps))) * dy)
                    lateral_slope = f32(f32(f32(f32(f32(4) * lc) * s) * s) * s) / ar4
                    psi = f32(f32(fold_angle + atanf(lateral_slope)) + f32(roll_curvature * s))
                    sum_cos = f32(sum_cos + cosf(psi))
                    sum_sin = f32(sum_sin + sinf(psi))
                direction_length = sqrtf(f32(f32(sum_cos * sum_cos) + f32(sum_sin * sum_sin)))
                # dy * make_vec2(c, s) / len: vec2 * scalar, then vec2::operator/ (multiply by 1/len)
                inv = f32(f32(1) / direction_length) if direction_length != 0 else f32(math.copysign(math.inf, direction_length))
                px, py = cross_section[i][k - 1]
                cross_section[i][k] = (f32(px + f32(f32(dy * sum_cos) * inv)), f32(py + f32(f32(dy * sum_sin) * inv)))

    ce = max(f32(1), f32(shape.longitudinal_curvature_exponent))       # fmaxf (operands never NaN here)
    lon = f32(shape.longitudinal_curvature)
    pr = f32(shape.petiole_roll)
    wp = f32(shape.wave_period)
    wa = f32(shape.wave_amplitude)
    offset = np.asarray(shape.lattice_offset, f32)
    nullorigin = np.zeros(3, f32)
    yaxis = _v3(0, 1, 0)

    lattice = np.zeros(((Nx + 1) * (Ny + 1), 3), f32)
    for j in range(Ny + 1):
        dtheta = f32(0)
        for i in range(Nx + 1):
            x = f32(f32(i) * dx)
            y = f32(f32(f32(j) * dy) - half_width)
            if pure_fold:
                local_fold = f32(f32(mf * f32(1.5)) * f32(f32(1) - f32(x * x)))
                arg = f32(float(f32(f32(0.5) * local_fold)) * M_PI)          # float*float, then *M_PI in double
                y_section = f32(cosf(arg) * y)
                z_section = f32(float(sinf(arg)) * abs(float(y)))           # sinf(..) * fabs(y): double product
            else:
                cells_out = abs(j - cells_per_side)
                side = f32(-1) if j < cells_per_side else f32(1)
                y_section = f32(side * cross_section[i][cells_out][0])
                z_section = cross_section[i][cells_out][1]

            z_petiole = f32(0)
            if pr != f32(0):
                # fmin(0.1f, pr * powf(7.f*y/ar, 4) * exp(-70.f*x)) - 0.01*pr/fabs(pr), all double after powf
                pw = powf(f32(f32(f32(7) * y) / ar), f32(4))
                a = float(f32(pr * pw)) * math.exp(float(f32(f32(-70) * x)))
                z_petiole = f32(min(float(f32(0.1)), a) - 0.01 * float(pr) / abs(float(pr)))

            z_wave = f32(0)
            if wp > f32(0) and wa > f32(0):
                jb = f32(1) if j >= 0.5 * Ny else f32(0)
                wave_phase = f32(float(f32(x + f32(wp * jb))) * M_PI / float(wp))
                z_wave = f32(((2.0 * abs(float(y))) * float(wa)) * float(sinf(wave_phase)))

            vertex = _v3(x, y_section, f32(z_section + z_petiole))
            rot_angle = f32(0)
            if lon != f32(0) and i > 0:
                arg = f32(f32(f32(ce * lon) * powf(x, f32(ce - f32(1)))) * dx)
                dtheta = f32(float(dtheta) - math.atan(float(arg)))
                vertex = V.rotate_point_about_line(vertex, nullorigin, yaxis, dtheta)
                rot_angle = f32(rot_angle + dtheta)

            if z_wave != f32(0):
                vertex[0] = f32(vertex[0] + f32(z_wave * sinf(rot_angle)))
                vertex[2] = f32(vertex[2] + f32(z_wave * cosf(rot_angle)))

            lattice[j * (Nx + 1) + i] = _vadd(vertex, offset)
    return lattice


# ---- texture path (PA:669-711) -------------------------------------------------------------------------

def _plugin_roots():
    """Directories whose ``plugins/plantarchitecture`` Helios would see: $HELIOS_BUILD / cwd / parent
    (``tryResolvePluginAsset``), then, as a Heliades fallback, the pinned Helios tree found by walking up
    from this package (``third_party/pyhelios_src/helios-core``)."""
    import os
    roots = []
    from ..core.texture import _build_directory
    roots.append(os.getcwd())
    roots.append(_build_directory())
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        cand = os.path.join(d, "third_party", "pyhelios_src", "helios-core")
        if os.path.isdir(os.path.join(cand, "plugins", "plantarchitecture")):
            roots.append(cand)
            break
        d = os.path.dirname(d)
    return roots


def _try_plugin_asset(asset_path: str) -> str:
    import os
    for r in _plugin_roots():
        p = os.path.join(r, "plugins", "plantarchitecture", asset_path)
        if os.path.exists(p):
            return os.path.realpath(p)
    return ""


@ports("PlantArchitecture::resolveTextureFile")
def resolveTextureFile(texture_file: str) -> str:
    import os
    if not texture_file:
        return ""
    if os.path.isabs(texture_file) and os.path.exists(texture_file):
        return texture_file
    r = try_resolve_file_path(texture_file)
    if r:
        return r
    r = _try_plugin_asset(texture_file)
    if r:
        return r
    if not texture_file.startswith("assets/"):
        filename = os.path.basename(texture_file)
        ext = os.path.splitext(texture_file)[1]
        sub = "assets/obj/" if ext in (".obj", ".mtl") else "assets/textures/"
        r = _try_plugin_asset(sub + filename)
        if r:
            return r
    raise PlantArchitectureError(f"ERROR (PlantArchitecture): Could not resolve asset file: {texture_file}. Tried: direct "
                                 "path, plugin asset path, and assets/ subdirectory prefix.")


# ---- GenericLeafPrototype (AS:640-924) ---------------------------------------------------------------------

@ports("GenericLeafPrototype")
def GenericLeafPrototype(context_ptr, prototype_parameters, compound_leaf_index: int) -> int:
    """Build one leaf prototype polymesh (textured blade + optional petiolule) and return its object ID."""
    pp = prototype_parameters
    ctx = context_ptr
    compound_leaf_index = int(compound_leaf_index)

    if pp.OBJ_model_file:
        resolved_obj = resolveTextureFile(pp.OBJ_model_file)
        return ctx.add_polymesh_object(ctx.load_obj(resolved_obj, pp.leaf_offset, f32(0), (0.0, 0.0), (0.0, 0.0, 0.0),
                                                    "ZUP", True))

    if pp.leaf_texture_series:
        leaf_texture = pp.getTextureSeriesVariant(compound_leaf_index, pp.texture_series_selector)[0]
    elif not pp.leaf_texture_file:
        raise PlantArchitectureError("ERROR (PlantArchitecture): Leaf prototype texture file was not specified.")
    elif len(pp.leaf_texture_file) == 1:
        leaf_texture = pp.leaf_texture_file[min(pp.leaf_texture_file)]
    elif compound_leaf_index not in pp.leaf_texture_file:
        raise PlantArchitectureError("ERROR (PlantArchitecture): Leaf prototype texture file for compound leaf index "
                                     f"{compound_leaf_index} was not found.")
    else:
        leaf_texture = pp.leaf_texture_file[compound_leaf_index]
    leaf_texture = resolveTextureFile(leaf_texture)

    if pp.leaf_texture_series:
        validate = getattr(ctx, "validate_color_texture", None)
        if validate is None:
            raise NotImplementedError("Context::validateColorTexture is not ported (leaf texture series)")
        for stage_texture in pp.getTextureSeriesVariant(compound_leaf_index, pp.texture_series_selector):
            validate(leaf_texture, resolveTextureFile(stage_texture))

    UUIDs: list[int] = []
    Nx = int(pp.subdivisions)
    Ny = int(math.ceil(f32(pp.leaf_aspect_ratio.val() * f32(Nx))))          # ceil(float) -> uint
    if Ny % 2 != 0:
        Ny = Ny + 1
    dx = f32(f32(1) / f32(Nx))
    dy = f32(pp.leaf_aspect_ratio.val() / f32(Ny))

    rfv = pp.roll_fraction.val()
    if not np.isfinite(rfv) or rfv < f32(0) or pp.roll_fraction.val() > f32(1):
        raise PlantArchitectureError("ERROR (PlantArchitecture): Leaf prototype roll_fraction must be between 0 (flat) and 1 "
                                     f"(rolled into a closed tube), but {pp.roll_fraction.val():f} was given.")

    shape = LeafShapeValues()
    shape.aspect_ratio = pp.leaf_aspect_ratio.val()
    shape.midrib_fold_fraction = pp.midrib_fold_fraction.val()
    shape.longitudinal_curvature = pp.longitudinal_curvature.val()
    shape.longitudinal_curvature_exponent = pp.longitudinal_curvature_exponent.val()
    shape.lateral_curvature = pp.lateral_curvature.val()
    shape.petiole_roll = pp.petiole_roll.val()
    shape.wave_period = pp.wave_period.val()
    shape.wave_amplitude = pp.wave_amplitude.val()
    shape.roll_fraction = pp.roll_fraction.val()
    shape.subdivisions_x = Nx
    shape.subdivisions_y = Ny
    shape.lattice_offset = (0, 0, 0)
    lattice = buildLeafBladeLattice(shape)

    ar = pp.leaf_aspect_ratio.val()
    mesh_vertices = [lattice[j * (Nx + 1) + i].copy() for j in range(Ny + 1) for i in range(Nx + 1)]
    mesh_vertex_uv = [(f32(f32(i) * dx), f32(f32(f32(j) * dy) / ar)) for j in range(Ny + 1) for i in range(Nx + 1)]

    def lvi(i, j):
        return j * (Nx + 1) + i

    mesh_faces: list[tuple] = []
    mesh_face_UUIDs: list[int] = []
    half = f32(f32(0.5) * ar)
    for j in range(Ny):
        for i in range(Nx):
            x = f32(f32(i) * dx)
            y = f32(f32(f32(j) * dy) - half)
            xr = f32(x + dx)
            ylo = f32(f32(y + half) / ar)
            yhi = f32(f32(f32(y + dy) + half) / ar)
            uv0, uv1, uv2, uv3 = (x, ylo), (xr, ylo), (xr, yhi), (x, yhi)
            v0 = lattice[lvi(i, j)]
            v1 = lattice[lvi(i + 1, j)]
            v2 = lattice[lvi(i + 1, j + 1)]
            v3 = lattice[lvi(i, j + 1)]

            u1 = ctx.add_triangle_textured(v0, v1, v2, leaf_texture, uv0, uv1, uv2)
            if ctx.primitive_area(u1) > 0:
                UUIDs.append(u1)
                mesh_faces.append((lvi(i, j), lvi(i + 1, j), lvi(i + 1, j + 1)))
                mesh_face_UUIDs.append(u1)
            else:
                ctx.delete_primitive(u1)

            u2 = ctx.add_triangle_textured(v0, v2, v3, leaf_texture, uv0, uv2, uv3)
            if ctx.primitive_area(u2) > 0:
                UUIDs.append(u2)
                mesh_faces.append((lvi(i, j), lvi(i + 1, j + 1), lvi(i, j + 1)))
                mesh_face_UUIDs.append(u2)
            else:
                ctx.delete_primitive(u2)

    leaf_offset = np.asarray(pp.leaf_offset, f32)
    _translate_primitives(ctx, UUIDs, leaf_offset)
    mesh_vertices = [_vadd(v, leaf_offset) for v in mesh_vertices]
    shape.lattice_offset = tuple(leaf_offset)

    if pp.build_petiolule:
        if not pp.petiolule_length:
            raise PlantArchitectureError("ERROR (PlantArchitecture): Leaf prototype builds a petiolule but "
                                         "LeafPrototype::petiolule_length is empty.")
        elif len(pp.petiolule_length) == 1:
            petiolule_length = f32(pp.petiolule_length[min(pp.petiolule_length)])
        elif compound_leaf_index not in pp.petiolule_length:
            raise PlantArchitectureError("ERROR (PlantArchitecture): LeafPrototype::petiolule_length has no entry for compound "
                                         f"leaf index {compound_leaf_index}. With more than one entry, every leaflet index needs "
                                         "its own; give a single entry to use one length for all leaflets.")
        else:
            petiolule_length = f32(pp.petiolule_length[compound_leaf_index])
        petiolule_radius = f32(pp.petiolule_radius_fraction)
        if petiolule_length <= f32(0) or petiolule_radius <= f32(0):
            raise PlantArchitectureError(f"ERROR (PlantArchitecture): A petiolule needs a positive length and radius (got length "
                                         f"{petiolule_length:f}, radius {petiolule_radius:f}). The length is "
                                         "LeafPrototype::petiolule_length; the radius is taken from the petiole, so the petiole must "
                                         "have a positive radius.")
        blade_shift = _v3(petiolule_length, 0, 0)
        _translate_primitives(ctx, UUIDs, blade_shift)
        shape.lattice_offset = tuple(_vadd(shape.lattice_offset, blade_shift))
        mesh_vertices = [_vadd(v, blade_shift) for v in mesh_vertices]

        join_length = f32(f32(4) * petiolule_radius)
        ring_x = [f32(0), petiolule_length]
        ring_half_width = [petiolule_radius, petiolule_radius]
        ring_half_thickness = [petiolule_radius, petiolule_radius]
        for ring in range(1, PETIOLULE_JOIN_RINGS):
            t = f32(f32(ring) / f32(PETIOLULE_JOIN_RINGS))
            ring_x.append(f32(petiolule_length + f32(t * join_length)))
            ring_half_width.append(f32(petiolule_radius * sqrtf(f32(f32(1) - f32(t * t)))))
            ring_half_thickness.append(f32(f32(petiolule_radius * f32(0.5)) * f32(f32(1) + cosf(f32(PI_F * t)))))

        first_ring_vertex = len(mesh_vertices)
        nr = PETIOLULE_RADIAL_SUBDIVISIONS
        for ring in range(len(ring_x)):
            for k in range(nr):
                theta = f32(f32(f32(f32(2) * PI_F) * f32(k)) / f32(nr))
                mesh_vertices.append(_vadd(leaf_offset, _v3(ring_x[ring], f32(ring_half_width[ring] * cosf(theta)),
                                                            f32(ring_half_thickness[ring] * sinf(theta)))))
                mesh_vertex_uv.append((f32(0), f32(0)))
        tip_vertex = len(mesh_vertices)
        mesh_vertices.append(_vadd(leaf_offset, _v3(f32(petiolule_length + join_length), 0, 0)))
        mesh_vertex_uv.append((f32(0), f32(0)))

        UUIDs_petiolule: list[int] = []

        def facet(i0, i1, i2):
            u = ctx.add_triangle(mesh_vertices[i0], mesh_vertices[i1], mesh_vertices[i2])
            UUIDs_petiolule.append(u)
            mesh_faces.append((i0, i1, i2))
            mesh_face_UUIDs.append(u)

        for ring in range(len(ring_x) - 1):
            for k in range(nr):
                k_next = (k + 1) % nr
                near_k = first_ring_vertex + ring * nr + k
                near_next = first_ring_vertex + ring * nr + k_next
                facet(near_k, near_next, near_next + nr)
                facet(near_k, near_next + nr, near_k + nr)
        last_ring_start = first_ring_vertex + (len(ring_x) - 1) * nr
        for k in range(nr):
            facet(last_ring_start + k, last_ring_start + (k + 1) % nr, tip_vertex)
        if UUIDs_petiolule:
            ctx.set_primitive_data(UUIDs_petiolule, "object_label", "petiolule")
        UUIDs.extend(UUIDs_petiolule)

    pp.leaf_aspect_ratio.resample()
    pp.midrib_fold_fraction.resample()
    pp.longitudinal_curvature.resample()
    pp.longitudinal_curvature_exponent.resample()
    pp.lateral_curvature.resample()
    pp.roll_fraction.resample()
    pp.petiole_roll.resample()

    pp.last_built_shape = shape
    pp.last_built_shape_valid = True
    pp.wave_period.resample()
    pp.wave_amplitude.resample()

    objID = ctx.add_polymesh_object(UUIDs)

    if mesh_faces and len(mesh_face_UUIDs) == len(ctx.object_primitive_uuids(objID)):
        from ..core.polymesh import NORMAL_SOURCE_NONE
        ctx.set_polymesh_object_topology(objID, np.array(mesh_vertices, f32).reshape(-1, 3),
                                         np.array(mesh_faces, np.int64).reshape(-1, 3), mesh_face_UUIDs, (),
                                         np.array(mesh_vertex_uv, f32).reshape(-1, 2), NORMAL_SOURCE_NONE)
        ctx.compute_polymesh_object_vertex_normals(objID, f32(180))

    class_map = ""
    if len(pp.leaf_class_map_file) == 1:
        class_map = pp.leaf_class_map_file[min(pp.leaf_class_map_file)]
    elif compound_leaf_index in pp.leaf_class_map_file:
        class_map = pp.leaf_class_map_file[compound_leaf_index]
    if class_map:
        if pp.leaf_texture_series:
            raise PlantArchitectureError("ERROR (PlantArchitecture): A leaf prototype has both a class map "
                                         "(LeafPrototype::leaf_class_map_file) and a texture series "
                                         "(LeafPrototype::leaf_texture_series), which cannot be combined.")
        raise NotImplementedError("PlantArchitecture::recordSpectralClassMap is not ported (leaf class maps)")

    return objID


def _translate_primitives(ctx, uuids, shift):
    """Context::translatePrimitive(vector, shift): one translation matrix applied to each primitive.
    (Heliades skips the product for a zero shift; multiplying by an identity matrix is exact apart from
    the sign of zeros, which no comparison distinguishes.)"""
    for u in uuids:
        ctx.translate_primitive(u, shift)


# ---- PlantArchitecture members (PA:181-276, 4532-4596) -----------------------------------------------------

@dataclass
class OrganMaterial:
    """PlantArchitecture::OrganMaterial (H:6707). Normal maps are not modelled by the Heliades material
    table, so their fields keep the defaults on both sides of every comparison."""
    label: str
    color: tuple
    texture_file: str = ""
    texture_color_overridden: bool = False
    normal_map_file: str = ""
    normal_map_kind: int = 0
    normal_map_strength: np.float32 = f32(1)
    normal_map_flip_green: bool = False


@dataclass
class LeafRestGeometry:
    """PlantArchitecture::LeafRestGeometry (H:6643)."""
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), f32))
    appended_vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), f32))
    subdivisions_x: int = 0
    subdivisions_y: int = 0
    shape: LeafShapeValues = field(default_factory=LeafShapeValues)
    has_shape: bool = False
    lattice_frame_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, f32))


def _pa_state(pa):
    """The PlantArchitecture members this module owns (created on first use)."""
    for name, default in (("unique_leaf_prototype_objIDs", dict), ("unique_leaf_prototype_rest_geometry", dict),
                          ("organ_material_labels", dict)):
        if not hasattr(pa, name):
            setattr(pa, name, default())
    return pa


@ports("PlantArchitecture::nameOrganMaterials")
def nameOrganMaterials(pa, objIDs, desired_base_name: str):
    """Give the organ's automatic materials the label ``desired_base_name`` (or a suffixed one), sharing
    a material already named for organs of the same kind and appearance (PA:181-264). ``objIDs`` may be one ID
    or a list (the vector overload)."""
    _pa_state(pa)
    if not np.isscalar(objIDs):
        for o in objIDs:
            nameOrganMaterials(pa, o, desired_base_name)
        return
    ctx = pa.context_ptr
    labels = pa.organ_material_labels
    prims_of_auto: dict[str, list[int]] = {}
    for u in ctx.object_primitive_uuids(int(objIDs)):
        lbl = ctx.get_primitive_material_label(u)
        if lbl[:7] == "__auto_":
            prims_of_auto.setdefault(lbl, []).append(u)

    for automatic_label in sorted(prims_of_auto):          # std::map iteration order
        primitives = prims_of_auto[automatic_label]
        key = (automatic_label, desired_base_name)
        named = labels.get(key)
        if named is not None and ctx.does_material_exist(named.label):
            ctx.assign_material_to_primitive(primitives, named.label)
            continue
        automatic = ctx.get_material(ctx.get_primitive_material_id(primitives[0]))
        a_color = np.asarray(automatic.color, f32)
        a_tex = automatic.texture_file or ""
        a_over = bool(automatic.texture_color_overridden)

        same = None
        for other_key in sorted(labels):
            other = labels[other_key]
            if (other_key[1] != desired_base_name or not ctx.does_material_exist(other.label) or other.texture_file != a_tex
                    or other.texture_color_overridden != a_over or other.normal_map_file != ""
                    or other.normal_map_kind != 0 or other.normal_map_strength != f32(1) or other.normal_map_flip_green):
                continue
            oc = np.asarray(other.color, f32)
            if all(abs(f32(oc[c] - a_color[c])) < f32(1e-4) for c in range(4)):
                same = other
                break
        if same is not None:
            ctx.assign_material_to_primitive(primitives, same.label)
            labels[key] = same
            continue

        label = desired_base_name
        suffix = 1
        while ctx.does_material_exist(label):
            label = f"{desired_base_name}_{suffix}"
            suffix += 1
        if automatic.reference_count == len(primitives):
            ctx.rename_material(automatic_label, label)
        else:
            ctx.add_material(label)
            ctx.set_material_color(label, a_color)
            if a_tex:
                ctx.set_material_texture(label, a_tex)
            ctx.set_material_texture_color_override(label, a_over)
            ctx.set_material_twosided_flag(label, automatic.twosided_flag)
            ctx.assign_material_to_primitive(primitives, label)
        labels[key] = OrganMaterial(label, tuple(a_color), a_tex, a_over)


@ports("PlantArchitecture::getCachedLeafPrototypeObjID")
def getCachedLeafPrototypeObjID(pa, prototype_identifier, leaves_per_petiole, prototype_index, leaf_index,
                                calling_function: str) -> int:
    _pa_state(pa)
    cached = pa.unique_leaf_prototype_objIDs.get((int(prototype_identifier), int(leaves_per_petiole)))
    if cached is None:
        raise PlantArchitectureError(f"ERROR ({calling_function}): No leaf blade prototypes have been cached for a petiole of "
                                     f"{leaves_per_petiole} leaflets. The cached blades of a leaf prototype are built for one "
                                     "leaflet count at a time, and this count was never built.")
    elif len(cached) <= prototype_index:
        raise PlantArchitectureError(f"ERROR ({calling_function}): Leaf prototype {prototype_index} was requested, but only "
                                     f"{len(cached)} unique leaf prototypes were cached for a petiole of {leaves_per_petiole} "
                                     "leaflets.")
    elif len(cached[prototype_index]) <= leaf_index:
        raise PlantArchitectureError(f"ERROR ({calling_function}): Leaflet {leaf_index} was requested, but the cached leaf "
                                     f"prototype holds blades for only {len(cached[prototype_index])} leaflet positions of a "
                                     f"petiole of {leaves_per_petiole} leaflets.")
    return cached[prototype_index][leaf_index]


@ports("PlantArchitecture::recordLeafPrototypeRestGeometry")
def recordLeafPrototypeRestGeometry(pa, prototype_params, leaves_per_petiole, prototype_index, objID_leaf,
                                    leaf_flexibility):
    _pa_state(pa)
    ctx = pa.context_ptr
    rg = LeafRestGeometry()
    rg.has_shape = bool(prototype_params.last_built_shape_valid)
    if rg.has_shape:
        rg.shape = prototype_params.last_built_shape.copy()
    T = ctx.object_transform(objID_leaf)
    origin = _v3(T[3], T[7], T[11])
    rg.lattice_frame_offset = origin
    if f32(leaf_flexibility) > f32(0):
        verts = [_vsub(v, origin) for v in ctx.polymesh_object_vertices(objID_leaf)]
        n_pet = petioluleVertexCount()
        if prototype_params.build_petiolule and len(verts) > n_pet:
            rg.appended_vertices = np.array(verts[-n_pet:], f32).reshape(-1, 3)
            verts = verts[:-n_pet]
        rg.subdivisions_x = int(prototype_params.subdivisions)
        if rg.subdivisions_x > 0 and len(verts) % (rg.subdivisions_x + 1) == 0:
            rg.subdivisions_y = len(verts) // (rg.subdivisions_x + 1) - 1
            rg.vertices = np.array(verts, f32).reshape(-1, 3)
        else:
            rg.vertices = np.zeros((0, 3), f32)
            rg.appended_vertices = np.zeros((0, 3), f32)
            rg.subdivisions_x = 0
    key = (int(prototype_params.unique_prototype_identifier), max(int(leaves_per_petiole), 0))
    pa.unique_leaf_prototype_rest_geometry[key][prototype_index].append(rg)


# ---- Phytomer members ----------------------------------------------------------------------------------

def _distribution_mean(parameter) -> np.float32:
    """static distributionMean (PA:4844): mean of uniform/normal/weibull, else ``val()`` (no draw for a constant)."""
    d, p = parameter.distribution, parameter.distribution_parameters
    if d == "uniform":
        return f32(f32(0.5) * f32(p[0] + p[1]))
    elif d == "normal":
        return f32(p[0])
    elif d == "weibull":
        return f32(f32(p[1]) * f32(math.gamma(float(f32(f32(1) + f32(f32(1) / f32(p[0])))))))   # std::tgamma(float)
    return parameter.val()


def _clamp_offset(count_per_axis: int, offset) -> np.float32:
    """static clampOffset (PA:266)."""
    offset = f32(offset)
    if count_per_axis > 2:
        denom = f32(f32(f32(0.5) * f32(count_per_axis)) - f32(1))
        if f32(offset * denom) > f32(1):
            offset = f32(f32(1) / denom)
    return offset


@ports("leafletSizeMax")
def leafletSizeMax(ind_from_tip, leaflet_scale, intercalary_leaflet_scale, prototype_scale) -> np.float32:
    """static leafletSizeMax (PA:309-323)."""
    slot = abs(f32(ind_from_tip))
    leaflet_scale, intercalary_leaflet_scale, prototype_scale = f32(leaflet_scale), f32(intercalary_leaflet_scale), f32(prototype_scale)
    if slot == f32(0) or leaflet_scale == f32(1):
        return prototype_scale
    if intercalary_leaflet_scale <= f32(0):
        return f32(powf(leaflet_scale, slot) * prototype_scale)
    pair_index = int(math.ceil(slot))
    size = f32(powf(leaflet_scale, f32((pair_index + 1) // 2)) * prototype_scale)
    return f32(intercalary_leaflet_scale * size) if pair_index % 2 == 0 else size


class PhytomerLeafMixin:
    """Phytomer methods of the leaf prototype path (PA:3449-3497, 4904-5014). See the module docstring for the
    attributes they need."""

    @ports("Phytomer::buildLeafPrototype")
    def buildLeafPrototype(self, leaves_per_petiole: int, ind_from_tip, prototype_index: int = 0,
                           leaflet_size_own=-1.0, petiole_radius_own=-1.0, petiolule_length_own=-1.0) -> int:
        pa = self.plantarchitecture_ptr
        prototype = self.phytomer_parameters.leaf.prototype
        shoot_type_label = self.parent_shoot_ptr.shoot_type_label
        ind_from_tip = f32(ind_from_tip)
        leaflet_size_own, petiole_radius_own = f32(leaflet_size_own), f32(petiole_radius_own)
        petiolule_length_own = f32(petiolule_length_own)

        prototype.validateTextureSeries(shoot_type_label)
        prototype.texture_series_selector = prototype_index

        if prototype.build_petiolule and leaflet_size_own > f32(0) and petiole_radius_own > f32(0):
            prototype.petiolule_radius_fraction = f32(petiole_radius_own / leaflet_size_own)
        elif prototype.build_petiolule:
            if shoot_type_label not in pa.shoot_types:
                raise PlantArchitectureError(f"ERROR (PlantArchitecture::Phytomer::buildLeafPrototype): Shoot type "
                                             f"'{shoot_type_label}' is not defined, but its leaf prototype builds a petiolule whose "
                                             "radius is taken from the shoot type's petiole.")
            tp = pa.shoot_types[shoot_type_label].phytomer_parameters
            attachment_fraction = f32(1)
            leaflet_offset_val = _clamp_offset(int(leaves_per_petiole), _distribution_mean(tp.leaf.leaflet_offset))
            if leaves_per_petiole > 1 and leaflet_offset_val > f32(0) and ind_from_tip != f32(0):
                attachment_fraction = f32(f32(1) - f32(f32(abs(ind_from_tip) - f32(0.5)) * leaflet_offset_val))
            petiole_radius = f32(_distribution_mean(tp.petiole.radius)
                                 * f32(f32(1) - f32(_distribution_mean(tp.petiole.taper) * attachment_fraction)))
            leaflet_size = leafletSizeMax(ind_from_tip, _distribution_mean(tp.leaf.leaflet_scale),
                                          _distribution_mean(tp.leaf.intercalary_leaflet_scale),
                                          _distribution_mean(tp.leaf.prototype_scale))
            if petiole_radius <= f32(0) or leaflet_size <= f32(0):
                raise PlantArchitectureError(f"ERROR (PlantArchitecture::Phytomer::buildLeafPrototype): The leaf prototype of shoot "
                                             f"type '{shoot_type_label}' builds a petiolule, whose radius is taken from the petiole, "
                                             f"but the petiole radius ({petiole_radius:f} m) or leaflet size ({leaflet_size:f} m) is "
                                             "not positive. Give the petiole a radius or set build_petiolule to false.")
            prototype.petiolule_radius_fraction = f32(petiole_radius / leaflet_size)

        petiolule_length_species = dict(prototype.petiolule_length)
        own_stalk = prototype.build_petiolule and petiolule_length_own > f32(0) and leaflet_size_own > f32(0)
        if own_stalk:
            prototype.petiolule_length = {0: f32(petiolule_length_own / leaflet_size_own)}

        prototype.last_built_shape_valid = False
        objID_leaf = prototype.prototype_function(self.context_ptr, prototype, int(ind_from_tip))
        if own_stalk:
            prototype.petiolule_length = petiolule_length_species
        self.last_built_leaf_shape_valid = bool(prototype.last_built_shape_valid)
        if self.last_built_leaf_shape_valid:
            self.last_built_leaf_shape = prototype.last_built_shape.copy()
        return objID_leaf

    @ports("Phytomer::labelLeafPrototype")
    def labelLeafPrototype(self, objID_leaf: int):
        ctx = self.context_ptr
        if self.phytomer_parameters.leaf.prototype.prototype_function is hooks.GenericLeafPrototype:
            object_UUIDs = ctx.object_primitive_uuids(objID_leaf)
            keep = set(ctx.filter_primitives_by_data(object_UUIDs, "object_label", "petiolule"))
            blade_UUIDs = [u for u in object_UUIDs if u not in keep]
            if blade_UUIDs:
                ctx.set_primitive_data(blade_UUIDs, "object_label", "leaf")
        self.assignPetioluleMaterial(objID_leaf)

    @ports("Phytomer::assignPetioluleMaterial")
    def assignPetioluleMaterial(self, objID_leaf: int):
        ctx = self.context_ptr
        pa = self.plantarchitecture_ptr
        petiolule_UUIDs = ctx.filter_primitives_by_data(ctx.object_primitive_uuids(objID_leaf), "object_label", "petiolule")
        if not petiolule_UUIDs:
            return
        material_label = ""
        for objID_petiole in self.petiole_objIDs:
            if ctx.does_object_exist(objID_petiole) and ctx.object_primitive_uuids(objID_petiole):
                material_label = ctx.get_primitive_material_label(ctx.object_primitive_uuids(objID_petiole)[0])
                break
        if not material_label:
            _pa_state(pa)
            material_label = (pa.plant_instances[self.plantID].plant_name + "_" + self.parent_shoot_ptr.shoot_type_label
                              + "_petiole")
            if not ctx.does_material_exist(material_label):
                c = self.phytomer_parameters.petiole.color
                rgba = (f32(c[0]), f32(c[1]), f32(c[2]), f32(1))
                ctx.add_material(material_label)
                ctx.set_material_color(material_label, rgba)
                pa.organ_material_labels[("", material_label)] = OrganMaterial(material_label, rgba, "", False)
        ctx.assign_material_to_primitive(petiolule_UUIDs, material_label)

    @ports("Phytomer::Phytomer (leaf prototype cache, PA:3449-3497)")
    def buildLeafPrototypeCache(self, leaves_per_petiole: int, leaf_flexibility=0.0) -> bool:
        """Build the hidden blade prototypes of ``(identifier, leaves_per_petiole)`` from the private stream if
        they are not cached yet. Returns whether it built them."""
        pa = _pa_state(self.plantarchitecture_ptr)
        ctx = self.context_ptr
        proto = self.phytomer_parameters.leaf.prototype
        assert proto.unique_prototype_identifier != 0
        key = (int(proto.unique_prototype_identifier), max(int(leaves_per_petiole), 0))
        if not (proto.unique_prototypes > 0 and key not in pa.unique_leaf_prototype_objIDs):
            return False
        n = int(proto.unique_prototypes)
        pa.unique_leaf_prototype_objIDs[key] = [[] for _ in range(n)]
        pa.unique_leaf_prototype_rest_geometry[key] = [[] for _ in range(n)]

        prototype_generator = MinstdRand0(proto.unique_prototype_identifier)
        context_generator = proto.setRandomGenerator(prototype_generator)
        material_base_name = (pa.plant_instances[self.plantID].plant_name + "_" + self.parent_shoot_ptr.shoot_type_label
                              + "_leaf")
        for prototype in range(n):
            for leaf in range(int(leaves_per_petiole)):
                ind_from_tip = f32(f32(leaf) - f32(f32(leaves_per_petiole - 1) / f32(2)))
                objID_leaf = self.buildLeafPrototype(leaves_per_petiole, ind_from_tip, prototype)
                self.labelLeafPrototype(objID_leaf)
                pa.unique_leaf_prototype_objIDs[key][prototype].append(objID_leaf)
                recordLeafPrototypeRestGeometry(pa, proto, leaves_per_petiole, prototype, objID_leaf, leaf_flexibility)
                nameOrganMaterials(pa, objID_leaf, material_base_name)
                ctx.hide_object(objID_leaf)
        proto.setRandomGenerator(context_generator)
        return True


__all__ = ["GenericLeafPrototype", "buildLeafBladeLattice", "deformLeafLattice", "petioluleVertexCount",
           "petioluleRadialSubdivisions", "resolveTextureFile", "nameOrganMaterials", "getCachedLeafPrototypeObjID",
           "recordLeafPrototypeRestGeometry", "leafletSizeMax", "PhytomerLeafMixin", "OrganMaterial", "LeafRestGeometry",
           "vec3"]
