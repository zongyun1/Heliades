# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/Assets.cpp:24-282, 517-924 and
# src/PlantArchitecture.cpp:181-276, 3449-3497, 4532-4596, 4844-5014 (Copyright (C) 2016-2026 Brian Bailey).
"""Leaf blade geometry (W3-A): ``GenericLeafPrototype``, ``buildLeafBladeLattice``, ``deformLeafLattice``.

``GenericLeafPrototype(scene, prototype, compound_leaf_index) -> objID`` is bound to
``hooks.GenericLeafPrototype`` when this module is imported (``phytomer.py`` imports it lazily). It draws only
from ``prototype``'s RandomParameters (the private ``minstd_rand0(identifier)`` during the cache build, which the
Phytomer installs), in Helios's order including the trailing resamples, records ``last_built_shape``, labels the
petiolule "petiolule" and returns the polymesh. The Phytomer side (buildLeafPrototype, labelLeafPrototype,
assignPetioluleMaterial, the cache, nameOrganMaterials) is W3-B's, in ``phytomer.py``.

Every float expression follows its C++ type, operation by operation:

* ``f``-suffixed libm calls (``sinf``, ``cosf``, ``atanf``, ``powf``, ``atan2f``, ``remainderf``, ``hypotf``)
  go through glibc with ctypes, because glibc's float functions are not correctly rounded;
* unsuffixed calls on a float argument (``atan`` in the longitudinal curvature, ``exp``/``fmin``/``fabs`` in the
  petiole roll and the wave) are the C ``double`` functions (no ``using namespace std`` reaches Assets.cpp), so
  they are evaluated in double and narrowed on assignment. Each choice is pinned by a mutation that the parity
  cases catch (notes/W3-A.md);
* ``float * M_PI`` is a double product.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math
import numpy as np

from ..core import vec as V
from ..core.texture import try_resolve_file_path
from ..migration_registry import ports
from . import hooks
from .params import LeafShapeValues
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

@hooks.GenericLeafPrototype.implement
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
        recordSpectralClassMap(ctx, ctx.object_primitive_uuids(objID), leaf_texture, class_map, "leaf",
                               " A class map is made for one leaf texture (see utilities/leaf_class_maps). If the leaf "
                               "texture was changed, clear LeafPrototype::leaf_class_map_file, or set it to a class map "
                               "made for the new texture.")

    return objID


# ---- spectral class maps (PA:714-787, global.cpp:2137-2220; W4-C) ------------------------------------------

_CLASS_MAP_CACHE: dict = {}


def _read_png_single_channel(path: str):
    """helios::readPNGSingleChannel: an 8- or 16-bit grayscale PNG as (height, width) uint16, row 0 the top row."""
    import os
    from PIL import Image
    from ..core.texture import _png_header
    if os.path.splitext(path)[1].lower() != ".png":
        raise PlantArchitectureError(f"ERROR (readPNGSingleChannel): File {path} is not PNG format. Save it as an 8-bit or "
                                     "16-bit grayscale PNG.")
    w, h, depth, ctype, trns = _png_header(path)
    if ctype == 3:
        raise PlantArchitectureError(f"ERROR (readPNGSingleChannel): File {path} is an indexed-color PNG.")
    if ctype != 0:
        raise PlantArchitectureError(f"ERROR (readPNGSingleChannel): File {path} has more than one channel.")
    if depth not in (8, 16):
        raise PlantArchitectureError(f"ERROR (readPNGSingleChannel): File {path} has a bit depth of {depth}.")
    if trns is not None:
        raise PlantArchitectureError(f"ERROR (readPNGSingleChannel): File {path} marks one gray value as transparent.")
    with Image.open(path) as im:
        im.load()
        a = np.asarray(im)
    return np.asarray(a, np.uint16).reshape(h, w)


def _clipped_polygon_area(polygon, u0, u1, v0, v1) -> float:
    """Area of a convex polygon inside [u0,u1] x [v0,v1] by Sutherland-Hodgman clipping (global.cpp:2144), double."""
    clipped = list(polygon)

    def clip(inside, intersect):
        nonlocal clipped
        out = []
        n = len(clipped)
        for i in range(n):
            cur = clipped[i]
            prev = clipped[(i + n - 1) % n]
            if inside(cur):
                if not inside(prev):
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(intersect(prev, cur))
        clipped = out

    def at_u(e):
        return lambda a, b: (e, a[1] + (b[1] - a[1]) * (e - a[0]) / (b[0] - a[0]))

    def at_v(e):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (e - a[1]) / (b[1] - a[1]), e)

    clip(lambda p: p[0] >= u0, at_u(u0))
    clip(lambda p: p[0] <= u1, at_u(u1))
    clip(lambda p: p[1] >= v0, at_v(v0))
    clip(lambda p: p[1] <= v1, at_v(v1))
    twice = 0.0
    n = len(clipped)
    for i in range(n):
        a, b = clipped[i], clipped[(i + 1) % n]
        twice += a[0] * b[1] - b[0] * a[1]
    return 0.5 * abs(twice)


def class_map_covers_a_class(classes: np.ndarray, width: int, height: int, uv) -> bool:
    """``!helios::classMapAreaFractions(classes, resolution, uv).empty()`` (global.cpp:2179): whether any texel of a
    non-zero class has a positive clipped area under the polygon. Same cell walk and double arithmetic; it stops
    at the first such texel, which does not change the answer."""
    polygon = [(float(f32(c[0])), float(f32(c[1]))) for c in np.asarray(uv, f32).reshape(-1, 2)]
    u_min = min([1e30] + [p[0] for p in polygon])
    u_max = max([-1e30] + [p[0] for p in polygon])
    v_min = min([1e30] + [p[1] for p in polygon])
    v_max = max([-1e30] + [p[1] for p in polygon])
    inf = math.inf

    def column_of(u):
        return 0 if width == 1 else min(max(int(math.floor((width - 1) * u)), 0), width - 1)

    def row_of(v):
        return 0 if height == 1 else min(max(int(math.floor((height - 1) * (1.0 - v))), 0), height - 1)

    for row in range(row_of(v_max), row_of(v_min) + 1):
        v_low = -inf if (height == 1 or row == height - 1) else 1.0 - float(row + 1) / float(height - 1)
        v_high = inf if (height == 1 or row == 0) else 1.0 - float(row) / float(height - 1)
        for column in range(column_of(u_min), column_of(u_max) + 1):
            if classes[row, column] == 0:
                continue
            u_low = -inf if (width == 1 or column == 0) else float(column) / float(width - 1)
            u_high = inf if (width == 1 or column == width - 1) else float(column + 1) / float(width - 1)
            if _clipped_polygon_area(polygon, max(u_low, u_min), min(u_high, u_max), max(v_low, v_min),
                                     min(v_high, v_max)) > 0:
                return True
    return False


@ports("PlantArchitecture::recordSpectralClassMap")
def recordSpectralClassMap(context, UUIDs, texture_file: str, class_map_file: str, organ: str, remedy: str):
    """Record ``spectral_class_map`` (the resolved class-map path) on each textured triangle that covers a class."""
    class_map = resolveTextureFile(class_map_file)
    classes = _CLASS_MAP_CACHE.get(class_map)
    if classes is None:
        classes = _read_png_single_channel(class_map)
        _CLASS_MAP_CACHE[class_map] = classes
    map_h, map_w = classes.shape

    textured = []
    for u in UUIDs:
        f = context.primitive_texture_file(u)
        if not f or (texture_file and f != texture_file):
            continue
        textured.append(u)
    if not textured:
        return
    texture = context.primitive_texture_file(textured[0])
    tw, th = context.primitive_texture_size(textured[0])
    if tw != map_w or th != map_h:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture): The {organ} class map {class_map} is {map_w} x {map_h} "
                                     f"texels, but the {organ} texture {texture} is {tw} x {th}.{remedy}")
    has_transparency = context.primitive_texture_has_transparency_channel(textured[0])
    if has_transparency:
        opaque = np.asarray(context.primitive_texture_transparency_data(textured[0]), bool)
        bad = opaque == (classes == 0)
        if bad.any():
            row, column = map(int, np.argwhere(bad)[0])
            raise PlantArchitectureError(f"ERROR (PlantArchitecture): The {organ} class map {class_map} does not match the "
                                         f"transparency of the {organ} texture {texture} (at column {column}, row {row}): class "
                                         f"0 must mark exactly its transparent texels.{remedy}")
    elif (classes == 0).any():
        texel = int(np.flatnonzero(classes.reshape(-1) == 0)[0])
        raise PlantArchitectureError(f"ERROR (PlantArchitecture): The {organ} class map {class_map} has a texel of class 0 (at "
                                     f"column {texel % map_w}, row {texel // map_w}), but the {organ} texture {texture} has "
                                     f"no transparency: every texel must have a class.{remedy}")
    for u in textured:
        if context.does_primitive_data_exist(u, "spectral_class_map"):
            continue
        if context.primitive_texture_file(u) != texture:
            raise PlantArchitectureError(f"ERROR (PlantArchitecture): The {organ} class map {class_map} was made for the texture "
                                         f"{texture}, but the {organ} is also mapped with {context.primitive_texture_file(u)}."
                                         f"{remedy}")
        if not has_transparency or class_map_covers_a_class(classes, map_w, map_h, context.primitive_texture_uv(u)):
            context.set_primitive_data(u, "spectral_class_map", class_map)


def _translate_primitives(ctx, uuids, shift):
    """Context::translatePrimitive(vector, shift): one translation matrix applied to each primitive.
    (Heliades skips the product for a zero shift; multiplying by an identity matrix is exact apart from
    the sign of zeros, which no comparison distinguishes.)"""
    for u in uuids:
        ctx.translate_primitive(u, shift)


__all__ = ["GenericLeafPrototype", "buildLeafBladeLattice", "deformLeafLattice", "petioluleVertexCount",
           "petioluleRadialSubdivisions", "resolveTextureFile"]
