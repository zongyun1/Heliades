# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantArchitecture.cpp:4616-4842 and 8090-8754, and core/src/global.cpp
# (cart2sphere, sphere2cart, atan2_2pi, evaluate_Beta_distribution_CDF) (Copyright (C) 2016-2026 Brian Bailey).
"""Leaf inclination distribution tracking (W5-B, first reached by the cowpea).

``buildPlantInstanceFromLibrary`` switches it on for a model registered with a leaf-inclination Beta (cowpea
(1.398, 1.574)), with ``library_leaf_angle_lambda_degrees = 180``, before the plant grows. Every growth sub-step,
``updatePlantLeafAngleDistributionTracking`` rebuilds the plant's facet-area histogram over 18 inclination bins, gives the
leaves of the phytomers created in that sub-step a target bin (cost = angular distance - lambda * kernel-weighted
shortfall; a target inside the bin drawn from the tracker's *own* ``minstd_rand0``, seeded from the plant ID and the Beta
parameters, so the Context stream is untouched), re-chooses the target of every still-expanding leaf, and turns each such
leaf along the great circle toward its target by the fraction of its remaining growth it just completed
(``Phytomer::setLeafNormal``: one minimal rotation about the leaf base, then the pose recorded as intrinsic yaw / pitch /
roll and the leaf marked prescribed).

Only elevation tracking is ported (the library's only use); azimuth tracking raises ``NotImplementedError``.

Float order follows the C++ exactly. The per-facet areas and normals of a blade are evaluated with numpy over all its
triangles at once (elementwise float32, the same operation sequence as ``Primitive.area`` / ``Primitive.normal``); the
area-weighted sums are sequential ``np.add.accumulate`` (bitwise the C++ loop), the bin deposits ``np.add.at`` (applied
in index order).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np

from ..core import vec as V
from ..core.random import MinstdRand0, UniformReal
from ..migration_registry import ports
from .random_parameter import PlantArchitectureError

f32 = np.float32
PI_F = f32(3.14159265358979323846)
_HALF_PI_F = f32(f32(0.5) * PI_F)

leaf_angle_inclination_bin_count = 18
leaf_angle_azimuth_bin_count = 24

#: PlantArchitecture::library_leaf_angle_lambda_degrees (PlantArchitecture.h:6617)
library_leaf_angle_lambda_degrees = f32(180)


def _libm():
    m = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    atanf = m.atanf
    atanf.restype, atanf.argtypes = ctypes.c_float, [ctypes.c_float]
    lgamma = m.lgamma
    lgamma.restype, lgamma.argtypes = ctypes.c_double, [ctypes.c_double]
    for name in ("log", "log1p", "exp"):
        f = getattr(m, name)
        f.restype, f.argtypes = ctypes.c_double, [ctypes.c_double]
    sqrtf = m.sqrtf
    sqrtf.restype, sqrtf.argtypes = ctypes.c_float, [ctypes.c_float]
    return atanf, lgamma, m.log, m.log1p, m.exp


_ATANF, _LGAMMA, _LOG, _LOG1P, _EXP = _libm()


def atanf(x):
    return f32(_ATANF(float(f32(x))))


def _sqrtf(x):
    return f32(np.sqrt(f32(x)))


# ---- helios:: (global.cpp) ----------------------------------------------------------------------

@ports("helios::atan2_2pi")
def atan2_2pi(y, x) -> np.float32:
    y, x = f32(y), f32(x)
    pi = PI_F
    v = f32(0)
    if x > 0:
        v = atanf(f32(y / x))
    if y >= 0 and x < 0:
        v = f32(pi + atanf(f32(y / x)))
    if y < 0 and x < 0:
        v = f32(-pi + atanf(f32(y / x)))
    if y > 0 and x == 0:
        v = f32(f32(0.5) * pi)
    if y < 0 and x == 0:
        v = f32(f32(-0.5) * pi)
    if v < 0:
        v = f32(v + f32(f32(2) * pi))
    return v


@ports("helios::cart2sphere")
def cart2sphere(c):
    """-> (radius, elevation, zenith, azimuth). ``SphericalCoord(r, elev, az)`` stores zenith = 0.5f*pi - elev."""
    x, y, z = (f32(c[0]), f32(c[1]), f32(c[2]))
    radius = _sqrtf(f32(f32(x * x + y * y) + z * z))
    x_safe, y_safe = x, y
    if abs(x_safe) < f32(1e-7) and abs(y_safe) < f32(1e-7):
        y_safe = f32(1e-7)
    elevation = V.asin_safe(f32(z / radius))
    azimuth = atan2_2pi(x_safe, y_safe)
    return radius, elevation, f32(_HALF_PI_F - elevation), azimuth


@ports("helios::sphere2cart")
def sphere2cart(radius, elevation, azimuth) -> np.ndarray:
    radius, elevation, azimuth = f32(radius), f32(elevation), f32(azimuth)
    ce = V.cosf(elevation)
    return np.array([f32(f32(radius * ce) * V.sinf(azimuth)), f32(f32(radius * ce) * V.cosf(azimuth)),
                     f32(radius * V.sinf(elevation))], f32)


def _sphere2cart_inclination(inclination, azimuth):
    """``sphere2cart(SphericalCoord(1.f, 0.5f * PI_F - inclination, azimuth))``."""
    return sphere2cart(f32(1), f32(_HALF_PI_F - f32(inclination)), azimuth)


def _regularized_incomplete_beta(x, a, b):
    """global.cpp:3844-3897 (Lentz continued fraction in double; libm lgamma / log / log1p / exp)."""
    x, a, b = f32(x), f32(a), f32(b)
    if x <= 0:
        return f32(0)
    if x >= 1:
        return f32(1)
    swap = bool(x > f32(f32(a + f32(1)) / f32(f32(a + b) + f32(2))))
    if swap:
        a, b = b, a
        x = f32(f32(1) - x)
    da, db, dx = float(a), float(b), float(x)
    log_prefactor = _LGAMMA(da + db) - _LGAMMA(da) - _LGAMMA(db) + da * _LOG(dx) + db * _LOG1P(-dx)
    prefactor = _EXP(log_prefactor) / da
    tiny, tolerance = 1e-30, 1e-9
    f, c, d = 1.0, 1.0, 0.0
    for i in range(301):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = float(m) * (db - float(m)) * dx / ((da + 2.0 * float(m) - 1.0) * (da + 2.0 * float(m)))
        else:
            numerator = -(da + float(m)) * (da + db + float(m)) * dx / ((da + 2.0 * float(m)) * (da + 2.0 * float(m) + 1.0))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < tolerance:
            break
    result = f32(prefactor * (f - 1.0))
    return f32(f32(1) - result) if swap else result


@ports("helios::evaluate_Beta_distribution_CDF")
def evaluate_Beta_distribution_CDF(theta, mu, nu) -> np.float32:
    mu, nu = f32(mu), f32(nu)
    if mu <= 0 or nu <= 0:
        raise PlantArchitectureError("ERROR (helios::evaluate_Beta_distribution_CDF): Beta distribution parameters must be positive.")
    x = f32(f32(theta) / _HALF_PI_F)
    return _regularized_incomplete_beta(x, nu, mu)


# ---- tracker state (PlantArchitecture.h:3523-3590, 2724-2731) ------------------------------------------

class BladeCacheEntry:
    __slots__ = ("primitive_count", "blade_UUIDs", "facet_inclination_offset", "facet_area", "facet_bin_kernel",
                 "facet_bin_kernel_origin")

    def __init__(self):
        self.primitive_count = 0
        self.blade_UUIDs = []
        self.facet_inclination_offset = np.zeros(0, f32)
        self.facet_area = np.zeros(0, f32)
        self.facet_bin_kernel = np.zeros(0, f32)
        self.facet_bin_kernel_origin = 0


class LeafAngleDistributionTracker:
    def __init__(self):
        self.track_elevation = False
        self.track_azimuth = False
        self.Beta_mu_inclination = f32(1)
        self.Beta_nu_inclination = f32(1)
        self.eccentricity_azimuth = f32(0)
        self.ellipse_rotation_azimuth_degrees = f32(0)
        self.lambda_ = f32(0)
        self.bin_weight = np.zeros(0, f32)
        self.bin_target_fraction = np.zeros(0, f32)
        self.blade_primitive_cache: dict[int, BladeCacheEntry] = {}
        self.pending_cohort: list[tuple[int, int]] = []
        self.generator = MinstdRand0(1)


class LeafAngleSteering:
    """Phytomer::LeafAngleSteering."""
    __slots__ = ("steered", "target_normal", "scale_at_last_step")

    def __init__(self):
        self.steered = False
        self.target_normal = np.zeros(3, f32)
        self.scale_at_last_step = f32(0)


def _steering(phytomer, petiole_index, leaf_index) -> LeafAngleSteering:
    """The leaf's steering record (Heliades stores None until first used: a default-constructed struct)."""
    row = phytomer.leaf_angle_steering[petiole_index]
    s = row[leaf_index]
    if s is None:
        s = row[leaf_index] = LeafAngleSteering()
    return s


# ---- enable / disable (PA:8100-8204, 8739-8753) ----------------------------------------------------------

def _llround(x: float) -> int:
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


@ports("PlantArchitecture::enablePlantLeafElevationAngleDistributionTracking(uint,float,float,float)")
def enablePlantLeafElevationAngleDistributionTracking(pa, plantID, Beta_mu_inclination, Beta_nu_inclination, lambda_degrees):
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError("ERROR (PlantArchitecture::enablePlantLeafElevationAngleDistributionTracking): Plant "
                                     f"with ID of {plantID} does not exist.")
    enablePlantLeafAngleDistributionTracking_private(pa, plantID, Beta_mu_inclination, Beta_nu_inclination, f32(0), f32(0),
                                                     lambda_degrees, True, False)


@ports("PlantArchitecture::enablePlantLeafAngleDistributionTracking_private")
def enablePlantLeafAngleDistributionTracking_private(pa, plantID, Beta_mu_inclination, Beta_nu_inclination, eccentricity,
                                                     ellipse_rotation_degrees, lambda_degrees, track_elevation, track_azimuth):
    mu, nu = f32(Beta_mu_inclination), f32(Beta_nu_inclination)
    eccentricity, ellipse_rotation_degrees, lambda_degrees = f32(eccentricity), f32(ellipse_rotation_degrees), f32(lambda_degrees)
    if track_elevation and (mu <= 0 or nu <= 0):
        raise PlantArchitectureError("ERROR (PlantArchitecture::enablePlantLeafAngleDistributionTracking): Beta distribution "
                                     "parameters must be positive.")
    if track_azimuth and (eccentricity < 0 or eccentricity > 1):
        raise PlantArchitectureError("ERROR (PlantArchitecture::enablePlantLeafAngleDistributionTracking): Eccentricity must be "
                                     "in [0,1].")
    if lambda_degrees < 0:
        raise PlantArchitectureError("ERROR (PlantArchitecture::enablePlantLeafAngleDistributionTracking): lambda must not be "
                                     "negative. Zero leaves the plant's own leaf angles unchanged; larger values pull the "
                                     "population onto the target.")
    if track_azimuth:
        raise NotImplementedError("leaf azimuth distribution tracking is not ported")

    inst = pa.plant_instances[plantID]
    tracker = LeafAngleDistributionTracker()
    inst.leaf_angle_tracker = tracker
    tracker.track_elevation = bool(track_elevation)
    tracker.track_azimuth = bool(track_azimuth)
    tracker.Beta_mu_inclination = mu
    tracker.Beta_nu_inclination = nu
    tracker.eccentricity_azimuth = eccentricity
    tracker.ellipse_rotation_azimuth_degrees = ellipse_rotation_degrees
    tracker.lambda_ = V.deg2rad(lambda_degrees)

    M = 2 ** 32
    seed = (int(plantID) * 2654435761 + (_llround(1000.0 * float(mu)) % M) * 40503
            + (_llround(1000.0 * float(nu)) % M) * 2246822519 + (_llround(1000.0 * float(eccentricity)) % M) * 3266489917
            + (_llround(1000.0 * float(ellipse_rotation_degrees)) % M) * 668265263) % M
    tracker.generator.seed(seed)

    n = leaf_angle_inclination_bin_count
    target = np.zeros(n, f32)
    for b in range(n):
        lower = f32(f32(_HALF_PI_F * f32(b)) / f32(n))
        upper = f32(f32(_HALF_PI_F * f32(b + 1)) / f32(n))
        fraction = f32(1)
        fraction = f32(fraction * max(f32(0), f32(evaluate_Beta_distribution_CDF(upper, mu, nu)
                                                   - evaluate_Beta_distribution_CDF(lower, mu, nu))))
        target[b] = fraction
    total = f32(0)
    for fr in target:
        total = f32(total + fr)
    if total <= 0:
        raise PlantArchitectureError("ERROR (PlantArchitecture::enablePlantLeafAngleDistributionTracking): The requested "
                                     "distribution puts no leaf area in any angle bin. Check the Beta parameters.")
    tracker.bin_target_fraction = (target / total).astype(f32)
    tracker.bin_weight = np.zeros(n, f32)
    tracker.pending_cohort = []
    inst.leaf_angle_tracking_enabled = True


@ports("PlantArchitecture::disablePlantLeafAngleDistributionTracking")
def disablePlantLeafAngleDistributionTracking(pa, plantID):
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError("ERROR (PlantArchitecture::disablePlantLeafAngleDistributionTracking): Plant with ID of "
                                     f"{plantID} does not exist.")
    inst = pa.plant_instances[plantID]
    inst.leaf_angle_tracking_enabled = False
    if getattr(inst, "leaf_angle_tracker", None) is not None:
        inst.leaf_angle_tracker.pending_cohort = []


def add_to_cohort(inst, shootID, node_index):
    """``plant_instance.leaf_angle_tracker.pending_cohort.emplace_back(shootID, node_index)``."""
    if getattr(inst, "leaf_angle_tracking_enabled", False):
        inst.leaf_angle_tracker.pending_cohort.append((int(shootID), int(node_index)))


# ---- blade geometry (PA:4658-4716, 8214-8297) ------------------------------------------------------------

_TRI = 1


def _facets(ctx, uuids):
    """Per primitive: (area, normal (3,), ok) as ``getPrimitiveArea`` / ``getPrimitiveNormal`` / ``facetAreaAndNormal``,
    triangles evaluated together (the op sequence of ``Primitive.vertices`` / ``area`` / ``normal``)."""
    n = len(uuids)
    areas = np.empty(n, f32)
    normals = np.empty((n, 3), f32)
    prims = [ctx.primitives[u] for u in uuids]
    tri = np.array([p.type == _TRI for p in prims], bool)
    if tri.any():
        idx = np.nonzero(tri)[0]
        T = np.stack([prims[i].transform for i in idx]).reshape(-1, 4, 4)
        sf = np.array([prims[i].solid_fraction for i in idx], f32)
        # canonical triangle (0,0,0), (0,1,0), (1,1,0): v = ((T0*y.x + T1*y.y) + T2*y.z) + T3
        Y = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], f32)
        verts = np.empty((len(idx), 3, 3), f32)
        for k in range(3):
            for r in range(3):
                verts[:, k, r] = ((T[:, r, 0] * Y[k, 0] + T[:, r, 1] * Y[k, 1]) + T[:, r, 2] * Y[k, 2]) + T[:, r, 3]
        v0, v1, v2 = verts[:, 0], verts[:, 1], verts[:, 2]

        def cross(a, b):
            return np.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1], a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                             a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], axis=1)

        ca = cross(v1 - v0, v2 - v0)
        mag = np.sqrt((ca[:, 0] * ca[:, 0] + ca[:, 1] * ca[:, 1]) + ca[:, 2] * ca[:, 2])
        areas[idx] = (f32(0.5) * mag) * sf
        cn = cross(v1 - v0, v2 - v1)
        m2 = (cn[:, 0] * cn[:, 0] + cn[:, 1] * cn[:, 1]) + cn[:, 2] * cn[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = f32(1) / np.sqrt(m2)
            normals[idx] = cn * inv[:, None]
    for i in np.nonzero(~tri)[0]:
        areas[i] = prims[i].area()
        normals[i] = prims[i].normal()
    ok = np.isfinite(areas) & (areas > 0) & np.isfinite(normals).all(axis=1)
    return areas, normals, ok


def _acos_safe_fabs(z):
    return np.array([V.acos_safe(abs(f32(v))) for v in z], f32)


@ports("leafBladeNormalCached")
def leafBladeNormalCached(ctx, objID_leaf, cache):
    """PA:8214-8296. -> (ok, normal, area, entry); ``entry`` is None when the blade faces no single direction (the
    C++ then leaves ``*entry_out`` unchanged, which no caller reads in that case)."""
    if not ctx.does_object_exist(objID_leaf):
        cache.pop(objID_leaf, None)
        return False, None, f32(0), None
    object_UUIDs = ctx.object_primitive_uuids(objID_leaf)
    entry = cache.get(objID_leaf)
    if entry is None or entry.primitive_count != len(object_UUIDs):
        fresh = BladeCacheEntry()
        fresh.primitive_count = len(object_UUIDs)
        fresh.blade_UUIDs = list(ctx.filter_primitives_by_data(object_UUIDs, "object_label", "leaf"))
        if not fresh.blade_UUIDs:
            fresh.blade_UUIDs = list(object_UUIDs)
        cache[objID_leaf] = entry = fresh

    areas, normals, ok = _facets(ctx, entry.blade_UUIDs)
    facet_area = np.where(ok, areas, f32(0)).astype(f32)
    offsets = np.zeros(len(areas), f32)
    if ok.any():
        offsets[ok] = _acos_safe_fabs(normals[ok, 2])
        wa = areas[ok]
        # normal = normal + primitive_area * facet_normal, area += primitive_area: sequential float sums
        nsum = np.add.accumulate(wa[:, None] * normals[ok], axis=0, dtype=f32)[-1]
        area = np.add.accumulate(wa, dtype=f32)[-1]
    else:
        nsum = np.zeros(3, f32)
        area = f32(0)
    entry.facet_area = facet_area
    entry.facet_inclination_offset = offsets
    normal = np.asarray(nsum, f32)
    area = f32(area)
    if area <= 0 or V.magnitude(normal) < f32(f32(1e-4) * area):
        return False, normal, area, None
    normal = V.normalize(normal)
    mean_inclination = V.acos_safe(abs(normal[2]))
    entry.facet_inclination_offset = (offsets - mean_inclination).astype(f32)

    bin_width = f32(_HALF_PI_F / f32(leaf_angle_inclination_bin_count))
    offset_bin = np.floor(entry.facet_inclination_offset / bin_width + f32(0.5)).astype(np.int64)
    lowest = min(0, int(offset_bin.min())) if len(offset_bin) else 0
    highest = max(0, int(offset_bin.max())) if len(offset_bin) else 0
    entry.facet_bin_kernel_origin = -lowest
    kernel = np.zeros(highest - lowest + 1, f32)
    np.add.at(kernel, offset_bin + entry.facet_bin_kernel_origin, entry.facet_area)
    entry.facet_bin_kernel = (kernel / area).astype(f32)
    return True, normal, area, entry


@ports("depositLeafFacets")
def depositLeafFacets(tracker, blade, mean_inclination, azimuth, sign):
    """PA:8333-8367 (elevation-only path)."""
    n = leaf_angle_inclination_bin_count
    bins_per_radian = f32(f32(n) / _HALF_PI_F)
    fi = (f32(mean_inclination) + blade.facet_inclination_offset).astype(f32)
    fi = np.where(fi < 0, -fi, fi)
    fi = np.where(fi > _HALF_PI_F, PI_F - fi, fi).astype(f32)
    b = np.clip((bins_per_radian * fi).astype(np.int64), 0, n - 1)      # int(float) truncates toward zero
    np.add.at(tracker.bin_weight, b, (f32(sign) * blade.facet_area).astype(f32))


@ports("chooseLeafAngleTarget")
def chooseLeafAngleTarget(tracker, inclination, azimuth, total_weight, blade=None):
    """PA:8397-8531 (elevation-only). -> target normal or None."""
    if tracker.lambda_ <= 0 or len(tracker.bin_target_fraction) == 0:
        return None
    n = leaf_angle_inclination_bin_count
    kernel = None
    origin = 0
    if blade is not None and tracker.track_elevation and len(blade.facet_bin_kernel) > 0:
        kernel = blade.facet_bin_kernel
        origin = blade.facet_bin_kernel_origin
    inclination, azimuth = f32(inclination), f32(azimuth)
    proposed = _sphere2cart_inclination(inclination, azimuth)

    sin_i = np.empty(n, f32)
    cos_i = np.empty(n, f32)
    for i in range(n):
        bin_inclination = f32(f32(_HALF_PI_F * f32(f32(i) + f32(0.5))) / f32(n))
        elevation = f32(_HALF_PI_F - bin_inclination)
        sin_i[i] = V.sinf(elevation)
        cos_i[i] = V.cosf(elevation)
    sin_a, cos_a = V.sinf(azimuth), V.cosf(azimuth)

    tw = max(f32(total_weight), f32(1e-9))
    best_bin, best_cost = -1, f32(0)
    for b in range(len(tracker.bin_target_fraction)):
        shortfall = f32(0)
        if kernel is None:
            target_weight = f32(tracker.bin_target_fraction[b] * tw)
            shortfall = V.clamp(f32(f32(target_weight - tracker.bin_weight[b]) / max(target_weight, f32(1e-9))), f32(-1), f32(1))
        else:
            for k in range(len(kernel)):
                if kernel[k] <= 0:
                    continue
                landed = b + k - origin
                if landed < 0:
                    landed = -landed - 1
                if landed >= n:
                    landed = 2 * n - landed - 1
                landed = min(max(landed, 0), n - 1)
                target_weight = f32(tracker.bin_target_fraction[landed] * tw)
                sf = V.clamp(f32(f32(target_weight - tracker.bin_weight[landed]) / max(target_weight, f32(1e-9))), f32(-1), f32(1))
                shortfall = f32(shortfall + f32(kernel[k] * sf))
        ce = cos_i[b]
        d = np.array([f32(ce * sin_a), f32(ce * cos_a), sin_i[b]], f32)
        angular_distance = V.acos_safe(V.clamp(V.dot(proposed, d), f32(-1), f32(1)))
        cost = f32(angular_distance - f32(tracker.lambda_ * shortfall))
        if best_bin < 0 or cost < best_cost:
            best_cost, best_bin = cost, b
    if best_bin < 0:
        return None
    within_bin = UniformReal(0.0, 1.0, real="float")
    lower = f32(f32(_HALF_PI_F * f32(best_bin)) / f32(n))
    upper = f32(f32(_HALF_PI_F * f32(best_bin + 1)) / f32(n))
    target_inclination = f32(lower + f32(within_bin(tracker.generator) * f32(upper - lower)))
    return _sphere2cart_inclination(target_inclination, azimuth)


def _folded_spherical(v):
    w = np.asarray(v, f32).copy()
    w[2] = abs(w[2])
    _, _, zenith, az = cart2sphere(w)
    return zenith, az


@ports("rodrigues")
def rodrigues(v, k, a):
    """``v * c + cross(k, v) * s + k * (kv * (1.f - c))`` (std::cos / std::sin on float)."""
    a = f32(a)
    c, s = V.cosf(a), V.sinf(a)
    v, k = np.asarray(v, f32), np.asarray(k, f32)
    kv = V.dot(k, v)
    return ((v * c + V.cross(k, v) * s) + k * f32(kv * f32(f32(1) - c))).astype(f32)


@ports("PlantArchitecture::updatePlantLeafAngleDistributionTracking")
def updatePlantLeafAngleDistributionTracking(pa, plantID):
    """PA:8533-8738."""
    inst = pa.plant_instances[plantID]
    if not getattr(inst, "leaf_angle_tracking_enabled", False):
        return
    ctx = pa.context_ptr
    tracker = inst.leaf_angle_tracker
    cache = tracker.blade_primitive_cache

    def leaf_angles(objID):
        ok, normal, area, entry = leafBladeNormalCached(ctx, objID, cache)
        if not ok:
            return False, f32(0), f32(0), area, None
        normal = normal.copy()
        normal[2] = abs(normal[2])
        _, _, zenith, az = cart2sphere(normal)
        return True, zenith, az, area, entry

    # ---- what the plant has now
    tracker.bin_weight[:] = 0
    total_weight = f32(0)
    for shoot in inst.shoot_tree:
        for ph in shoot.phytomers:
            for p in range(len(ph.leaf_objIDs)):
                for leaf in range(len(ph.leaf_objIDs[p])):
                    ok, incl, az, area, entry = leaf_angles(ph.leaf_objIDs[p][leaf])
                    if not ok:
                        continue
                    depositLeafFacets(tracker, entry, incl, az, f32(1))
                    total_weight = f32(total_weight + area)

    # ---- give this sub-step's new leaves a target
    for shootID, node_index in tracker.pending_cohort:
        if shootID < 0 or shootID >= len(inst.shoot_tree):
            continue
        shoot = inst.shoot_tree[shootID]
        if node_index >= len(shoot.phytomers):
            continue
        ph = shoot.phytomers[node_index]
        if not ph.hasLeaf():
            continue
        for p in range(len(ph.leaf_objIDs)):
            for leaf in range(len(ph.leaf_objIDs[p])):
                if p >= len(ph.leaf_angle_steering) or leaf >= len(ph.leaf_angle_steering[p]):
                    continue
                steering = _steering(ph, p, leaf)
                if steering.steered:
                    continue
                ok, incl, az, area, entry = leaf_angles(ph.leaf_objIDs[p][leaf])
                if not ok:
                    continue
                chosen = chooseLeafAngleTarget(tracker, incl, az, total_weight, entry)
                if chosen is None:
                    steering.steered = False
                    continue
                depositLeafFacets(tracker, entry, incl, az, f32(-1))
                tz, ta = _folded_spherical(chosen)
                depositLeafFacets(tracker, entry, tz, ta, f32(1))
                steering.steered = True
                steering.target_normal = chosen
                steering.scale_at_last_step = (ph.current_leaf_scale_factor[p] if p < len(ph.current_leaf_scale_factor)
                                               else f32(0))
    tracker.pending_cohort = []

    # ---- turn the expanding leaves a little further toward their targets
    for shoot in inst.shoot_tree:
        for ph in shoot.phytomers:
            for p in range(len(ph.leaf_angle_steering)):
                if p >= len(ph.current_leaf_scale_factor):
                    continue
                scale = f32(ph.current_leaf_scale_factor[p])
                for leaf in range(len(ph.leaf_angle_steering[p])):
                    steering = ph.leaf_angle_steering[p][leaf]
                    if steering is None or not steering.steered:
                        continue
                    if leaf >= len(ph.leaf_objIDs[p]) or not ctx.does_object_exist(ph.leaf_objIDs[p][leaf]):
                        continue
                    previous_scale = steering.scale_at_last_step
                    remaining = f32(f32(1) - previous_scale)
                    if scale >= 1 or remaining <= f32(1e-4):
                        step_fraction = f32(1)
                    else:
                        step_fraction = V.clamp(f32(f32(scale - previous_scale) / remaining), f32(0), f32(1))
                    if step_fraction <= 0:
                        continue
                    ok, current_normal, _, current_blade = leafBladeNormalCached(ctx, ph.leaf_objIDs[p][leaf], cache)
                    if not ok:
                        continue
                    measured_normal = current_normal.copy()
                    current_normal = current_normal.copy()
                    current_normal[2] = abs(current_normal[2])
                    if scale < 1:
                        _, _, cz, ca = cart2sphere(current_normal)
                        rechosen = chooseLeafAngleTarget(tracker, cz, ca, total_weight, current_blade)
                        if rechosen is not None:
                            pz, pa_ = _folded_spherical(steering.target_normal)
                            depositLeafFacets(tracker, current_blade, pz, pa_, f32(-1))
                            nz, na = _folded_spherical(rechosen)
                            depositLeafFacets(tracker, current_blade, nz, na, f32(1))
                            steering.target_normal = rechosen
                    separation = V.acos_safe(V.clamp(V.dot(current_normal, steering.target_normal), f32(-1), f32(1)))
                    step_normal = np.asarray(steering.target_normal, f32).copy()
                    if separation > f32(1e-4) and step_fraction < 1:
                        axis = V.cross(current_normal, steering.target_normal)
                        if V.magnitude(axis) > f32(1e-6):
                            step_normal = rodrigues(current_normal, _member_normalize(axis), f32(separation * step_fraction))
                    setLeafNormal(ph, p, leaf, step_normal, measured_normal)
                    steering.scale_at_last_step = scale
                    if scale >= 1:
                        steering.steered = False


def _member_normalize(v):
    """vec3::normalize() (member): unchanged when m2 <= 0, else times 1/sqrtf(m2)."""
    return V.normalize(v)


# ---- Phytomer::setLeafNormal (PA:4616-4842) ---------------------------------------------------------------

def _orthonormal_axis(v):
    ax = V.cross(v, np.array([1, 0, 0], f32))
    if V.magnitude(ax) < f32(1e-6):
        ax = V.cross(v, np.array([0, 1, 0], f32))
    return V.normalize(ax)


@ports("minimalRotationBetween")
def minimalRotationBetween(v, u):
    dot = V.clamp(V.dot(v, u), f32(-1), f32(1))
    angle = V.acos_safe(dot)
    if angle < f32(1e-6):
        return None
    axis = V.cross(v, u)
    if V.magnitude(axis) < f32(1e-6):
        axis = _orthonormal_axis(v)
    else:
        axis = V.normalize(axis)
    return axis, angle


def _free_normalize(v):
    """helios::normalize(const vec3&): no zero guard."""
    v = np.asarray(v, f32)
    m2 = f32(f32(v[0] * v[0] + v[1] * v[1]) + v[2] * v[2])
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = f32(f32(1) / f32(np.sqrt(m2)))
    return (v * inv).astype(f32)


def _leaf_blade_normal(ctx, objID_leaf):
    """leafBladeNormal (PA:4683-4716), uncached."""
    if not ctx.does_object_exist(objID_leaf):
        return None
    uu = ctx.object_primitive_uuids(objID_leaf)
    blade = list(ctx.filter_primitives_by_data(uu, "object_label", "leaf")) or list(uu)
    areas, normals, ok = _facets(ctx, blade)
    if not ok.any():
        return None
    wa = areas[ok]
    nsum = np.add.accumulate(wa[:, None] * normals[ok], axis=0, dtype=f32)[-1]
    area = f32(np.add.accumulate(wa, dtype=f32)[-1])
    if area <= 0 or V.magnitude(nsum) < f32(f32(1e-4) * area):
        return None
    return V.normalize(nsum)


@ports("Phytomer::setLeafNormal")
def setLeafNormal(phytomer, petiole_index, leaf_index, target_normal, current_normal=None):
    from .tube_utils import clamp_offset, compound_leaf_rotation
    from .phytomer import atan2f
    ph, ctx = phytomer, phytomer.context_ptr
    if petiole_index >= len(ph.leaf_objIDs):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::Phytomer::setLeafNormal): Invalid petiole index {petiole_index}.")
    if leaf_index >= len(ph.leaf_objIDs[petiole_index]):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::Phytomer::setLeafNormal): Invalid leaf index {leaf_index}.")
    objID_leaf = ph.leaf_objIDs[petiole_index][leaf_index]
    if not ctx.does_object_exist(objID_leaf):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::Phytomer::setLeafNormal): Leaf {leaf_index} on petiole "
                                     f"{petiole_index} has no geometry in the Context.")
    target = np.asarray(target_normal, f32)
    if V.magnitude(target) < f32(1e-6):
        raise PlantArchitectureError("ERROR (PlantArchitecture::Phytomer::setLeafNormal): Target normal has zero length.")
    target = V.normalize(target)

    T = ctx.object_transform(objID_leaf)
    if (V.magnitude((T[0], T[4], T[8])) < f32(1e-9) or V.magnitude((T[1], T[5], T[9])) < f32(1e-9)
            or V.magnitude((T[2], T[6], T[10])) < f32(1e-9)):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::Phytomer::setLeafNormal): Leaf {leaf_index} on petiole "
                                     f"{petiole_index} has a degenerate transformation matrix.")
    if current_normal is not None and V.magnitude(current_normal) > f32(1e-6):
        world_normal = _free_normalize(current_normal)
    else:
        world_normal = _leaf_blade_normal(ctx, objID_leaf)
        if world_normal is None:
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::Phytomer::setLeafNormal): Leaf {leaf_index} on petiole "
                                         f"{petiole_index} has no single direction it faces; its facet normals cancel. It "
                                         "cannot be aimed at a target normal.")
    leaves_per_petiole = len(ph.leaf_objIDs[petiole_index])

    rot = minimalRotationBetween(world_normal, target)
    if rot is not None:
        axis, angle = rot
        ctx.rotate_object(objID_leaf, angle, axis, ph.leaf_bases[petiole_index][leaf_index])

    def rotate_about_y(a, v):
        c, s = V.cosf(a), V.sinf(a)
        return np.array([f32(f32(v[0] * c) + f32(v[2] * s)), v[1], f32(f32(-v[0] * s) + f32(v[2] * c))], f32)

    def rotate_about_z(a, v):
        c, s = V.cosf(a), V.sinf(a)
        return np.array([f32(f32(v[0] * c) - f32(v[1] * s)), f32(f32(v[0] * s) + f32(v[1] * c)), v[2]], f32)

    Tn = ctx.object_transform(objID_leaf)
    axis_x = _free_normalize((Tn[0], Tn[4], Tn[8]))
    axis_y = _free_normalize((Tn[1], Tn[5], Tn[9]))
    axis_z = _free_normalize((Tn[2], Tn[6], Tn[10]))
    tip_axis = V.normalize(ph.getPetioleAxisVector(f32(1), petiole_index))
    elevation = V.asin_safe(tip_axis[2])
    horizontal = _sqrtf(f32(tip_axis[0] * tip_axis[0] + tip_axis[1] * tip_axis[1]))
    azimuth = f32(0) if horizontal < f32(1e-6) else atan2f(tip_axis[1], tip_axis[0])
    if leaves_per_petiole > 1:
        compound_rotation = compound_leaf_rotation(leaves_per_petiole, int(leaf_index),
                                                   clamp_offset(leaves_per_petiole, ph.phytomer_parameters.leaf.leaflet_offset.val()))
    else:
        compound_rotation = f32(0)

    def undo(v):
        r = rotate_about_z(f32(-azimuth), v)
        r = rotate_about_y(elevation, r)
        return rotate_about_z(f32(-compound_rotation), r)

    axis_x, axis_y, axis_z = undo(axis_x), undo(axis_y), undo(axis_z)
    recorded_yaw = atan2f(axis_x[1], axis_x[0])
    recorded_pitch = atan2f(axis_x[2], _sqrtf(f32(axis_x[0] * axis_x[0] + axis_x[1] * axis_x[1])))
    recorded_roll = atan2f(axis_y[2], axis_z[2])
    r = ph.leaf_rotation[petiole_index][leaf_index]
    r.yaw, r.pitch, r.roll = recorded_yaw, recorded_pitch, recorded_roll
    if petiole_index < len(ph.leaf_pose_prescribed) and leaf_index < len(ph.leaf_pose_prescribed[petiole_index]):
        ph.leaf_pose_prescribed[petiole_index][leaf_index] = True


@ports("nearestPointOnPolyline")
def nearestPointOnPolyline(line, point):
    """PA:278-294."""
    point = np.asarray(point, f32)
    nearest = np.asarray(line[0], f32).copy()
    best = V.magnitude(point - nearest)
    for i in range(len(line) - 1):
        a = np.asarray(line[i], f32)
        segment = np.asarray(line[i + 1], f32) - a
        sls = V.dot(segment, segment)
        t = f32(V.dot(point - a, segment) / sls) if sls > 0 else f32(0)
        t = min(max(t, f32(0)), f32(1))
        candidate = a + t * segment
        distance = V.magnitude(point - candidate)
        if distance < best:
            best, nearest = distance, candidate.astype(f32)
    return nearest


__all__ = ["enablePlantLeafElevationAngleDistributionTracking", "disablePlantLeafAngleDistributionTracking",
           "updatePlantLeafAngleDistributionTracking", "setLeafNormal", "add_to_cohort", "nearestPointOnPolyline",
           "evaluate_Beta_distribution_CDF", "cart2sphere", "sphere2cart", "atan2_2pi"]
