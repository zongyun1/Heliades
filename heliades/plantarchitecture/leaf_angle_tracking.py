# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantArchitecture.cpp (Copyright (C) 2016-2026 Brian Bailey):
#   rodrigues / orthonormal_axis / minimalRotationBetween / facetAreaAndNormal / leafBladeNormal   PA:4617-4716
#   Phytomer::setLeafNormal                                     PA:4718-4834
#   enablePlantLeafAngleDistributionTracking_private            PA:8126-8206
#   leafBladeNormalCached / leafAngleBinIndex / depositLeafFacets / chooseLeafAngleTarget         PA:8214-8531
#   PlantArchitecture::updatePlantLeafAngleDistributionTracking  PA:8533-8740
# and core/src/global.cpp: regularized_incomplete_beta / evaluate_Beta_distribution_CDF (3842-3907),
# cart2sphere / sphere2cart / atan2_2pi (863-904).
"""Leaf elevation-angle distribution tracking (W5-D; needed by ``pistachio``).

A library model that declares a leaf-inclination Beta (``registerPlantModel(..., make_vec2(mu, nu))``: cowpea,
pistachio, eastern redbud) is built with ``enablePlantLeafElevationAngleDistributionTracking(plantID, mu, nu,
180 deg)``. Every growth sub-step, after the shoot loop, the tracker histograms the blade facets of every leaf
into 18 inclination bins, gives each leaf created this sub-step (the *pending cohort*) a target inclination that
trades the distance from its own orientation against the population's shortfall in each bin, and turns every
still-expanding steered leaf toward its target in proportion to the growth it just completed. Targets inside a
bin are drawn from the tracker's own ``std::minstd_rand0``, seeded from the plant ID and the Beta parameters, so
the Context stream is untouched.

Only elevation tracking is ported (azimuth tracking, the ellipsoidal CDF, is not used by any library model).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np

from ..core import vec as V
from ..core.polymesh import acos_safe
from ..core.random import MinstdRand0, UniformReal
from ..migration_registry import ports
from .random_parameter import PlantArchitectureError
from .tube_utils import PI_F, clamp_offset, compound_leaf_rotation

f32 = np.float32
INCLINATION_BINS = 18
AZIMUTH_BINS = 24
HALF_PI = f32(f32(0.5) * PI_F)

_libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
_lgamma = _libm.lgamma
_lgamma.restype, _lgamma.argtypes = ctypes.c_double, [ctypes.c_double]
_atanf = _libm.atanf
_atanf.restype, _atanf.argtypes = ctypes.c_float, [ctypes.c_float]


def atanf(x) -> np.float32:
    return f32(_atanf(float(f32(x))))


# ---- helios:: helpers (global.cpp) ---------------------------------------------------------------------------

@ports("helios::atan2_2pi")
def atan2_2pi(y, x) -> np.float32:
    y, x = f32(y), f32(x)
    pi = f32(math.pi)
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
    """Returns (radius, elevation, zenith, azimuth)."""
    x, y, z = f32(c[0]), f32(c[1]), f32(c[2])
    radius = f32(np.sqrt(f32(f32(f32(x * x) + f32(y * y)) + f32(z * z))))
    xs, ys = x, y
    if abs(xs) < f32(1e-7) and abs(ys) < f32(1e-7):
        ys = f32(1e-7)
    elevation = V.asin_safe(f32(z / radius))
    zenith = f32(HALF_PI - elevation)                     # SphericalCoord: 0.5f * float(M_PI) - elevation
    return radius, elevation, zenith, atan2_2pi(xs, ys)


@ports("helios::sphere2cart")
def sphere2cart(radius, elevation, azimuth) -> np.ndarray:
    radius, elevation, azimuth = f32(radius), f32(elevation), f32(azimuth)
    ce = f32(radius * V.cosf(elevation))
    return np.array([f32(ce * V.sinf(azimuth)), f32(ce * V.cosf(azimuth)), f32(radius * V.sinf(elevation))], f32)


def _regularized_incomplete_beta(x, a, b) -> np.float32:
    x, a, b = f32(x), f32(a), f32(b)
    if x <= 0:
        return f32(0)
    if x >= 1:
        return f32(1)
    swap = x > f32(f32(a + f32(1)) / f32(f32(a + b) + f32(2)))
    if swap:
        a, b = b, a
        x = f32(f32(1) - x)
    da, db, dx = float(a), float(b), float(x)
    log_prefactor = _lgamma(da + db) - _lgamma(da) - _lgamma(db) + da * math.log(dx) + db * math.log1p(-dx)
    prefactor = math.exp(log_prefactor) / da
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
        raise PlantArchitectureError("ERROR (helios::evaluate_Beta_distribution_CDF): Beta distribution parameters must "
                                     "be positive.")
    x = f32(f32(theta) / HALF_PI)
    return _regularized_incomplete_beta(x, nu, mu)


# ---- tracker state (H:3515-3590, 2724-2734) -------------------------------------------------------------------

class LeafAngleSteering:
    """Phytomer::LeafAngleSteering."""
    __slots__ = ("steered", "target_normal", "scale_at_last_step")

    def __init__(self):
        self.steered = False
        self.target_normal = np.zeros(3, f32)
        self.scale_at_last_step = f32(0)


class BladeCacheEntry:
    __slots__ = ("primitive_count", "blade_UUIDs", "facet_inclination_offset", "facet_area", "facet_bin_kernel",
                 "facet_bin_kernel_origin")

    def __init__(self):
        self.primitive_count = 0
        self.blade_UUIDs = []
        self.facet_inclination_offset = []
        self.facet_area = []
        self.facet_bin_kernel = []
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
        self.bin_weight = []
        self.bin_target_fraction = []
        self.blade_primitive_cache = {}
        self.pending_cohort = []
        self.generator = MinstdRand0(1)


def _tracker(inst):
    t = getattr(inst, "leaf_angle_tracker", None)
    if t is None:
        t = inst.leaf_angle_tracker = LeafAngleDistributionTracker()
    return t


def add_to_cohort(inst, shootID, node_index):
    """``plant_instance.leaf_angle_tracker.pending_cohort.emplace_back(shootID, node)`` (PA:12221, 12249, 12285)."""
    _tracker(inst).pending_cohort.append((int(shootID), int(node_index)))


def _llround(x: float) -> int:
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


@ports("PlantArchitecture::enablePlantLeafElevationAngleDistributionTracking")
def enablePlantLeafElevationAngleDistributionTracking(pa, plantID, Beta_mu_inclination, Beta_nu_inclination,
                                                      lambda_degrees):
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError("ERROR (PlantArchitecture::enablePlantLeafElevationAngleDistributionTracking): Plant "
                                     f"with ID of {plantID} does not exist.")
    _enable(pa, plantID, f32(Beta_mu_inclination), f32(Beta_nu_inclination), f32(0), f32(0), f32(lambda_degrees), True,
            False)


@ports("PlantArchitecture::enablePlantLeafAngleDistributionTracking_private")
def _enable(pa, plantID, mu, nu, eccentricity, ellipse_rotation_degrees, lambda_degrees, track_elevation, track_azimuth):
    if track_elevation and (mu <= 0 or nu <= 0):
        raise PlantArchitectureError("ERROR (PlantArchitecture::enablePlantLeafAngleDistributionTracking): Beta "
                                     "distribution parameters must be positive.")
    if track_azimuth:
        raise NotImplementedError("leaf azimuth distribution tracking is not ported")
    if lambda_degrees < 0:
        raise PlantArchitectureError("ERROR (PlantArchitecture::enablePlantLeafAngleDistributionTracking): lambda must not "
                                     "be negative. Zero leaves the plant's own leaf angles unchanged; larger values pull "
                                     "the population onto the target.")
    inst = pa.plant_instances[plantID]
    t = _tracker(inst)
    t.track_elevation, t.track_azimuth = track_elevation, track_azimuth
    t.Beta_mu_inclination, t.Beta_nu_inclination = mu, nu
    t.eccentricity_azimuth, t.ellipse_rotation_azimuth_degrees = eccentricity, ellipse_rotation_degrees
    t.lambda_ = V.deg2rad(lambda_degrees)
    seed = (plantID * 2654435761 + (_llround(1000.0 * float(mu)) & 0xFFFFFFFF) * 40503
            + (_llround(1000.0 * float(nu)) & 0xFFFFFFFF) * 2246822519
            + (_llround(1000.0 * float(eccentricity)) & 0xFFFFFFFF) * 3266489917
            + (_llround(1000.0 * float(ellipse_rotation_degrees)) & 0xFFFFFFFF) * 668265263) & 0xFFFFFFFF
    t.generator.seed(seed)
    bin_count = INCLINATION_BINS if track_elevation else 1
    fractions = []
    for b in range(bin_count):
        fraction = f32(1)
        if track_elevation:
            lower = f32(f32(HALF_PI * f32(b)) / f32(INCLINATION_BINS))
            upper = f32(f32(HALF_PI * f32(b + 1)) / f32(INCLINATION_BINS))
            fraction = f32(fraction * max(f32(0), f32(evaluate_Beta_distribution_CDF(upper, mu, nu)
                                                      - evaluate_Beta_distribution_CDF(lower, mu, nu))))
        fractions.append(fraction)
    total = f32(0)
    for f in fractions:
        total = f32(total + f)
    if total <= 0:
        raise PlantArchitectureError("ERROR (PlantArchitecture::enablePlantLeafAngleDistributionTracking): The requested "
                                     "distribution puts no leaf area in any angle bin. Check the Beta parameters.")
    t.bin_target_fraction = [f32(f / total) for f in fractions]
    t.bin_weight = [f32(0)] * bin_count
    t.pending_cohort = []
    inst.leaf_angle_tracking_enabled = True


# ---- blade geometry ----------------------------------------------------------------------------------------

def _facet_area_and_normal(ctx, uuid):
    area = ctx.primitive_area(uuid)
    if not math.isfinite(float(area)) or area <= 0:
        return False, area, None
    normal = np.asarray(ctx.primitive_normal(uuid), f32)
    return bool(np.all(np.isfinite(normal))), area, normal


def _blade_entry(ctx, objID, cache):
    """The cache half of leafBladeNormalCached: the entry (None if the object is gone)."""
    if not ctx.does_object_exist(objID):
        cache.pop(objID, None)
        return None
    uuids = ctx.object_primitive_uuids(objID)
    entry = cache.get(objID)
    if entry is None or entry.primitive_count != len(uuids):
        entry = BladeCacheEntry()
        entry.primitive_count = len(uuids)
        entry.blade_UUIDs = ctx.filter_primitives_by_data(uuids, "object_label", "leaf") or list(uuids)
        cache[objID] = entry
    return entry


def _blade_normal_cached(ctx, objID, cache, facets=None):
    """leafBladeNormalCached: (ok, normal, area, entry). ``facets`` optionally gives the blade facets' (areas,
    normals), precomputed in one batch (``prim_batch.areas_normals``, bit-identical to the per-primitive calls)."""
    entry = _blade_entry(ctx, objID, cache)
    if entry is None:
        return False, None, f32(0), None
    n = len(entry.blade_UUIDs)
    if facets is None:
        from .prim_batch import areas_normals
        facets = areas_normals(ctx, entry.blade_UUIDs)
    f_areas, f_normals = facets
    entry.facet_inclination_offset = [f32(0)] * n
    entry.facet_area = [f32(0)] * n
    normal = np.zeros(3, f32)
    area = f32(0)
    for i in range(n):
        pa_ = f_areas[i]
        fn = f_normals[i]
        ok = math.isfinite(float(pa_)) and pa_ > 0 and bool(np.all(np.isfinite(fn)))
        if not ok:
            entry.facet_area[i] = f32(0)
            entry.facet_inclination_offset[i] = f32(0)
            continue
        normal = normal + f32(pa_) * fn
        area = f32(area + pa_)
        entry.facet_area[i] = f32(pa_)
        entry.facet_inclination_offset[i] = acos_safe(abs(fn[2]))
    if area <= 0 or V.magnitude(normal) < f32(f32(1e-4) * area):
        return False, normal, area, entry
    normal = V.normalize(normal)
    mean_inclination = acos_safe(abs(normal[2]))
    entry.facet_inclination_offset = [f32(o - mean_inclination) for o in entry.facet_inclination_offset]
    bin_width = f32(HALF_PI / f32(INCLINATION_BINS))
    bins = [int(math.floor(float(f32(f32(o / bin_width) + f32(0.5))))) for o in entry.facet_inclination_offset]
    lowest = min([0] + bins)
    highest = max([0] + bins)
    entry.facet_bin_kernel_origin = -lowest
    kernel = [f32(0)] * (highest - lowest + 1)
    for i, b in enumerate(bins):
        kernel[b + entry.facet_bin_kernel_origin] = f32(kernel[b + entry.facet_bin_kernel_origin] + entry.facet_area[i])
    entry.facet_bin_kernel = [f32(w / area) for w in kernel]
    return True, normal, area, entry


def _deposit(t, blade, mean_inclination, azimuth, sign):
    """depositLeafFacets (elevation-only branch)."""
    bins_per_radian = f32(f32(INCLINATION_BINS) / HALF_PI)
    sign = f32(sign)
    for off, area in zip(blade.facet_inclination_offset, blade.facet_area):
        fi = f32(mean_inclination + off)
        if fi < 0:
            fi = f32(-fi)
        if fi > HALF_PI:
            fi = f32(PI_F - fi)
        b = min(max(int(f32(bins_per_radian * fi)), 0), INCLINATION_BINS - 1)
        t.bin_weight[b] = f32(t.bin_weight[b] + f32(sign * area))


def _clamp(x, lo, hi):
    return min(max(x, lo), hi)


def _choose_target(t, inclination, azimuth, total_weight, blade):
    """chooseLeafAngleTarget (elevation only): (ok, target_normal)."""
    if t.lambda_ <= 0 or not t.bin_target_fraction:
        return False, None
    kernel = None
    origin = 0
    if blade is not None and t.track_elevation and blade.facet_bin_kernel:
        kernel = blade.facet_bin_kernel
        origin = blade.facet_bin_kernel_origin
    inclination, azimuth = f32(inclination), f32(azimuth)
    proposed = sphere2cart(f32(1), f32(HALF_PI - inclination), azimuth)
    sin_inc, cos_inc = [], []
    for i in range(INCLINATION_BINS):
        bin_inclination = f32(f32(HALF_PI * f32(f32(i) + f32(0.5))) / f32(INCLINATION_BINS))
        elevation = f32(HALF_PI - bin_inclination)
        sin_inc.append(V.sinf(elevation))
        cos_inc.append(V.cosf(elevation))
    sin_az, cos_az = V.sinf(azimuth), V.cosf(azimuth)
    tw_scale = max(f32(total_weight), f32(1e-9))
    best_bin, best_cost = -1, f32(0)
    for b in range(len(t.bin_target_fraction)):
        shortfall = f32(0)
        if kernel is None:
            tw = f32(t.bin_target_fraction[b] * tw_scale)
            shortfall = _clamp(f32(f32(tw - t.bin_weight[b]) / max(tw, f32(1e-9))), f32(-1), f32(1))
        else:
            for ki, kw in enumerate(kernel):
                if kw <= 0:
                    continue
                landed = b + ki - origin
                if landed < 0:
                    landed = -landed - 1
                if landed >= INCLINATION_BINS:
                    landed = 2 * INCLINATION_BINS - landed - 1
                landed = _clamp(landed, 0, INCLINATION_BINS - 1)
                tw = f32(t.bin_target_fraction[landed] * tw_scale)
                shortfall = f32(shortfall + f32(kw * _clamp(f32(f32(tw - t.bin_weight[landed]) / max(tw, f32(1e-9))),
                                                            f32(-1), f32(1))))
        ce = cos_inc[b]
        bin_direction = np.array([f32(ce * sin_az), f32(ce * cos_az), sin_inc[b]], f32)
        angular_distance = acos_safe(_clamp(V.dot(proposed, bin_direction), f32(-1), f32(1)))
        cost = f32(angular_distance - f32(t.lambda_ * shortfall))
        if best_bin < 0 or cost < best_cost:
            best_cost, best_bin = cost, b
    if best_bin < 0:
        return False, None
    within_bin = UniformReal(0.0, 1.0, "float")
    lower = f32(f32(HALF_PI * f32(best_bin)) / f32(INCLINATION_BINS))
    upper = f32(f32(HALF_PI * f32(best_bin + 1)) / f32(INCLINATION_BINS))
    target_inclination = f32(lower + f32(within_bin(t.generator) * f32(upper - lower)))
    return True, sphere2cart(f32(1), f32(HALF_PI - target_inclination), azimuth)


def _folded(v):
    w = np.asarray(v, f32).copy()
    w[2] = abs(w[2])
    return w


def _rodrigues(v, k, a):
    c, s = V.cosf(a), V.sinf(a)
    kv = V.dot(k, v)
    return (np.asarray(v, f32) * c + V.cross(k, v) * s) + np.asarray(k, f32) * f32(kv * f32(f32(1) - c))


# ---- per-sub-step update ---------------------------------------------------------------------------------------

@ports("PlantArchitecture::updatePlantLeafAngleDistributionTracking")
def updatePlantLeafAngleDistributionTracking(pa, plantID):
    inst = pa.plant_instances[plantID]
    if not getattr(inst, "leaf_angle_tracking_enabled", False):
        return
    ctx = pa.context_ptr
    t = _tracker(inst)
    cache = t.blade_primitive_cache

    # One batched area/normal pass over every blade facet of the plant. Nothing moves a leaf before the turning
    # loop below, and that loop turns each leaf once, after reading it, so every read sees this geometry.
    from .prim_batch import areas_normals
    leaf_ids = [objID for shoot in inst.shoot_tree for ph in shoot.phytomers for petiole in ph.leaf_objIDs
                for objID in petiole]
    entries = [_blade_entry(ctx, objID, cache) for objID in leaf_ids]
    all_uuids = [u for e in entries if e is not None for u in e.blade_UUIDs]
    batch_areas, batch_normals = areas_normals(ctx, all_uuids)
    facets = {}
    k = 0
    for objID, e in zip(leaf_ids, entries):
        if e is not None:
            m = len(e.blade_UUIDs)
            facets[objID] = (batch_areas[k:k + m], batch_normals[k:k + m])
            k += m

    def leaf_angles(objID):
        ok, normal, area, entry = _blade_normal_cached(ctx, objID, cache, facets.get(objID))
        if not ok:
            return False, f32(0), f32(0), area, entry
        _, _, zenith, az = cart2sphere(_folded(normal))
        return True, zenith, az, area, entry

    t.bin_weight = [f32(0)] * len(t.bin_weight)
    total_weight = f32(0)
    for shoot in inst.shoot_tree:
        for ph in shoot.phytomers:
            for petiole in ph.leaf_objIDs:
                for objID in petiole:
                    ok, inc, az, area, entry = leaf_angles(objID)
                    if not ok:
                        continue
                    _deposit(t, entry, inc, az, 1)
                    total_weight = f32(total_weight + area)

    tree = inst.shoot_tree
    for shootID, node_index in t.pending_cohort:
        if shootID < 0 or shootID >= len(tree):
            continue
        shoot = tree[shootID]
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
                ok, inc, az, area, entry = leaf_angles(ph.leaf_objIDs[p][leaf])
                if not ok:
                    continue
                chosen_ok, chosen = _choose_target(t, inc, az, total_weight, entry)
                if not chosen_ok:
                    steering.steered = False
                    continue
                _deposit(t, entry, inc, az, -1)
                _, _, tz, ta = cart2sphere(_folded(chosen))
                _deposit(t, entry, tz, ta, 1)
                steering.steered = True
                steering.target_normal = chosen
                steering.scale_at_last_step = (ph.current_leaf_scale_factor[p] if p < len(ph.current_leaf_scale_factor)
                                               else f32(0))
    t.pending_cohort = []

    for shoot in tree:
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
                    previous = f32(steering.scale_at_last_step)
                    remaining = f32(f32(1) - previous)
                    if scale >= 1 or remaining <= f32(1e-4):
                        step_fraction = f32(1)
                    else:
                        step_fraction = _clamp(f32(f32(scale - previous) / remaining), f32(0), f32(1))
                    if step_fraction <= 0:
                        continue
                    ok, current, _area, blade = _blade_normal_cached(ctx, ph.leaf_objIDs[p][leaf], cache,
                                                                     facets.get(ph.leaf_objIDs[p][leaf]))
                    if not ok:
                        continue
                    measured = current.copy()
                    current = _folded(current)
                    if scale < 1:
                        _, _, cz, caz = cart2sphere(current)
                        re_ok, rechosen = _choose_target(t, cz, caz, total_weight, blade)
                        if re_ok:
                            _, _, pz, paz = cart2sphere(_folded(steering.target_normal))
                            _deposit(t, blade, pz, paz, -1)
                            _, _, nz, naz = cart2sphere(_folded(rechosen))
                            _deposit(t, blade, nz, naz, 1)
                            steering.target_normal = rechosen
                    separation = acos_safe(_clamp(V.dot(current, steering.target_normal), f32(-1), f32(1)))
                    step_normal = np.asarray(steering.target_normal, f32).copy()
                    if separation > f32(1e-4) and step_fraction < 1:
                        axis = V.cross(current, steering.target_normal)
                        if V.magnitude(axis) > f32(1e-6):
                            step_normal = _rodrigues(current, V.normalize(axis), f32(separation * step_fraction))
                    setLeafNormal(ph, p, leaf, step_normal, measured)
                    steering.scale_at_last_step = scale
                    if scale >= 1:
                        steering.steered = False


def _steering(ph, p, leaf):
    s = ph.leaf_angle_steering[p][leaf]
    if s is None:
        s = ph.leaf_angle_steering[p][leaf] = LeafAngleSteering()
    return s


# ---- Phytomer::setLeafNormal (PA:4718-4834) -------------------------------------------------------------------

def _orthonormal_axis(v):
    ax = V.cross(v, np.array([1, 0, 0], f32))
    if V.magnitude(ax) < f32(1e-6):
        ax = V.cross(v, np.array([0, 1, 0], f32))
    return V.normalize(ax)


@ports("Phytomer::setLeafNormal")
def setLeafNormal(ph, petiole_index, leaf_index, target_normal, current_normal=None):
    ctx = ph.context_ptr
    if petiole_index >= len(ph.leaf_objIDs):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::Phytomer::setLeafNormal): Invalid petiole index "
                                     f"{petiole_index}.")
    if leaf_index >= len(ph.leaf_objIDs[petiole_index]):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::Phytomer::setLeafNormal): Invalid leaf index {leaf_index}.")
    objID = ph.leaf_objIDs[petiole_index][leaf_index]
    if not ctx.does_object_exist(objID):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::Phytomer::setLeafNormal): Leaf {leaf_index} on petiole "
                                     f"{petiole_index} has no geometry in the Context.")
    target = np.asarray(target_normal, f32)
    if V.magnitude(target) < f32(1e-6):
        raise PlantArchitectureError("ERROR (PlantArchitecture::Phytomer::setLeafNormal): Target normal has zero length.")
    target = V.normalize(target)
    T = ctx.object_transform(objID).reshape(-1)
    for col in range(3):
        if V.magnitude(np.array([T[col], T[col + 4], T[col + 8]], f32)) < f32(1e-9):
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::Phytomer::setLeafNormal): Leaf {leaf_index} on "
                                         f"petiole {petiole_index} has a degenerate transformation matrix.")
    if current_normal is not None and V.magnitude(current_normal) > f32(1e-6):
        world_normal = V.normalize(current_normal)        # helios::normalize (no zero guard)
    else:
        raise NotImplementedError("setLeafNormal without a measured normal is not ported")
    leaves_per_petiole = len(ph.leaf_objIDs[petiole_index])

    def rot_y(angle, v):
        c, s = V.cosf(angle), V.sinf(angle)
        return np.array([f32(f32(v[0] * c) + f32(v[2] * s)), v[1], f32(f32(-v[0] * s) + f32(v[2] * c))], f32)

    def rot_z(angle, v):
        c, s = V.cosf(angle), V.sinf(angle)
        return np.array([f32(f32(v[0] * c) - f32(v[1] * s)), f32(f32(v[0] * s) + f32(v[1] * c)), v[2]], f32)

    dot = _clamp(V.dot(world_normal, target), f32(-1), f32(1))
    angle = acos_safe(dot)
    if not angle < f32(1e-6):
        axis = V.cross(world_normal, target)
        axis = _orthonormal_axis(world_normal) if V.magnitude(axis) < f32(1e-6) else V.normalize(axis)
        ctx.rotate_object(objID, angle, axis, ph.leaf_bases[petiole_index][leaf_index])

    Tn = ctx.object_transform(objID).reshape(-1)
    axis_x = V.normalize(np.array([Tn[0], Tn[4], Tn[8]], f32))
    axis_y = V.normalize(np.array([Tn[1], Tn[5], Tn[9]], f32))
    axis_z = V.normalize(np.array([Tn[2], Tn[6], Tn[10]], f32))
    tip_axis = V.normalize(ph.getPetioleAxisVector(f32(1), petiole_index))
    elevation = V.asin_safe(tip_axis[2])
    horizontal = f32(np.sqrt(f32(f32(tip_axis[0] * tip_axis[0]) + f32(tip_axis[1] * tip_axis[1]))))
    azimuth = f32(0) if horizontal < f32(1e-6) else V.atan2f(tip_axis[1], tip_axis[0])
    compound = (compound_leaf_rotation(leaves_per_petiole, leaf_index,
                                       clamp_offset(leaves_per_petiole, ph.phytomer_parameters.leaf.leaflet_offset.val()))
                if leaves_per_petiole > 1 else f32(0))

    def undo(v):
        r = rot_z(f32(-azimuth), v)
        r = rot_y(elevation, r)
        return rot_z(f32(-compound), r)

    axis_x, axis_y, axis_z = undo(axis_x), undo(axis_y), undo(axis_z)
    rec = ph.leaf_rotation[petiole_index][leaf_index]
    rec.yaw = V.atan2f(axis_x[1], axis_x[0])
    rec.pitch = V.atan2f(axis_x[2], f32(np.sqrt(f32(f32(axis_x[0] * axis_x[0]) + f32(axis_x[1] * axis_x[1])))))
    rec.roll = V.atan2f(axis_y[2], axis_z[2])
    if petiole_index < len(ph.leaf_pose_prescribed) and leaf_index < len(ph.leaf_pose_prescribed[petiole_index]):
        ph.leaf_pose_prescribed[petiole_index][leaf_index] = True


__all__ = ["enablePlantLeafElevationAngleDistributionTracking", "updatePlantLeafAngleDistributionTracking",
           "add_to_cohort", "setLeafNormal", "evaluate_Beta_distribution_CDF", "LeafAngleSteering", "cart2sphere",
           "sphere2cart"]
