# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for sphere/disk/cone/box objects (W1-C), colored overloads and their getters."""
from __future__ import annotations

import math

import numpy as np

from ...cases import case

_NVEC = np.array([0.1817, 0.6198, 0.7634])  # addConeObject's seed vector (hits the cross-product fallback)


def _rng(seed):
    return np.random.default_rng(1000 + seed)


def _rot(r):
    return (0.0, 0.0) if r.random() < 0.25 else (r.uniform(-math.pi, math.pi), r.uniform(-2 * math.pi, 2 * math.pi))


def _color(r):
    u = r.random()
    if u < 0.25:
        return None
    return tuple(r.random(4)) if u < 0.6 else tuple(r.random(3))


def _getters(B, oids):
    return [float(x) for o in oids for x in B.shape_getters(o)]


# ---- sphere --------------------------------------------------------------------------------

def _spheres(B, r):
    oids = []
    for _ in range(5):
        nd = int(r.integers(3, 16))
        c = r.uniform(-5, 5, 3)
        rad = r.uniform(0.01, 3) if r.random() < 0.5 else r.uniform(0.01, 3, 3)
        col = None if r.random() < 0.3 else r.random(3)
        oids.append(B.add_sphere(nd, c, rad, col))
    return oids


@case("core.shape.sphere", tier="exact", max_ulp=0, fuzz=8)
def sphere_fuzz(B, seed):
    _spheres(B, _rng(seed))
    return B.snapshot_rgba()


@case("core.shape.sphere_edges", tier="exact", max_ulp=0, fuzz=1)
def sphere_edges(B, seed):
    """Ndivs 1 and 2 (caps only, from Helios's Test_data.h), 3 (PyHelios's minimum), tiny and huge radii."""
    B.add_sphere(1, (2, 0, 0), 1.0)
    B.add_sphere(2, (0, 0, 0), (1.0, 2.0, 3.0), (1, 0, 0))
    B.add_sphere(3, (0, 0, 1), 0.5)
    B.add_sphere(10, (1, 2, 3), 5.0)
    B.add_sphere(7, (0, 0, 0), 1e-4)
    B.add_sphere(5, (100, -50, 3), (1e3, 1e-3, 2.0))
    return B.snapshot_rgba()


@case("core.shape.sphere_getters", tier="exact", max_ulp=0, fuzz=6)
def sphere_getters(B, seed):
    return _getters(B, _spheres(B, _rng(seed)))


# ---- disk ----------------------------------------------------------------------------------

def _disks(B, r):
    oids = []
    for _ in range(6):
        c = r.uniform(-5, 5, 3)
        s = r.uniform(0.01, 3, 2)
        rot = None if r.random() < 0.2 else _rot(r)
        if r.random() < 0.5:
            col = _color(r)
            oids.append(B.add_disk(int(r.integers(3, 25)), c, s, rot, col))
        else:
            col = _color(r) or tuple(r.random(3))
            oids.append(B.add_disk((int(r.integers(3, 20)), int(r.integers(1, 5))), c, s, rot, col))
    return oids


@case("core.shape.disk", tier="exact", max_ulp=0, fuzz=8)
def disk_fuzz(B, seed):
    _disks(B, _rng(seed))
    return B.snapshot_rgba()


@case("core.shape.disk_edges", tier="exact", max_ulp=0, fuzz=1)
def disk_edges(B, seed):
    """Minimum subdivisions, zero size (Helios has no size check), and Helios's own test inputs."""
    B.add_disk(10, (0, 0, 0), (1, 1))
    B.add_disk((8, 16), (1, 2, 3), (4, 5), (0.0, 0.0), (1, 0, 0))
    B.add_disk((6, 3), (0, 4, 1), (2, 2), (0.0, 0.0), (1, 0, 0))
    B.add_disk(7, (0, 6, 0), (0.5, 0.5))
    B.add_disk(1, (0, 0, 0), (1, 1), (0.3, 0.2))
    B.add_disk((2, 1), (0, 0, 0), (1, 2), (0.0, 0.0), (0.1, 0.2, 0.3, 0.4))
    B.add_disk((3, 2), (1, 1, 1), (0.5, 0.25), (0.5 * math.pi, math.pi), (0.5, 0.5, 0.5))
    B.add_disk(5, (0, 0, 1), (0.0, 0.0))
    B.add_disk(4, (0, 0, 1), (1.0, 0.0), (0.7, 0.0))
    B.add_disk(6, (3, 0, 0), (1.0, 1.0), (-1.2, 2.5), (0, 0, 1, 0.5))
    return B.snapshot_rgba()


@case("core.shape.disk_getters", tier="exact", max_ulp=0, fuzz=6)
def disk_getters(B, seed):
    return _getters(B, _disks(B, _rng(seed)))


# ---- cone ----------------------------------------------------------------------------------

def _cones(B, r):
    oids = []
    for _ in range(6):
        n0 = r.uniform(-3, 3, 3)
        u = r.random()
        if u < 0.15:
            d = np.array([0, 0, r.uniform(0.1, 3)]) * (1 if r.random() < 0.5 else -1)
        elif u < 0.25:
            d = np.eye(3)[r.integers(2)] * r.uniform(0.1, 3)
        else:
            d = r.normal(size=3)
        r0, r1 = r.uniform(0.0, 1.0, 2)
        if r.random() < 0.15:
            r1 = 0.0
        col = None if r.random() < 0.3 else r.random(3)
        oids.append(B.add_cone(int(r.integers(3, 25)), n0, n0 + d, r0, r1, col))
    return oids


@case("core.shape.cone", tier="exact", max_ulp=0, fuzz=8)
def cone_fuzz(B, seed):
    _cones(B, _rng(seed))
    return B.snapshot_rgba()


def _cone_edge_set(B):
    oids = [
        B.add_cone(10, (0, 0, 0), (0, 0, 5), 2.0, 1.0),          # Test_context.h addConeObject
        B.add_cone(1, (3, 0, 0), (3, 0, 2), 1.0, 2.0),           # Test_data.h, Ndivs = 1
        B.add_cone(2, (0, 0, 0), (0, 0, 1), 0.5, 0.3),
        B.add_cone(5, (0, 0, 1), (0, 0, 0), 0.1, 0.05),          # pointing down
        B.add_cone(6, (0, 0, 0), (2, 0, 0), 0.3, 0.3),           # horizontal x (cylinder)
        B.add_cone(6, (0, 0, 0), (0, -2, 0), 0.3, 0.0),          # horizontal y, sharp apex
        B.add_cone(8, (1, 1, 1), (1, 1, 1), 0.2, 0.1),           # coincident nodes -> axis (0,0,1)
        B.add_cone(8, (0, 0, 0), tuple(_NVEC), 0.2, 0.1),        # axis parallel to the seed vector
        B.add_cone(8, (0, 0, 0), tuple(-2 * _NVEC), 0.2, 0.1),
        B.add_cone(8, (0, 0, 0), (3e-6, 0, 0), 0.2, 0.1),        # tiny axis: second-cross fallback branch
        B.add_cone(8, (0, 0, 0), (0.95, 0, 1e-7), 0.2, 0.1),
        B.add_cone(7, (0, 0, 0), (0, 0, 1), 0.0, 0.0),           # zero radii
        B.add_cone(7, (0, 0, 0), (0, 0, 1), -0.2, 0.3),          # negative radius is accepted
    ]
    return oids


@case("core.shape.cone_edges", tier="exact", max_ulp=0, fuzz=1)
def cone_edges(B, seed):
    _cone_edge_set(B)
    return B.snapshot_rgba()


@case("core.shape.cone_getters", tier="exact", max_ulp=0, fuzz=6)
def cone_getters(B, seed):
    return _getters(B, _cones(B, _rng(seed)))


@case("core.shape.cone_edge_getters", tier="exact", max_ulp=0, fuzz=1)
def cone_edge_getters(B, seed):
    # coincident nodes give a 0/0 axis (NaN in both); every other getter is compared
    return _getters(B, [o for i, o in enumerate(_cone_edge_set(B)) if i != 6])


# ---- box -----------------------------------------------------------------------------------

def _boxes(B, r):
    oids = []
    for _ in range(5):
        c = r.uniform(-5, 5, 3)
        s = r.uniform(0.01, 3, 3)
        sd = tuple(int(x) for x in r.integers(1, 5, 3))
        col = None if r.random() < 0.3 else r.random(3)
        oids.append(B.add_box(c, s, sd, col, bool(r.random() < 0.4)))
    return oids


@case("core.shape.box", tier="exact", max_ulp=0, fuzz=8)
def box_fuzz(B, seed):
    _boxes(B, _rng(seed))
    return B.snapshot_rgba()


@case("core.shape.box_edges", tier="exact", max_ulp=0, fuzz=1)
def box_edges(B, seed):
    """Helios's own test boxes, minimum subdivisions, reversed normals, thin slabs."""
    B.add_box((1, 2, 3), (3, 2, 1), (1, 1, 1))
    B.add_box((0, 0, 0), (1, 1, 1), (2, 3, 2))
    B.add_box((0, 0, 0), (2, 3, 4), (1, 1, 1), (1, 0, 0), True)
    B.add_box((1, 2, 3), (2, 4, 6), (3, 1, 2), None, True)
    B.add_box((0, 0, 0), (1e-3, 5, 2e-3), (1, 7, 1))
    return B.snapshot_rgba()


@case("core.shape.box_getters", tier="exact", max_ulp=0, fuzz=6)
def box_getters(B, seed):
    return _getters(B, _boxes(B, _rng(seed)))


# ---- mixed: interleaved creation keeps UUID / object-ID order ----------------------------

@case("core.shape.mixed", tier="exact", max_ulp=0, fuzz=6)
def mixed_fuzz(B, seed):
    r = _rng(seed + 50)
    for _ in range(8):
        k = r.integers(5)
        if k == 0:
            _spheres(B, r) if r.random() < 0.2 else B.add_sphere(int(r.integers(3, 8)), r.uniform(-2, 2, 3), r.uniform(0.1, 1))
        elif k == 1:
            B.add_disk(int(r.integers(3, 12)), r.uniform(-2, 2, 3), r.uniform(0.1, 1, 2), _rot(r), r.random(3))
        elif k == 2:
            B.add_cone(int(r.integers(3, 12)), r.uniform(-2, 2, 3), r.uniform(-2, 2, 3), r.uniform(0, 0.5), r.uniform(0, 0.5))
        elif k == 3:
            B.add_box(r.uniform(-2, 2, 3), r.uniform(0.1, 1, 3), (1, 2, 1), r.random(3), bool(r.random() < 0.5))
        else:
            B.add_patch(r.uniform(-2, 2, 3), r.uniform(0.1, 1, 2))
            B.add_tile(r.uniform(-2, 2, 3), (1.0, 1.0), _rot(r), (2, 2))
    return B.snapshot_rgba()


# ---- raw streams: no 1e-6 absolute floor, so a 1-ulp change on O(1) geometry is caught -------

def _raw(s):
    pad = np.arange(4)[None, :] < s.nverts[:, None]
    parts = [s.vertices[pad].ravel(), s.normal.ravel(), s.area, s.color.ravel(),
             s.objects.center.ravel(), s.objects.transform.ravel()]
    return np.concatenate([np.asarray(p, np.float32) for p in parts])


@case("core.shape.raw_stream", tier="exact", max_ulp=0, fuzz=8)
def raw_stream(B, seed):
    r = _rng(seed + 100)
    _spheres(B, r)
    _disks(B, r)
    _cones(B, r)
    _boxes(B, r)
    return _raw(B.snapshot_rgba())
