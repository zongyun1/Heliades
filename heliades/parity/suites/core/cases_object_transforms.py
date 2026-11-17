# SPDX-License-Identifier: LGPL-2.1-or-later
"""W1-B parity cases: copy / delete / object transforms / arbitrary-axis primitive rotation.

The fuzz cases draw a random op sequence (create patch/triangle/tile/tube/polymesh -> copy ->
transform -> delete) with numpy's generator; every random choice of an ID is made from the ID list
the backend itself reports, so a divergence in IDs shows up as a divergence in the result.
"""
from __future__ import annotations

import math

import numpy as np

from ...cases import _stem, case


def _unit(r):
    a = r.normal(size=3)
    if r.random() < 0.2:                               # axis-aligned axes hit the exact-zero entries
        a = np.eye(3)[r.integers(3)] * (1 if r.random() < 0.5 else -1)
    return a * r.uniform(0.3, 3.0)                     # unnormalised: makeRotationMatrix normalises


def _angle(r):
    return 0.0 if r.random() < 0.08 else float(r.uniform(-2 * math.pi, 2 * math.pi))


def _scale(r):
    u = r.random()
    if u < 0.08:
        return (1.0, 1.0, 1.0)                         # early-return path
    if u < 0.4:
        s = r.uniform(0.5, 2.0)
        return (s, s, s)                               # uniform, as plantarchitecture uses
    s = r.uniform(0.5, 2.0, 3)
    if r.random() < 0.1:
        s[r.integers(3)] *= -1                         # mirror
    return tuple(s)


def _shift(r):
    return (0.0, 0.0, 0.0) if r.random() < 0.08 else tuple(r.uniform(-2, 2, 3))


class _World:
    """Tracks free (object-less) primitives on the case side; objects are asked of the backend."""

    def __init__(self, B, r):
        self.B, self.r, self.free = B, r, []

    def alive_free(self):
        uu = set(self.B.all_uuids())
        self.free = [u for u in self.free if u in uu]
        return self.free

    def pick(self, xs, k=None):
        if not xs:
            return None
        if k is None:
            return xs[int(self.r.integers(len(xs)))]
        k = min(k, len(xs))
        return [xs[int(i)] for i in self.r.choice(len(xs), k, replace=False)]

    def create(self):
        B, r = self.B, self.r
        k = r.integers(5)
        if k == 0:
            rot = (0.0, 0.0) if r.random() < 0.3 else (r.uniform(-math.pi, math.pi), r.uniform(0, 2 * math.pi))
            self.free.append(B.add_patch(r.uniform(-2, 2, 3), r.uniform(0.05, 1.5, 2), rot, r.random(3)))
        elif k == 1:
            c = r.uniform(-2, 2, 3)
            self.free.append(B.add_triangle(c + r.normal(0, .5, 3), c + r.normal(0, .5, 3), c + r.normal(0, .5, 3),
                                            r.random(3)))
        elif k == 2:
            rot = (0.0, 0.0) if r.random() < 0.3 else (r.uniform(-math.pi, math.pi), r.uniform(0, 2 * math.pi))
            B.add_tile(r.uniform(-2, 2, 3), r.uniform(0.05, 1.5, 2), rot,
                       (int(r.integers(1, 4)), int(r.integers(1, 4))), r.random(3))
        elif k == 3:
            n = int(r.integers(2, 7))
            B.add_tube(int(r.integers(3, 8)), _stem(r, n), r.uniform(0.002, 0.05, n),
                       r.random((n, 3)) if r.random() < 0.5 else None)
        else:
            c = r.uniform(-2, 2, 3)
            uu = [B.add_triangle(c + r.normal(0, .4, 3), c + r.normal(0, .4, 3), c + r.normal(0, .4, 3), r.random(3))
                  for _ in range(int(r.integers(2, 6)))]
            free = self.alive_free()
            if free and r.random() < 0.3:
                uu.append(self.pick(free))             # absorb an existing free primitive
            B.add_polymesh(uu)
            gone = set(uu)
            self.free = [u for u in self.free if u not in gone]

    def step(self):
        B, r = self.B, self.r
        objs, prims = B.object_ids(), B.all_uuids()
        k = r.integers(16)
        if k <= 2 or not prims:
            self.create()
        elif k == 3 and objs:
            B.copy_object(self.pick(objs) if r.random() < 0.7 else self.pick(objs, 3))
        elif k == 4:
            c = B.copy_primitive(self.pick(prims) if r.random() < 0.7 else self.pick(prims, 3))
            self.free += c if isinstance(c, list) else [c]
        elif k == 5 and objs:
            B.translate_object(self.pick(objs) if r.random() < 0.8 else self.pick(objs, 3), _shift(r))
        elif k == 6 and objs:
            B.rotate_object(self.pick(objs), _angle(r), "xyz"[r.integers(3)])
        elif k == 7 and objs:
            B.rotate_object(self.pick(objs) if r.random() < 0.8 else self.pick(objs, 3), _angle(r), _unit(r))
        elif k == 8 and objs:
            B.rotate_object(self.pick(objs), _angle(r), _unit(r), r.uniform(-2, 2, 3))
        elif k == 9 and objs:
            B.rotate_object_about_origin(self.pick(objs), _angle(r), _unit(r))
        elif k == 10 and objs:
            mode = ("world", "center", "point", "origin")[r.integers(4)]
            B.scale_object(self.pick(objs), _scale(r), mode, r.uniform(-2, 2, 3))
        elif k == 11:
            # any primitive: free ones rotate, members of tiles/tubes are locked, polymesh members rotate
            origin = None if r.random() < 0.5 else r.uniform(-2, 2, 3)
            B.rotate_primitive_axis(self.pick(prims) if r.random() < 0.7 else self.pick(prims, 4), _angle(r), _unit(r), origin)
        elif k == 12:
            s = _scale(r)
            B.scale_primitive_about_point(self.pick(prims), s, r.uniform(-2, 2, 3))
        elif k == 13:
            u = r.random()
            if u < 0.2 and objs:                       # every member of an object: the object goes too
                B.delete_primitive(B.object_uuids(self.pick(objs)))
            elif u < 0.6:
                B.delete_primitive(self.pick(prims))
            else:
                B.delete_primitive(self.pick(prims, int(r.integers(2, 6))))
        elif k == 14 and objs:
            B.delete_object(self.pick(objs) if r.random() < 0.7 else self.pick(objs, 2))
        else:
            self.create()


@case("core.object_transforms.fuzz", tier="ulp", max_ulp=1, fuzz=12)
def object_transforms_fuzz(B, seed):
    r = np.random.default_rng(1000 + seed)
    W = _World(B, r)
    for _ in range(4):
        W.create()
    for _ in range(70):
        W.step()
    return B.snapshot()


def _near_zero(v):
    """Values that should be 0 come out as +-1e-9 depending on rounding (huge in ulps); the snapshot
    comparison has an absolute floor for this, the stream comparison does not."""
    v = np.asarray(v, np.float32)
    return np.where(np.abs(v) < 1e-6, np.float32(0), v)


@case("core.object_transforms.stored_data", tier="ulp", max_ulp=1, fuzz=8)
def object_transforms_stored_data(B, seed):
    """Type-specific data read back through the object transform (tube nodes/radii, tile corners and
    center) and every primitive's transformation matrix, after a fuzzed op sequence."""
    r = np.random.default_rng(2000 + seed)
    W = _World(B, r)
    for _ in range(6):
        W.create()
    for _ in range(50):
        W.step()
    out = []
    for oid in B.object_ids():
        t = B.object_type(oid)
        if t == 2:
            out += B.tube_nodes(oid) + B.tube_radii(oid)
        elif t == 0:
            out += B.tile_vertices(oid) + B.tile_center(oid)
    for u in B.all_uuids():
        out += B.primitive_transform(u)
    return _near_zero(out)


@case("core.object_transforms.origin", tier="ulp", max_ulp=1, fuzz=4)
def object_transforms_origin(B, seed):
    """Object origin semantics: rotate/scale about the stored origin, which translateObject does not
    move and which a copied object has reset to (0,0,0)."""
    r = np.random.default_rng(3000 + seed)
    tile = B.add_tile(r.uniform(-2, 2, 3), (0.6, 0.4), (0.3, 0.7), (2, 3))
    tube = B.add_tube(5, _stem(r, 5), r.uniform(0.01, 0.05, 5))
    pm = B.add_polymesh([B.add_triangle(r.normal(size=3), r.normal(size=3), r.normal(size=3)) for _ in range(4)])
    for o in (tile, tube, pm):
        B.translate_object(o, r.uniform(-1, 1, 3))
        B.rotate_object_about_origin(o, r.uniform(-3, 3), _unit(r))
        B.scale_object(o, _scale(r), "origin")
        c = B.copy_object(o)
        B.rotate_object_about_origin(c, r.uniform(-3, 3), _unit(r))
        B.scale_object(c, (1.3, 0.7, 1.1), "origin")
        cc = B.copy_object(c)
        B.translate_object(cc, (0.5, -0.25, 1.0))
        B.scale_object(cc, (0.8, 0.8, 0.8), "center")
    return B.snapshot()


@case("core.object_transforms.primitive_axis", tier="ulp", max_ulp=1, fuzz=6)
def primitive_axis_rotation(B, seed):
    """rotatePrimitive / scalePrimitiveAboutPoint with a vec3 axis: free primitives move, tile and tube
    members are locked by their parent, polymesh members (deformable) move."""
    r = np.random.default_rng(4000 + seed)
    free = [B.add_patch(r.uniform(-1, 1, 3), r.uniform(0.1, 1, 2), (r.uniform(-1, 1), r.uniform(0, 6))) for _ in range(6)]
    free += [B.add_triangle(r.normal(size=3), r.normal(size=3), r.normal(size=3)) for _ in range(6)]
    tile = B.add_tile((0, 0, 1), (1, 1), (0.2, 0.4), (2, 2))
    tube = B.add_tube(4, _stem(r, 4), [0.02] * 4)
    pm_uu = [B.add_triangle(r.normal(size=3), r.normal(size=3), r.normal(size=3)) for _ in range(4)]
    B.add_polymesh(pm_uu)
    members = B.object_uuids(tile) + B.object_uuids(tube)[:3] + pm_uu
    for _ in range(30):
        u = free[r.integers(len(free))] if r.random() < 0.6 else members[r.integers(len(members))]
        k = r.integers(3)
        if k == 0:
            B.rotate_primitive_axis(u, _angle(r), _unit(r))
        elif k == 1:
            B.rotate_primitive_axis(u, _angle(r), _unit(r), r.uniform(-2, 2, 3))
        else:
            B.scale_primitive_about_point(u, _scale(r), r.uniform(-2, 2, 3))
    B.rotate_primitive_axis(free[:5], 0.7, (1.0, 2.0, -0.5), (0.1, 0.2, 0.3))   # vector overload
    return B.snapshot()


@case("core.object_transforms.delete", tier="exact", fuzz=4)
def delete_semantics(B, seed):
    """Topology after deletes: emptied objects disappear, partial deletes keep member order, deleted
    objects take their primitives, IDs are never reused. Compared as an integer stream."""
    r = np.random.default_rng(5000 + seed)
    tiles = [B.add_tile(r.uniform(-1, 1, 3), (1, 1), (0, 0), (3, 2)) for _ in range(3)]
    tube = B.add_tube(4, _stem(r, 5), [0.02] * 5)
    pm = B.add_polymesh([B.add_triangle(r.normal(size=3), r.normal(size=3), r.normal(size=3)) for _ in range(5)])
    B.delete_primitive(B.object_uuids(tiles[0])[1::2])            # partial, batch
    B.delete_primitive(B.object_uuids(tiles[1]))                  # all members -> object deleted
    B.delete_primitive(B.object_uuids(pm)[2])                     # single
    c = B.copy_object([tiles[0], tube])
    B.delete_object(tiles[2])
    B.delete_object([tube, c[0]])
    B.copy_object(c[1])
    u = B.copy_primitive(B.object_uuids(pm))
    B.delete_primitive(u[1:])
    B.add_patch((0, 0, 0), (1, 1))                                # UUIDs are not reused
    out = []
    for oid in B.object_ids():
        out += [oid, B.object_type(oid), -1] + B.object_uuids(oid) + [-2]
    out += [-3] + B.all_uuids()
    return [float(x) for x in out]


@case("core.object_transforms.leaf_pattern", tier="ulp", max_ulp=1, fuzz=6)
def leaf_pattern(B, seed):
    """The plantarchitecture leaf/fruit pattern: copy a prototype object, scale it, roll/pitch/yaw it
    about x/y/z, rotate about the leaf base along arbitrary axes, translate into place, then
    rescale about the base point; finally delete the prototype."""
    r = np.random.default_rng(6000 + seed)
    tri = []
    for i in range(6):                                   # a small strip "leaf" along +x
        x0, x1 = i / 6, (i + 1) / 6
        w0, w1 = 0.2 * math.sin(math.pi * x0) + 0.01, 0.2 * math.sin(math.pi * x1) + 0.01
        tri.append(B.add_triangle((x0, -w0, 0), (x1, -w1, 0), (x1, w1, 0.01)))
        tri.append(B.add_triangle((x0, -w0, 0), (x1, w1, 0.01), (x0, w0, 0)))
    proto = B.add_polymesh(tri)
    fruit_proto = B.add_tube(6, [(0, 0, 0), (0, 0, 0.05), (0, 0.01, 0.1)], [0.01, 0.03, 0.005])
    for _ in range(8):
        base = r.uniform(-1, 1, 3)
        leaf = B.copy_object(proto)
        B.scale_object(leaf, (r.uniform(0.3, 1.5),) * 3)
        B.rotate_object(leaf, r.uniform(-0.5, 0.5), "x")
        B.rotate_object(leaf, -r.uniform(0, 1), "y")
        B.rotate_object(leaf, r.uniform(0, 2 * math.pi), "z")
        B.translate_object(leaf, base)
        B.rotate_object(leaf, r.uniform(-1, 1), _unit(r), base)
        B.scale_object(leaf, (r.uniform(0.8, 1.2),) * 3, "point", base)
        fruit = B.copy_object(fruit_proto)
        B.scale_object(fruit, (r.uniform(0.5, 2),) * 3)
        B.rotate_object(fruit, r.uniform(0, 2 * math.pi), "x")
        B.translate_object(fruit, base + (0, 0, 0.2))
        B.rotate_object(fruit, r.uniform(-1, 1), _unit(r), base)
    B.delete_object([proto, fruit_proto])
    return B.snapshot()


@case("core.object_transforms.primitive_data", tier="exact", fuzz=3)
def copy_primitive_data(B, seed):
    """copyPrimitive / copyObject carry primitive data to the copies; deletes drop it."""
    r = np.random.default_rng(7000 + seed)
    tile = B.add_tile((0, 0, 0), (1, 1), (0, 0), (2, 2))
    free = [B.add_patch(r.uniform(-1, 1, 3), (0.5, 0.5)) for _ in range(3)]
    for u in B.object_uuids(tile) + free:
        if r.random() < 0.8:
            B.set_pdata_float(u, "a", r.uniform(-5, 5))
        if r.random() < 0.5:
            B.set_pdata_float(u, "b", r.uniform(-5, 5))
    c = B.copy_object(tile)
    p = B.copy_primitive(free)
    B.set_pdata_float(p[0], "a", 42.0)                   # the copy's data is its own
    B.delete_primitive(free[1])
    B.copy_primitive(B.object_uuids(c)[0])
    out = []
    for u in B.all_uuids():
        for lab in ("a", "b"):
            v = B.get_pdata_float(u, lab)
            out += [u, -1.0 if v is None else v]
    return [float(x) for x in out]
