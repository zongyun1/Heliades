"""Sphere/disk/cone/box objects (W1-C): topology, getters, and error parity with Helios."""
import math

import numpy as np
import pytest

from heliades.core.objects import OBJECT_BOX, OBJECT_CONE, OBJECT_DISK, OBJECT_SPHERE
from heliades.core.scene import Scene


def test_sphere_topology_and_getters():
    S = Scene()
    o = S.add_sphere_object(10, (1, 2, 3), 5.0)
    assert S.object_type(o) == OBJECT_SPHERE
    assert len(S.object_primitive_uuids(o)) == 2 * 10 + 2 * 10 * 8
    assert np.array_equal(S.sphere_object_center(o), np.float32([1, 2, 3]))
    assert np.array_equal(S.sphere_object_radius(o), np.float32([5, 5, 5]))
    assert S.sphere_object_subdivision_count(o) == 10
    assert S.sphere_object_volume(o) == pytest.approx(4 / 3 * math.pi * 125, rel=1e-6)
    for u in S.object_primitive_uuids(o):  # outward-facing triangles
        v = S.primitive_vertices(u)
        assert np.dot(S.primitive_normal(u), v.mean(0) - [1, 2, 3]) > 0
    e = S.add_sphere_object(3, (0, 0, 0), (1, 2, 3))
    assert np.array_equal(S.sphere_object_radius(e), np.float32([1, 2, 3]))
    assert len(S.object_primitive_uuids(S.add_sphere_object(1, (0, 0, 0), 1.0))) == 2


def test_disk_topology_and_getters():
    S = Scene()
    o = S.add_disk_object((8, 16), (1, 2, 3), (4, 5), (0.0, 0.0), (1, 0, 0))
    assert S.object_type(o) == OBJECT_DISK
    assert len(S.object_primitive_uuids(o)) == 8 + 8 * 15 * 2
    assert np.array_equal(S.disk_object_center(o), np.float32([1, 2, 3]))
    assert np.array_equal(S.disk_object_size(o), np.float32([4, 5]))
    assert S.disk_object_subdivision_count(o) == 8
    d = S.add_disk_object(6, (0, 0, 0), (1, 1))
    assert np.array_equal(S.primitives[S.object_primitive_uuids(d)[0]].color, np.float32([1, 0, 0, 1]))
    a = S.add_disk_object(6, (0, 0, 0), (1, 1), (0.3, 0.1), (0.1, 0.2, 0.3, 0.4))
    assert S.primitives[S.object_primitive_uuids(a)[0]].color[3] == np.float32(0.4)


def test_cone_topology_and_getters():
    S = Scene()
    o = S.add_cone_object(10, (0, 0, 0), (0, 0, 5), 2.0, 1.0)
    assert S.object_type(o) == OBJECT_CONE
    assert len(S.object_primitive_uuids(o)) == 20
    assert np.array_equal(S.cone_object_node(o, 0), np.float32([0, 0, 0]))
    assert np.array_equal(S.cone_object_node(o, 1), np.float32([0, 0, 5]))
    assert S.cone_object_node_radius(o, 0) == 2 and S.cone_object_node_radius(o, 1) == 1
    assert np.array_equal(S.cone_object_axis_unit_vector(o), np.float32([0, 0, 1]))
    assert S.cone_object_length(o) == 5
    assert S.cone_object_volume(o) == pytest.approx(math.pi * 5 / 3 * 7, rel=1e-6)
    with pytest.raises(ValueError):
        S.cone_object_node(o, 2)


def test_box_topology_normals_and_getters():
    S = Scene()
    for rev in (False, True):
        o = S.add_box_object((1, 2, 3), (3, 2, 1), (2, 3, 4), reverse_normals=rev)
        assert S.object_type(o) == OBJECT_BOX
        assert len(S.object_primitive_uuids(o)) == 2 * (4 * (2 + 3) + 2 * 3)
        for u in S.object_primitive_uuids(o):
            out = np.dot(S.primitive_normal(u), S.primitive_vertices(u).mean(0) - [1, 2, 3])
            assert (out < 0) if rev else (out > 0)
        assert np.array_equal(S.box_object_center(o), np.float32([1, 2, 3]))
        assert np.array_equal(S.box_object_size(o), np.float32([3, 2, 1]))
        assert S.box_object_subdivision_count(o) == (2, 3, 4)
        assert S.box_object_volume(o) == 6


def test_ids_and_type_checks():
    S = Scene()
    s = S.add_sphere_object(4, (0, 0, 0), 1.0)
    b = S.add_box_object((0, 0, 0), (1, 1, 1), (1, 1, 1))
    assert (s, b) == (1, 2)
    assert all(S.primitive_parent_object(u) == b for u in S.object_primitive_uuids(b))
    with pytest.raises(ValueError):
        S.box_object_size(s)


@pytest.mark.parametrize("call", [
    lambda S: S.add_sphere_object(5, (0, 0, 0), 0.0),
    lambda S: S.add_sphere_object(5, (0, 0, 0), (1.0, -1.0, 1.0)),
    lambda S: S.add_box_object((0, 0, 0), (1, 0, 1), (1, 1, 1)),
    lambda S: S.add_box_object((0, 0, 0), (1, 1, 1), (1, 0, 1)),
])
def test_errors_match_helios(call):
    pytest.importorskip("pyhelios")
    from heliades.parity.backends import HeliosBackend
    from heliades.parity.ops import load_plugins
    load_plugins()
    with pytest.raises(ValueError):
        call(Scene())

    class Shim:  # the same call against the oracle
        def __init__(self, B):
            self.B = B

        def add_sphere_object(self, n, c, r):
            return self.B.add_sphere(n, c, r)

        def add_box_object(self, c, s, d):
            return self.B.add_box(c, s, d)
    with pytest.raises(Exception, match="(?i)positive"):
        call(Shim(HeliosBackend()))
