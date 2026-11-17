"""Heliades-only checks of W1-B semantics (the Helios parity cases live in heliades/parity/suites/core/cases_object_transforms.py)."""
import math

import numpy as np
import pytest

from heliades.core.object_transforms import (make_rotation_matrix_about, make_rotation_matrix_axis,
                                             make_scale_matrix_about_point)
from heliades.core.primitives import make_rotation_matrix
from heliades.core.scene import Scene

f32 = np.float32


def _apply(T, p):
    return (T.reshape(4, 4) @ np.r_[np.asarray(p, float), 1.0])[:3]


def test_axis_matrix_matches_named_axes():
    for name, ax in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
        np.testing.assert_allclose(make_rotation_matrix_axis(0.7, np.array(ax) * 3.0), make_rotation_matrix(0.7, name),
                                   atol=1e-7)


def test_rotation_about_origin_fixes_the_line():
    o, ax = np.array([1.0, -2.0, 0.5]), np.array([0.3, 1.0, -0.2])
    T = make_rotation_matrix_about(1.1, o, ax)
    np.testing.assert_allclose(_apply(T, o + 2 * ax), o + 2 * ax, atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(T.reshape(4, 4)[:3, :3]), 1.0, atol=1e-6)


def test_scale_about_point_fixes_the_point():
    p = (0.5, -1.0, 2.0)
    np.testing.assert_allclose(_apply(make_scale_matrix_about_point((2, 3, 0.5), p), p), p, atol=1e-6)


def test_copy_primitive_preserves_every_field_and_detaches():
    S = Scene()
    oid = S.add_tile_object((0, 0, 0), (1, 1), (0, 0), (2, 1))
    u = S.objects[oid].uuids[0]
    p = S.primitives[u]
    p.uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], f32)
    p.some_future_field = {"k": [1, 2]}                 # fields other packages may add
    S.set_primitive_data(u, "w", 3.5)
    c = S.copy_primitive(u)
    q = S.primitives[c]
    assert c == u + 2 and q.parent_object == 0 and q.uuid == c
    np.testing.assert_array_equal(q.transform, p.transform)
    np.testing.assert_array_equal(q.uv, p.uv)
    assert q.some_future_field == p.some_future_field and q.some_future_field is not p.some_future_field
    assert S.get_primitive_data(c, "w") == 3.5
    assert S.copy_primitive([u, c]) == [c + 1, c + 2]


def test_copy_object_deep_copies_local_data_and_resets_origin():
    S = Scene()
    t = S.add_tube_object(4, [(0, 0, 0), (0, 0, 1), (0, 1, 2)], [0.1, 0.1, 0.05])
    S.translate_object(t, (1, 0, 0))
    c = S.copy_object(t)
    assert S.objects[c].type == S.objects[t].type
    assert all(S.primitives[u].parent_object == c for u in S.objects[c].uuids)
    np.testing.assert_array_equal(S.objects[c].origin, 0)          # Helios default-constructs it
    assert np.any(S.objects[t].origin != 0)
    S.objects[c].extra["nodes"][0, 0] = 99
    assert S.objects[t].extra["nodes"][0, 0] == 0
    np.testing.assert_array_equal(S.tube_object_nodes(t)[0], [1, 0, 0])   # stored local, read through transform


def test_transforms_leave_local_data_and_origin_alone():
    S = Scene()
    t = S.add_tube_object(4, [(0, 0, 0), (0, 0, 1)], [0.1, 0.1])
    nodes, origin = S.objects[t].extra["nodes"].copy(), S.objects[t].origin.copy()
    S.translate_object(t, (1, 2, 3))
    S.rotate_object(t, 0.4, (1, 1, 0), (0, 0, 1))
    S.scale_object(t, (2, 2, 2))
    np.testing.assert_array_equal(S.objects[t].extra["nodes"], nodes)
    np.testing.assert_array_equal(S.objects[t].origin, origin)
    np.testing.assert_allclose(S.tube_object_node_radii(t), [0.2, 0.2], rtol=1e-6)


def test_delete_primitives_removes_emptied_objects():
    S = Scene()
    a = S.add_tile_object((0, 0, 0), (1, 1), (0, 0), (2, 2))
    b = S.add_tile_object((0, 0, 1), (1, 1), (0, 0), (2, 1))
    S.delete_primitives(S.objects[a].uuids[1:3])
    assert S.objects[a].uuids == [0, 3]
    S.delete_primitives(list(S.objects[b].uuids))
    assert b not in S.objects
    with pytest.raises(ValueError):
        S.delete_primitives([0, 0])                      # the second one no longer exists
    assert 0 not in S.primitives


def test_delete_object_takes_its_primitives():
    S = Scene()
    a = S.add_tile_object((0, 0, 0), (1, 1), (0, 0), (2, 2))
    free = S.add_patch((0, 0, 0), (1, 1))
    S.delete_object(a)
    assert S.all_uuids() == [free] and S.all_object_ids() == []
    with pytest.raises(ValueError):
        S.delete_object(a)


def test_member_primitives_are_locked_except_polymesh():
    S = Scene()
    t = S.add_tile_object((0, 0, 0), (1, 1), (0, 0), (1, 1))
    u = S.objects[t].uuids[0]
    before = S.primitives[u].transform.copy()
    S.rotate_primitive_about_axis(u, 0.5, (0, 1, 1))
    S.scale_primitive_about_point(u, (2, 2, 2), (1, 0, 0))
    np.testing.assert_array_equal(S.primitives[u].transform, before)
    tri = S.add_triangle((0, 0, 0), (1, 0, 0), (0, 1, 0))
    S.add_polymesh_object([tri])
    before = S.primitives[tri].transform.copy()
    S.rotate_primitive_about_axis(tri, 0.5, (0, 1, 1), (1, 1, 1))
    assert np.any(S.primitives[tri].transform != before)


def test_rotate_object_axis_vector_is_about_world_origin():
    S = Scene()
    t = S.add_tile_object((5, 0, 0), (1, 1), (0, 0), (1, 1))
    S.rotate_object(t, math.pi / 2, (0, 0, 1))
    np.testing.assert_allclose(S.object_center(t), (0, 5, 0), atol=1e-5)
    S.rotate_object_about_origin(t, math.pi / 2, (0, 0, 1))      # about the creation center (5,0,0)
    np.testing.assert_allclose(S.object_center(t), (0, -5, 0), atol=1e-5)
    np.testing.assert_allclose(S.tile_object_center(t), (0, -5, 0), atol=1e-5)
    with pytest.raises(ValueError):
        S.rotate_object(t, 0.1, "w")
