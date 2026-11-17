"""W2-D: tube editing, hidden primitives/objects, typed data, object area, materials, scalar helpers."""
import numpy as np
import pytest

from heliades.core import data as D
from heliades.core import vec as V
from heliades.core.materials import generate_material_label
from heliades.core.scene import Scene, resolve_plugin_asset, try_resolve_plugin_asset

f32 = np.float32


def _tube(S, n=4, nr=6):
    nodes = [(0, 0, 0.1 * i) for i in range(n)]
    return S.add_tube_object(nr, nodes, [0.02] * n, [(0.2, 0.5, 0.1)] * n)


# ---- tubes -----------------------------------------------------------------------------------

def test_append_tube_segment_grows_and_keeps_radial_block_order():
    S = Scene()
    t = _tube(S, 3, 5)
    before = S.object_primitive_uuids(t)
    S.append_tube_segment(t, (0.0, 0.01, 0.35), 0.01, (1.0, 0.0, 0.0))
    after = S.object_primitive_uuids(t)
    assert len(after) == len(before) + 2 * 5 and S.tube_object_node_count(t) == 4
    # radial slot j holds segments 0..n-2 in order: the old ones first, the 2 new ones last
    blocks = [after[6 * j:6 * j + 6] for j in range(5)]
    for j, blk in enumerate(blocks):
        assert blk[:4] == before[4 * j:4 * j + 4] and min(blk[4:]) >= len(before)
    assert all(S.primitive_parent_object(u) == t for u in after)


def test_set_tube_radii_is_path_dependent():
    S = Scene()
    t = _tube(S)
    S.set_tube_radii(t, [0.0, 0.03, 0.03, 0.03])        # node 0 ring collapses onto the node
    S.set_tube_radii(t, [0.05, 0.05, 0.05, 0.05])       # ... and cannot be re-inflated by radius alone
    tv = S.tube_object_triangle_vertices(t)
    assert np.all(tv[0] == tv[0][0]) and np.allclose(np.linalg.norm(tv[1] - (0, 0, 0.1), axis=1), 0.05, atol=1e-6)
    S.set_tube_nodes(t, S.tube_object_nodes(t))         # setTubeNodes recomputes from scratch
    assert np.allclose(np.linalg.norm(S.tube_object_triangle_vertices(t)[0], axis=1), 0.05, atol=1e-6)


def test_tube_setters_bake_the_object_transform():
    S = Scene()
    t = _tube(S)
    S.translate_object(t, (1.0, 2.0, 3.0))
    nodes = S.tube_object_nodes(t)
    S.set_tube_radii(t, S.tube_object_node_radii(t))
    assert np.array_equal(S.object_transform(t), np.eye(4, dtype=f32).reshape(16))
    assert np.array_equal(S.tube_object_nodes(t), nodes)


def test_tube_segment_volume_and_errors():
    S = Scene()
    t = _tube(S)
    v = S.tube_object_segment_volume(t, 0)
    assert abs(float(v) - np.pi * 0.1 / 3 * 3 * 0.02 ** 2) < 1e-8
    with pytest.raises(ValueError):
        S.tube_object_segment_volume(t, 3)
    with pytest.raises(ValueError):
        S.set_tube_radii(t, [0.1, 0.1])
    with pytest.raises(ValueError):
        S.append_tube_segment(t, (0, 0, 1), -1.0, (0, 0, 0))


def test_prune_tube_nodes():
    S = Scene()
    t = _tube(S, 5, 4)
    S.prune_tube_nodes(t, 3)
    assert S.tube_object_node_count(t) == 3 and len(S.object_primitive_uuids(t)) == 2 * 2 * 4
    S.prune_tube_nodes(t, 1)
    assert t not in S.objects


# ---- hidden ----------------------------------------------------------------------------------

def test_hidden_object_keeps_ids_and_copies_are_visible():
    S = Scene()
    t = _tube(S)
    u = S.add_patch((0, 0, 0), (1, 1))
    S.hide_object(t)
    assert S.all_uuids() == [u] and S.all_object_ids() == []
    assert S.primitive_count() == len(S.primitives) and S.object_count() == 1
    c = S.copy_object(t)
    assert not S.is_object_hidden(c) and S.all_object_ids() == [c]
    assert all(not S.is_primitive_hidden(x) for x in S.object_primitive_uuids(c))
    snap = S.snapshot(include_hidden=True)
    assert snap.hidden.sum() == len(S.object_primitive_uuids(t)) and list(snap.objects.hidden) == [True, False]
    S.show_object(t)
    assert S.all_object_ids() == [t, c]
    with pytest.raises(ValueError):
        S.is_primitive_hidden(10 ** 6)


# ---- data ------------------------------------------------------------------------------------

def test_typed_data_round_trip_and_registry():
    S = Scene()
    uu = [S.add_patch((i, 0, 0), (1, 1)) for i in range(4)]
    S.set_primitive_data(uu, "n", 3)
    assert type(S.get_primitive_data(uu[0], "n")) is np.int32
    S.set_primitive_data(uu[1], "n", 2.7)                 # cast to the registered int
    assert S.get_primitive_data(uu[1], "n") == 2 and S.get_primitive_data_type(uu[1], "n") == D.TYPE_INT
    with pytest.raises(TypeError):
        S.set_primitive_data(uu[0], "n", "text")
    S.set_primitive_data(uu[0], "v", (1.0, 2.0, 3.0))
    assert S.get_primitive_data(uu[0], "v").dtype == np.float32
    S.set_primitive_data(uu[0], "i3", (1, 2, 3))
    assert S.get_primitive_data_type(uu[0], "i3") == D.TYPE_INT3
    S.set_primitive_data(uu[0], "b", True)
    assert S.get_primitive_data(uu[0], "b") is True
    S.set_primitive_data(uu[0], "d", np.float64(0.1))
    assert S.get_primitive_data(uu[0], "d") == 0.1 and S.get_primitive_data_type(uu[0], "d") == D.TYPE_DOUBLE
    S.set_primitive_data_vector(uu[0], "fv", [0.5, 1.5], "float")
    assert S.get_primitive_data_size(uu[0], "fv") == 2 and S.get_primitive_data(uu[0], "fv") == f32(0.5)
    S.delete_primitive(uu)                                # registry released with the last holder
    u = S.add_patch((0, 0, 0), (1, 1))
    S.set_primitive_data(u, "n", "now a string")
    with pytest.raises(ValueError):
        S.set_primitive_data([u, 999], "x", 1)            # every UUID is checked first
    assert not S.does_primitive_data_exist(u, "x")


def test_filter_swap_and_pop_order():
    S = Scene()
    uu = [S.add_patch((i, 0, 0), (1, 1)) for i in range(6)]
    for u, v in zip(uu, [1, 5, 2, 5, 3, 5]):
        S.set_primitive_data(u, "k", v)
    # Helios walks from the back and swaps each rejected ID with the last kept one
    assert S.filter_primitives_by_data(uu, "k", 5, "==") == [uu[3], uu[1], uu[5]]
    assert S.filter_primitives_by_data(uu, "k", 5.0, "==") == []        # float filter on int data
    S.set_primitive_data(uu, "u", np.uint32(7))
    S.set_primitive_data(uu, "dd", np.float64(1.0))
    assert sorted(S.filter_primitives_by_data(uu, "u", 7, ">=", dtype="uint")) == uu
    assert sorted(S.filter_primitives_by_data(uu, "dd", 0.5, ">", dtype="double")) == uu
    S.set_primitive_data(uu[:2], "object_label", "petiolule")
    assert sorted(S.filter_primitives_by_data(uu, "object_label", "petiolule")) == uu[:2]
    with pytest.raises(ValueError):
        S.filter_primitives_by_data(uu, "k", 1, "=")


def test_object_data_copied_and_deleted():
    S = Scene()
    t = _tube(S)
    S.set_object_data(t, "leaf_area", f32(0.25))
    c = S.copy_object(t)
    assert S.get_object_data(c, "leaf_area") == f32(0.25)
    S.delete_object(t)
    assert S.list_all_object_data_labels() == ["leaf_area"] and S.filter_objects_by_data([c], "leaf_area", 0.2, ">") == [c]


def test_object_area_double_accumulation():
    S = Scene()
    t = S.add_tile_object((0, 0, 0), (1.0, 1.0), (0.0, 0.0), (3, 3))
    areas = [float(S.primitive_area(u)) for u in S.object_primitive_uuids(t)]
    assert S.object_area(t) == f32(sum(areas))


# ---- materials -------------------------------------------------------------------------------

def test_automatic_material_labels_match_helios():
    # values read from the oracle (PyHelios getPrimitiveMaterialLabel)
    assert generate_material_label((0, 0, 0, 1)) == "__auto_3783893658918629045"
    assert generate_material_label((0.5, 0.25, 1, 1)) == "__auto_8026340529729172239"


def test_named_material_recolors_users_and_copy_on_write():
    S = Scene()
    a, b = S.add_triangle((0, 0, 0), (1, 0, 0), (0, 1, 0)), S.add_triangle((0, 0, 1), (1, 0, 1), (0, 1, 1))
    S.add_material("petiole")
    S.set_material_color("petiole", (0.1, 0.2, 0.3, 1.0))
    S.assign_material_to_primitive([a, b], "petiole")
    assert np.allclose(S.primitives[a].color, (0.1, 0.2, 0.3, 1))
    assert S.get_material(S.get_primitive_material_id(a)).reference_count == 2
    S.set_primitive_color(a, (0.9, 0.9, 0.9))            # shared: a gets a __copy_ material
    assert S.get_primitive_material_label(a).startswith("__copy_") and S.get_primitive_material_label(b) == "petiole"
    S.delete_primitive(b)
    assert S.get_material(S.material_label_to_id["petiole"]).reference_count == 0


# ---- helios:: helpers ------------------------------------------------------------------------

def test_scalar_helpers():
    assert V.deg2rad(180.0) == f32(np.pi) and V.rad2deg(f32(np.pi)) == f32(180.0)
    assert V.acos_safe(1.5) == 0 and V.asin_safe(-2.0) == V.asinf(-1.0)
    assert V.clamp(f32(2), f32(0), f32(1)) == 1 and np.isnan(V.clamp(f32("nan"), f32(0), f32(1)))
    assert V.sum_f([f32(0.1)] * 3) == f32(f32(f32(0.1) + f32(0.1)) + f32(0.1))
    with pytest.raises(ValueError):
        V.sum_f([])


def test_resolve_plugin_asset(tmp_path, monkeypatch):
    d = tmp_path / "plugins" / "plantarchitecture" / "assets" / "obj"
    d.mkdir(parents=True)
    (d / "TomatoFruit.obj").write_text("")
    monkeypatch.setenv("HELIOS_BUILD", str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)
    assert resolve_plugin_asset("plantarchitecture", "assets/obj/TomatoFruit.obj") == str((d / "TomatoFruit.obj").resolve())
    assert try_resolve_plugin_asset("plantarchitecture", "nope.obj") == ""
    with pytest.raises(FileNotFoundError):
        resolve_plugin_asset("plantarchitecture", "nope.obj")


# ---- mutation checks against the oracle ------------------------------------------------------

def _case_fails(name):
    from heliades.parity import run_case
    from heliades.parity.cases import CASES
    return any(not run_case(name, s).ok for s in range(CASES[name]["fuzz"]))


def test_mutation_radius_only_rescale_is_caught(monkeypatch):
    """Recomputing the rings from scratch in setTubeRadii (dropping the path dependence) must fail."""
    pytest.importorskip("pyhelios")
    from heliades.core import objects as O
    from heliades.parity.ops import load_plugins
    load_plugins()
    orig = O.TubeEditMixin.set_tube_radii

    def from_scratch(self, oid, radii):
        orig(self, oid, radii)
        self.set_tube_nodes(oid, self.objects[oid].extra["nodes"])
    monkeypatch.setattr(O.TubeEditMixin, "set_tube_radii", from_scratch)
    assert _case_fails("core_gaps.tube.radius_only_repeat")


def test_mutation_filter_order_is_caught(monkeypatch):
    """Returning filter results in input order instead of Helios's swap-and-pop order must fail."""
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    load_plugins()
    orig = D._TypedStore.filter

    def stable(self, ids, label, value, comparator="==", dtype=None):
        kept = set(orig(self, ids, label, value, comparator, dtype))
        return [i for i in D._ids(ids) if i in kept]
    monkeypatch.setattr(D._TypedStore, "filter", stable)
    assert _case_fails("core_gaps.data.primitive")
