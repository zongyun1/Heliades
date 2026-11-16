"""W2-E: loadOBJ and the material API share one material table (MaterialsMixin)."""
import numpy as np
import pytest

from heliades.core import materials as M
from heliades.core.scene import Scene
from heliades.parity.cases_fileio import OBJ_DIR, TOMATO

f32 = np.float32


def _users_match_reference_counts(S):
    for i, m in S.materials.items():
        assert m.reference_count == len(S.get_primitives_using_material(m.label)), m.label
    assert sum(m.reference_count for m in S.materials.values()) == len(S.primitives)


def test_scene_has_a_single_material_registry():
    S = Scene()
    for gone in ("primitive_material_label", "_mat_registry", "_fileio_add_material", "_fileio_assign_material",
                 "_fileio_does_material_exist"):
        assert not hasattr(S, gone)
    S.load_obj(TOMATO[0], (0, 0, 0), 0.75, (0, 0), (0, 0, 0), "ZUP")
    assert not hasattr(S, "_fileio_materials")


def test_load_obj_materials_are_context_materials():
    S = Scene()
    pre = S.add_patch((0, 0, 0), (1, 1), (0, 0), (f32(0.26), f32(0.38), f32(0.1)))     # Material.003's Kd
    auto = S.get_primitive_material_label(pre)
    uu = S.load_obj(TOMATO[0], (0, 0, 0), 0.75, (0, 0), (0, 0, 0), "ZUP")
    assert S.list_materials() == ["Material.001", "Material.003"]
    labels = {S.get_primitive_material_label(u) for u in uu}
    assert labels == {"Material.001", "Material.003"}
    # each loaded triangle passed through the automatic material of its colour before its named one
    assert S.get_material(S.get_material_id_from_label(auto)).reference_count == 1
    _users_match_reference_counts(S)
    S.set_material_color("Material.001", (0.1, 0.2, 0.3, 1))
    petals = S.get_primitives_using_material("Material.001")
    assert petals and all(np.array_equal(S.primitives[u].color, np.array([0.1, 0.2, 0.3, 1], f32)) for u in petals)
    # a renamed label frees the MTL name; a still-used one makes the next load take a suffix
    S.rename_material("Material.001", "petal")
    S.load_obj(TOMATO[0], (0, 0, 0), 0.75, (0, 0), (0, 0, 0), "ZUP")
    assert S.list_materials() == ["Material.001", "Material.003", "Material.003_1", "petal"]
    _users_match_reference_counts(S)


def test_map_d_only_material_overrides_and_copies_split_it():
    S = Scene()
    uu = S.load_obj(OBJ_DIR + "BougainvilleaFlower.obj")
    over = [u for u in uu if S.is_primitive_texture_color_overridden(u)]
    assert over and all(S.get_primitive_material_label(u) == "Material.001" for u in over)
    assert S.is_material_texture_color_overridden("Material.001")
    # copyPrimitive calls overrideTextureColor() on the copy: a shared named material is copied for it
    c = S.copy_primitive(over[0])
    assert S.get_primitive_material_label(c) == f"__copy_{S.current_material_id - 1}_{c}"
    assert S.get_primitive_material_label(over[0]) == "Material.001"
    _users_match_reference_counts(S)
    S.use_primitive_texture_color(c)
    assert not S.is_primitive_texture_color_overridden(c)


def test_primitive_order_matches_libstdcxx():
    """Three patches, delete UUID 0, copy one, add 20 more: Helios lists the last 20 as [23..14, 4..13]."""
    o = M.UnorderedUIntMapOrder()
    for u in range(3):
        o.insert(u)
    o.erase(0)
    for u in range(3, 24):
        o.insert(u)
    assert list(o) == list(range(23, 13, -1)) + list(range(1, 14))


def test_primitive_order_against_helios():
    pytest.importorskip("pyhelios")
    from pyhelios import Context
    r = np.random.default_rng(3)
    c, o, alive = Context(), M.UnorderedUIntMapOrder(), []
    for step in range(150):
        k = r.random()
        if k < 0.6 or not alive:
            for _ in range(int(r.integers(1, 40))):
                u = c.addPatch()
                o.insert(u)
                alive.append(u)
        elif k < 0.85:
            d = sorted({int(x) for x in r.choice(alive, size=min(len(alive), int(r.integers(1, 30))), replace=False)})
            c.deletePrimitive(d)
            for u in d:
                o.erase(u)
                alive.remove(u)
        else:
            for u in c.copyPrimitive([int(x) for x in r.choice(alive, size=min(len(alive), 5), replace=False)]):
                o.insert(u)
                alive.append(u)
    assert list(c.getPrimitivesUsingMaterial(c.getPrimitiveMaterialLabel(alive[0]))) == list(o)


def test_scene_primitive_order_follows_creation_copy_and_delete():
    S = Scene()
    o = M.UnorderedUIntMapOrder()
    for _ in range(30):
        o.insert(S.add_patch())
    S.delete_primitive(5)
    o.erase(5)
    for u in S.copy_primitive([1, 2, 3]):
        o.insert(u)
    assert list(S.primitive_order) == list(o)
    assert S.get_primitives_using_material(S.get_primitive_material_label(0)) == list(o)


# ---- mutation checks: the interplay case must catch these ---------------------------------------

def _caught(name):
    from heliades.parity import run_case
    from heliades.parity.cases import CASES
    return any(not run_case(name, s).ok for s in range(CASES[name]["fuzz"]))


def test_mutation_copy_ignoring_texture_override_is_caught(monkeypatch):
    pytest.importorskip("pyhelios")

    def plain(self, p):
        self.materials[p.material_id].reference_count += 1

    monkeypatch.setattr(M.MaterialsMixin, "_material_reference_copied", plain)
    assert _caught("core.materials.obj_interplay")


def test_mutation_material_refresh_in_uuid_order_is_caught(monkeypatch):
    pytest.importorskip("pyhelios")
    orig = M.MaterialsMixin._refresh_material_users

    def uuid_order(self, mid, texture_changed):
        saved = self.primitive_order
        self.primitive_order = sorted(self.primitives)
        try:
            orig(self, mid, texture_changed)
        finally:
            self.primitive_order = saved

    monkeypatch.setattr(M.MaterialsMixin, "_refresh_material_users", uuid_order)
    assert _caught("core.materials.obj_interplay")
