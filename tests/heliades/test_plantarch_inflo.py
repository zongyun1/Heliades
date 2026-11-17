"""W3-D inflorescence: unit tests, the age-0 forced-flowering integration test, and mutation checks."""
import inspect

import numpy as np
import pytest

from heliades.core.random import MinstdRand0
from heliades.core.scene import Scene
from heliades.plantarchitecture import BUD_ACTIVE, BUD_DEAD, BUD_DORMANT, BUD_FLOWER_CLOSED, BUD_FLOWER_OPEN, \
    BUD_FRUITING, PlantArchitecture, hooks
from heliades.plantarchitecture import inflorescence as INF
from heliades.plantarchitecture.architecture import context_generator

f32 = np.float32


def _age0(seed=3):
    sc = Scene()
    sc.seed_random_generator(seed)
    pa = PlantArchitecture(sc)
    pa.loadPlantModelFromLibrary("tomato")
    pid = pa.buildPlantInstanceFromLibrary((0.0, 0.0, 0.0), 0.0)
    return sc, pa, pid


def _calls(seed, gen):
    g = MinstdRand0(seed)
    for n in range(10000):
        if g.x == gen.x:
            return n
        g()
    return None


# ---- helpers ------------------------------------------------------------------------------------------

def test_bend_peduncle_axis_branches():
    a = np.array([0.6, 0.0, 0.8], f32)
    assert np.array_equal(INF.bend_peduncle_axis(a, 0, 0.036), a)                       # no curvature
    up = np.array([0, 0, 1], f32)
    assert np.array_equal(INF.bend_peduncle_axis(up, -900, 0.036), up)                 # vertical: left as it is
    assert np.array_equal(INF.bend_peduncle_axis(up, 900, 0.036), up)                  # already at the target
    down = np.array([0, 0, -1], f32)
    near = np.array([0.01, 0.0, -0.99995], f32)
    assert np.array_equal(INF.bend_peduncle_axis(near, -900, 0.036), down)             # clamped to the target
    b = INF.bend_peduncle_axis(a, -900, 0.036)
    assert b.dtype == f32 and b[2] < a[2] and abs(float(np.linalg.norm(b)) - 1) < 1e-6


def test_fruit_growth_fraction():
    assert INF.fruit_growth_fraction(0, 30) == f32(0.25)
    assert INF.fruit_growth_fraction(30, 30) == f32(1)
    assert INF.fruit_growth_fraction(40, 30) == f32(1)
    assert INF.fruit_growth_fraction(6, 30) == f32(f32(0.25) + f32(f32(f32(0.75) * f32(6)) / f32(30)))


def test_hooks_bound_on_first_use():
    INF._bind_library_hooks()
    from heliades.plantarchitecture.library import tomato_inflorescence as T
    assert hooks.TomatoFlowerPrototype.impl is T.TomatoFlowerPrototype
    assert hooks.TomatoFruitPrototype.impl is T.TomatoFruitPrototype


def test_storage_index_and_shortcut_states():
    sc, pa, pid = _age0()
    ph = pa.plant_instances[pid].shoot_tree[0].phytomers[0]
    fb = ph.floral_buds[0][0]
    assert INF.get_floral_bud_storage_index(ph, fb) == (0, 0)
    with pytest.raises(Exception):
        INF.get_floral_bud_storage_index(ph, fb.copy())
    n = sc.object_count()
    INF.set_floral_bud_state(pa, ph, fb, BUD_DORMANT)
    INF.set_floral_bud_state(pa, ph, fb, BUD_ACTIVE)
    assert fb.state == BUD_ACTIVE and sc.object_count() == n and not fb.peduncle_objIDs


# ---- integration: a real age-0 Phytomer forced to flower ------------------------------------------------

def test_force_age0_phytomer_through_all_states():
    sc, pa, pid = _age0()
    ph = pa.plant_instances[pid].shoot_tree[0].phytomers[0]
    fb = ph.floral_buds[0][0]
    hidden_before = sum(sc.is_object_hidden(o) for o in sc.all_object_ids(include_hidden=True))
    gen = context_generator(sc)
    seen = []
    for k, state in enumerate((BUD_FLOWER_CLOSED, BUD_FLOWER_OPEN, BUD_FRUITING)):
        sc.seed_random_generator(100 + k)
        INF.set_floral_bud_state(pa, ph, fb, state)
        assert _calls(100 + k, gen) == 6                           # SPEC §3.3: one phyllotactic draw per flower
        assert fb.state == state and len(fb.peduncle_objIDs) == 1 and len(fb.inflorescence_objIDs) == 6
        assert len(fb.inflorescence_rotation) == len(fb.inflorescence_bases) == len(fb.inflorescence_base_scales) == 6
        seen += list(fb.inflorescence_objIDs)
        ped = fb.peduncle_objIDs[0]
        assert all(sc.get_primitive_data(u, "object_label") == "peduncle" for u in sc.object_primitive_uuids(ped))
        assert np.array_equal(sc.tube_object_nodes(ped)[0], np.asarray(fb.base_position, f32))
        assert len(ph.peduncle_vertices[0][0]) == 6
    # the three prototypes were built once, hidden, and the old copies were deleted at each rebuild
    hidden = sum(sc.is_object_hidden(o) for o in sc.all_object_ids(include_hidden=True))
    assert hidden - hidden_before == 3
    assert not any(sc.does_object_exist(o) for o in seen[:12])
    assert fb.current_fruit_scale_factor == f32(0.25)
    assert [f32(s) for s in fb.inflorescence_base_scales] == [f32(0.15)] * 6
    INF.set_floral_bud_state(pa, ph, fb, BUD_DEAD)
    assert not fb.inflorescence_objIDs and not fb.peduncle_objIDs and not any(sc.does_object_exist(o) for o in seen)


def test_scale_fraction_scales_about_each_base():
    sc, pa, pid = _age0()
    ph = pa.plant_instances[pid].shoot_tree[0].phytomers[0]
    fb = ph.floral_buds[0][0]
    INF.set_floral_bud_state(pa, ph, fb, BUD_FRUITING)
    T0 = [sc.object_transform(o) for o in fb.inflorescence_objIDs]
    INF.set_inflorescence_scale_fraction(ph, fb, f32(0.25))         # unchanged: no-op
    assert all(np.array_equal(a, sc.object_transform(o)) for a, o in zip(T0, fb.inflorescence_objIDs))
    INF.set_inflorescence_scale_fraction(ph, fb, f32(0.5))
    for a, o, base in zip(T0, fb.inflorescence_objIDs, fb.inflorescence_bases):
        T = sc.object_transform(o).reshape(4, 4)
        assert np.allclose(T[:3, :3], 2 * a.reshape(4, 4)[:3, :3], rtol=1e-6)
        p = a.reshape(4, 4)[:3, 3]
        assert np.allclose(T[:3, 3], base + 2 * (p - base), atol=1e-7)


def test_update_peduncle_radii_reanchors_and_matches_culm():
    sc, pa, pid = _age0()
    ph = pa.plant_instances[pid].shoot_tree[0].phytomers[0]
    fb = ph.floral_buds[0][0]
    INF.set_floral_bud_state(pa, ph, fb, BUD_FLOWER_CLOSED)
    ped = fb.peduncle_objIDs[0]
    nodes = sc.tube_object_nodes(ped).copy()
    INF.update_peduncle_radii(ph)                                    # nothing to do
    assert np.array_equal(sc.tube_object_nodes(ped), nodes)
    fb.base_position = tuple(np.asarray(fb.base_position, f32) + f32(0.01))
    INF.update_peduncle_radii(ph)
    assert np.allclose(sc.tube_object_nodes(ped), nodes + f32(0.01), atol=1e-7)
    ph.shoot_index[2] = ph.shoot_index[0] + 1                       # last node of a max-node shoot: match the culm
    INF.update_peduncle_radii(ph)
    r = INF._internode_radius_at(ph, f32(1))
    assert np.array_equal(sc.tube_object_node_radii(ped), np.full(6, f32(f32(0.9) * r), f32))
    assert ph.peduncle_radius[0][0] == f32(f32(0.9) * r)


# ---- oracle: driver dump agrees with PyHelios, mutations are caught ---------------------------------------

def test_driver_plant_matches_pyhelios_xml():
    """The trace driver links the same libhelios.so as PyHelios: its plant (bud states, flower yaws) is PyHelios's."""
    pytest.importorskip("pyhelios")
    from heliades.parity.suites.plantarch import ops_inflo as M
    from heliades.parity.backends import HeliosBackend
    seed, age, reseed = M.grid(0)
    buds = M.oracle_dump(seed, age, reseed)
    B = HeliosBackend()
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load("tomato")
    B.plantarch_build((0.0, 0.0, 0.0), age)
    root = B.plantarch_xml()
    xml = []
    for shoot in root.iter("shoot"):
        for ph in shoot.findall("phytomer"):
            for fb in ph.iter("floral_bud"):
                st = int(fb.find("bud_state").text)
                if st in (2, 3, 4):
                    xml.append((st, [float(f.find("flower_yaw").text) for f in fb.iter("flower")]))
    from heliades.plantarchitecture.params import rad2deg
    got = [(b["state"], [float("%g" % float(rad2deg(o["rot"][1]))) for o in b["grown"]["objs"]]) for b in buds]
    assert got == xml and len(got) >= 1


def _mutate(monkeypatch, old, new):
    src = inspect.getsource(INF)
    assert src.count(old) == 1, old
    ns = {"__name__": INF.__name__, "__file__": INF.__file__, "__package__": INF.__package__}
    exec(compile(src.replace(old, new), INF.__file__, "exec"), ns)
    for k, v in ns.items():
        if not k.startswith("__") and callable(v) and getattr(v, "__module__", None) == INF.__name__:
            monkeypatch.setattr(INF, k, v)


def _fails(monkeypatch, names, seeds=(0,)):
    from heliades.parity.suites.plantarch import ops_inflo as M
    from heliades.parity import run_case
    monkeypatch.setattr(M, "_SMALL", {})
    return [n for n in names for s in seeds if not run_case(n, s).ok]


PLACE, LABELS, DRAWS = "plantarch.inflo.placement", "plantarch.inflo.placement_labels", "plantarch.inflo.draws"

MUTATIONS = {
    # placement chain
    "yaw_before_translate": (
        "    ctx.translate_object(objID_fruit, fruit_base)\n    if abs(float(yaw_compound)) > _TINY:\n"
        "        ctx.rotate_object(objID_fruit, yaw_compound, peduncle_axis, origin=fruit_base)\n",
        "    if abs(float(yaw_compound)) > _TINY:\n"
        "        ctx.rotate_object(objID_fruit, yaw_compound, peduncle_axis, origin=fruit_base)\n"
        "    ctx.translate_object(objID_fruit, fruit_base)\n", [PLACE]),
    "pitch_before_roll": (
        '    if abs(float(roll)) > _TINY:\n        ctx.rotate_object(objID_fruit, roll, "x")\n'
        '    if abs(float(pitch)) > _TINY:\n        ctx.rotate_object(objID_fruit, pitch, "y")\n',
        '    if abs(float(pitch)) > _TINY:\n        ctx.rotate_object(objID_fruit, pitch, "y")\n'
        '    if abs(float(roll)) > _TINY:\n        ctx.rotate_object(objID_fruit, roll, "x")\n', [PLACE]),
    "ind_from_tip_unsigned": ("ind_from_tip = f32(abs(f32(f32(fruit) - f32(f32(flowers_per_peduncle - 1) / petioles_per_internode))))",
                              "ind_from_tip = f32(fruit)", [PLACE]),
    "bend_sign": ("theta_curvature = V.deg2rad(f32(curvature_degrees_per_meter * f32(segment_length)))",
                  "theta_curvature = V.deg2rad(f32(-curvature_degrees_per_meter * f32(segment_length)))", [PLACE]),
    "azimuth_negated": ("        azimuth = _atan2f(axis[1], axis[0])", "        azimuth = -_atan2f(axis[1], axis[0])", [PLACE]),
    "scale_about_origin": ("scale_object_about_point(objID, (delta_scale, delta_scale, delta_scale), base)",
                           "scale_object_about_point(objID, (delta_scale, delta_scale, delta_scale), (0, 0, 0))", [PLACE]),
    "no_initial_quarter_scale": ("            set_inflorescence_scale_fraction(phytomer, fbud, f32(0.25))", "            pass",
                                 [PLACE]),
    # RNG
    "drop_phyllotactic_resample": ("                pp.internode.phyllotactic_angle.resample()\n", "", [PLACE, DRAWS]),
    "extra_roll_resample": ("    peduncle_roll = ped.roll.val()\n",
                            "    peduncle_roll = ped.roll.val()\n    pp.internode.phyllotactic_angle.resample()\n",
                            [PLACE, DRAWS]),
    # materials / labels
    "fruit_two_sided": ("            make_closed_fruit_surfaces_one_sided(ctx, prototypes[i], plant_name + \"_fruit\")\n",
                        "", [LABELS]),
    "peduncle_unlabelled": ('        ctx.set_primitive_data(ctx.object_primitive_uuids(objID), "object_label", "peduncle")\n',
                            "", [LABELS]),
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    load_plugins()
    old, new, cases = MUTATIONS[mutation]
    _mutate(monkeypatch, old, new)
    assert _fails(monkeypatch, cases) == cases


def test_unmutated_source_passes(monkeypatch):
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    load_plugins()
    _mutate(monkeypatch, "_TINY = 1e-6", "_TINY = 1e-6")
    assert _fails(monkeypatch, [PLACE, LABELS, DRAWS]) == []
