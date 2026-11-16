# SPDX-License-Identifier: LGPL-2.1-or-later
"""W4-C: strawberry and maize library plants (unit tests + oracle mutation checks)."""
from __future__ import annotations

import types

import numpy as np
import pytest

from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture, hooks
from heliades.plantarchitecture import leaf_prototype as LP
from heliades.plantarchitecture import phytomer as PH
from heliades.plantarchitecture.library import maize as MZ
from heliades.plantarchitecture.library import strawberry as SB

f32 = np.float32


def _plant(model, seed=1, age=0.0):
    s = Scene()
    s.seed_random_generator(seed)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary(model)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), age)
    return s, pa, pid


def test_models_registered():
    s = Scene()
    pa = PlantArchitecture(s)
    for name in ("strawberry", "maize", "tomato"):
        pa.loadPlantModelFromLibrary(name)
        assert "mainstem" in pa.shoot_types


def test_hooks_are_shared_objects():
    assert SB.named_hook("MaizeEarPrototype") is hooks.MaizeEarPrototype
    assert hooks.StrawberryFlowerPrototype.impl is SB.StrawberryFlowerPrototype
    assert hooks.MaizePhytomerCreationFunction.impl is MZ.MaizePhytomerCreationFunction


def test_strawberry_age0_layout():
    s, pa, pid = _plant("strawberry")
    shoot = pa.plant_instances[pid].shoot_tree[0]
    ph = shoot.phytomers[0]
    assert len(ph.leaf_objIDs[0]) == 3 and not ph.leaf_is_sessile
    assert pa.plant_instances[pid].max_age == f32(120)
    assert not pa.leaf_geometry_is_stub


def test_maize_age0_layout():
    s, pa, pid = _plant("maize")
    ph = pa.plant_instances[pid].shoot_tree[0].phytomers[0]
    assert ph.leaf_is_sessile and len(ph.leaf_objIDs[0]) == 1
    assert ph.leaf_flexibility > 0
    rest = pa.unique_leaf_prototype_rest_geometry[(int(ph.phytomer_parameters.leaf.prototype.unique_prototype_identifier), 1)]
    assert len(rest) == 10 and rest[0][0].subdivisions_x == 50 and rest[0][0].subdivisions_y == 8
    assert len(rest[0][0].vertices) == 51 * 9
    # the class map is recorded on the blade triangles that cover a class
    uu = s.object_primitive_uuids(ph.leaf_objIDs[0][0])
    tagged = [u for u in uu if s.does_primitive_data_exist(u, "spectral_class_map")]
    assert 0 < len(tagged) <= len(uu)


def test_maize_leaves_droop_and_ear_bud():
    s, pa, pid = _plant("maize", age=30)
    shoot = pa.plant_instances[pid].shoot_tree[0]
    ph = shoot.phytomers[2]
    assert ph.leaf_last_deformed_scale[0][0] > 0
    states = [fb.state for p in shoot.phytomers for petiole in p.floral_buds for fb in petiole]
    from heliades.plantarchitecture.params import BUD_DEAD
    assert all(st == BUD_DEAD for st in states)          # the ear is on node 14, not yet grown


def test_flexibility_aging():
    assert PH.flexibility_aging_multiplier(0, 11, 6) == f32(1)
    m = [float(PH.flexibility_aging_multiplier(a, 11, 6)) for a in (0, 10, 22, 40, 100)]
    assert m == sorted(m) and abs(m[-1] - 6) < 1e-3


def test_class_map_cover():
    classes = np.array([[0, 1, 1]] * 3, np.uint16)
    # column c owns c/2 <= u < (c+1)/2 (helios::classMapAreaFractions): column 0 (class 0) owns u < 0.5
    assert LP.class_map_covers_a_class(classes, 3, 3, [(0.6, 0.9), (0.95, 0.9), (0.95, 0.6)])
    assert not LP.class_map_covers_a_class(classes, 3, 3, [(0.0, 0.0), (0.2, 0.0), (0.0, 0.2)])


# ---- mutation checks (oracle) -------------------------------------------------------------------

def _fails(name, seeds=range(8)):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


def _patch_source(monkeypatch, module, owner, attr, old, new):
    """Recompile ``module`` with one source edit and install ``attr`` (of class ``owner``, or of the module when
    ``owner`` is None) from the edited copy. Hook bindings made by the re-executed decorators are undone, and the
    hook whose implementation is ``attr`` (if any) is pointed at the edited function instead."""
    src = open(module.__file__).read()
    attrs = attr if isinstance(attr, tuple) else (attr,)
    assert src.count(old) == len(attrs), old
    saved = {k: h.impl for k, h in vars(hooks).items() if isinstance(h, hooks.Hook)}
    mod = types.ModuleType(module.__name__)
    mod.__dict__.update(__name__=module.__name__, __package__=module.__package__, __file__=module.__file__)
    exec(compile(src.replace(old, new), module.__file__, "exec"), mod.__dict__)
    for k, impl in saved.items():
        getattr(hooks, k).impl = impl
    for attr in attrs:
        if owner is None:
            original = getattr(module, attr)
            monkeypatch.setattr(module, attr, getattr(mod, attr))
            for k, impl in saved.items():
                if impl is original:
                    monkeypatch.setattr(getattr(hooks, k), "impl", getattr(mod, attr))
        else:
            monkeypatch.setattr(getattr(module, owner), attr, getattr(getattr(mod, owner), attr))


MUTATIONS = {
    # maize base rotation: GCC draws roll, yaw, then pitch (right to left); pitch first is caught at age 0
    "maize_rotation_left_to_right": (MZ, None, "buildMaizePlant",
                                     "    roll = ctx.randu(f32(0), f32(2.0 * np.pi))\n"
                                     "    yaw = ctx.randu(f32(0), f32(2.0 * np.pi))\n"
                                     "    pitch = ctx.randu(f32(0), f32(float(f32(0.035)) * np.pi))\n",
                                     "    pitch = ctx.randu(f32(0), f32(float(f32(0.035)) * np.pi))\n"
                                     "    yaw = ctx.randu(f32(0), f32(2.0 * np.pi))\n"
                                     "    roll = ctx.randu(f32(0), f32(2.0 * np.pi))\n",
                                     "plantarch.maize.age0_full"),
    # 1.f - 0.75f * fmax(0.f, r - 0.45f) / 0.55f: C fmax returns double, so the chain is double
    "maize_flex_factor_float": (MZ, None, "MaizePhytomerCreationFunction",
                                "        factor = 1.0 - float(f32(0.75)) * fm / float(f32(0.55))\n",
                                "        factor = float(f32(f32(1) - f32(f32(f32(0.75) * f32(fm)) / f32(0.55))))\n",
                                "plantarch.maize.flag_leaf"),
    # the leaf flexibility taper with rank
    "maize_no_flex_taper": (MZ, None, "MaizePhytomerCreationFunction",
                            "        phytomer.leaf_flexibility = f32(float(phytomer.leaf_flexibility) * factor)\n", "",
                            "plantarch.maize.flag_leaf"),
    # the flag leaf turned to 23 degrees off the culm
    "maize_no_flag_leaf_angle": (MZ, None, "MaizePhytomerCreationFunction",
                                 "        setGrassFlagLeafAngle(phytomer, 23.0)\n", "        pass\n",
                                 "plantarch.maize.flag_leaf"),
    # rotateLeaf turns about -cross(internode, petiole), not +cross
    "rotate_leaf_axis_sign": (PH, "Phytomer", "rotateLeaf",
                              "pitch_axis = _smul(-1, V.cross(internode_axis, petiole_axis))",
                              "pitch_axis = V.cross(internode_axis, petiole_axis)", "plantarch.maize.flag_leaf"),
    # the blade keeps softening as it ages (flexibility_aging)
    "no_flexibility_aging": (PH, "Phytomer", "deformLeafUnderSelfWeight", "        if aging_doubling_days > 0:\n",
                             "        if False:\n", "plantarch.maize.steps"),
    # the deflected lattice is written back to the leaf (setPolymeshObjectVertices)
    "no_droop": (PH, "Phytomer", "deformLeafUnderSelfWeight",
                 "        set_polymesh_object_vertices(ctx, objID_leaf, deformed_global)\n", "",
                 "plantarch.maize.steps"),
    # glibc expf (not correctly rounded) in flexibilityAgingProgress, vs the correctly rounded double exp
    "aging_exp_double": (PH, None, "flexibility_aging_progress",
                         "    sigmoid = f32(f32(1) / f32(f32(1) + expf(f32(f32(-2) * f32(t - f32(2))))))\n",
                         "    sigmoid = f32(1.0 / (1.0 + math.exp(-2.0 * (float(t) - 2.0))))\n",
                         "plantarch.maize.steps"),
    # recordSpectralClassMap tags only the triangles that cover a class texel
    "class_map_every_triangle": (LP, None, "recordSpectralClassMap",
                                 "        if not has_transparency or class_map_covers_a_class(",
                                 "        if True or class_map_covers_a_class(", "plantarch.maize.age0_attrs"),
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity import cases_plantarch_w4c as CW
    from heliades.parity.ops import load_plugins
    load_plugins()
    module, owner, attr, old, new, case_name = MUTATIONS[mutation]
    monkeypatch.setitem(CW.STEP_END, "maize", 20)
    _patch_source(monkeypatch, module, owner, attr, old, new)
    assert _fails(case_name)
