# SPDX-License-Identifier: LGPL-2.1-or-later
"""W4-B: bean and soybean (library entries, appendShoot) and the mutation checks of their parity cases."""
from __future__ import annotations

import types

import numpy as np
import pytest

from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture
from heliades.plantarchitecture import phytomer as PH
from heliades.plantarchitecture import shoot as SM
from heliades.plantarchitecture.library import bean as BEAN
from heliades.plantarchitecture.library import soybean as SOY
from heliades.plantarchitecture.random_parameter import PlantArchitectureError

f32 = np.float32


def _pa(model, seed=1):
    s = Scene()
    s.seed_random_generator(seed)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary(model)
    return s, pa


@pytest.mark.parametrize("model,types_", [("bean", {"unifoliate", "trifoliate"}),
                                          ("soybean", {"unifoliate", "trifoliate", "branch"})])
def test_shoot_types(model, types_):
    _, pa = _pa(model)
    assert set(pa.shoot_types) == types_
    assert pa.shoot_types["unifoliate"].phytomer_parameters.phytomer_creation_function is None
    assert pa.shoot_types["trifoliate"].phytomer_parameters.phytomer_creation_function is not None
    uni = pa.shoot_types["unifoliate"].phytomer_parameters.leaf.prototype
    tri = pa.shoot_types["trifoliate"].phytomer_parameters.leaf.prototype
    assert uni.unique_prototype_identifier != tri.unique_prototype_identifier


def test_bean_textures():
    _, pa = _pa("bean")
    tri = pa.shoot_types["trifoliate"].phytomer_parameters.leaf.prototype
    uni = pa.shoot_types["unifoliate"].phytomer_parameters.leaf.prototype
    assert tri.leaf_texture_file == {0: "BeanLeaf_tip.png", -1: "BeanLeaf_right_centered.png",
                                     1: "BeanLeaf_left_centered.png"}
    assert uni.leaf_texture_file == {0: "BeanLeaf_unifoliate_centered.png"} and uni.unique_prototypes == 2


@pytest.mark.parametrize("model", ["bean", "soybean"])
def test_age0_structure(model):
    _, pa = _pa(model)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 0)
    tree = pa.plant_instances[pid].shoot_tree
    assert [s.shoot_type_label for s in tree] == ["unifoliate", "trifoliate"]
    uni, tri = tree
    assert len(uni.phytomers[0].petiole_objIDs) == 2 and len(uni.phytomers[0].leaf_objIDs[0]) == 1
    assert tri.is_axis_continuation and tri.parent_shoot_ID == 0 and tri.parent_node_index == 0 and tri.rank == 0
    assert uni.growth_state.max_nodes_growth_cap == 1 and not uni.meristem_is_alive
    assert np.array_equal(tri.shoot_internode_vertices[0][0], uni.shoot_internode_vertices[-1][-1])
    assert uni.childIDs == {0: [1]}


def test_append_shoot_errors():
    _, pa = _pa("bean")
    pid = pa.addPlantInstance((0, 0, 0), 0)
    from heliades.plantarchitecture.params import make_AxisRotation
    with pytest.raises(PlantArchitectureError, match="empty shoot"):
        pa.appendShoot(pid, 0, 1, make_AxisRotation(0, 0, 0), 0.001, 0.02, 0.1, 0.1, 0, "trifoliate")
    with pytest.raises(PlantArchitectureError, match="does not exist"):
        pa.appendShoot(pid, 0, 1, make_AxisRotation(0, 0, 0), 0.001, 0.02, 0.1, 0.1, 0, "nope")
    pa.addBaseStemShoot(pid, 1, make_AxisRotation(0, 0, 0), 0.0005, 0.03, 0.01, 0.01, 0, "unifoliate")
    with pytest.raises(PlantArchitectureError, match="Parent with ID"):
        pa.appendShoot(pid, 5, 1, make_AxisRotation(0, 0, 0), 0.001, 0.02, 0.1, 0.1, 0, "trifoliate")
    with pytest.raises(PlantArchitectureError, match="max node number"):
        pa.appendShoot(pid, 0, 9, make_AxisRotation(0, 0, 0), 0.001, 0.02, 0.1, 0.1, 0, "trifoliate")


def test_models_registered():
    _, pa = _pa("tomato")
    assert "bean" in pa.getAvailablePlantModels() and "soybean" in pa.getAvailablePlantModels()


# ---- mutation checks (oracle) -------------------------------------------------------------------

def _fails(name, seeds=range(8)):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


def _patch_source(monkeypatch, module, owner, attr, old, new):
    """Recompile ``module`` with one source edit; install ``attr`` (of class ``owner``, or the module itself
    when ``owner`` is None) from the edited copy."""
    src = open(module.__file__).read()
    assert src.count(old) == 1, old
    mod = types.ModuleType(module.__name__)
    mod.__dict__.update(__name__=module.__name__, __package__=module.__package__, __file__=module.__file__)
    exec(compile(src.replace(old, new), module.__file__, "exec"), mod.__dict__)
    if owner is None:
        monkeypatch.setattr(module, attr, getattr(mod, attr))
    else:
        monkeypatch.setattr(getattr(module, owner), attr, getattr(getattr(mod, owner), attr))


MUTATIONS = {
    # appendShoot: an axis continuation, not a lateral seated 0.9 parent radii off the axis
    "no_axis_continuation": (SM, None, "appendShoot", "    shoot.is_axis_continuation = True\n", "",
                             "plantarch.bean.steps"),
    # appendShoot: the parent keeps growing (no growth cap)
    "no_parent_growth_cap": (SM, None, "appendShoot",
                             "    parent.growth_state.max_nodes_growth_cap = int(parent.current_node_number)\n", "",
                             "plantarch.soybean.steps"),
    # base rotation drawn left to right (pitch first)
    "base_rotation_left_to_right": (BEAN, None, "base_rotation_draws",
                                    "    roll = pa.context_ptr.randu(f32(0), hi)\n    yaw = pa.context_ptr.randu(f32(0), hi)\n"
                                    "    pitch = pa.context_ptr.randu(f32(0), f32(float(f32(0.05)) * np.pi))\n",
                                    "    pitch = pa.context_ptr.randu(f32(0), f32(float(f32(0.05)) * np.pi))\n"
                                    "    roll = pa.context_ptr.randu(f32(0), hi)\n    yaw = pa.context_ptr.randu(f32(0), hi)\n",
                                    "plantarch.bean.age0_full"),
    # (0.05f * M_PI in float instead of double gives the same bits, f32 0x1.41b2f8p-3: not observable, so not listed)
    # the second petiole of a node (unifoliate) is rotated 2*pi*p/npet about the internode
    "no_second_petiole_rotation": (PH, "Phytomer", "__init__", "            if petiole > 0:\n                budrot",
                                   "            if petiole > 1:\n                budrot", "plantarch.bean.age0_full"),
    # leaves_per_petiole == 1: the blade elevation correction about the petiole tip axis (its node sign is always +1
    # for the unifoliate, phytomer 0, so the direction is mutated instead)
    "single_leaf_correction_reversed": (PH, "Phytomer", "orientLeaf", "f32(blade_correction * f32(sign)), tip)",
                                        "f32(-blade_correction * f32(sign)), tip)", "plantarch.soybean.age0_full"),
    # the bean creation function's leaf scale in float arithmetic instead of double
    "bean_leaf_scale_float": (BEAN, None, "_bean_phytomer_creation", "leaf_scale = f32(min(1.0, 0.6 + 0.4 * age / 8.0))",
                              "leaf_scale = min(f32(1), f32(f32(0.6) + f32(f32(f32(0.4) * f32(age)) / f32(8))))",
                              "plantarch.bean.steps"),
    # the soybean creation function's node threshold
    "soybean_branch_nodes": (SOY, None, "_soybean_phytomer_creation", "shoot_node_index >= 5:", "shoot_node_index >= 4:",
                             "plantarch.soybean.steps"),
    # the soybean's closed flower is the bean's mesh (AS:4042), not its own open-flower mesh
    "soybean_closed_flower_mesh": (SOY, None, "_soybean_flower", '"assets/obj/BeanFlower_closed_white.obj"',
                                   '"assets/obj/SoybeanFlower_open_white.obj"', "plantarch.soybean.flowering_full"),
    # open and closed bean flowers are different meshes (the loadOBJ overload of the pod is not observable: every
    # face of BeanPod.obj follows a usemtl, so the default colour is never used)
    "bean_flower_open_closed_swapped": (BEAN, None, "_bean_flower", "    if flower_is_open:\n",
                                        "    if not flower_is_open:\n", "plantarch.bean.flowering_full"),
}

_HOOKS = {"_bean_phytomer_creation": ("BeanPhytomerCreationFunction", BEAN),
          "_soybean_phytomer_creation": ("SoybeanPhytomerCreationFunction", SOY),
          "_soybean_flower": ("SoybeanFlowerPrototype", SOY), "_bean_flower": ("BeanFlowerPrototype", BEAN)}


SLOW_MUTATIONS = {"soybean_closed_flower_mesh", "bean_flower_open_closed_swapped"}   # >= 5 s


@pytest.mark.parametrize("mutation", [pytest.param(m, marks=pytest.mark.slow) if m in SLOW_MUTATIONS else m for m in sorted(MUTATIONS)])
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity.suites.plantarch import cases_bean_soybean as CW
    from heliades.parity.ops import load_plugins
    load_plugins()
    monkeypatch.setattr(CW, "STEP_AGES", tuple(range(2, 21, 2)))
    module, owner, attr, old, new, case_name = MUTATIONS[mutation]
    _patch_source(monkeypatch, module, owner, attr, old, new)
    if attr in _HOOKS:
        # the hook holds the function object: rebind it to the patched copy
        hook_name, mod = _HOOKS[attr]
        monkeypatch.setattr(getattr(mod, hook_name), "impl", getattr(mod, attr))
    assert _fails(case_name)
