# SPDX-License-Identifier: LGPL-2.1-or-later
"""W5-A: the library grasses (maize tassel, wheat, sorghum; rice in test_cereals_rice.py): unit tests and oracle
mutation checks."""
from __future__ import annotations

import numpy as np
import pytest

from heliades.core import scene as SC
from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture, hooks
from heliades.plantarchitecture.library import build_parameters as BP
from heliades.plantarchitecture.library import grass_inflorescence as GI
from heliades.plantarchitecture.library import maize as MZ
from heliades.plantarchitecture.library import sorghum as SG
from heliades.plantarchitecture.library import wheat as WH
from heliades.plantarchitecture.params import BUD_DEAD
from heliades.plantarchitecture.random_parameter import PlantArchitectureError

from test_strawberry_maize import _patch_source  # noqa: E402  (tests/plantarch is on sys.path under pytest)

f32 = np.float32


def _plant(model, seed=1, age=0.0, build_parameters=None):
    s = Scene()
    s.seed_random_generator(seed)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary(model)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), age, build_parameters)
    return s, pa, pid


def test_models_registered_with_build_parameters():
    s = Scene()
    pa = PlantArchitecture(s)
    for name, n_names in (("wheat", 1), ("sorghum", 1 + 2 + 25 + 3 + 6), ("maize", 2 + 27 + 2 + 9)):
        pa.loadPlantModelFromLibrary(name)
        assert "mainstem" in pa.shoot_types
        assert len(BP.getPlantModelBuildParameterNames(pa, name)) == n_names
    assert hooks.MaizeTasselPrototype.impl is MZ.MaizeTasselPrototype
    assert hooks.SorghumPaniclePrototype.impl is SG.SorghumPaniclePrototype
    assert hooks.WheatSpikePrototype.impl is WH.WheatSpikePrototype


def test_tassel_parameters_presets_and_keys():
    p = MZ.MaizeTasselParameters.fromBuildParameters({})
    assert p.primary_branch_count == 10 and p.spikelet_subdivisions == 4
    few = MZ.MaizeTasselParameters.fromBuildParameters({"tassel_type": 0.0})
    assert few.primary_branch_count == 6 and few.length == f32(0.36)
    many = MZ.MaizeTasselParameters.fromBuildParameters({"tassel_type": 1.0, "tassel_length": 0.5})
    assert many.primary_branch_count == 22 and many.length == f32(0.5)      # a key overrides the preset
    mid = MZ.MaizeTasselParameters.fromBuildParameters({"tassel_type": 0.75})
    assert mid.primary_branch_count == 16                                   # round(0.5 * 10 + 0.5 * 22)
    with pytest.raises(PlantArchitectureError, match="whole number"):
        MZ.MaizeTasselParameters.fromBuildParameters({"tassel_branch_count": 2.5})
    with pytest.raises(PlantArchitectureError, match="outside the valid range"):
        MZ.MaizeTasselParameters.fromBuildParameters({"anther_color_r": 1.2})


def test_tassel_mesh_topology():
    s = Scene()
    s.seed_random_generator(3)
    params = MZ.MaizeTasselParameters.fromBuildParameters({"tassel_branch_count": 2, "tassel_secondary_branch_count": 1})
    oid = MZ.buildMaizeTassel(s, params)
    uu = s.object_primitive_uuids(oid)
    assert len(uu) > 1000
    assert all(s.get_primitive_data(u, "object_label") == "fruit" for u in uu[:10])
    # colors are clamped to [0, 1] as Context::addTriangle(RGBcolor) does
    colors = np.array([s.get_primitive_color(u) for u in uu])
    assert colors.max() <= 1 and colors.min() >= 0


def test_context_add_tube_counts():
    s = Scene()
    nodes = [GI.vec(0, 0, 0), GI.vec(0, 0, 0.1), GI.vec(0.01, 0, 0.2)]
    uu = GI.context_add_tube(s, 5, nodes, [f32(0.01)] * 3, [GI.rgb(0.2, 0.3, 0.4)] * 3)
    assert len(uu) == 2 * 2 * 5 and s.object_count() == 0          # loose triangles, no object


def test_wheat_creation_function_profiles():
    assert WH.wheatBladeLengthFraction(0, 8) == f32(0.4)
    assert WH.wheatBladeLengthFraction(7, 8) == f32(0.88)          # the flag leaf
    assert WH.wheatBladeLengthFraction(6, 8) == f32(1)
    s, pa, pid = _plant("wheat", age=36, build_parameters={"tiller_count": 2})
    tree = pa.plant_instances[pid].shoot_tree
    buds = [vb.state for ph in tree[0].phytomers[2:] for petiole in ph.axillary_vegetative_buds for vb in petiole]
    assert buds and all(st == BUD_DEAD for st in buds)                 # only the two lowest axils tiller
    assert len(tree) == 3 and {t.parent_node_index for t in tree[1:]} == {0, 1}
    assert all(int(t.shoot_parameters.max_nodes.val()) == 8 - (t.parent_node_index + 1) for t in tree[1:])


def test_sorghum_tiller_order_and_rachis_profile():
    assert SG._TILLER_LEAF_NUMBERS == (3, 4, 2, 1, 5, 6)
    assert SG.interpolateRachisProfile(0.2, 0.21, 0.12, 0.0) == f32(0.2)
    assert SG.interpolateRachisProfile(0.2, 0.21, 0.12, 1.0) == f32(0.12)
    assert abs(float(SG.interpolateRachisProfile(0.2, 0.21, 0.12, 0.5)) - 0.21) < 1e-7


def test_resolve_overlaps_matches_loop():
    """The vectorised overlap pass equals the C++ triple loop (grid cells, i ascending, j > i)."""
    rng = np.random.default_rng(0)
    c = rng.uniform(0, 0.01, (60, 3)).astype(f32)
    scale = rng.uniform(0.8, 1.2, 60).astype(f32)
    cr, cs = f32(0.0018), f32(f32(f32(2) * f32(1.3)) * f32(0.0018))
    got = SG._resolve_overlaps(c, scale, cr, cs)
    cells = np.floor(c / cs).astype(int)
    grid: dict = {}
    for i in range(len(c)):
        grid.setdefault(tuple(cells[i]), []).append(i)
    disp = np.zeros_like(c)
    for i in range(len(c)):
        for dx, dy, dz in SG._OFFSETS:
            for j in grid.get((cells[i][0] + dx, cells[i][1] + dy, cells[i][2] + dz), []):
                if j <= i:
                    continue
                sep = c[i] - c[j]
                d = f32(np.sqrt(f32(f32(sep[0] * sep[0] + sep[1] * sep[1]) + sep[2] * sep[2])))
                ov = f32(f32(cr * f32(scale[i] + scale[j])) - d)
                if ov > 0 and d > f32(1e-9):
                    push = sep * f32(f32(f32(0.5) * ov) / d)
                    disp[i] = disp[i] + push
                    disp[j] = disp[j] - push
    assert np.array_equal(got, c + disp)


# ---- mutation checks (oracle) ---------------------------------------------------------------------------------------

def _fails(name, seeds):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


MUTATIONS = {
    # make_vec3(0, randu, randu) is evaluated right to left: z drawn before y
    "tassel_anther_scatter_left_to_right": (
        MZ, None, "buildMaizeTassel",
        "                sz = ctx.randu(f32(-0.25), f32(0.25))\n                sy = ctx.randu(f32(-0.25), f32(0.25))\n",
        "                sy = ctx.randu(f32(-0.25), f32(0.25))\n                sz = ctx.randu(f32(-0.25), f32(0.25))\n",
        "plantarch.maize.tassel", (0,)),
    # Context::addTriangle(RGBcolor) clamps the brightened colors to [0, 1] (done by the core since P1)
    "tassel_no_color_clamp": (
        SC, None, "_color_rgba", "    return np.where(c < 0, f32(0), np.where(c > 1, f32(1), c)).astype(f32)\n",
        "    return c\n", "plantarch.maize.tassel", (0,)),
    # secondary branches: the droop argument (right) is drawn before the insertion angle
    "tassel_secondary_draws_left_to_right": (
        MZ, None, "buildMaizeTassel",
        "        droop = deg2rad(ctx.randu(f32(0), f32(15)))                     # arguments right to left\n"
        "        insertion_angle = deg2rad(ctx.randu(f32(20), f32(35)))\n",
        "        insertion_angle = deg2rad(ctx.randu(f32(20), f32(35)))\n"
        "        droop = deg2rad(ctx.randu(f32(0), f32(15)))\n",
        "plantarch.maize.tassel", (0,)),
    # the internode runs from ligule to ligule: less the sheath of the leaf below
    "wheat_no_sheath_below": (
        WH, None, "WheatPhytomerCreationFunction",
        "    setInternodeMaxLength(phytomer, f32(f32(culm_internode_length + sheath_length) - sheath_length_below))\n",
        "    setInternodeMaxLength(phytomer, f32(culm_internode_length + sheath_length))\n",
        "plantarch.wheat.vegetative", (0, 1)),
    # the wheat flag leaf is turned to 33 degrees off the culm
    "wheat_flag_leaf_angle": (
        WH, None, "WheatPhytomerCreationFunction", "        setGrassFlagLeafAngle(phytomer, f32(33))\n",
        "        setGrassFlagLeafAngle(phytomer, f32(30))\n", "plantarch.wheat.heading", (0, 1)),
    # a tiller has one leaf fewer for each main-stem leaf below its own
    "wheat_tiller_leaf_number": (
        WH, None, "WheatPhytomerCreationFunction",
        "int(main_stem.shoot_parameters.max_nodes.val()) - int(parent_shoot_node_index + 1))",
        "int(main_stem.shoot_parameters.max_nodes.val()) - int(parent_shoot_node_index))",
        "plantarch.wheat.tillers_heading", (2, 3)),
    # sorghum tillers are added in the order T3, T4, T2, T1, T5, T6
    "sorghum_tiller_order": (SG, None, "_TILLER_LEAF_NUMBERS", "_TILLER_LEAF_NUMBERS = (3, 4, 2, 1, 5, 6)",
                             "_TILLER_LEAF_NUMBERS = (1, 2, 3, 4, 5, 6)", "plantarch.sorghum.tillers_vegetative", (0, 1)),
    # each plant's internode length is drawn, then resampled for the next plant
    "sorghum_no_resample": (SG, None, "buildSorghumPlant",
                            "    pa.shoot_types[\"mainstem\"].internode_length_max.resample()\n", "",
                            "plantarch.sorghum.vegetative", (0,)),
    # the packing pass: a grain's displacement takes the subtractions (pairs (i, k), i < k) before its own additions
    "sorghum_packing_order": (SG, None, "_resolve_overlaps",
                              "    phase = np.concatenate([np.zeros(len(I), np.int64), np.ones(len(I), np.int64)])\n",
                              "    phase = np.concatenate([np.ones(len(I), np.int64), np.zeros(len(I), np.int64)])\n",
                              "plantarch.sorghum.panicle", (0,)),
    # glume tip offset: std::sin(float) of deg2rad(glume_spread) (sinf); here dropped altogether
    "sorghum_glume_spread": (SG, None, "addSorghumGlumes",
                             "    spread_term = f32(f32(f32(0.5) * sinf(deg2rad(glume_spread))) * glume_length)\n",
                             "    spread_term = f32(0)\n", "plantarch.sorghum.panicle", (0,)),
}

FAST_MUTATIONS = {"wheat_no_sheath_below", "wheat_flag_leaf_angle", "wheat_tiller_leaf_number"}


@pytest.mark.parametrize("mutation", [m if m in FAST_MUTATIONS else pytest.param(m, marks=pytest.mark.slow)
                                      for m in sorted(MUTATIONS)])
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    load_plugins()
    module, owner, attr, old, new, case_name, seeds = MUTATIONS[mutation]
    if attr == "_TILLER_LEAF_NUMBERS":
        src_new = tuple(int(x) for x in new.split("(")[1].rstrip(")").split(","))
        assert old in open(module.__file__).read()
        monkeypatch.setattr(module, attr, src_new)
    else:
        _patch_source(monkeypatch, module, owner, attr, old, new)
    assert _fails(case_name, seeds)
