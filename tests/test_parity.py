import pytest

pytest.importorskip("pyhelios")

from heliades.parity import run_case  # noqa: E402
from heliades.parity.cases import CASES  # noqa: E402
from heliades.parity.ops import load_plugins  # noqa: E402

load_plugins()


# Cases that take >= 5 s (all seeds) on the reference machine (RTX 5060 Ti box): large or old plants,
# long growth sequences, the tomato acceptance grid, full-blade builds, OBJ asset sweeps, plus the
# `*_state` / `*_labels` cases that reuse a cached build of a slow case (they would pay for it).
# Skipped unless --runslow.
SLOW_CASES = {
    "core.materials.obj_interplay", "fileio.obj.assets", "fileio.obj.assets_topology", "plantarch.bean.age0_full",
    "plantarch.bean.age0_full_attrs", "plantarch.bean.flowering", "plantarch.bean.flowering_full",
    "plantarch.bean.gallery", "plantarch.bean.gallery_full", "plantarch.bean.steps", "plantarch.bean.steps_full",
    "plantarch.grape.age120", "plantarch.grape.age76", "plantarch.grape.age84", "plantarch.grape.load",
    "plantarch.grape.steps", "plantarch.inflo.draws", "plantarch.inflo.placement",
    "plantarch.inflo.placement_labels", "plantarch.inflo.prototypes_attrs", "plantarch.leafproto.edges",
    "plantarch.leafproto.edges_attrs", "plantarch.leafproto.tomato", "plantarch.leafproto.tomato_attrs",
    "plantarch.maize.age0_attrs", "plantarch.maize.age0_full", "plantarch.maize.flag_leaf",
    "plantarch.maize.gallery", "plantarch.maize.gallery_vegetative", "plantarch.maize.last_vegetative",
    "plantarch.maize.steps", "plantarch.soybean.age0_full", "plantarch.soybean.age0_full_attrs",
    "plantarch.soybean.flowering", "plantarch.soybean.flowering_full", "plantarch.soybean.gallery",
    "plantarch.soybean.gallery_full", "plantarch.soybean.steps", "plantarch.soybean.steps_full",
    "plantarch.strawberry.fruiting", "plantarch.strawberry.gallery", "plantarch.strawberry.gallery_vegetative",
    "plantarch.strawberry.last_vegetative", "plantarch.strawberry.steps", "plantarch.tomato.age0_full",
    "plantarch.tomato.age0_full_attrs", "plantarch.tomato.age50", "plantarch.tomato.age75_vegetative",
    "plantarch.tomato.flowering", "plantarch.tomato.split", "plantarch.tomato.steps",
    "plantarch.tomato.steps_replay", "plantarch.w4a.capsicum.reproductive",
    "plantarch.w4a.capsicum.reproductive_full", "plantarch.w4a.capsicum.steps_late",
    "plantarch.w4a.cherrytomato.age0_full", "plantarch.w4a.cherrytomato.age0_full_attrs",
    "plantarch.w4a.cherrytomato.gallery", "plantarch.w4a.cherrytomato.gallery_full",
    "plantarch.w4a.cherrytomato.reproductive", "plantarch.w4a.cherrytomato.reproductive_full",
    "plantarch.w4a.cherrytomato.steps", "plantarch.w4a.cherrytomato.steps_late", "plantarchitecture.tomato",
    "plantarchitecture.tomato.canopy", "plantarchitecture.tomato.canopy_state", "plantarchitecture.tomato.staggered",
    "plantarchitecture.tomato.staggered_state", "plantarchitecture.tomato.state",
    "plantarchitecture.tomato.two_plants", "plantarchitecture.tomato.two_plants_state",
    # W5-B (herbaceous vegetables)
    "plantarch.butterlettuce.age0_attrs", "plantarch.butterlettuce.age0_full", "plantarch.butterlettuce.ages",
    "plantarch.butterlettuce.steps", "plantarch.sugarbeet.age0_attrs", "plantarch.sugarbeet.age0_full",
    "plantarch.sugarbeet.ages", "plantarch.sugarbeet.steps", "plantarch.asparagus.ages", "plantarch.asparagus.steps",
    "plantarch.cucumber.age0_attrs", "plantarch.cucumber.age0_full", "plantarch.cucumber.ages",
    "plantarch.cucumber.steps", "plantarch.cucumber.bump_maps", "plantarch.cowpea.age0_attrs",
    "plantarch.cowpea.age0_full", "plantarch.cowpea.ages", "plantarch.cowpea.steps",
    # W5-C
    *(f"plantarch.{m}.{c}" for m in ("bindweed", "cheeseweed", "groundcherryweed", "puncturevine", "bougainvillea")
      for c in ("steps", "ages")),
    "plantarch.puncturevine.ages_late", "plantarch.grapewye.load", "plantarch.grapewye.age0",
    "plantarch.grapewye.age0_build_parameters", "plantarch.grapewye.steps", "plantarch.grapewye.age76",
    "plantarch.grapewye.age84", "plantarch.grapewye.age100", "plantarch.grapewye.small_age150",
    # W5-A: the library grasses
    "plantarch.rice.steps",
    "plantarch.rice.tillering",
    "plantarch.rice.stage1",
    "plantarch.rice.heading",
    "plantarch.rice.anthesis",
    "plantarch.rice.ripening",
    "plantarch.rice.mature",
    "plantarch.rice.gallery365",
    "plantarch.rice.tillers_vegetative",
    "plantarch.rice.tillers_ripening",
    "plantarch.rice.panicle_build_parameters",
    "plantarch.maize.tassel",
    "plantarch.maize.anthesis",
    "plantarch.maize.mature",
    "plantarch.maize.gallery365",
    "plantarch.maize.tassel_steps",
    "plantarch.maize.tassel_build_parameters",
    "plantarch.wheat.steps",
    "plantarch.wheat.vegetative",
    "plantarch.wheat.heading",
    "plantarch.wheat.ear",
    "plantarch.wheat.ripe",
    "plantarch.wheat.gallery365",
    "plantarch.wheat.tillers_vegetative",
    "plantarch.wheat.tillers_heading",
    "plantarch.wheat.tillers_ripe",
    "plantarch.sorghum.age0",
    "plantarch.sorghum.steps",
    "plantarch.sorghum.vegetative",
    "plantarch.sorghum.flag_leaf",
    "plantarch.sorghum.panicle",
    "plantarch.sorghum.grain_fill",
    "plantarch.sorghum.mature",
    "plantarch.sorghum.gallery365",
    "plantarch.sorghum.tillers_vegetative",
    "plantarch.sorghum.tillers_panicle",
    "plantarch.sorghum.panicle_steps",
    "plantarch.sorghum.panicle_build_parameters",
}
# W5-D nut trees: everything but the parameter trees, the build-parameter checks and two age-0 builds
from heliades.parity.suites.plantarch import cases_nut_trees as _NT  # noqa: E402

SLOW_CASES |= {n for n in CASES if n.split(".")[1] in _NT.MODELS
               and n not in {f"plantarch.{m}.params" for m in _NT.MODELS}
               and n not in {"plantarch.almond_wood_colony.age0", "plantarch.pistachio.age0"}}
# W5-E fruit trees: likewise, keeping the parameter trees, the build-parameter checks and two age-0 builds fast
from heliades.parity.suites.plantarch import cases_fruit_trees as _FT  # noqa: E402

SLOW_CASES |= {n for n in CASES if n.split(".")[1] in _FT.MODELS
               and n not in {f"plantarch.{m}.params" for m in _FT.MODELS}
               and n not in {"plantarch.olive.age0", "plantarch.easternredbud.age0"}}


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.slow) if n in SLOW_CASES else n
                                  for n in sorted(CASES)])
def test_parity_with_helios(name):
    for seed in range(CASES[name]["fuzz"]):
        r = run_case(name, seed)
        assert r.ok, f"seed {seed}\n{r}"


def test_compare_detects_divergence():
    """The harness must catch a one-ulp-scale-changing bug, not just pass everything."""
    import numpy as np

    from heliades.parity.backends import ForgeBackend
    from heliades.parity.compare import compare_snapshots
    a, b = ForgeBackend(), ForgeBackend()
    for B in (a, b):
        B.add_patch((0.1, 0.2, 0.3), (1.0, 2.0), (0.4, 0.5))
    sb = b.snapshot()
    sb.vertices[0, 2, 1] = np.nextafter(sb.vertices[0, 2, 1], np.float32(10)) + np.float32(1e-5)
    r = compare_snapshots(a.snapshot(), sb)
    assert not r.ok and "vertices" in r.first_divergence
