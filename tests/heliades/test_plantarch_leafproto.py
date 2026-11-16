"""W3-A: leaf blade geometry -- GenericLeafPrototype, the blade lattice, the petiolule, deformLeafLattice,
the blades built by W3-B's prototype cache, and mutation checks against the oracle."""
import inspect

import numpy as np
import pytest

from heliades.core.random import MinstdRand0, unit_uniform
from heliades.core.scene import Scene
from heliades.parity.tools.draw_count import calls_between
from heliades.plantarchitecture import LeafShapeValues, PlantArchitecture, PlantArchitectureError, hooks
from heliades.plantarchitecture import leaf_prototype as LPM

f32 = np.float32


def _tomato(seed=3):
    sc = Scene()
    sc.seed_random_generator(seed)
    pa = PlantArchitecture(sc)
    pa.loadPlantModelFromLibrary("tomato")
    return sc, pa


# ---- unit ------------------------------------------------------------------------------------------

def test_petiolule_counts():
    assert LPM.petioluleVertexCount() == 57 and LPM.petioluleRadialSubdivisions() == 8


def test_hook_bound_on_import():
    assert hooks.GenericLeafPrototype.impl is LPM.GenericLeafPrototype
    sc, pa = _tomato()
    proto = pa.shoot_types["mainstem"].phytomer_parameters.leaf.prototype.copy()
    proto.build_petiolule = False
    oid = hooks.GenericLeafPrototype(sc, proto, 0)
    assert hooks.GenericLeafPrototype.impl is LPM.GenericLeafPrototype
    assert sc.object_type(oid) == 5 and proto.last_built_shape_valid


def test_lattice_flat_and_errors():
    s = LeafShapeValues()
    s.subdivisions_x, s.subdivisions_y, s.aspect_ratio = 4, 2, 0.5
    L = LPM.buildLeafBladeLattice(s)
    assert L.shape == (15, 3) and np.all(L[:, 2] == 0)
    assert np.array_equal(L[:5, 0], np.arange(5, dtype=f32) * f32(0.25)) and L[0, 1] == f32(-0.25)
    for nx, ny in ((0, 2), (3, 1), (3, 3)):
        s.subdivisions_x, s.subdivisions_y = nx, ny
        with pytest.raises(PlantArchitectureError):
            LPM.buildLeafBladeLattice(s)


def test_deform_rigid_is_identity_and_flexible_droops():
    s = LeafShapeValues()
    s.subdivisions_x, s.subdivisions_y, s.aspect_ratio = 6, 4, 0.5
    L = LPM.buildLeafBladeLattice(s)
    assert np.array_equal(LPM.deformLeafLattice(L, 6, 4, 1.0, 1.0, 0.0), L)      # flexibility 0 (tomato)
    assert np.array_equal(LPM.deformLeafLattice(L, 6, 4, 0.0, 1.0, 5.0), L)      # scale 0
    D = LPM.deformLeafLattice(L, 6, 4, 0.2, 0.2, 20.0, 0.5)
    tip = D[2 * 7 + 6]
    assert tip[2] < 0 and np.allclose(D[2 * 7], L[2 * 7] * f32(0.2))           # clamped base, drooping tip
    seg = np.linalg.norm(np.diff(D[14:21], axis=0), axis=1)
    assert np.allclose(seg, f32(0.2) / 6, rtol=1e-5)                              # bends without stretching
    with pytest.raises(PlantArchitectureError):
        LPM.deformLeafLattice(L[:-1], 6, 4, 1.0, 1.0, 1.0)
    with pytest.raises(PlantArchitectureError):
        LPM.deformLeafLattice(L, 6, 4, 1.0, 0.0, 1.0)
    with pytest.raises(PlantArchitectureError):
        LPM.deformLeafLattice(L, 6, 4, 1.0, 1.0, 1.0, 1.0, (0, 0, 0))


def _built(seed=3):
    sc, pa = _tomato(seed)
    pa.buildPlantInstanceFromLibrary((0.0, 0.0, 0.0), 0.0)
    assert not getattr(pa, "leaf_geometry_is_stub", False)
    return sc, pa


def test_tomato_cache_layout_and_private_stream():
    sc, pa = _built(3)
    proto = pa.shoot_types["mainstem"].phytomer_parameters.leaf.prototype
    key = (proto.unique_prototype_identifier, 7)
    cache = pa.unique_leaf_prototype_objIDs[key]
    assert [len(c) for c in cache] == [7] * 5
    oids = [o for c in cache for o in c]
    assert oids == list(range(3, 38)) and all(sc.is_object_hidden(o) for o in oids)
    assert sc.object_primitive_uuids(3)[0] == 84 and all(sc.object_primitive_count(o) == 144 for o in oids)
    assert sc.all_object_ids() == [1, 2] + list(range(38, 45))
    # blade b = p*7 + l uses draw b of longitudinal_curvature from minstd_rand0(identifier)
    g = MinstdRand0(proto.unique_prototype_identifier)
    lo, hi = f32(-0.45), f32(-0.2)
    draws = [f32(lo + f32(unit_uniform(g, "float") * f32(hi - lo))) for _ in range(36)]
    rest = pa.unique_leaf_prototype_rest_geometry[key]
    got = [rest[p][l].shape.longitudinal_curvature for p in range(5) for l in range(7)]
    assert got == draws[:35]
    labels = {sc.get_primitive_material_label(u) for u in sc.object_primitive_uuids(3)}
    assert labels == {"tomato_mainstem_leaf", "tomato_mainstem_petiole"}
    assert sc.polymesh_object_vertex_count(3) == 35 + 57 and sc.polymesh_object_face_count(3) == 144
    labs = [sc.get_primitive_data(u, "object_label") for u in sc.object_primitive_uuids(3)]
    assert labs.count("leaf") == 40 and labs.count("petiolule") == 104


def test_hook_draws_only_from_the_prototype():
    """One val() and one resample() per distributed shape parameter, none from anywhere else."""
    sc, pa = _tomato(4)
    proto = pa.shoot_types["mainstem"].phytomer_parameters.leaf.prototype.copy()
    proto.petiolule_radius_fraction = 0.01
    ctx_before = pa.generator.copy()
    g = MinstdRand0(12345)
    proto.setRandomGenerator(g)
    LPM.GenericLeafPrototype(sc, proto, 0)
    LPM.GenericLeafPrototype(sc, proto, 1)
    assert calls_between(12345, g, nmax=10) == 3 and pa.generator.x == ctx_before.x
    assert proto.last_built_shape_valid and proto.last_built_shape.subdivisions_y == 4


# ---- mutation checks (oracle) -----------------------------------------------------------------------

def _mutate(monkeypatch, old, new):
    """Re-execute leaf_prototype.py with one source edit and patch its functions/classes in."""
    src = inspect.getsource(LPM)
    assert src.count(old) == 1, old
    ns = {"__name__": LPM.__name__, "__file__": LPM.__file__, "__package__": LPM.__package__}
    orig_impl = hooks.GenericLeafPrototype.impl
    exec(compile(src.replace(old, new), LPM.__file__, "exec"), ns)      # its decorator rebinds the hook...
    hooks.GenericLeafPrototype.impl = orig_impl                          # ...so undo that and let monkeypatch do it
    for k, v in ns.items():
        if not k.startswith("__") and callable(v) and getattr(v, "__module__", None) == LPM.__name__:
            monkeypatch.setattr(LPM, k, v)
    monkeypatch.setattr(hooks.GenericLeafPrototype, "impl", ns["GenericLeafPrototype"])


def _fails(name, seeds):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


MUTATIONS = {
    # libm / type choices (each proves the ported one)
    "atan_in_float": ("dtheta = f32(float(dtheta) - math.atan(float(arg)))", "dtheta = f32(dtheta - atanf(arg))",
                      "plantarch.leafproto.tomato", range(2)),
    "wave_in_float": ("z_wave = f32(((2.0 * abs(float(y))) * float(wa)) * float(sinf(wave_phase)))",
                      "z_wave = f32(f32(f32(f32(2) * abs(y)) * wa) * sinf(wave_phase))", "plantarch.leafproto.tomato", range(2)),
    "wave_phase_in_float": ("wave_phase = f32(float(f32(x + f32(wp * jb))) * M_PI / float(wp))",
                            "wave_phase = f32(f32(f32(x + f32(wp * jb)) * PI_F) / wp)", "plantarch.leafproto.tomato", range(2)),
    "fold_pi_in_float": ("arg = f32(float(f32(f32(0.5) * local_fold)) * M_PI)",
                         "arg = f32(f32(f32(0.5) * local_fold) * PI_F)", "plantarch.leafproto.edges", [0, 1]),
    "petiole_roll_in_float": ("z_petiole = f32(min(float(f32(0.1)), a) - 0.01 * float(pr) / abs(float(pr)))",
                              "z_petiole = f32(min(f32(0.1), f32(a)) - f32(f32(f32(0.01) * pr) / abs(pr)))",
                              "plantarch.leafproto.edges", [7]),
    "atanf_correctly_rounded": ('return f32(_M["atanf"](float(f32(x))))', "return f32(math.atan(float(f32(x))))",
                                "plantarch.leafproto.fuzz", [3, 5]),
    "cosf_correctly_rounded": ('return f32(_M["cosf"](float(f32(x))))', "return f32(math.cos(float(f32(x))))",
                               "plantarch.leafproto.edges", [2, 7]),
    "sinf_correctly_rounded": ('return f32(_M["sinf"](float(f32(x))))', "return f32(math.sin(float(f32(x))))",
                               "plantarch.leafproto.fuzz", range(12)),
    # RNG and structure
    "drop_trailing_curvature_resample": ("    pp.longitudinal_curvature.resample()\n", "", "plantarch.leafproto.tomato", range(1)),
    "wave_resample_first": ("    pp.leaf_aspect_ratio.resample()\n    pp.midrib_fold_fraction.resample()\n",
                            "    pp.wave_amplitude.resample()\n    pp.leaf_aspect_ratio.resample()\n    pp.midrib_fold_fraction.resample()\n",
                            "plantarch.leafproto.edges", [7]),
    "keep_transparent_facets": ("            if ctx.primitive_area(u1) > 0:", "            if True:",
                                "plantarch.leafproto.tomato", range(1)),
    "petiolule_unlabelled": ('            ctx.set_primitive_data(UUIDs_petiolule, "object_label", "petiolule")', "            pass",
                             "plantarch.leafproto.tomato_attrs", range(1)),
    "petiolule_radius_halved": ("        petiolule_radius = f32(pp.petiolule_radius_fraction)",
                                "        petiolule_radius = f32(f32(0.5) * pp.petiolule_radius_fraction)",
                                "plantarch.tomato.age0_full", range(1)),
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    load_plugins()
    old, new, case_name, seeds = MUTATIONS[mutation]
    _mutate(monkeypatch, old, new)
    assert _fails(case_name, seeds)


def test_unmutated_source_passes(monkeypatch):
    """The source re-execution itself does not perturb anything."""
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    load_plugins()
    _mutate(monkeypatch, "M_PI = math.pi\n", "M_PI = math.pi\n")
    assert not _fails("plantarch.leafproto.tomato", range(1))


def test_hook_restored_after_mutation_tests():
    assert hooks.GenericLeafPrototype.impl is LPM.GenericLeafPrototype
