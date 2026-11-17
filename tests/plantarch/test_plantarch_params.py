"""W2-C: PlantArchitecture data layer -- RandomParameter semantics, identifier draws, resample order,
the tomato library entry, the draw-count probe, and mutation checks against the oracle."""
import numpy as np
import pytest

from heliades.core.random import MinstdRand0, Normal, UniformInt, Weibull, uniform_integer, unit_uniform
from heliades.core.scene import Scene
from heliades.parity.tools.draw_count import calls_between, engine_calls
from heliades.plantarchitecture import (BUD_DEAD, AxisRotation, BudState, LeafPrototype, PhytomerParameters,
                                        PlantArchitecture, PlantArchitectureError, RandomParameterFloat,
                                        RandomParameterInt, ShootParameters, VegetativeBud, make_AxisRotation)
from heliades.plantarchitecture import hooks
from heliades.plantarchitecture.params import LeafParameters, deg2rad, rad2deg

f32 = np.float32


def _calls(seed, g):
    return calls_between(seed, g, nmax=100_000)


# ---- RandomParameter --------------------------------------------------------------------------

def test_constant_never_draws():
    g = MinstdRand0(7)
    p = RandomParameterFloat()
    p.initialize(3.5, g)
    assert p.val() == f32(3.5) and p.resample() == f32(3.5) and _calls(7, g) == 0
    q = RandomParameterInt()
    q.initialize(4, g)
    assert q.val() == 4 and q.resample() == 4 and _calls(7, g) == 0


def test_uniform_lazy_cache_and_value():
    g = MinstdRand0(11)
    p = RandomParameterFloat()
    p.initialize(0, g)
    p.uniformDistribution(97.5, 177.5)
    h = MinstdRand0(11)
    v = p.val()
    assert v == f32(f32(97.5) + f32(unit_uniform(h, "float") * f32(80))) and _calls(11, g) == 1
    assert p.val() == v and _calls(11, g) == 1          # cached
    v2 = p.resample()                                    # always draws
    assert _calls(11, g) == 2 and v2 == f32(f32(97.5) + f32(unit_uniform(h, "float") * f32(80)))


def test_float_distributions_use_fresh_distribution_objects():
    g, h = MinstdRand0(3), MinstdRand0(3)
    p = RandomParameterFloat(generator=g)
    p.normalDistribution(10, 3)
    a, b = p.resample(), p.resample()
    # no polar spare carried between resamples: each call is a fresh normal_distribution
    assert a == Normal(10, 3, "float")(h) and b == Normal(10, 3, "float")(h)
    p.weibullDistribution(2, 0.5)
    assert p.resample() == Weibull(f32(2), f32(0.5), "float")(h)
    assert g.x == h.x


def test_int_uniform_equal_bounds_still_draws_and_discrete():
    g = MinstdRand0(9)
    q = RandomParameterInt()
    q.initialize(g)
    q.uniformDistribution(1, 1)
    assert q.val() == 1 and _calls(9, g) == 1           # no Context::randu(a,a) shortcut
    q.discreteValues([5, 6, 7])
    h = MinstdRand0(9)
    h()
    assert q.val() == [5, 6, 7][UniformInt(0, 2)(h)] and g.x == h.x
    with pytest.raises(PlantArchitectureError):
        q.uniformDistribution(3, 2)


def test_assignment_and_copy_semantics():
    g = MinstdRand0(5)
    p = RandomParameterFloat(generator=g)
    p.uniformDistribution(0, 1)
    v = p.val()
    c = p.copy()
    assert c.sampled and c.constval == v and c.generator is g      # cache and pointer travel with copies
    p.set(2)
    assert p.distribution == "constant" and not p.sampled and p.generator is g and p.val() == f32(2)
    c.setRandomGenerator(MinstdRand0(1))
    assert not c.sampled and c.distribution == "uniform"
    r = RandomParameterFloat(1.0)
    r.uniformDistribution(0, 1)
    with pytest.raises(PlantArchitectureError):
        r.val()                                           # null generator


def test_struct_attribute_assignment_is_cpp_assignment():
    pp = PhytomerParameters()
    p = pp.internode.pitch
    pp.internode.pitch = 10                                # operator=(float), same object
    assert pp.internode.pitch is p and p.val() == f32(10)
    other = RandomParameterFloat(3.0)
    pp.internode.pitch = other                             # copy assignment, still distinct objects
    assert pp.internode.pitch is p and p.constval == f32(3) and p is not other
    pp.internode.length_segments = -1
    assert pp.internode.length_segments == 2**32 - 1       # uint


# ---- LeafPrototype ----------------------------------------------------------------------------

def test_leaf_prototype_identifier_draws():
    g = MinstdRand0(1)
    lp = LeafPrototype(g)
    h = MinstdRand0(1)
    assert lp.unique_prototype_identifier == uniform_integer(h, 2**32 - 1) and g.x == h.x
    assert lp.prototype_function is hooks.GenericLeafPrototype
    n = _calls(1, g)
    c = lp.copy()                                          # copy ctor: no draw
    d = LeafPrototype()
    d.duplicate(lp)                                        # duplicate: no draw
    assert _calls(1, g) == n and c.unique_prototype_identifier == lp.unique_prototype_identifier
    assert d.generator is g
    e = LeafPrototype()                                    # default ctor: nothing initialised
    assert e.prototype_function is None and e.generator is None
    e._assign_from(lp)                                     # operator=: copies, then redraws
    assert e.unique_prototype_identifier == uniform_integer(h, 2**32 - 1) and g.x == h.x
    lp._assign_from(lp)                                    # self-assignment: no-op
    assert g.x == h.x
    z = LeafPrototype(None)                                # explicit ctor with null generator
    assert z.prototype_function is hooks.GenericLeafPrototype and z.unique_prototype_identifier == 0
    z2 = LeafPrototype()
    z2._assign_from(z)                                     # null generator: operator= does not draw
    assert g.x == h.x


def test_phytomer_parameters_draws_two_identifiers():
    g = MinstdRand0(2)
    pp = PhytomerParameters(g)
    h = MinstdRand0(2)
    uniform_integer(h, 2**32 - 1)
    assert pp.leaf.prototype.unique_prototype_identifier == uniform_integer(h, 2**32 - 1) and g.x == h.x
    n = _calls(2, g)
    q = pp.copy()
    r = PhytomerParameters()
    r._assign_from(pp)
    lp = LeafParameters()
    lp._assign_from(pp.leaf)                               # LeafParameters::operator= uses duplicate()
    assert _calls(2, g) == n
    assert r.leaf.prototype.unique_prototype_identifier == pp.leaf.prototype.unique_prototype_identifier
    assert q.leaf.prototype is not pp.leaf.prototype
    assert pp.leaf.leaflet_scale.generator is None         # set with operator=(float) in the C++ ctor
    assert tuple(pp.peduncle.color) == (0, 0, 0)            # never set by the C++ ctor


def test_resample_order():
    g = MinstdRand0(4)
    pp = PhytomerParameters(g)
    floats = []
    for path in PhytomerParameters.RESAMPLE_ORDER:
        p = pp.get(path)
        p.generator = g        # (RandomParameter_int has no setRandomGenerator; some members start with null)
        if isinstance(p, RandomParameterFloat):
            p.uniformDistribution(len(floats), len(floats) + 1)
        else:
            p.uniformDistribution(len(floats) * 10, len(floats) * 10 + 5)
        floats.append(p)
    pp.leaf.prototype.leaf_aspect_ratio.set(1.0)            # a constant in the middle draws nothing
    start = g.x
    pp.resample()
    h = MinstdRand0(1)
    h.x = start
    for p in floats:
        if p is pp.leaf.prototype.leaf_aspect_ratio:
            assert p.constval == f32(1)
            continue
        if isinstance(p, RandomParameterFloat):
            lo, hi = p.distribution_parameters
            assert p.constval == f32(lo + f32(unit_uniform(h, "float") * f32(hi - lo)))
        else:
            assert p.constval == UniformInt(*p.distribution_parameters)(h)
    assert g.x == h.x


def test_shoot_parameters_no_draws_and_child_types():
    g = MinstdRand0(6)
    sp = ShootParameters(g)
    c = sp.copy()
    sp._assign_from(c)
    assert _calls(6, g) == 0
    assert sp.max_nodes.val() == 10 and sp.leaf_expansion_rate_max.val() == f32(-1)
    sp.defineChildShootTypes(["a", "b"], [0.3, 0.7])
    with pytest.raises(PlantArchitectureError):
        sp.defineChildShootTypes(["a"], [0.5])
    with pytest.raises(PlantArchitectureError):
        sp.defineChildShootTypes([], [])


def test_axis_rotation_and_buds():
    a = make_AxisRotation(1, 2, 3)
    b = AxisRotation(0.5, 0.25, 0.125)
    assert (a + b).pitch == f32(1.5)
    assert (a - b).pitch == f32(-0.5)                       # Helios: x - a == a - x (H:479)
    assert tuple(a.peduncle_axis) == (0, 0, 1)
    assert BudState.BUD_DEAD == BUD_DEAD == 5
    assert VegetativeBud().shoot_ID == 2**32 - 1
    assert deg2rad(180) == f32(np.pi) and rad2deg(f32(np.pi)) == f32(180)


# ---- PlantArchitecture + tomato ---------------------------------------------------------------

def test_plantarch_holds_scene_generator_and_load_draws():
    for seed in (1, 2, 5):
        s1 = Scene()
        s1.seed_random_generator(seed)
        pa1 = PlantArchitecture(s1)
        assert pa1.generator is s1.rng.generator
        pa1.loadPlantModelFromLibrary("tomato")
        s2 = Scene()
        pa2 = PlantArchitecture(s2)
        s2.seed_random_generator(seed)                     # seeding after the ctor is equivalent
        pa2.loadPlantModelFromLibrary("tomato")
        i1 = pa1.shoot_types["mainstem"].phytomer_parameters.leaf.prototype.unique_prototype_identifier
        i2 = pa2.shoot_types["mainstem"].phytomer_parameters.leaf.prototype.unique_prototype_identifier
        assert i1 == i2
        h = MinstdRand0(seed)
        ids = [uniform_integer(h, 2**32 - 1) for _ in range(4)]
        assert ids[3] == i1 and s1.rng.generator.x == h.x  # exactly 4 identifiers
    assert _calls(5, s1.rng.generator) == 10                # SPEC §3.3: seed 5 -> 10 engine calls


def test_seed_zero_equals_seed_one():
    a, b = Scene(), Scene()
    a.seed_random_generator(0)
    b.seed_random_generator(1)
    pa, pb = PlantArchitecture(a), PlantArchitecture(b)
    pa.loadPlantModelFromLibrary("tomato")
    pb.loadPlantModelFromLibrary("tomato")
    assert (pa.shoot_types["mainstem"].phytomer_parameters.leaf.prototype.unique_prototype_identifier
            == pb.shoot_types["mainstem"].phytomer_parameters.leaf.prototype.unique_prototype_identifier)


def test_registry_and_errors():
    pa = PlantArchitecture(Scene())
    models = pa.getAvailablePlantModels()
    assert models == sorted(models) and "tomato" in models and len(models) == 32
    assert pa.plant_type_map["almond"] == "tree" and pa.getPlantModelLeafInclinationDistribution("cowpea")[0] == f32(1.398)
    with pytest.raises(PlantArchitectureError):
        pa.loadPlantModelFromLibrary("nope")
    with pytest.raises(NotImplementedError):
        pa.loadPlantModelFromLibrary("cowpea")
    with pytest.raises(PlantArchitectureError):
        pa.registerPlantModel("x", None, None, "herbaceous", (1.0, 0.0))
    with pytest.raises(PlantArchitectureError):
        PlantArchitecture(Scene()).buildPlantInstanceFromLibrary((0, 0, 0), 0)


def test_tomato_values_and_shoot_type_semantics():
    s = Scene()
    s.seed_random_generator(3)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary("tomato")
    sp = pa.shoot_types["mainstem"]
    pp = sp.phytomer_parameters
    assert pp.internode.phyllotactic_angle.distribution_parameters == [f32(97.5), f32(177.5)]
    assert pp.leaf.prototype_scale.distribution_parameters == [f32(f32(0.12) * f32(0.9031)), f32(f32(0.18) * f32(0.9031))]
    assert pp.leaf.prototype.leaf_aspect_ratio.val() == f32(f32(0.5) / f32(0.9031))
    assert pp.phytomer_creation_function is hooks.TomatoPhytomerCreationFunction
    assert pp.inflorescence.fruit_prototype_function is hooks.TomatoFruitPrototype
    assert sp.child_shoot_type_labels == ["mainstem"] and sp.max_nodes.val() == 16
    # getCurrentShootParameters returns a copy; defineShootType on an existing label assigns in place
    x = pa.getCurrentShootParameters("mainstem")
    x.max_nodes = 3
    assert sp.max_nodes.val() == 16
    n = _calls(3, s.rng.generator)
    pa.defineShootType("mainstem", x)
    assert pa.shoot_types["mainstem"] is sp and sp.max_nodes.val() == 3 and _calls(3, s.rng.generator) == n
    pa.loadPlantModelFromLibrary("tomato")                   # reload: cleared, 4 more identifiers
    assert pa.shoot_types["mainstem"] is not sp and _calls(3, s.rng.generator) > n


# ---- draw-count probe -------------------------------------------------------------------------

def test_engine_calls_probe():
    s = Scene()
    for seed, n in ((5, 0), (5, 37), (2, 70_000), (8, 131_073)):
        s.seed_random_generator(seed)
        for _ in range(n):
            s.rng.generator()
        assert engine_calls(seed, [s.randu() for _ in range(3)]) == n
    assert engine_calls(1, [f32(0.5)] * 3, nmax=1 << 17) is None


# ---- mutation checks (oracle) -----------------------------------------------------------------

def _fails(name, seeds):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


@pytest.mark.parametrize("mutation", ["drop_operator_eq_draw", "extra_identifier_draw", "drop_wasted_curvature_resample",
                                      "left_to_right_rotation"])
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    load_plugins()
    from heliades.plantarchitecture import params as P
    from heliades.plantarchitecture.library import tomato as T
    if mutation == "drop_operator_eq_draw":
        # PhytomerParameters(gen): skip the operator= identifier draw of PA:983 (3 identifiers on load)
        orig = P.LeafPrototype._assign_from

        def no_draw(self, a):
            g = a.generator
            a._raw_set("generator", None)
            try:
                return orig(self, a)
            finally:
                a._raw_set("generator", g)
                self._raw_set("generator", g)
        orig_init = P.PhytomerParameters.__init__

        def init(self, generator=None):
            monkeypatch.setattr(P.LeafPrototype, "_assign_from", no_draw)
            try:
                orig_init(self, generator)
            finally:
                monkeypatch.setattr(P.LeafPrototype, "_assign_from", orig)
        monkeypatch.setattr(P.PhytomerParameters, "__init__", init)
        assert _fails("plantarch.load.tomato", range(8))
    elif mutation == "extra_identifier_draw":
        # make LeafParameters::operator= draw (operator= instead of duplicate()): ShootParameters assignment draws
        monkeypatch.setattr(P.LeafParameters, "_assign_from",
                            lambda self, a: (self._memberwise_assign(a, self._FIELDS[:-1]),
                                             self.prototype._assign_from(a.prototype), self)[-1])
        assert _fails("plantarch.load.tomato", range(8))
    elif mutation == "drop_wasted_curvature_resample":
        # the wasted (f) draw of PhytomerParameters::resample: leaflet prototype choices shift
        order = tuple(p for p in P.PhytomerParameters.RESAMPLE_ORDER if p != "leaf.prototype.longitudinal_curvature")
        monkeypatch.setattr(P.PhytomerParameters, "RESAMPLE_ORDER", order)
        assert _fails("plantarch.load.identifier", range(4))
        assert not _fails("plantarch.load.tomato", range(2))   # load itself is unaffected
    elif mutation == "left_to_right_rotation":
        def l2r(pa):
            hi = f32(2.0 * np.pi)
            yaw = pa.context_ptr.randu(f32(0), hi)
            roll = pa.context_ptr.randu(f32(0), hi)
            return make_AxisRotation(0, yaw, roll)
        monkeypatch.setattr(T, "base_rotation_draws", l2r)
        assert _fails("plantarch.load.sampled_age0", range(2))
