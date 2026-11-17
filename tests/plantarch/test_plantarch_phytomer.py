# SPDX-License-Identifier: LGPL-2.1-or-later
"""W3-B: Phytomer constructor, tube helpers, the age-0 tomato build, and mutation checks against the oracle."""
from __future__ import annotations

import types

import numpy as np
import pytest

from heliades.core.scene import Scene
from heliades.parity.tools.draw_count import calls_between
from heliades.plantarchitecture import PlantArchitecture, hooks
from heliades.plantarchitecture import tube_utils as TU
from heliades.plantarchitecture.params import BUD_DEAD

f32 = np.float32


def _tomato(seed=2):
    s = Scene()
    s.seed_random_generator(seed)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary("tomato")
    return s, pa


# ---- helpers ------------------------------------------------------------------------------------

def test_interpolate_tube_endpoints_and_fplus():
    P = [np.array([0, 0, 0], f32), np.array([1, 0, 0], f32), np.array([1, 2, 0], f32)]
    assert np.array_equal(TU.interpolate_tube(P, 0), P[0])
    # the last segment ends at fplus = float(1.f + 1e-3), so frac = 1 lands just short of the end node
    end = TU.interpolate_tube(P, 1)
    assert end[0] == 1 and f32(1.99) < end[1] < 2
    assert TU.interpolate_tube_float([0.0, 1.0, 3.0], 0.5) == f32(1)


def test_axis_vector_uses_backward_difference_at_tip():
    P = [np.array([0, 0, 0], f32), np.array([0, 0, 1], f32)]
    assert np.array_equal(TU.get_axis_vector(1, P), np.array([0, 0, 1], f32))


def test_make_petiole_tube_drops_coincident_and_tiny():
    s = Scene()
    c = (0.1, 0.2, 0.3)
    v = [(0, 0, 0), (0, 0, 0.5e-6), (0, 0, 0.01)]
    oid = TU.make_petiole_tube(5, v, [0.001, 0.001, 0.0], [c] * 3, s)
    assert s.tube_object_node_count(oid) == 2
    assert s.tube_object_node_radii(oid)[-1] == TU.MIN_TUBE_RADIUS_FOR_GEOMETRY
    assert TU.make_petiole_tube(5, [(0, 0, 0), (0, 0, 1e-5)], [0.001] * 2, [c] * 2, s) == TU.NO_PETIOLE_OBJID
    assert TU.make_petiole_tube(5, [(0, 0, 0), (0, 0, 1)], [1e-6] * 2, [c] * 2, s) == TU.NO_PETIOLE_OBJID


def test_compound_rotation_and_offsets():
    assert TU.compound_leaf_rotation(7, 3, 0.15) == 0
    assert TU.compound_leaf_rotation(7, 0, 0.15) == f32(-0.5 * float(TU.PI_F))
    assert TU.compound_leaf_rotation(7, 6, 0.15) == f32(0.5 * float(TU.PI_F))
    assert TU.compound_leaf_rotation(1, 0, 0.15) == 0
    assert TU.clamp_offset(7, 0.15) == f32(0.15)
    assert TU.clamp_offset(7, 0.9) == f32(f32(1) / f32(2.5))
    assert TU.leaflet_size_max(-3, 0.7, 0, 0.1) == f32(TU.powf(0.7, 3) * f32(0.1))
    # interruptedly pinnate: pairs 1, 3 major, 2 intercalary
    assert TU.leaflet_size_max(2, 0.7, 0.5, 1.0) == f32(0.5 * TU.powf(0.7, 1))
    assert TU.leaflet_size_max(3, 0.7, 0.5, 1.0) == TU.powf(0.7, 2)


# ---- the age-0 tomato ---------------------------------------------------------------------------

def test_age0_build_structure_and_draws():
    s, pa = _tomato(2)
    n0 = calls_between(2, s.rng.generator)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 0)
    assert calls_between(2, s.rng.generator) - n0 in (26, 27)
    shoot = pa.plant_instances[pid].shoot_tree[0]
    ph = shoot.phytomers[0]
    assert len(shoot.phytomers) == 1 and shoot.internode_tube_objID == 1 and ph.petiole_objIDs == [2]
    assert pa.leaf_geometry_is_stub == (hooks.GenericLeafPrototype.impl is None)
    hidden = sorted(s.hidden_objects)
    assert hidden == list(range(3, 38))                         # 35 cached blades (5 prototypes x 7 leaflets)
    assert ph.leaf_objIDs == [list(range(38, 45))]
    assert all(fb.state == BUD_DEAD for p in ph.floral_buds for fb in p)       # node < 8, rank 0
    assert ph.internode_length_max == f32(f32(0.04) * f32(0.7))               # scaleInternodeMaxLength(0.7)
    assert ph.current_internode_scale_factor == f32(f32(0.01) / f32(0.7))
    assert s.get_primitive_data(0, "object_label") == "shoot"
    assert s.get_primitive_data(s.object_primitive_uuids(2)[0], "object_label") == "petiole"


def test_constructor_draws_23_calls():
    """Phytomer 0 alone: resample (6) + petiole pitch/curvature (2) + 7 x (randu prototype + leaf.pitch) + prototype_scale."""
    from heliades.plantarchitecture import phytomer as PH
    s, pa = _tomato(3)
    counts = []
    orig = PH.Phytomer.__init__

    def init(self, *a, **k):
        n = calls_between(3, s.rng.generator)
        orig(self, *a, **k)
        counts.append(calls_between(3, s.rng.generator) - n)
    PH.Phytomer.__init__ = init
    try:
        pa.buildPlantInstanceFromLibrary((0, 0, 0), 0)
    finally:
        PH.Phytomer.__init__ = orig
    assert counts == [23]


def test_age_past_zero_grows():
    """W3-C replaced the age > 0 NotImplementedError with advanceTime (tests/heliades/test_plantarch_growth.py)."""
    _, pa = _tomato(1)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 2.0)
    assert len(pa.plant_instances[pid].shoot_tree[0].phytomers) == 2


def test_creation_function_hook_is_bound():
    from heliades.plantarchitecture.library.tomato import TomatoPhytomerCreationFunction
    assert hooks.TomatoPhytomerCreationFunction.impl is TomatoPhytomerCreationFunction


def test_prototype_function_override_frames():
    """The stand-in blade puts the polymesh frame at the requested first vertex (used by age0_leaf_frames)."""
    from heliades.plantarchitecture.phytomer import stub_generic_leaf_prototype
    s, pa = _tomato(4)
    pa.leaf_prototype_function_override = (
        lambda ctx, proto, idx: stub_generic_leaf_prototype(ctx, proto, idx, first_vertex=(0.085, 0.25, -0.125)))
    pa.buildPlantInstanceFromLibrary((0, 0, 0), 0)
    T = s.object_transform(3)
    assert (T[3], T[7], T[11]) == (f32(0.085), f32(0.25), f32(-0.125))


# ---- mutation checks (oracle) -------------------------------------------------------------------

def _fails(name, seeds=range(8)):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


def _patch_source(monkeypatch, module, cls_name, attr, old, new):
    """Recompile ``module`` with one source edit and install ``cls.attr`` from the edited copy."""
    src = open(module.__file__).read()
    assert src.count(old) == 1, old
    mod = types.ModuleType(module.__name__)
    mod.__dict__.update(__name__=module.__name__, __package__=module.__package__, __file__=module.__file__)
    exec(compile(src.replace(old, new), module.__file__, "exec"), mod.__dict__)
    monkeypatch.setattr(getattr(module, cls_name), attr, getattr(getattr(mod, cls_name), attr))


@pytest.mark.parametrize("mutation", ["drop_curvature_resample", "drop_breakdormancy_draw", "petiole_radius_order",
                                      "interpolate_fplus", "scale_about_origin", "pitch_before_roll"])
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    load_plugins()
    from heliades.plantarchitecture import phytomer as PH
    from heliades.plantarchitecture import shoot as SM
    if mutation == "drop_curvature_resample":
        # the unused petiole.curvature.resample() of PA:3340: every later draw shifts by one
        _patch_source(monkeypatch, PH, "Phytomer", "__init__", "                pp.petiole.curvature.resample()\n", "")
        assert _fails("plantarch.tomato.age0_draws")
    elif mutation == "drop_breakdormancy_draw":
        # breakPlantDormancy's bud-break randu (the 26/27 variance)
        monkeypatch.setattr(SM.Shoot, "breakDormancy", lambda self: None)
        assert _fails("plantarch.tomato.age0_draws")
    elif mutation == "petiole_radius_order":
        # frac * (radius * taper) instead of (frac * radius) * taper (PA:3417)
        _patch_source(monkeypatch, PH, "Phytomer", "__init__",
                      "f32(f32(radius_fraction * pp.petiole.radius.val()) * taper_term)",
                      "f32(radius_fraction * f32(pp.petiole.radius.val() * taper_term))")
        assert _fails("plantarch.tomato.age0_tubes")
    elif mutation == "interpolate_fplus":
        # fplus = 1 instead of float(1.f + 1e-3) on the last segment (PA:367): leaflet bases move
        orig = TU.interpolate_tube

        def bad(P, frac):
            n = len(P)
            dl = f32(0)
            for i in range(n - 1):
                dl = f32(dl + TU.V.magnitude(np.asarray(P[i + 1], f32) - P[i]))
            f = f32(0)
            for i in range(n - 1):
                fplus = f32(f + f32(TU.V.magnitude(np.asarray(P[i + 1], f32) - P[i]) / dl))
                fplus = f32(1) if fplus >= 1 else fplus
                if f32(frac) >= f and f32(frac) <= fplus:
                    t = f32(f32(f32(frac) - f) / f32(fplus - f))
                    return np.asarray(P[i], f32) + t * (np.asarray(P[i + 1], f32) - P[i])
                f = fplus
            return orig(P, frac)
        monkeypatch.setattr(PH, "interpolate_tube", bad)
        assert _fails("plantarch.tomato.age0_leaf_frames")
        assert _fails("plantarch.tomato.age0_params")
    elif mutation == "scale_about_origin":
        # scaleLeafPrototypeScale must scale about the leaf base (scaleObjectAboutPoint), not the origin
        def about_origin(self, *args):
            if len(args) == 1:
                for p in range(len(self.leaf_objIDs)):
                    about_origin(self, p, args[0])
                return
            p, k = args
            for leaf in range(len(self.leaf_objIDs[p])):
                self.leaf_size_max[p][leaf] = f32(self.leaf_size_max[p][leaf] * f32(k))
                self.context_ptr.scale_object(self.leaf_objIDs[p][leaf], (k, k, k))
        monkeypatch.setattr(PH.Phytomer, "scaleLeafPrototypeScale", about_origin)
        assert _fails("plantarch.tomato.age0_leaf_frames")
    elif mutation == "pitch_before_roll":
        # orientLeaf applies roll about x before pitch about y (PA:5148-5157)
        orig = PH.Phytomer.orientLeaf

        def swapped(self, objID, *a, **k):
            ctx = self.context_ptr
            calls = []
            real = ctx.rotate_object
            ctx.rotate_object = lambda *x: calls.append(x)
            try:
                orig(self, objID, *a, **k)
            finally:
                ctx.rotate_object = real
            calls[0], calls[1] = calls[1], calls[0]
            for c in calls:
                real(*c)
        monkeypatch.setattr(PH.Phytomer, "orientLeaf", swapped)
        assert _fails("plantarch.tomato.age0_leaf_frames")
