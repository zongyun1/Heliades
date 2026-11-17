# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the PlantArchitecture data layer (W2-C).

Seeds: the case seed ``s`` maps to Context seed ``s + 1`` (seed 0 equals seed 1 in minstd_rand0, SPEC §3.4),
so ``fuzz=8`` covers Context seeds 1..8.

* ``plantarch.load.tomato``: engine calls consumed by ``PlantArchitecture(ctx)``, by
  ``loadPlantModelFromLibrary("tomato")``, and by a second load (4 more identifiers), via the external
  draw-count probe (``parity/tools/draw_count.py``).
* ``plantarch.load.tomato_params``: the whole ``mainstem`` ShootParameters tree, as PyHelios serialises it,
  compared as float32 values (the bytes of a canonical JSON).
* ``plantarch.load.models``: the registry (``getAvailablePlantModels``).
* ``plantarch.load.identifier``: the 4th identifier, checked through what it seeds: for each of the 7
  leaflets of phytomer 0, the prototype index (XML ``<leaf_prototype>``) and the blade's
  ``longitudinal_curvature`` (``getLeafShape``, full float) drawn from ``minstd_rand0(identifier)``.
* ``plantarch.load.sampled_age0``: RandomParameter samples on the Context stream after load: the base
  rotation (``randu`` right to left) and the first ``PhytomerParameters::resample`` values exposed by the
  XML (6 significant digits: the Heliades value is printed the way ``std::ostream << float`` prints it).
"""
from __future__ import annotations

import numpy as np

from ...cases import case
from ...tools.draw_count import engine_calls

f32 = np.float32


def _seed(s):
    return s + 1


def _probe(B, seed):
    return engine_calls(seed, [B.randu() for _ in range(3)])


@case("plantarch.load.tomato", tier="exact", fuzz=8)
def load_tomato(B, s):
    seed = _seed(s)
    out = []
    for stages in (0, 1, 2, 3):
        B.seed(seed)
        if stages >= 1:
            B.plantarch_new()
        if stages >= 2:
            B.plantarch_load("tomato")
        if stages >= 3:
            B.plantarch_load("tomato")
        n = _probe(B, seed)
        out.append(-1 if n is None else n)
    return out


@case("plantarch.load.tomato_params", tier="exact", fuzz=2)
def load_tomato_params(B, s):
    from ....plantarchitecture.structure import canonical
    B.seed(_seed(s))
    B.plantarch_new()
    B.plantarch_load("tomato")
    txt = canonical(B.plantarch_shoot_params("mainstem"))
    return list(txt.encode())


@case("plantarch.load.models", tier="exact")
def load_models(B, s):
    B.plantarch_new()
    return list(",".join(B.plantarch_available_models()).encode())


@case("plantarch.load.identifier", tier="exact", fuzz=8)
def load_identifier(B, s):
    B.seed(_seed(s))
    B.plantarch_new()
    B.plantarch_load("tomato")
    o = B.plantarch_age0_observables()
    out = []
    for p, c in zip(o["leaf_prototype"], o["leaf_longitudinal_curvature"]):
        out += [float(p), c]
    return out


def _cout(v) -> float:
    """``std::ostream << float`` with the default precision 6, read back."""
    return float("%g" % float(f32(v)))


@case("plantarch.load.sampled_age0", tier="exact", fuzz=8)
def load_sampled_age0(B, s):
    B.seed(_seed(s))
    B.plantarch_new()
    B.plantarch_load("tomato")
    o = B.plantarch_age0_observables()
    vals = list(o["base_rotation_deg"]) + [o["internode_phyllotactic_angle_deg"], o["petiole_pitch_deg"],
                                           o["petiole_curvature"]]
    return [_cout(v) for v in vals]


# Distributed parameters the age-0 Phytomer constructor reads but does not resample a second time
# before the leaflet loop (PA:2694-3592), so the stand-in replay stays the same draw plan. Every kind of
# RandomParameter draw is covered: float normal/weibull/uniform, int uniform with a == b (still one draw:
# no Context-style shortcut) and discretevalues.
REDEFINE_SPEC = [
    ("phytomer_parameters.internode.pitch", "normal", (10.0, 3.0)),
    ("phytomer_parameters.internode.phyllotactic_angle", "weibull", (3.0, 140.0)),
    ("phytomer_parameters.internode.max_floral_buds_per_petiole", "uniform", (1, 1)),
    ("phytomer_parameters.petiole.pitch", "normal", (50.0, 5.0)),
    ("phytomer_parameters.petiole.length", "uniform", (0.19, 0.21)),
    ("phytomer_parameters.petiole.curvature", "normal", (-100.0, 20.0)),
    ("phytomer_parameters.leaf.leaves_per_petiole", "discretevalues", (7,)),
    ("phytomer_parameters.peduncle.length", "weibull", (2.0, 0.18)),
    ("phytomer_parameters.inflorescence.flowers_per_peduncle", "discretevalues", (5, 6, 7)),
    ("phytomer_parameters.inflorescence.pitch", "normal", (80.0, 5.0)),
]


@case("plantarch.load.redefined_distributions", tier="exact", fuzz=8)
def load_redefined(B, s):
    """getCurrentShootParameters/defineShootType round trip with every distribution kind, then the
    first PhytomerParameters::resample sequence as seen in the age-0 XML. The Context is reseeded after
    defineShootType so the case does not depend on the PyHelios JSON wrapper's internals (measured with
    the probe: the 1.3.98 wrapper makes no draws there, because it seeds from the existing entry)."""
    B.seed(_seed(s))
    B.plantarch_new()
    B.plantarch_load("tomato")
    B.plantarch_redefine("mainstem", REDEFINE_SPEC)
    B.seed(1000 + s)
    o = B.plantarch_age0_observables()
    vals = list(o["base_rotation_deg"]) + [o["internode_pitch_deg"], o["internode_phyllotactic_angle_deg"],
                                           o["petiole_pitch_deg"], o["petiole_curvature"]]
    return [_cout(v) for v in vals] + [float(p) for p in o["leaf_prototype"]]
