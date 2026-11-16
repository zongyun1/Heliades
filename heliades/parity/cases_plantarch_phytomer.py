# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the Phytomer constructor and the age-0 tomato (W3-B).

Seeds: case seed ``s`` -> Context seed ``s + 1`` (seeds 0 and 1 are the same minstd_rand0 stream), so
``fuzz=8`` covers Context seeds 1..8. Every case builds one tomato per PlantArchitecture at age 0 with
``buildPlantInstanceFromLibrary((0, 0, 0), 0)``.

* ``plantarch.tomato.age0_draws`` (exact): engine calls consumed by load and by the age-0 build (26 or 27,
  SPEC §3.3), via the external probe.
* ``plantarch.tomato.age0_params`` (exact): the per-phytomer values ``writePlantStructureXML`` prints
  (6 significant digits), then the same state at full float32 precision through the PyHelios getters
  (internode nodes, petiole nodes and radii, the 7 leaflet bases, petiole/internode axis vectors).
* ``plantarch.tomato.age0_tubes`` (ulp, max 0): every primitive labelled ``shoot`` or ``petiole``: UUID,
  label, vertices and color. Leaves are excluded until W3-A merges.
* ``plantarch.tomato.age0_tube_materials`` (exact): the material label of those primitives
  (``nameOrganMaterials``).
* ``plantarch.tomato.age0_leaf_frames`` (ulp, max 0): object ID and 4x4 transform of the 7 leaflet copies.
  The Heliades blades are stand-ins placed in the oracle prototypes' frames (see ``ops_plantarch_phytomer``),
  so this checks the placement chain alone: copyObject, scaleObject, orientLeaf, translateObject and the
  creation function's scaleLeafPrototypeScale.
"""
from __future__ import annotations

import numpy as np

from .cases import case
from .tools.draw_count import engine_calls

f32 = np.float32


def _seed(s):
    return s + 1


def _probe(B, seed):
    n = engine_calls(seed, [B.randu() for _ in range(3)])
    return -1 if n is None else n


def _build(B, s):
    B.seed(_seed(s))
    B.plantarch_new()
    B.plantarch_load("tomato")
    B.plantarch_build((0.0, 0.0, 0.0), 0.0)


@case("plantarch.tomato.age0_draws", tier="exact", fuzz=8)
def tomato_age0_draws(B, s):
    seed = _seed(s)
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load("tomato")
    n_load = _probe(B, seed)
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load("tomato")
    B.plantarch_build((0.0, 0.0, 0.0), 0.0)
    n_build = _probe(B, seed)
    return [n_load, n_build, n_build - n_load]


@case("plantarch.tomato.age0_params", tier="exact", fuzz=8)
def tomato_age0_params(B, s):
    _build(B, s)
    return list(B.pa_age0_xml_values()) + list(B.pa_age0_full_values())


@case("plantarch.tomato.age0_tubes", tier="ulp", max_ulp=0, fuzz=8)
def tomato_age0_tubes(B, s):
    _build(B, s)
    return B.pa_tube_prims()


@case("plantarch.tomato.age0_tube_materials", tier="exact", fuzz=2)
def tomato_age0_tube_materials(B, s):
    _build(B, s)
    return [float(b) for b in "\n".join(B.pa_tube_materials()).encode()]


@case("plantarch.tomato.age0_leaf_frames", tier="ulp", max_ulp=0, fuzz=8)
def tomato_age0_leaf_frames(B, s):
    B.seed(_seed(s))
    B.plantarch_new()
    B.plantarch_load("tomato")
    B.pa_build_with_oracle_prototype_frames(_seed(s))
    return B.pa_leaf_frames()
