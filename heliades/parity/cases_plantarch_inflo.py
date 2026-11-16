# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the inflorescence geometry (W3-D).

Seeds: the placement cases sweep case seed ``s = 0..23`` -> Context seed ``s % 8 + 1`` (1..8) x age
``(100, 110, 120)[s // 8]``; see ``ops_plantarch_inflo.grid``. Every floral bud carrying flowers or fruit in that
oracle plant is forced through DEAD -> FLOWER_CLOSED -> FLOWER_OPEN -> FRUITING (Context reseeded with
``1000*seed + age + k`` before step k), then fruit growth tc = 2..32 and ``updatePeduncleRadii``.

* ``plantarch.inflo.prototypes`` (ulp, max 0): the three hidden inflorescence prototypes as Helios builds them on
  first use (an oracle tomato grown to 100 / 110 / 120 days vs a Heliades ``ensureInflorescencePrototypesInitialized``):
  per object its primitive count and transform, then per primitive vertices, UVs, colour and area.
* ``plantarch.inflo.prototypes_attrs`` (exact): material (label, two-sided flag, colour, texture, override) and
  ``object_label`` of every prototype primitive, object data labels, hidden flags.
* ``plantarch.inflo.placement`` (ulp, max 0): after every forced step: bud state / counts / fruit scale, peduncle tube
  nodes, radii and triangles + colours, each flower / fruit object's transform, ``inflorescence_rotation`` (pitch, yaw,
  roll, azimuth, peduncle axis, gravity), base, base scale and triangles + colours, and the phytomer's stored
  peduncle vertices / radii / sampled parameters.
* ``plantarch.inflo.placement_labels`` (exact): material and ``object_label`` of those primitives.
* ``plantarch.inflo.draws`` (exact): engine calls consumed by each ``setFloralBudState`` rebuild (SPEC §3.3: 6, the
  per-flower ``phyllotactic_angle.resample()``; ``randu(0, 0)`` draws nothing) and the phyllotactic-angle cache
  (value, sampled) the rebuild starts from -- the last resample of the previous rebuild.
* ``plantarch.tomato.flowering`` (ulp, max 0): end to end, ``buildPlantInstanceFromLibrary`` at 75 / 100 / 120 days,
  seeds 1..8 (real W3-A blades): the full W3-C observation (engine calls, whole XML with ``<flower>`` records, tubes,
  leaflets), then every peduncle / flower / fruit object. **Registered only once growth exists**
  (``growth_available()``: ``PlantArchitecture.advanceTime`` is defined and a 2-day build succeeds), or with
  ``HELIADES_PARITY_FLOWERING=1``; ``=0`` disables it.
"""
from __future__ import annotations

import functools
import os

import numpy as np

from .cases import case

f32 = np.float32


@case("plantarch.inflo.prototypes", tier="ulp", max_ulp=0, fuzz=3)
def inflo_prototypes(B, s):
    (floats, _strings), _hidden = B.inflo_prototypes(s * 3 + 1, (100.0, 110.0, 120.0)[s])
    return floats


@case("plantarch.inflo.prototypes_attrs", tier="exact", fuzz=3)
def inflo_prototypes_attrs(B, s):
    (_floats, strings), hidden = B.inflo_prototypes(s * 3 + 1, (100.0, 110.0, 120.0)[s])
    return [float(len(hidden))] + [float(h) for h in hidden] + [float(b) for b in "\n".join(strings).encode()]


@case("plantarch.inflo.placement", tier="ulp", max_ulp=0, fuzz=24)
def inflo_placement(B, s):
    return B.inflo_forced(s, "floats")


@case("plantarch.inflo.placement_labels", tier="exact", fuzz=24)
def inflo_placement_labels(B, s):
    return B.inflo_forced(s, "labels")


@case("plantarch.inflo.draws", tier="exact", fuzz=24)
def inflo_draws(B, s):
    return B.inflo_forced(s, "draws")


# ---- end to end (gated on growth) ----------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def growth_available() -> bool:
    """True once Heliades can build a tomato past age 0 (W3-C's ``advanceTime``)."""
    flag = os.environ.get("HELIADES_PARITY_FLOWERING")
    if flag is not None:
        return flag not in ("", "0")
    from ..plantarchitecture import PlantArchitecture
    if not callable(getattr(PlantArchitecture, "advanceTime", None)):
        return False
    try:
        from ..core.scene import Scene
        sc = Scene()
        sc.seed_random_generator(1)
        pa = PlantArchitecture(sc)
        pa.loadPlantModelFromLibrary("tomato")
        pa.buildPlantInstanceFromLibrary((0.0, 0.0, 0.0), 2.0)
    except NotImplementedError:
        return False
    return True


FLOWERING_AGES = (75.0, 100.0, 120.0)


def _flowering(B, s):
    """Full end-to-end build with real inflorescence geometry and the real W3-A blades (replayed oracle blades,
    as in ``plantarch.tomato.age50``, only if ``GenericLeafPrototype`` is unbound), then the whole W3-C observation (engine calls, complete XML including the
    ``<flower>`` / ``<peduncle>`` records, every shoot's nodes and radii, every internode / petiole tube and leaflet by
    object ID / UUID) and every peduncle / flower / fruit object (ID, transform, triangles, colours)."""
    from .cases_plantarch_growth import _hook_bound, _observe
    seed = s % 8 + 1
    age = FLOWERING_AGES[s // 8]
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load("tomato")
    if _hook_bound():
        B.plantarch_build((0.0, 0.0, 0.0), age)
    else:
        B.pa_build_replay_blades(seed, age)
    return np.concatenate([np.asarray(_observe(B, True, draws_seed=seed), f32), np.array([-120.0], f32),
                           B.inflo_plant_organs()])


if growth_available():
    case("plantarch.tomato.flowering", tier="ulp", max_ulp=0, fuzz=24)(_flowering)
