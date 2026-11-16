# SPDX-License-Identifier: LGPL-2.1-or-later
"""Operations for the strawberry / maize parity cases (W4-C), both backends.

* ``w4c_build(age)``: ``buildPlantInstanceFromLibrary((0,0,0), age)``; on Heliades the real W3-A blades are bound and
  a stub build is refused;
* ``w4c_polymesh_objects()``: every polymesh object of the Context (blade prototypes, leaves), ascending;
* ``w4c_inflorescence_gate(crop, seed, age)``: the number of flower + fruit objects an oracle plant of that seed
  carries at that age (computed with PyHelios on both backends and cached, so the gate decision is identical).
"""
from __future__ import annotations

from .ops import op

_GATE: dict = {}


@op("w4c_build", "helios")
def _(B, age=0.0):
    return B.plantarch_build((0.0, 0.0, 0.0), age)


@op("w4c_build", "heliades")
def _(B, age=0.0):
    from ..plantarchitecture import leaf_prototype  # noqa: F401  (binds hooks.GenericLeafPrototype)
    pid = B.plantarch_build((0.0, 0.0, 0.0), age)
    if getattr(B.pa, "leaf_geometry_is_stub", False):
        raise RuntimeError("the build used the stub leaf prototype")
    return pid


#: helios::OBJECT_TYPE_POLYMESH (the snapshot's object type code on both backends)
OBJECT_TYPE_POLYMESH = 5


def _polymesh_objects(B):
    O = B.snapshot(include_hidden=True).objects
    return [int(o) for o, t in zip(O.oid, O.type) if int(t) == OBJECT_TYPE_POLYMESH]


op("w4c_polymesh_objects", "helios")(_polymesh_objects)
op("w4c_polymesh_objects", "heliades")(_polymesh_objects)


def inflorescence_gate(crop, seed, age):
    key = (crop, int(seed), float(age))
    if key not in _GATE:
        from pyhelios import Context, PlantArchitecture
        from pyhelios.types import vec3
        ctx = Context()
        ctx.seedRandomGenerator(int(seed))
        pa = PlantArchitecture(ctx)
        try:
            pa.disableMessages()
        except Exception:
            pass
        pa.loadPlantModelFromLibrary(crop)
        pid = pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), float(age))
        _GATE[key] = len(pa.getPlantFlowerObjectIDs(pid)) + len(pa.getPlantFruitObjectIDs(pid))
    return _GATE[key]


@op("w4c_inflorescence_gate", "helios")
def _(B, crop, seed, age):
    return inflorescence_gate(crop, seed, age)


@op("w4c_inflorescence_gate", "heliades")
def _(B, crop, seed, age):
    return inflorescence_gate(crop, seed, age)
