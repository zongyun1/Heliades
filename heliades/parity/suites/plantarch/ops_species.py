# SPDX-License-Identifier: LGPL-2.1-or-later
"""Operations for the species-file parity cases (S0).

* ``species_new(spec, label)``: a fresh PlantArchitecture; on Heliades the species file ``spec`` is registered as
  plant model ``label`` (replacing the library model of that name on this PlantArchitecture), on Helios nothing is
  registered, so ``plantarch_load(label)`` loads the Helios library model ``label`` there and the species file here;
* ``species_build(spec, label, age)``: ``species_new``, load ``label``, ``buildPlantInstanceFromLibrary((0,0,0), age)``.
"""
from __future__ import annotations

from ...ops import op


@op("species_new", "helios")
def _(B, spec, label):
    return B.plantarch_new()


@op("species_new", "heliades")
def _(B, spec, label):
    from ....species import register
    pa = B.plantarch_new()
    register(pa, spec, label)
    return pa


def _build(B, spec, label, age):
    B.species_new(spec, label)
    B.plantarch_load(label)
    return B.plantarch_build((0.0, 0.0, 0.0), float(age))


@op("species_build", "helios")
def _(B, spec, label, age=0.0):
    return _build(B, spec, label, age)


@op("species_build", "heliades")
def _(B, spec, label, age=0.0):
    from ....plantarchitecture import leaf_prototype  # noqa: F401  (binds hooks.GenericLeafPrototype)
    pid = _build(B, spec, label, age)
    if getattr(B.pa, "leaf_geometry_is_stub", False):
        raise RuntimeError("the build used the stub leaf prototype")
    return pid
