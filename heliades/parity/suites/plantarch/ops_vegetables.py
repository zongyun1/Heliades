# SPDX-License-Identifier: LGPL-2.1-or-later
"""Operations for the herbaceous-vegetable parity cases (W5-B: cowpea, cucumber, butterlettuce, sugarbeet, asparagus).

* ``w5b_build(model, age)``: a fresh PlantArchitecture, ``loadPlantModelFromLibrary(model)``,
  ``buildPlantInstanceFromLibrary((0,0,0), age)``; on Heliades the real W3-A blades are bound and a stub build is
  refused;
* ``w5b_bud_states()``: every floral bud state of the plant, in plant order (from the XML on Helios);
* ``w5b_leaf_normal_maps()``: per leaf primitive, its material label and that material's normal map.
"""
from __future__ import annotations

import numpy as np

from ...ops import op

f32 = np.float32


@op("w5b_build", "helios")
def _(B, model, age=0.0):
    B.plantarch_new()
    B.plantarch_load(model)
    return B.plantarch_build((0.0, 0.0, 0.0), float(age))


@op("w5b_build", "heliades")
def _(B, model, age=0.0):
    from ....plantarchitecture import leaf_prototype  # noqa: F401  (binds hooks.GenericLeafPrototype)
    B.plantarch_new()
    B.plantarch_load(model)
    pid = B.plantarch_build((0.0, 0.0, 0.0), float(age))
    if getattr(B.pa, "leaf_geometry_is_stub", False):
        raise RuntimeError("the build used the stub leaf prototype")
    return pid


@op("w5b_bud_states", "helios")
def _(B):
    return [int(el.text) for el in B.plantarch_xml().iter("bud_state")]


@op("w5b_bud_states", "heliades")
def _(B):
    out = []
    for shoot in B.pa.plant_instances[B.pid].shoot_tree:
        for ph in shoot.phytomers:
            for petiole in ph.floral_buds:
                out.extend(int(b.state) for b in petiole)
    return out


def _normal_maps(B, labels):
    import os
    out = []
    for m in labels:
        if B.name == "helios":
            f, k, st = B.ctx.getMaterialNormalMapFile(m), int(B.ctx.getMaterialNormalMapKind(m)), B.ctx.getMaterialNormalMapStrength(m)
        else:
            f, k, st = (B.scene.get_material_normal_map_file(m), int(B.scene.get_material_normal_map_kind(m)),
                        B.scene.get_material_normal_map_strength(m))
        out.append((m, os.path.basename(f), k, float(np.float32(st))))
    return out


@op("w5b_leaf_normal_maps", "helios")
def _(B):
    """For every primitive of every leaf object of the plant: its material label and that material's normal map
    (file basename, kind, strength)."""
    labels = [B.ctx.getPrimitiveMaterialLabel(int(u)) for o in B.pa.getPlantLeafObjectIDs(B.pid)
              for u in B.ctx.getObjectPrimitiveUUIDs(int(o))]
    return _normal_maps(B, labels)


@op("w5b_leaf_normal_maps", "heliades")
def _(B):
    sc = B.scene
    labels = [sc.get_primitive_material_label(u) for shoot in B.pa.plant_instances[B.pid].shoot_tree
              for ph in shoot.phytomers for petiole in ph.leaf_objIDs for o in petiole if sc.does_object_exist(o)
              for u in sc.object_primitive_uuids(o)]
    return _normal_maps(B, labels)
