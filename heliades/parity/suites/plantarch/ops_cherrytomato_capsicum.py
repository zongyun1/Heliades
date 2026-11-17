# SPDX-License-Identifier: LGPL-2.1-or-later
"""Operations for the W4-A crops (cherry tomato, capsicum) on both backends.

The W3-C ops (``pa_advance``, ``pa_xml_text``, ``pa_shoot_internodes``, ``pa_leaf_objects``) are reused. New ones:

* ``w4a_full_build``: load + build at an age with the real blades (refuses the stub on Heliades);
* ``w4a_direct_shoot(crop, age, nodes)``: ``addPlantInstance(pos, age)`` + ``addBaseStemShoot`` with the builder's
  thresholds, then 2 days of growth: the creation function then sees plant ages a library build never gives it
  (the literal-type probe);
* ``w4a_tubes(kind)``: the internode or petiole tube objects in shoot order (``getPlantInternodeObjectIDs`` /
  ``getPlantPetioleObjectIDs``) with their object ID, primitive UUIDs and triangle vertices. Unlike
  ``pa_tube_objects`` it does not attach a shoot index to the petioles: capsicum sheds the petioles of its first six
  nodes (``Phytomer::removeLeaf``), so the petiole count is no longer one per node;
* ``w4a_inflorescence_objects``: the number of peduncle, flower and fruit objects of the plant (the W3-D gate).
"""
from __future__ import annotations

import numpy as np

from ...ops import op

f32 = np.float32


@op("w4a_full_build", "helios")
def _(B, crop, age=0.0):
    B.plantarch_new()
    B.plantarch_load(crop)
    return B.plantarch_build((0.0, 0.0, 0.0), age)


@op("w4a_full_build", "heliades")
def _(B, crop, age=0.0):
    from ....plantarchitecture import leaf_prototype  # noqa: F401  (binds hooks.GenericLeafPrototype)
    B.plantarch_new()
    B.plantarch_load(crop)
    pid = B.plantarch_build((0.0, 0.0, 0.0), age)
    if B.pa.leaf_geometry_is_stub:
        raise RuntimeError("the build used the stub leaf prototype")
    if B.pa.inflorescence_geometry_is_stub:
        raise RuntimeError("the build used the draws-only inflorescence")
    return pid


@op("w4a_tubes", "helios")
def _(B, kind):
    ctx, pa, pid = B.ctx, B.pa, B.pid
    oids = pa.getPlantInternodeObjectIDs(pid) if kind == "internode" else pa.getPlantPetioleObjectIDs(pid)
    out = []
    for oid in [int(o) for o in oids]:
        uu = [int(u) for u in ctx.getObjectPrimitiveUUIDs(oid)]
        flat, _ = ctx.getPrimitiveVertices(uu)
        out.append((oid, uu, [float(x) for x in np.asarray(flat, f32).reshape(-1)]))
    return out


@op("w4a_tubes", "heliades")
def _(B, kind):
    s = B.scene
    out = []
    for shoot in B.pa.plant_instances[B.pid].shoot_tree:
        oids = ([shoot.internode_tube_objID] if kind == "internode"
                else [o for ph in shoot.phytomers for o in ph.getExistingPetioleObjIDs()])
        for oid in oids:
            if not s.does_object_exist(oid):
                continue
            uu = list(s.object_primitive_uuids(oid))
            out.append((oid, uu, [float(x) for u in uu for x in s.primitive_vertices(u).reshape(-1)]))
    return out


@op("w4a_inflorescence_objects", "helios")
def _(B):
    pa, pid = B.pa, B.pid
    return (len(pa.getPlantPeduncleObjectIDs(pid)) + len(pa.getPlantFlowerObjectIDs(pid))
            + len(pa.getPlantFruitObjectIDs(pid)))


@op("w4a_inflorescence_objects", "heliades")
def _(B):
    n_ped = n_inf = 0
    for shoot in B.pa.plant_instances[B.pid].shoot_tree:
        for ph in shoot.phytomers:
            for petiole in ph.floral_buds:
                for fbud in petiole:
                    n_ped += len(fbud.peduncle_objIDs)
                    n_inf += len(fbud.inflorescence_objIDs)
    return n_ped + n_inf


# ---- direct one-shoot build (literal-type probe) ----------------------------------------------------------

#: the library builders' thresholds (buildCherryTomatoPlant / buildCapsicumPlant)
THRESHOLDS = {"cherrytomato": (0, 50, 10, 5, 30, 1000), "capsicum": (0, -1, -1, 75, 14, 1000)}


@op("w4a_direct_shoot", "helios")
def _(B, crop, age, nodes):
    from pyhelios.types import vec3
    from pyhelios.wrappers.DataTypes import AxisRotation
    B.plantarch_new()
    B.plantarch_load(crop)
    pa = B.pa
    B.pid = pid = pa.addPlantInstance(vec3(0, 0, 0), float(age))
    pa.addBaseStemShoot(pid, int(nodes), AxisRotation(0, 0, 0), 0.002, 0.06, 0.01, 0.01, 0, "mainstem")
    pa.breakPlantDormancy(pid)
    pa.setPlantPhenologicalThresholds(pid, *THRESHOLDS[crop])
    pa.advanceTime(2.0, plant_id=pid)
    return pid


@op("w4a_direct_shoot", "heliades")
def _(B, crop, age, nodes):
    from ....plantarchitecture import leaf_prototype  # noqa: F401
    from ....plantarchitecture.params import make_AxisRotation
    B.plantarch_new()
    B.plantarch_load(crop)
    pa = B.pa
    B.pid = pid = pa.addPlantInstance((0, 0, 0), age)
    pa.addBaseStemShoot(pid, int(nodes), make_AxisRotation(0, 0, 0), 0.002, 0.06, 0.01, 0.01, 0, "mainstem")
    pa.breakPlantDormancy(pid)
    pa.setPlantPhenologicalThresholds(pid, *THRESHOLDS[crop])
    pa.advanceTime(pid, 2.0)
    return pid


# ---- inflorescence objects (W3-D) ----------------------------------------------------------------------

def _object_geometry(ids, uuids_of, transform_of, vertices_of):
    out = []
    for oid in sorted(ids):
        uu = list(uuids_of(oid))
        out += [float(oid), float(len(uu))] + [float(u) for u in uu] + transform_of(oid)
        for u in uu:
            out += vertices_of(u)
    return out


@op("w4a_inflorescence_geometry", "helios")
def _(B):
    """Every peduncle, flower and fruit object, sorted by ID: ID, UUIDs, 4x4 transform, triangle vertices."""
    ctx, pa, pid = B.ctx, B.pa, B.pid
    out = []
    for ids in (pa.getPlantPeduncleObjectIDs(pid), pa.getPlantFlowerObjectIDs(pid), pa.getPlantFruitObjectIDs(pid)):
        out.append(-1.0)
        out += _object_geometry(
            [int(o) for o in ids], lambda o: [int(u) for u in ctx.getObjectPrimitiveUUIDs(o)],
            lambda o: [float(x) for x in np.asarray(ctx.getObjectTransformationMatrix(o), f32).reshape(16)],
            lambda u: [float(x) for x in np.asarray(ctx.getPrimitiveVertices([u])[0], f32).reshape(-1)])
    return out


@op("w4a_inflorescence_geometry", "heliades")
def _(B):
    from ....plantarchitecture.params import BUD_FLOWER_CLOSED, BUD_FLOWER_OPEN, BUD_FRUITING
    s = B.scene
    ped, flo, fru = [], [], []
    for shoot in B.pa.plant_instances[B.pid].shoot_tree:
        for ph in shoot.phytomers:
            for petiole in ph.floral_buds:
                for fbud in petiole:
                    ped += [o for o in fbud.peduncle_objIDs if s.does_object_exist(o)]
                    objs = [o for o in fbud.inflorescence_objIDs if s.does_object_exist(o)]
                    if fbud.state in (BUD_FLOWER_CLOSED, BUD_FLOWER_OPEN):
                        flo += objs
                    elif fbud.state == BUD_FRUITING:
                        fru += objs
    out = []
    for ids in (ped, flo, fru):
        out.append(-1.0)
        out += _object_geometry(ids, s.object_primitive_uuids,
                                lambda o: [float(x) for x in s.object_transform(o).reshape(16)],
                                lambda u: [float(x) for x in s.primitive_vertices(u).reshape(-1)])
    return out
