# SPDX-License-Identifier: LGPL-2.1-or-later
"""Operations for the bean / soybean parity cases (W4-B).

* ``w4b_build(model, age)``: ``loadPlantModelFromLibrary(model)`` + ``buildPlantInstanceFromLibrary((0,0,0), age)``.
  On Heliades the real ``GenericLeafPrototype`` is bound (the build refuses the stub blade), and, while W3-D's
  inflorescence geometry is not merged, ``updateInflorescence`` runs in its draws-only mode
  (``w4b_inflorescence_ported`` reports which);
* ``w4b_inflorescence_ported``: whether this backend builds flower / fruit geometry (always on Helios; on Heliades
  once ``Phytomer.updateInflorescence`` stops raising the W3-D seam error);
* ``w4b_floral_states``: per shoot and phytomer, the floral bud states (so a case can tell whether a plant has any
  inflorescence geometry: states 2..4 = closed flower / open flower / fruiting);
* ``w4b_shoot_types``: the shoot type label of every shoot.
"""
from __future__ import annotations

import numpy as np

from .ops import op

f32 = np.float32


def inflorescence_ported() -> bool:
    """True once W3-D's ``updateInflorescence`` is merged (the seam no longer raises)."""
    from ..plantarchitecture import phytomer as P
    try:
        from ..plantarchitecture import inflorescence  # noqa: F401
    except ImportError:
        return False
    src = getattr(P.Phytomer, "updateInflorescence", None)
    return src is not None and "W3-D" not in (getattr(src, "__doc__", "") or "")


@op("w4b_inflorescence_ported", "helios")
def _(B):
    return True


@op("w4b_inflorescence_ported", "heliades")
def _(B):
    return inflorescence_ported()


@op("w4b_build", "helios")
def _(B, model, age=0.0):
    B.plantarch_new()
    B.plantarch_load(model)
    return B.plantarch_build((0.0, 0.0, 0.0), float(age))


@op("w4b_build", "heliades")
def _(B, model, age=0.0):
    from ..plantarchitecture import leaf_prototype  # noqa: F401  (binds hooks.GenericLeafPrototype)
    B.plantarch_new()
    B.plantarch_load(model)
    if not inflorescence_ported():
        B.pa.inflorescence_geometry_draws_only = True
    pid = B.plantarch_build((0.0, 0.0, 0.0), float(age))
    if B.pa.leaf_geometry_is_stub:
        raise RuntimeError("the build used the stub leaf prototype")
    return pid


@op("w4b_floral_states", "helios")
def _(B):
    return [int(el.text) for el in B.plantarch_xml().iter("bud_state")]


@op("w4b_floral_states", "heliades")
def _(B):
    out = []
    for shoot in B.pa.plant_instances[B.pid].shoot_tree:
        for ph in shoot.phytomers:
            for petiole in ph.floral_buds:
                out.extend(int(b.state) for b in petiole)
    return out


@op("w4b_shoot_types", "helios")
def _(B):
    root = B.plantarch_xml()
    return [s.find("shoot_type_label").text.strip() for s in root.iter("shoot")]


@op("w4b_shoot_types", "heliades")
def _(B):
    return [s.shoot_type_label for s in B.pa.plant_instances[B.pid].shoot_tree]


# ---- petiole tubes in plant order ------------------------------------------------------------------
# ``pa_tube_objects("petiole")`` (W3-C) labels each petiole object with a shoot index taken from the node counts
# and zips the two lists, which truncates the list on a plant with more petiole objects than phytomers (the
# unifoliate node carries two petioles). This op lists them in plant order (``getPlantPetioleObjectIDs``), no label.

@op("w4b_petiole_objects", "helios")
def _(B):
    ctx = B.ctx
    out = []
    for oid in [int(o) for o in B.pa.getPlantPetioleObjectIDs(B.pid)]:
        uu = [int(u) for u in ctx.getObjectPrimitiveUUIDs(oid)]
        flat, _ = ctx.getPrimitiveVertices(uu)
        out.append((oid, uu, [float(x) for x in np.asarray(flat, f32).reshape(-1)]))
    return out


@op("w4b_petiole_objects", "heliades")
def _(B):
    s = B.scene
    out = []
    for shoot in B.pa.plant_instances[B.pid].shoot_tree:
        for ph in shoot.phytomers:
            for oid in ph.getExistingPetioleObjIDs():
                uu = s.object_primitive_uuids(oid)
                out.append((oid, list(uu), [float(x) for u in uu for x in s.primitive_vertices(u).reshape(-1)]))
    return out
