# SPDX-License-Identifier: LGPL-2.1-or-later
"""Growth operations for both backends (W3-C).

* ``pa_advance(days)``: ``advanceTime(plantID, days)``;
* ``pa_xml_text``: ``writePlantStructureXML`` as text (Helios: the file; Heliades: ``plant.plant_structure_xml``,
  a text-identical port);
* ``pa_shoot_internodes``: per shoot, the stored internode nodes and radii (``getShootInternodeVertices`` /
  ``getShootInternodeRadii``) at full precision;
* ``pa_tube_objects``: the internode and petiole tube objects in shoot order (``getPlantInternodeObjectIDs`` /
  ``getPlantPetioleObjectIDs``) with their shoot, object ID, primitive UUIDs and triangle vertices;
* ``pa_leaf_objects``: the leaflet objects in shoot order with their 4x4 transforms and, optionally, their
  primitives' vertices;
* ``pa_area_exact``: whether this backend's blade areas are the real ones (always on Helios; on Heliades when the
  real ``GenericLeafPrototype`` is bound, or the blades are an exact replay of the oracle's);
* ``pa_build_replay_blades(seed, age)``: Heliades builds with *replayed* blades: the 35 hidden prototypes are
  rebuilt primitive by primitive from an oracle build of the same seed (transforms, UVs, texture, labels),
  while the private-stream draws are those of the stub. Every blade area, hence girth, child-shoot offsets and
  UUID numbering, is then the oracle's, which lets the growth cases check everything that depends on blade
  area before W3-A lands. On Helios it is a plain build.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from ...ops import op

f32 = np.float32


# ---- advance / state ---------------------------------------------------------------------------------

@op("pa_fresh_context", "helios")
def _(B):
    from pyhelios import Context
    B.ctx = Context()


@op("pa_fresh_context", "heliades")
def _(B):
    from ....core.scene import Scene
    B.scene = Scene()


@op("pa_advance", "helios")
def _(B, days):
    B.pa.advanceTime(float(days), plant_id=B.pid)


@op("pa_advance", "heliades")
def _(B, days):
    B.pa.advanceTime(B.pid, float(days))


@op("pa_xml_text", "helios")
def _(B):
    d = tempfile.mkdtemp(prefix="heliades_pa_")
    path = os.path.join(d, "plant.xml")
    B.pa.writePlantStructureXML(B.pid, path)
    with open(path) as f:
        text = f.read()
    os.remove(path)
    os.rmdir(d)
    return text


@op("pa_xml_text", "heliades")
def _(B):
    from ....plantarchitecture.xml_io import plant_structure_xml
    return plant_structure_xml(B.pa, B.pid)


@op("pa_area_exact", "helios")
def _(B):
    return True


@op("pa_area_exact", "heliades")
def _(B):
    return (not B.pa.leaf_geometry_is_stub) or bool(getattr(B.pa, "leaf_geometry_is_replay", False))


@op("pa_shoot_ids", "helios")
def _(B):
    return [int(s) for s in B.pa.getAllShootIDs(B.pid)]


@op("pa_shoot_ids", "heliades")
def _(B):
    return [s.ID for s in B.pa.plant_instances[B.pid].shoot_tree]


@op("pa_shoot_internodes", "helios")
def _(B, shoot_ids):
    out = []
    for sid in shoot_ids:
        verts = B.pa.getShootInternodeVertices(B.pid, sid)
        radii = B.pa.getShootInternodeRadii(B.pid, sid)
        out.append(([float(f32(c)) for v in verts for c in (v[0], v[1], v[2])], [float(f32(r)) for r in radii]))
    return out


@op("pa_shoot_internodes", "heliades")
def _(B, shoot_ids):
    tree = B.pa.plant_instances[B.pid].shoot_tree
    out = []
    for sid in shoot_ids:
        s = tree[sid]
        out.append(([float(c) for row in s.shoot_internode_vertices for v in row for c in v],
                    [float(r) for row in s.shoot_internode_radii for r in row]))
    return out


@op("pa_shoot_node_counts", "helios")
def _(B):
    return [int(B.pa.getShootNodeCount(B.pid, s)) for s in B.pa.getAllShootIDs(B.pid)]


@op("pa_shoot_node_counts", "heliades")
def _(B):
    return [len(s.phytomers) for s in B.pa.plant_instances[B.pid].shoot_tree]


# ---- tubes and leaves in shoot order -------------------------------------------------------------------

def _shoot_of(counts):
    """Expand per-shoot counts into the shoot index of each object, in shoot order."""
    return [sid for sid, n in enumerate(counts) for _ in range(n)]


@op("pa_tube_objects", "helios")
def _(B, kind):
    ctx, pa, pid = B.ctx, B.pa, B.pid
    if kind == "internode":
        oids = [int(o) for o in pa.getPlantInternodeObjectIDs(pid)]
        shoots = list(range(len(oids)))
    else:
        oids = [int(o) for o in pa.getPlantPetioleObjectIDs(pid)]
        shoots = _shoot_of([int(pa.getShootNodeCount(pid, s)) for s in pa.getAllShootIDs(pid)])
    out = []
    for sid, oid in zip(shoots, oids):
        uu = [int(u) for u in ctx.getObjectPrimitiveUUIDs(oid)]
        flat, _ = ctx.getPrimitiveVertices(uu)
        out.append((sid, oid, uu, [float(x) for x in np.asarray(flat, f32).reshape(-1)]))
    return out


@op("pa_tube_objects", "heliades")
def _(B, kind):
    s = B.scene
    out = []
    for shoot in B.pa.plant_instances[B.pid].shoot_tree:
        oids = ([shoot.internode_tube_objID] if kind == "internode"
                else [o for ph in shoot.phytomers for o in ph.getExistingPetioleObjIDs()])
        for oid in oids:
            if not s.does_object_exist(oid):
                continue
            uu = s.object_primitive_uuids(oid)
            out.append((shoot.ID, oid, list(uu), [float(x) for u in uu for x in s.primitive_vertices(u).reshape(-1)]))
    return out


@op("pa_leaf_objects", "helios")
def _(B, with_vertices=False):
    ctx, pa, pid = B.ctx, B.pa, B.pid
    out = []
    for oid in [int(o) for o in pa.getPlantLeafObjectIDs(pid)]:
        T = [float(x) for x in np.asarray(ctx.getObjectTransformationMatrix(oid), f32).reshape(16)]
        verts = []
        if with_vertices:
            uu = [int(u) for u in ctx.getObjectPrimitiveUUIDs(oid)]
            flat, _ = ctx.getPrimitiveVertices(uu)
            verts = [float(x) for x in np.asarray(flat, f32).reshape(-1)]
        out.append((oid, T, verts))
    return out


@op("pa_leaf_objects", "heliades")
def _(B, with_vertices=False):
    s = B.scene
    out = []
    for shoot in B.pa.plant_instances[B.pid].shoot_tree:
        for ph in shoot.phytomers:
            for petiole in ph.leaf_objIDs:
                for oid in petiole:
                    if not s.does_object_exist(oid):
                        continue
                    T = [float(x) for x in s.object_transform(oid).reshape(16)]
                    verts = []
                    if with_vertices:
                        verts = [float(x) for u in s.object_primitive_uuids(oid) for x in s.primitive_vertices(u).reshape(-1)]
                    out.append((oid, T, verts))
    return out


# ---- replayed blades (Heliades) -------------------------------------------------------------------------

_ORACLE_PROTOTYPES: dict = {}


def oracle_prototypes(seed):
    """The 35 hidden leaf prototypes of an oracle age-0 tomato of Context seed ``seed``, in creation order:
    per object, its primitives as (type, transform[16], texture file, uv, object_label, color, UUID)."""
    if seed in _ORACLE_PROTOTYPES:
        return _ORACLE_PROTOTYPES[seed]
    from pyhelios import Context, PlantArchitecture
    from pyhelios.types import vec3
    ctx = Context()
    ctx.seedRandomGenerator(int(seed))
    pa = PlantArchitecture(ctx)
    try:
        pa.disableMessages()
    except Exception:
        pass
    pa.loadPlantModelFromLibrary("tomato")
    pid = pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), 0.0)
    first_leaf = min(min(p) for p in pa.getPhytomerLeafObjectIDs(pid, 0, 0))
    protos = []
    for oid in range(1, first_leaf):
        if not (ctx.doesObjectExist(oid) and ctx.isObjectHidden(oid)):
            continue
        prims = []
        for u in ctx.getObjectPrimitiveUUIDs(oid):
            u = int(u)
            T = np.asarray(ctx.getPrimitiveTransformationMatrix(u), f32).reshape(16).copy()
            tex = ctx.getPrimitiveTextureFile(u) or ""
            uv = np.asarray([[p.x, p.y] for p in ctx.getPrimitiveTextureUV(u)], f32) if tex else None
            label = ctx.getPrimitiveData(u, "object_label") if ctx.doesPrimitiveDataExist(u, "object_label") else ""
            c = ctx.getPrimitiveColor(u)
            prims.append((int(ctx.getPrimitiveType(u)), T, tex, uv, label, (c.r, c.g, c.b), u))
        protos.append(prims)
    _ORACLE_PROTOTYPES[seed] = protos
    return protos


def replay_leaf_prototype_function(protos):
    """A ``GenericLeafPrototype`` stand-in that makes the stub's private-stream draws, then rebuilds the next
    oracle prototype primitive by primitive (transform set directly, texture + UV -> solid fraction computed by
    Heliades) under the oracle's UUIDs (its builder leaves gaps: 152 UUIDs for 144 primitives), labels the
    petiolule, and returns ``addPolymeshObject`` of it."""
    from ....core.primitives import TRIANGLE
    from ....plantarchitecture.phytomer import stub_draws_only
    it = iter(protos)

    def fn(scene, prototype, compound_leaf_index):
        stub_draws_only(prototype)
        uu = []
        for ptype, T, tex, uv, label, color, oracle_uuid in next(it):
            if ptype != int(TRIANGLE):
                raise NotImplementedError("replayed prototypes hold triangles only")
            if oracle_uuid < scene.current_uuid:
                raise RuntimeError("replayed blade UUIDs out of step with the oracle")
            scene.current_uuid = oracle_uuid      # the oracle blade builder consumes UUIDs it deletes again
            if tex:
                scene.add_texture(tex)
                u = scene._new_textured(TRIANGLE, T, tex, uv)
            else:
                u = scene._new(TRIANGLE, T, color)
            if label == "petiolule":
                scene.set_primitive_data(u, "object_label", "petiolule")
            uu.append(u)
        return scene.add_polymesh_object(uu)
    return fn


@op("pa_build_replay_blades", "heliades")
def _(B, seed, age=0.0):
    B.pa.leaf_prototype_function_override = replay_leaf_prototype_function(oracle_prototypes(seed))
    B.pa.leaf_geometry_is_replay = True
    return B.plantarch_build((0.0, 0.0, 0.0), age)


@op("pa_build_replay_blades", "helios")
def _(B, seed, age=0.0):
    return B.plantarch_build((0.0, 0.0, 0.0), age)


@op("pa_draws_only_inflorescence", "heliades")
def _(B):
    B.pa.inflorescence_geometry_draws_only = True


@op("pa_draws_only_inflorescence", "helios")
def _(B):
    pass
