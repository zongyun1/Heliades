# SPDX-License-Identifier: LGPL-2.1-or-later
"""Leaf-prototype operations for both backends (W3-A).

``plantarch_leafproto_build`` builds the tomato leaf prototype cache the way the tomato path does:

* Helios: ``buildPlantInstanceFromLibrary(pos, 0)``. The first Phytomer builds its internode tube (object 1,
  UUIDs 0-13), its petiole tube (object 2, UUIDs 14-83), then the hidden blade prototypes (objects 3.., in
  prototype-major, leaflet-minor order), then the visible leaflet copies. The prototypes are exactly the
  hidden objects of the Context at age 0 (the inflorescence prototypes only appear with the first
  peduncle), so they are found by their hidden flag, not by position.
* Heliades: the same Context state up to the cache, reproduced with the public Scene API until W3-B lands:
  the identifier from ``loadPlantModelFromLibrary``, a stand-in internode tube (7 radial subdivisions, 2
  nodes: 14 triangles, labelled "shoot", material ``tomato_mainstem_stem``) and petiole tube (6 nodes: 70
  triangles, labelled "petiole", material ``tomato_mainstem_petiole`` in the petiole color), then
  :meth:`PhytomerLeafMixin.buildLeafPrototypeCache` on a phytomer stand-in. The stand-in tubes are never
  compared; only their UUID/object-ID footprint and the petiole material label matter to the prototypes.

``plantarch_leafproto_redefine`` edits the mainstem leaf prototype (and petiole) through the PyHelios JSON
round trip on Helios and through the ported structs on Heliades (each RandomParameter rebuilt with
``initialize(gen)``, as the wrapper's ``jsonToRandomParameterFloat`` does).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .ops import op

f32 = np.float32


# ---- build ---------------------------------------------------------------------------------------

@op("plantarch_leafproto_build", "helios")
def _(B):
    B.plantarch_build((0.0, 0.0, 0.0), 0.0)


class _PhytomerStandIn:
    """Only the members PhytomerLeafMixin reads (see leaf_prototype module docstring)."""

    def __init__(self, pa, shoot_type_label, petiole_objIDs, plantID=0):
        from ..plantarchitecture.leaf_prototype import PhytomerLeafMixin
        self.__class__ = type("_Phytomer", (_PhytomerStandIn, PhytomerLeafMixin), {})
        self.context_ptr = pa.context_ptr
        self.plantarchitecture_ptr = pa
        sp = pa.shoot_types[shoot_type_label].copy()        # Shoot's copy -> Phytomer's own copy (no draws)
        self.phytomer_parameters = sp.phytomer_parameters.copy()
        self.parent_shoot_ptr = SimpleNamespace(shoot_type_label=shoot_type_label)
        self.petiole_objIDs = list(petiole_objIDs)
        self.plantID = plantID


def heliades_tomato_prototype_cache(pa, plant_name="tomato", shoot_type_label="mainstem"):
    """The Context state of the tomato path up to the blade cache, then the cache itself (see module doc)."""
    from ..plantarchitecture.leaf_prototype import nameOrganMaterials
    from ..plantarchitecture.library.tomato import base_rotation_draws
    sc = pa.context_ptr
    if not hasattr(pa, "plant_instances"):
        pa.plant_instances = {0: SimpleNamespace(plant_name=plant_name)}
    base_rotation_draws(pa)
    pp = pa.shoot_types[shoot_type_label].phytomer_parameters
    ic, pc = pp.internode.color, pp.petiole.color
    stem = sc.add_tube_object(7, [(0, 0, 0), (0, 0, 0.004)], [0.001, 0.001], [ic, ic])
    sc.set_primitive_data(sc.object_primitive_uuids(stem), "object_label", "shoot")
    nameOrganMaterials(pa, stem, f"{plant_name}_{shoot_type_label}_stem")
    nodes = [(0, 0, 0.004 + 0.01 * k) for k in range(6)]
    petiole = sc.add_tube_object(7, nodes, [0.002 * (1 - 0.14 * k) for k in range(6)], [pc] * 6)
    sc.set_primitive_data(sc.object_primitive_uuids(petiole), "object_label", "petiole")
    nameOrganMaterials(pa, petiole, f"{plant_name}_{shoot_type_label}_petiole")
    ph = _PhytomerStandIn(pa, shoot_type_label, [petiole])
    nleaf = int(ph.phytomer_parameters.leaf.leaves_per_petiole.val())
    ph.buildLeafPrototypeCache(nleaf, ph.phytomer_parameters.leaf.prototype.resolveFlexibility())
    return ph


@op("plantarch_leafproto_build", "heliades")
def _(B):
    return heliades_tomato_prototype_cache(B.pa)


# ---- observations of the prototype objects --------------------------------------------------------

def _hidden_objects(B, snap):
    return [int(o) for o, h in zip(snap.objects.oid, snap.objects.hidden) if h]


@op("plantarch_leafproto_snapshot", "helios")
def _(B):
    return B.snapshot(include_hidden=True)


@op("plantarch_leafproto_snapshot", "heliades")
def _(B):
    return B.snapshot(include_hidden=True)


def _texkey(t):
    from .compare import texture_key
    return texture_key(t)


@op("plantarch_leafproto_attrs", "helios")
def _(B, oids):
    """Per prototype object: material label / color / texture / two-sided flag and object_label per
    primitive, object data labels, and the polymesh face set."""
    ctx = B.ctx
    out = []
    for o in oids:
        uu = [int(u) for u in ctx.getObjectPrimitiveUUIDs(o)]
        prims = []
        for u in uu:
            m = ctx.getPrimitiveMaterialLabel(u)
            c = ctx.getMaterialColor(m)
            prims.append(dict(material=m, color=[c.r, c.g, c.b, c.a], mtex=_texkey(ctx.getMaterialTexture(m)),
                              twosided=int(ctx.getMaterialTwosidedFlag(m)),
                              label=ctx.getPrimitiveData(u, "object_label") if ctx.doesPrimitiveDataExist(u, "object_label") else "",
                              data=sorted(ctx.listPrimitiveData(u))))
        V = [(v.x, v.y, v.z) for v in ctx.getPolymeshObjectVertices(o)]
        N = [(v.x, v.y, v.z) for v in ctx.getPolymeshObjectVertexNormals(o)]
        UV = [(v.x, v.y) for v in ctx.getPolymeshObjectVertexUV(o)]
        F = [(f.x, f.y, f.z) for f in ctx.getPolymeshObjectFaces(o)]
        nf = ctx.getPolymeshObjectFaceCount(o)
        fu = [int(ctx.getPolymeshObjectPrimitiveUUIDForFace(o, k)) for k in range(nf)]
        out.append(dict(oid=o, prims=prims, objdata=sorted(ctx.listObjectData(o)), vertices=V, normals=N, uv=UV,
                        faces=F, face_uuids=fu, normal_source=int(ctx.getPolymeshObjectVertexNormalSource(o))))
    return out


@op("plantarch_leafproto_attrs", "heliades")
def _(B, oids):
    sc = B.scene
    out = []
    for o in oids:
        prims = []
        for u in sc.object_primitive_uuids(o):
            m = sc.get_primitive_material_label(u)
            prims.append(dict(material=m, color=[float(x) for x in sc.get_material_color(m)],
                              mtex=_texkey(sc.get_material_texture(m)), twosided=int(sc.get_material_twosided_flag(m)),
                              label=sc.get_primitive_data(u, "object_label") if sc.does_primitive_data_exist(u, "object_label") else "",
                              data=sorted(sc.list_primitive_data(u))))
        nf = sc.polymesh_object_face_count(o)
        out.append(dict(oid=o, prims=prims, objdata=sorted(sc.list_object_data(o)),
                        vertices=[tuple(v) for v in sc.polymesh_object_vertices(o)],
                        normals=[tuple(v) for v in sc.polymesh_object_vertex_normals(o)],
                        uv=[tuple(v) for v in sc.polymesh_object_vertex_uv(o)],
                        faces=[tuple(int(i) for i in f) for f in sc.polymesh_object_faces(o)],
                        face_uuids=[sc.polymesh_object_primitive_uuid_for_face(o, k) for k in range(nf)],
                        normal_source=int(sc.polymesh_object_vertex_normal_source(o))))
    return out


# ---- redefining the leaf prototype ------------------------------------------------------------------
# spec: dict path -> value, path relative to ShootParameters. A value is either a scalar/dict/vec3 (plain
# field) or ("constant"|"uniform"|"normal", params) for a RandomParameter_float.

@op("plantarch_leafproto_redefine", "helios")
def _(B, label, spec):
    d = B.pa.getCurrentShootParameters(label)
    for path, value in spec.items():
        node = d
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[k]
        if isinstance(value, tuple) and value[0] == "int":
            node[keys[-1]] = {"distribution": "constant", "parameters": [int(value[1])]}
        elif isinstance(value, tuple) and isinstance(value[0], str):
            node[keys[-1]] = {"distribution": value[0], "parameters": [float(x) for x in value[1]]}
        elif isinstance(value, dict):
            node[keys[-1]] = {str(k): (v if isinstance(v, str) else float(v)) for k, v in value.items()}
        elif isinstance(value, tuple):
            node[keys[-1]] = {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}
        else:
            node[keys[-1]] = value
    B.pa.defineShootType(label, d)


@op("plantarch_leafproto_redefine", "heliades")
def _(B, label, spec):
    from ..plantarchitecture import RandomParameterFloat
    sp = B.pa.getCurrentShootParameters(label)
    for path, value in spec.items():
        keys = path.split(".")
        parent = sp
        for k in keys[:-1]:
            parent = getattr(parent, k)
        if isinstance(value, tuple) and value[0] == "int":
            setattr(parent, keys[-1], int(value[1]))      # RandomParameter_int::operator=(int)
        elif isinstance(value, tuple) and isinstance(value[0], str):
            new = RandomParameterFloat()
            new.initialize(B.pa.generator)
            dist, params = value
            if dist == "constant":
                new.set(params[0])
            elif dist == "uniform":
                new.uniformDistribution(*params)
            elif dist == "normal":
                new.normalDistribution(*params)
            else:
                raise ValueError(dist)
            setattr(parent, keys[-1], new)
        elif isinstance(value, dict):
            parent._raw_set(keys[-1], {int(k): (v if isinstance(v, str) else f32(v)) for k, v in value.items()})
        else:
            setattr(parent, keys[-1], value)
    B.pa.defineShootType(label, sp)
