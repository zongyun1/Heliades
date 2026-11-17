# SPDX-License-Identifier: LGPL-2.1-or-later
"""Acceptance operations for the library-built tomato (W3-E), both backends.

* ``pa_accept_build(seed, age)``: seed, ``PlantArchitecture``, load ``tomato``, ``buildPlantInstanceFromLibrary``
  (Heliades with the real W3-A blades); records the build wall time in :data:`TIMINGS`;
* ``pa_accept_attrs()``: materials and data of every primitive and object, hidden ones included: per primitive
  its material label, the material's color / texture basename / two-sided flag, and each primitive data label with
  its value; per object each object data label with its type and value. Encoded as one number stream;
* ``pa_accept_api(pids)``: the public query API (``api.py`` on Heliades, PyHelios on Helios) for the given plants:
  plant scalars and lists, shoot topology, per-phytomer getters, organ object IDs, canopy-wide UUID lists and the
  ``object_label`` selections. Returned as a nested structure, flattened by :func:`flatten`;
* ``pa_accept_xml(pid)``: ``writePlantStructureXML`` text; ``pa_accept_structure(pid)``: the structured dict
  (Heliades ``plant_structure_dict``; Helios: the same parser over its XML plus the organ IDs from the PyHelios
  getters);
* ``pa_accept_build_two(seed, age, positions)``, ``pa_accept_canopy(...)``, ``pa_accept_staggered(seed)``: the
  multi-plant builds.

Flowering pairs: :func:`oracle_has_inflorescence` builds the oracle plant and reports whether any inflorescence
geometry (a peduncle object, or the hidden flower / fruit prototypes, which outlive a flower that died) was ever
made; it only labels the grid now. :func:`inflorescence_geometry_implemented` is the capability check (true since
W3-D merged); ``HELIADES_PARITY_FLOWERING=0`` turns it off, and then the flowering pairs are gated (both backends
return the same sentinel), which is how the grid ran before W3-D.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import tempfile
import time
import zlib

import numpy as np

from ...ops import op

f32 = np.float32

#: (backend, label) -> seconds, filled by the build ops (the performance report reads it)
TIMINGS: dict = {}

LEAF_PROTOTYPE_OBJECTS = 35          # tomato: 5 unique prototypes x 7 leaflets, all hidden


def _code(s: str) -> float:
    return float(zlib.crc32(str(s).encode()) & 0xFFFFFF)


# ---- gating -----------------------------------------------------------------------------------------

def inflorescence_geometry_implemented() -> bool:
    """True when Heliades builds inflorescence geometry (W3-D, merged): the module exists and the Phytomer seam
    delegates to it. ``HELIADES_PARITY_FLOWERING=0/1`` (shared with ``cases_inflo``) overrides: ``0``
    gates the flowering pairs of the grid again."""
    flag = os.environ.get("HELIADES_PARITY_FLOWERING")
    if flag is not None:
        return flag not in ("", "0")
    if importlib.util.find_spec("heliades.plantarchitecture.inflorescence") is None:
        return False
    from ....plantarchitecture.phytomer import Phytomer
    try:
        src = inspect.getsource(Phytomer.updateInflorescence)
    except (OSError, TypeError):
        return True
    return "is W3-D" not in src


_ORACLE_INFL: dict = {}


def oracle_has_inflorescence(seed: int, age: float) -> bool:
    """Did the oracle tomato of Context seed ``seed`` grown to ``age`` ever build inflorescence geometry?"""
    key = (int(seed), float(age))
    if key not in _ORACLE_INFL:
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
        pid = pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), float(age))
        from ...backends import _all_ids_including_hidden
        oids = _all_ids_including_hidden(ctx.getObjectCount(), ctx.doesObjectExist, 1)
        hidden = sum(1 for o in oids if ctx.isObjectHidden(o))
        _ORACLE_INFL[key] = bool(pa.getPlantPeduncleObjectIDs(pid)) or hidden > LEAF_PROTOTYPE_OBJECTS
    return _ORACLE_INFL[key]


def pair_is_gated(seed: int, age: float) -> bool:
    return oracle_has_inflorescence(seed, age) and not inflorescence_geometry_implemented()


# ---- flattening ---------------------------------------------------------------------------------------

def flatten(x, out=None) -> list:
    """Nested lists / tuples / dicts / vec3-likes / scalars / strings -> one number stream with length markers,
    so structure differences show up as a divergence and not as a silent shift."""
    if out is None:
        out = []
    if isinstance(x, dict):
        out += [-7.0, float(len(x))]
        for k in sorted(x, key=lambda k: (isinstance(k, str), k)):
            out.append(_code(k) if isinstance(k, str) else float(k))
            flatten(x[k], out)
    elif isinstance(x, str):
        out += [-6.0, _code(x)]
    elif hasattr(x, "x") and hasattr(x, "y") and hasattr(x, "z") and not isinstance(x, np.ndarray):
        out += [-5.0, 3.0, float(f32(x.x)), float(f32(x.y)), float(f32(x.z))]      # a vec3 is a 3-list
    elif isinstance(x, (list, tuple, np.ndarray)):
        out += [-5.0, float(len(x))]
        for v in x:
            flatten(v, out)
    elif isinstance(x, (bool, np.bool_)):
        out.append(float(bool(x)))
    else:
        out.append(float(f32(x)) if isinstance(x, (float, np.floating)) else float(x))
    return out


# ---- build ---------------------------------------------------------------------------------------------

def _new_pa(B, seed):
    B.seed(int(seed))
    B.plantarch_new()
    B.plantarch_load("tomato")
    if B.name == "heliades":
        from ....plantarchitecture import leaf_prototype  # noqa: F401  (binds hooks.GenericLeafPrototype)


def _check_real_blades(B):
    if B.name == "heliades" and getattr(B.pa, "leaf_geometry_is_stub", False):
        raise RuntimeError("the build used the stub leaf prototype")


def _timed(B, label, fn):
    t = time.perf_counter()
    r = fn()
    TIMINGS[(B.name, label)] = time.perf_counter() - t
    return r


def _vec3(B, p):
    if B.name == "helios":
        from pyhelios.types import vec3
        return vec3(*map(float, p))
    return tuple(map(float, p))


@op("pa_accept_build", "helios")
def _(B, seed, age):
    _new_pa(B, seed)
    B.pid = _timed(B, ("build", float(age), int(seed)), lambda: B.pa.buildPlantInstanceFromLibrary(_vec3(B, (0, 0, 0)), float(age)))
    return B.pid


@op("pa_accept_build", "heliades")
def _(B, seed, age):
    _new_pa(B, seed)
    B.pid = _timed(B, ("build", float(age), int(seed)), lambda: B.pa.buildPlantInstanceFromLibrary((0.0, 0.0, 0.0), float(age)))
    _check_real_blades(B)
    return B.pid


@op("pa_accept_build_two", "helios")
def _(B, seed, age, positions):
    _new_pa(B, seed)
    B.pids = [B.pa.buildPlantInstanceFromLibrary(_vec3(B, p), float(age)) for p in positions]
    return B.pids


@op("pa_accept_build_two", "heliades")
def _(B, seed, age, positions):
    _new_pa(B, seed)
    B.pids = [B.pa.buildPlantInstanceFromLibrary(p, float(age)) for p in positions]
    _check_real_blades(B)
    return B.pids


@op("pa_accept_canopy", "helios")
def _(B, seed, center, spacing, count, age, germination):
    from pyhelios.types import int2, vec2
    _new_pa(B, seed)
    B.pids = list(B.pa.buildPlantCanopyFromLibrary(_vec3(B, center), vec2(*map(float, spacing)),
                                                   int2(*map(int, count)), float(age), float(germination)))
    return B.pids


@op("pa_accept_canopy", "heliades")
def _(B, seed, center, spacing, count, age, germination):
    _new_pa(B, seed)
    B.pids = B.pa.buildPlantCanopyFromLibrary(center, spacing, count, float(age), float(germination))
    _check_real_blades(B)
    return B.pids


def _staggered(B, build, advance):
    """Plant A at age 0, all plants +6 days, plant B at age 4, both +5 (one call), B alone +3, all +2."""
    a = build((0.0, 0.0, 0.0), 0.0)
    advance(6.0, None)
    b = build((0.4, -0.3, 0.0), 4.0)
    advance(5.0, [a, b])
    advance(3.0, [b])
    advance(2.0, None)
    B.pids = [a, b]
    return B.pids


@op("pa_accept_staggered", "helios")
def _(B, seed):
    _new_pa(B, seed)

    def advance(dt, pids):
        if pids is None:
            B.pa.advanceTime(float(dt))
        elif len(pids) == 1:
            B.pa.advanceTime(float(dt), plant_id=pids[0])
        else:
            B.pa.advanceTime(float(dt), plant_ids=list(pids))
    return _staggered(B, lambda p, age: B.pa.buildPlantInstanceFromLibrary(_vec3(B, p), age), advance)


@op("pa_accept_staggered", "heliades")
def _(B, seed):
    _new_pa(B, seed)

    def advance(dt, pids):
        if pids is None:
            B.pa.advanceTime(float(dt))
        elif len(pids) == 1:
            B.pa.advanceTime(float(dt), plant_id=pids[0])        # PyHelios keyword form
        else:
            B.pa.advanceTime(list(pids), float(dt))              # C++ form
    out = _staggered(B, lambda p, age: B.pa.buildPlantInstanceFromLibrary(p, age), advance)
    _check_real_blades(B)
    return out


# ---- observations -----------------------------------------------------------------------------------------

@op("pa_accept_xml", "helios")
def _(B, pid):
    d = tempfile.mkdtemp(prefix="heliades_pa_")
    path = os.path.join(d, "plant.xml")
    B.pa.writePlantStructureXML(int(pid), path)
    with open(path) as f:
        text = f.read()
    os.remove(path)
    os.rmdir(d)
    return text


@op("pa_accept_xml", "heliades")
def _(B, pid):
    return B.pa.writePlantStructureXML(int(pid))


@op("pa_accept_structure", "helios")
def _(B, pid):
    """The oracle side of ``plant_structure_dict``: its XML through the same parser, organ IDs from PyHelios."""
    from ....plantarchitecture.structure import xml_to_structure_dict
    pa, pid = B.pa, int(pid)
    d = xml_to_structure_dict(B.pa_accept_xml(pid))
    tubes = [int(o) for o in pa.getPlantInternodeObjectIDs(pid)]
    for sd, oid in zip(d["shoots"], tubes):
        sd["organs"] = {"internode": oid}
        for k, pd in enumerate(sd["phytomers"]):
            pd["organs"] = {"leaves": [[int(o) for o in pet] for pet in pa.getPhytomerLeafObjectIDs(pid, sd["ID"], k)]}
    d["organs"] = {"internode": tubes, "petiole": [int(o) for o in pa.getPlantPetioleObjectIDs(pid)],
                   "leaf": [int(o) for o in pa.getPlantLeafObjectIDs(pid)],
                   "peduncle": [int(o) for o in pa.getPlantPeduncleObjectIDs(pid)],
                   "flower": [int(o) for o in pa.getPlantFlowerObjectIDs(pid)],
                   "fruit": [int(o) for o in pa.getPlantFruitObjectIDs(pid)]}
    return d


@op("pa_accept_structure", "heliades")
def _(B, pid):
    d = B.pa.plant_structure_dict(int(pid))
    for sd in d["shoots"]:                       # keep what the oracle side can observe
        for pd in sd["phytomers"]:
            pd["organs"] = {"leaves": pd["organs"]["leaves"]}
    return d


@op("pa_accept_attrs", "helios")
def _(B):
    from ...backends import _all_ids_including_hidden
    from ...compare import texture_key
    ctx = B.ctx
    uu = _all_ids_including_hidden(ctx.getPrimitiveCount(), ctx.doesPrimitiveExist, 0)
    labels = ctx.getPrimitiveMaterialLabel(uu) if uu else []
    mats = {}
    for m in sorted(set(labels)):
        c = ctx.getMaterialColor(m)
        mats[m] = (c.r, c.g, c.b, c.a, texture_key(ctx.getMaterialTexture(m)), int(ctx.getMaterialTwosidedFlag(m)))
    prims = [(u, m, [(lab, ctx.getPrimitiveData(u, lab)) for lab in sorted(ctx.listPrimitiveData(u))])
             for u, m in zip(uu, labels)]
    oids = _all_ids_including_hidden(ctx.getObjectCount(), ctx.doesObjectExist, 1)
    objs = [(o, [(lab, int(ctx.getObjectDataType(o, lab)), ctx.getObjectData(o, lab)) for lab in sorted(ctx.listObjectData(o))])
            for o in oids]
    return _encode_attrs(prims, mats, objs)


@op("pa_accept_attrs", "heliades")
def _(B):
    from ...compare import texture_key
    sc = B.scene
    uu = sc.all_uuids(include_hidden=True)
    labels = [sc.get_primitive_material_label(u) for u in uu]
    mats = {}
    for m in sorted(set(labels)):
        c = sc.get_material_color(m)
        mats[m] = (*[float(x) for x in c], texture_key(sc.get_material_texture(m)), int(sc.get_material_twosided_flag(m)))
    prims = [(u, m, [(lab, sc.get_primitive_data(u, lab)) for lab in sorted(sc.list_primitive_data(u))])
             for u, m in zip(uu, labels)]
    objs = [(o, [(lab, int(sc.get_object_data_type(o, lab)), sc.get_object_data(o, lab)) for lab in sorted(sc.list_object_data(o))])
            for o in sc.all_object_ids(include_hidden=True)]
    return _encode_attrs(prims, mats, objs)


def _encode_attrs(prims, mats, objs) -> list:
    out = [-20.0, float(len(mats))]
    for m in sorted(mats):
        out += [_code(m)] + [float(f32(x)) for x in mats[m][:4]] + [_code(mats[m][4]), float(mats[m][5])]
    out += [-21.0, float(len(prims))]
    for u, m, data in prims:
        out += [float(u), _code(m), float(len(data))]
        for lab, val in data:
            out.append(_code(lab))
            flatten(val, out)
    out += [-22.0, float(len(objs))]
    for o, data in objs:
        out += [float(o), float(len(data))]
        for lab, typ, val in data:
            out += [_code(lab), float(typ)]
            flatten(val, out)
    return out


@op("pa_accept_api", "helios")
def _(B, pids):
    pa, ctx = B.pa, B.ctx

    def by_label(uuids, label):
        # Context::filterPrimitivesByData: a swap-and-pop permutation of the kept UUIDs, not a stable filter
        return [int(u) for u in ctx.filterPrimitivesByData([int(u) for u in uuids], "object_label", label)] if uuids else []

    obs = {"plants": [int(p) for p in pa.getAllPlantIDs()]}
    for pid in pids:
        obs[pid] = _api_common(pa, pid, by_label, lambda oid: pa.getLeafBladeArea(oid))
    obs["canopy"] = _api_canopy(pa, by_label)
    return obs


@op("pa_accept_api", "heliades")
def _(B, pids):
    pa, sc = B.pa, B.scene

    def by_label(uuids, label):
        return sc.filter_primitives_by_data(uuids, "object_label", label)

    obs = {"plants": pa.getAllPlantIDs()}
    for pid in pids:
        obs[pid] = _api_common(pa, pid, by_label, pa.getLeafBladeArea)
        # the Heliades-only object_label getters must agree with the generic filter
        for label in ("leaf", "petiolule", "petiole", "shoot"):
            assert pa.getPlantUUIDsByObjectLabel(pid, label) == by_label(pa.getAllPlantUUIDs(pid), label)
            assert (pa.getPlantUUIDsByObjectLabel(pid, label, include_hidden=True)
                    == by_label(pa.getAllPlantUUIDs(pid, True), label))
        # and the snake_case aliases are the same functions
        assert pa.get_plant_leaf_area(pid) == pa.getPlantLeafArea(pid)
        assert pa.get_all_plant_uuids(pid, True) == pa.getAllPlantUUIDs(pid, True)
    obs["canopy"] = _api_canopy(pa, by_label)
    for label in ("leaf", "petiolule"):
        assert pa.getAllUUIDsByObjectLabel(label) == by_label(pa.getAllUUIDs(), label)
    return obs


def _api_canopy(pa, by_label):
    allu = [int(u) for u in pa.getAllUUIDs()]
    return {"all_uuids": allu, "leaf_uuids": [int(u) for u in pa.getAllLeafUUIDs()],
            "internode_uuids": [int(u) for u in pa.getAllInternodeUUIDs()],
            "petiole_uuids": [int(u) for u in pa.getAllPetioleUUIDs()],
            "peduncle_uuids": [int(u) for u in pa.getAllPeduncleUUIDs()],
            "flower_uuids": [int(u) for u in pa.getAllFlowerUUIDs()],
            "fruit_uuids": [int(u) for u in pa.getAllFruitUUIDs()],
            "object_ids": [int(o) for o in pa.getAllObjectIDs()],
            "by_label": {lab: [int(u) for u in by_label(allu, lab)] for lab in ("leaf", "petiolule", "petiole", "shoot")}}


def _api_common(pa, pid, by_label, blade_area):
    """The same calls on PyHelios and on Heliades (identical names and argument orders)."""
    pid = int(pid)
    shoots = [int(s) for s in pa.getAllShootIDs(pid)]
    o = {
        "age": pa.getPlantAge(pid), "max_age": pa.getPlantMaxAge(pid), "name": pa.getPlantName(pid),
        "height": pa.getPlantHeight(pid), "stem_height": pa.getPlantStemHeight(pid),
        "leaf_area": pa.getPlantLeafArea(pid), "leaf_count": int(pa.getPlantLeafCount(pid)),
        "leaf_areas": list(pa.getPlantLeafAreas(pid)), "internode_lengths": list(pa.getPlantInternodeLengths(pid)),
        "leaf_bases": list(pa.getPlantLeafBases(pid)), "base": pa.getPlantBasePosition(pid),
        "dormant": bool(pa.isPlantDormant(pid)), "shoots": shoots,
        "by_rank": {int(k): [int(x) for x in v] for k, v in pa.getShootIDsByRank(pid).items()},
        "hierarchy": {int(k): [int(x) for x in v] for k, v in pa.getShootHierarchyMap(pid).items()},
        "terminal": [int(x) for x in pa.getTerminalShootIDs(pid)],
        "object_ids": [int(x) for x in pa.getAllPlantObjectIDs(pid)],
        "uuids": [int(x) for x in pa.getAllPlantUUIDs(pid)],
        "uuids_hidden": [int(x) for x in pa.getAllPlantUUIDs(pid, True)],
        "internode_ids": [int(x) for x in pa.getPlantInternodeObjectIDs(pid)],
        "internode_ids_mainstem": [int(x) for x in pa.getPlantInternodeObjectIDs(pid, "mainstem")],
        "petiole_ids": [int(x) for x in pa.getPlantPetioleObjectIDs(pid)],
        "leaf_ids": [int(x) for x in pa.getPlantLeafObjectIDs(pid)],
        "peduncle_ids": [int(x) for x in pa.getPlantPeduncleObjectIDs(pid)],
        "flower_ids": [int(x) for x in pa.getPlantFlowerObjectIDs(pid)],
        "fruit_ids": [int(x) for x in pa.getPlantFruitObjectIDs(pid)],
    }
    o["blade_areas"] = [blade_area(x) for x in o["leaf_ids"][:50]]
    o["by_label"] = {lab: [int(u) for u in by_label(o["uuids"], lab)] for lab in ("leaf", "petiolule", "petiole", "shoot")}
    per_shoot = []
    for sid in shoots:
        n = int(pa.getShootNodeCount(pid, sid))
        s = {"nodes": n, "parent": int(pa.getParentShootID(pid, sid)), "rank": int(pa.getShootRank(pid, sid)),
             "depth": int(pa.getShootDepth(pid, sid)), "path": [int(x) for x in pa.getPathToRoot(pid, sid)],
             "children": [int(x) for x in pa.getChildShootIDs(pid, sid)],
             "descendants": [int(x) for x in pa.getAllDescendantShootIDs(pid, sid)],
             "pruned": bool(pa.isShootPruned(pid, sid)), "prescribed": bool(pa.isShootGeometryPrescribed(pid, sid)),
             "vertices": [tuple(f32(c) for c in v) for v in pa.getShootInternodeVertices(pid, sid)],
             "radii": [f32(r) for r in pa.getShootInternodeRadii(pid, sid)]}
        phys = []
        for k in sorted({0, n // 2, n - 1}) if n else []:
            leaves = pa.getPhytomerLeafObjectIDs(pid, sid, k)
            p = {"age": pa.getPhytomerAge(pid, sid, k), "length": pa.getInternodeLength(pid, sid, k),
                 "radius": pa.getInternodeRadius(pid, sid, k),
                 "nodes": list(pa.getInternodeNodePositions(pid, sid, k)),
                 "axis": [pa.getInternodeAxisVector(pid, sid, k, f) for f in (0.0, 0.5, 1.0)],
                 "leaves": [[int(x) for x in pet] for pet in leaves],
                 "petiole_length_mean": pa.getPetioleLength(pid, sid, k)}
            if leaves:
                p["petiole"] = {"vertices": list(pa.getPetioleVertices(pid, sid, k, 0)),
                                "radii": [f32(r) for r in pa.getPetioleRadii(pid, sid, k, 0)],
                                "length": pa.getPetioleLength(pid, sid, k, 0),
                                "axis": [pa.getPetioleAxisVector(pid, sid, k, f, 0) for f in (0.0, 0.5, 1.0)],
                                "leaf_bases": [pa.getLeafBasePosition(pid, sid, k, 0, l) for l in range(len(leaves[0]))]}
            phys.append(p)
        s["phytomers"] = phys
        per_shoot.append(s)
    o["per_shoot"] = per_shoot
    return o
