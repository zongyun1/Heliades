# SPDX-License-Identifier: LGPL-2.1-or-later
"""W2-D parity cases: tube editing, hidden primitives/objects, typed data and filters, object area,
the material table and the helios:: scalar helpers.

Every case is 0 ulp / exact. Fuzz cases draw their op sequence with numpy's generator; IDs are always
picked from the lists the backend itself reports.
"""
from __future__ import annotations

import ctypes
import math

import numpy as np

from ...cases import _stem, case
from .cases_object_transforms import _World, _angle, _scale, _shift, _unit
from .cases_texture import ALPHA_TEXTURES, TEXTURES


def _tubes(B):
    return [o for o in B.object_ids() if B.object_type(o) == 2]


def _perturb(r, nodes, amp):
    nodes = np.asarray(nodes, float).reshape(-1, 3)
    k = r.random()
    if k < 0.1:
        return nodes                                    # same nodes (recompute from scratch)
    if k < 0.2:
        return nodes + r.uniform(-amp, amp, 3)          # rigid shift
    return nodes + r.normal(0, amp, nodes.shape)


def _radii(r, n):
    k = r.random()
    if k < 0.05:
        return np.zeros(n)                              # 0 radius: rings collapse onto the nodes
    if k < 0.1:
        return np.full(n, 5e-6)                         # below the 1e-5 clamp of setTubeNodes
    return r.uniform(0.001, 0.06, n)


def _tube_step(B, r, oid):
    """One random tube edit, using only what the backend reports about the tube."""
    k = r.integers(12)
    nodes = np.asarray(B.tube_nodes(oid), float).reshape(-1, 3)
    n = len(nodes)
    if k <= 2:
        d = nodes[-1] - nodes[-2]
        nd = np.linalg.norm(d)
        d = d / nd if nd > 0 else np.array([0, 0, 1.0])
        if r.random() < 0.15:
            d = r.normal(size=3)                        # kink
        step = 0.0 if r.random() < 0.05 else r.uniform(0.005, 0.08)
        rad = 0.0 if r.random() < 0.05 else r.uniform(0.001, 0.05)
        B.append_tube_segment(oid, nodes[-1] + step * d, rad, r.random(3))
    elif k <= 4:
        B.set_tube_nodes(oid, _perturb(r, nodes, 0.01))
    elif k <= 7:
        for _ in range(int(r.integers(1, 4))):          # repeated radius-only updates (path dependent)
            B.set_tube_radii(oid, _radii(r, n))
    elif k == 8:
        B.translate_object(oid, _shift(r))
    elif k == 9:
        B.rotate_object(oid, _angle(r), _unit(r), r.uniform(-1, 1, 3) if r.random() < 0.5 else None)
    elif k == 10:
        B.scale_object(oid, _scale(r), "point", r.uniform(-1, 1, 3))
    else:
        if r.random() < 0.5:
            B.scale_tube_girth(oid, r.uniform(0.5, 1.5))
        else:
            B.scale_tube_length(oid, r.uniform(0.5, 1.5))


def _tube_world(B, r):
    for _ in range(int(r.integers(1, 4))):
        n = int(r.integers(2, 6))
        B.add_tube(int(r.integers(3, 9)), _stem(r, n), r.uniform(0.002, 0.05, n),
                   r.random((n, 3)) if r.random() < 0.7 else None)


@case("core_gaps.tube.edit_fuzz", tier="ulp", max_ulp=0, fuzz=12)
def tube_edit_fuzz(B, seed):
    """appendTubeSegment / setTubeNodes / setTubeRadii (repeated) / girth / length interleaved with
    object transforms (which the next edit bakes) and copies."""
    r = np.random.default_rng(7000 + seed)
    _tube_world(B, r)
    for _ in range(40):
        tubes = _tubes(B)
        if r.random() < 0.06:
            B.copy_object(tubes[int(r.integers(len(tubes)))])
            continue
        _tube_step(B, r, tubes[int(r.integers(len(tubes)))])
    return B.snapshot()


@case("core_gaps.tube.stored_data", tier="ulp", max_ulp=0, fuzz=8)
def tube_stored_data(B, seed):
    """Nodes, radii (through the object transform), colors, volume and segment volumes after edits."""
    r = np.random.default_rng(7100 + seed)
    _tube_world(B, r)
    for _ in range(25):
        tubes = _tubes(B)
        _tube_step(B, r, tubes[int(r.integers(len(tubes)))])
    out = []
    for o in _tubes(B):
        out += B.tube_nodes(o) + B.tube_radii(o) + B.tube_info(o)
    for u in B.all_uuids():
        out += B.primitive_transform(u)
    return out


@case("core_gaps.tube.radius_only_repeat", tier="ulp", max_ulp=0, fuzz=8)
def tube_radius_only(B, seed):
    """N radius-only updates in a row: each rescales the current ring offsets, so the result depends on
    the whole history (and on the 0-radius updates that destroy the offset direction)."""
    r = np.random.default_rng(7200 + seed)
    n = int(r.integers(3, 8))
    oid = B.add_tube(int(r.integers(3, 9)), _stem(r, n), r.uniform(0.002, 0.05, n))
    out = []
    for k in range(int(r.integers(5, 25))):
        B.set_tube_radii(oid, _radii(r, n))
        if k % 4 == 3:
            out += B.tube_radii(oid)
    s = B.snapshot()
    return out + s.vertices.reshape(-1).tolist()


@case("core_gaps.tube.append_chain", tier="ulp", max_ulp=0, fuzz=8)
def tube_append_chain(B, seed):
    """The internode pattern: one tube grown by appendTubeSegment, then updateShootNodes-style
    setTubeRadii + setTubeNodes, repeated."""
    r = np.random.default_rng(7300 + seed)
    nodes = _stem(r, 2)
    oid = B.add_tube(7, nodes, [0.003, 0.003], [(0.3, 0.6, 0.1)] * 2)
    for _ in range(int(r.integers(4, 12))):
        p = np.asarray(B.tube_nodes(oid), float).reshape(-1, 3)
        d = p[-1] - p[-2]
        d = d / max(np.linalg.norm(d), 1e-9) + 0.3 * r.normal(size=3)
        B.append_tube_segment(oid, p[-1] + 0.03 * d / np.linalg.norm(d), 0.003, (0.3, 0.6, 0.1))
        if r.random() < 0.5:
            n = len(p) + 1
            rr = np.asarray(B.tube_radii(oid)) * r.uniform(1.0, 1.3, n)
            B.set_tube_radii(oid, rr)
            B.set_tube_nodes(oid, _perturb(r, B.tube_nodes(oid), 0.002))
    return B.snapshot()


@case("core_gaps.tube.textured_append", tier="ulp", max_ulp=0, fuzz=6)
def tube_textured_append(B, seed):
    """Textured appendTubeSegment. Opaque textures (JPEG) keep every triangle; with alpha textures
    zero-area triangles are deleted at creation, after which Helios's positional UUID bookkeeping
    fails on append (both backends raise; see migration/notes/W2-D.md)."""
    r = np.random.default_rng(7400 + seed)
    tex = [t for t in TEXTURES if t.endswith((".jpg", ".jpeg"))][seed % 2] if seed % 3 else \
        ALPHA_TEXTURES[seed % len(ALPHA_TEXTURES)]
    n = 3
    oid = B.add_tube_textured(6, _stem(r, n), r.uniform(0.01, 0.05, n), tex)
    for _ in range(4):
        p = np.asarray(B.tube_nodes(oid), float).reshape(-1, 3)
        u0 = r.uniform(0, 0.5)
        try:
            B.append_tube_segment_textured(oid, p[-1] + r.uniform(0.01, 0.05) * np.array([0, 0, 1.0]),
                                           r.uniform(0.01, 0.05), tex, (u0, r.uniform(u0, 1)))
        except Exception:            # a tube that lost zero-area triangles cannot be appended to in Helios
            break
    try:                             # ... nor re-radiused: the triangles before the failing index are updated
        B.set_tube_radii(oid, np.asarray(B.tube_radii(oid)) * 1.3)
    except Exception:
        pass
    return B.snapshot()


# ---- hidden ---------------------------------------------------------------------------------

def _hidden_world(B, r):
    W = _World(B, r)
    for _ in range(8):
        W.create()
    objs = B.object_ids()
    B.hide_object([o for o in objs if r.random() < 0.4])
    uu = B.all_uuids()
    B.hide_primitive([u for u in uu if r.random() < 0.15])
    for _ in range(12):
        k = r.integers(6)
        objs, uu = B.object_ids(), B.all_uuids()
        if k == 0 and objs:
            B.copy_object(objs[int(r.integers(len(objs)))])
        elif k == 1 and uu:
            B.copy_primitive(uu[int(r.integers(len(uu)))])
        elif k == 2:
            W.create()
        elif k == 3 and objs:
            B.translate_object(objs[int(r.integers(len(objs)))], _shift(r))
        elif k == 4 and uu:
            B.delete_primitive(uu[int(r.integers(len(uu)))])
        elif objs:
            B.hide_object(objs[int(r.integers(len(objs)))])
    return W


@case("core_gaps.hidden.visible_snapshot", tier="ulp", max_ulp=0, fuzz=8)
def hidden_visible_snapshot(B, seed):
    """What Helios's bulk getters see: hidden primitives and objects are left out."""
    _hidden_world(B, np.random.default_rng(8000 + seed))
    return B.snapshot()


@case("core_gaps.hidden.full_snapshot", tier="ulp", max_ulp=0, fuzz=8)
def hidden_full_snapshot(B, seed):
    """Every primitive and object, hidden included, with the hidden flags."""
    _hidden_world(B, np.random.default_rng(8000 + seed))
    return B.snapshot(include_hidden=True)


@case("core_gaps.hidden.queries", tier="exact", fuzz=6)
def hidden_queries(B, seed):
    """getAllUUIDs / getAllObjectIDs / counts / isHidden, including the prototype pattern: hide an
    object, copy it (the copy is visible), transform the copy; and show again."""
    r = np.random.default_rng(8100 + seed)
    _hidden_world(B, r)
    out = [-1] + B.all_uuids() + [-2] + B.object_ids() + [-3] + B.counts()
    full = B.snapshot(include_hidden=True)
    out += [-4] + [int(B.is_primitive_hidden(u)) for u in full.uuid]
    out += [-5] + [int(B.is_object_hidden(o)) for o in full.objects.oid]
    hidden_objs = [int(o) for o, h in zip(full.objects.oid, full.objects.hidden) if h]
    if hidden_objs:
        c = B.copy_object(hidden_objs[0])
        B.rotate_object(c, 0.7, "z")
        B.show_object(hidden_objs[-1])
    hp = [int(u) for u, h in zip(full.uuid, full.hidden) if h]
    if hp:
        B.show_primitive(hp[: len(hp) // 2 + 1])
    out += [-6] + B.all_uuids() + [-7] + B.object_ids() + [-8] + B.counts()
    return out


# ---- typed data -----------------------------------------------------------------------------

_DTYPES = ("int", "uint", "float", "double", "vec2", "vec3", "vec4", "int2", "int3", "int4", "string")
_WORDS = ("leaf", "petiole", "petiolule", "shoot", "peduncle", "", "fruit", "flower")


def _value(r, dtype):
    if dtype == "int":
        return int(r.integers(-5, 6)) if r.random() < 0.7 else int(r.integers(-2**31, 2**31))
    if dtype == "uint":
        return int(r.integers(0, 6)) if r.random() < 0.7 else int(r.integers(0, 2**32))
    if dtype == "float":
        return float(np.float32(r.choice([0.5, 1.0, 2.25]) if r.random() < 0.4 else r.normal()))
    if dtype == "double":
        return float(r.choice([0.5, 1.0, 2.25]) if r.random() < 0.4 else r.normal())
    if dtype.startswith("vec"):
        return [float(np.float32(x)) for x in r.normal(size=int(dtype[-1]))]
    if dtype.startswith("int"):
        return [int(x) for x in r.integers(-100, 100, int(dtype[-1]))]
    return str(_WORDS[int(r.integers(len(_WORDS)))])


def _data_case(B, r, kind):
    W = _World(B, r)
    for _ in range(6):
        W.create()
    ids = B.all_uuids() if kind == "primitive" else B.object_ids()
    labels = {f"L{k}": _DTYPES[int(r.integers(len(_DTYPES)))] for k in range(6)}
    labels["object_label"] = "string"
    out = []
    for lbl, dt in labels.items():                       # registers each label's type
        B.set_data(kind, ids[int(r.integers(len(ids)))], lbl, _value(r, dt), dt)
    for _ in range(40):
        lbl = list(labels)[int(r.integers(len(labels)))]
        dt = labels[lbl]
        sel = [ids[int(i)] for i in r.choice(len(ids), int(r.integers(1, min(6, len(ids)) + 1)), replace=False)]
        if dt in ("int", "uint", "float", "double") and r.random() < 0.2:
            # a value of another numeric type: Helios casts it to the label's registered type
            other = ("int", "uint", "float", "double")[int(r.integers(4))]
            B.set_data(kind, sel if r.random() < 0.5 else sel[0], lbl, abs(_value(r, other)) if other == "uint"
                       else _value(r, other), other)
        else:
            B.set_data(kind, sel if r.random() < 0.5 else sel[0], lbl, _value(r, dt), dt)
    for lbl, dt in labels.items():
        for i in ids:
            v = B.get_data(kind, i, lbl, dt)
            out += [-1.0] if v is None else [float(B.data_type(kind, i, lbl))] + v
    # filters, every comparator, values present and absent
    for lbl, dt in labels.items():
        if dt not in ("int", "uint", "float", "double", "string"):
            continue
        if kind == "primitive" and dt in ("uint", "double"):
            continue        # PyHelios exports no uint/double primitive filter (tested in tests/heliades only)
        probe = [ids[int(i)] for i in r.permutation(len(ids))]
        vals = [_value(r, dt) for _ in range(2)]
        first = B.get_data(kind, probe[0], lbl, dt)
        for v in vals:
            if dt == "string":
                out += [-2.0] + [float(x) for x in B.filter_data(kind, probe, lbl, v, dt)]
                continue
            for comp in ("==", ">", "<", ">=", "<="):
                out += [-3.0] + [float(x) for x in B.filter_data(kind, probe, lbl, v, dt, comp)]
        if dt == "string":
            for w in ("leaf", "petiolule"):
                out += [-4.0] + [float(x) for x in B.filter_data(kind, probe, lbl, w, dt)]
        del first
    # a filter with a mismatched value type keeps nothing
    lbl = "object_label"
    out += [-5.0] + [float(x) for x in B.filter_data(kind, ids, lbl, 1.0, "float", "==")]
    return out


@case("core_gaps.data.primitive", tier="exact", fuzz=8)
def data_primitive(B, seed):
    """setPrimitiveData (every type, one UUID or a list, numeric casts) -> get / type, then
    filterPrimitivesByData with every comparator (output order is Helios's swap-and-pop order); data
    survives copyPrimitive and goes with deletePrimitive."""
    r = np.random.default_rng(9000 + seed)
    out = _data_case(B, r, "primitive")
    uu = B.all_uuids()
    c = B.copy_primitive(uu[: 3])
    B.delete_primitive(uu[3:5])
    for u in B.all_uuids():
        v = B.get_data("primitive", u, "object_label", "string")
        out += [-6.0, float(u)] + ([-1.0] if v is None else v)
    del c
    return out


@case("core_gaps.data.object", tier="exact", fuzz=8)
def data_object(B, seed):
    """setObjectData / getObjectData / doesObjectDataExist / filterObjectsByData, and copyObject's
    copy of the object data."""
    r = np.random.default_rng(9100 + seed)
    out = _data_case(B, r, "object")
    objs = B.object_ids()
    B.copy_object(objs[: 2])
    B.delete_object(objs[-1])
    for o in B.object_ids():
        v = B.get_data("object", o, "object_label", "string")
        out += [-6.0, float(o)] + ([-1.0] if v is None else v)
    return out


# ---- area -----------------------------------------------------------------------------------

@case("core_gaps.object_area", tier="ulp", max_ulp=0, fuzz=8)
def object_area(B, seed):
    """getObjectArea (double accumulation) and getPrimitiveArea, textured tiles included, hidden
    objects included."""
    r = np.random.default_rng(9200 + seed)
    W = _World(B, r)
    for _ in range(8):
        W.create()
    tex = ALPHA_TEXTURES[seed % len(ALPHA_TEXTURES)]
    B.add_tile_textured(r.uniform(-1, 1, 3), (0.5, 0.3), (0.2, 0.4), (3, 2), tex)
    for _ in range(10):
        W.step()
    objs = B.object_ids()
    B.hide_object(objs[: len(objs) // 2])
    full = B.snapshot(include_hidden=True)
    return [B.object_area(o) for o in full.objects.oid] + [B.primitive_area(u) for u in full.uuid]


# ---- materials ------------------------------------------------------------------------------

@case("core_gaps.materials", tier="exact", fuzz=6)
def materials_case(B, seed):
    """Material labels / IDs (automatic labels hash as libstdc++ does), named materials, rename of an
    automatic material (nameOrganMaterials), assign to primitives/objects (petiolule pattern),
    setMaterialColor / setPrimitiveColor (copy-on-write of a shared named material) and the
    keepNamedTubeMaterial rule of appendTubeSegment. Colors are then compared via the snapshot."""
    r = np.random.default_rng(9300 + seed)
    palette = [tuple(np.float32(x) for x in r.random(3)) for _ in range(3)]
    uu = [B.add_triangle(r.normal(size=3), r.normal(size=3), r.normal(size=3), palette[int(r.integers(3))])
          for _ in range(10)]
    t1 = B.add_tube(5, _stem(r, 3), [0.01, 0.01, 0.01], [palette[0]] * 3)
    t2 = B.add_tube(5, _stem(r, 3), [0.01, 0.01, 0.01], [palette[1]] * 3)
    out = []
    labels = B.material_labels(B.all_uuids())
    out += [-1] + labels + B.material_ids(B.all_uuids())
    B.rename_material(B.material_labels([B.object_uuids(t1)[0]])[0], "petiole")    # every palette[0] primitive
    B.add_material("stem")
    B.set_material_color("stem", (*palette[2], 1.0))
    B.assign_material_to_object(t2, "stem")
    B.add_material("leaf")
    B.set_material_color("leaf", (0.2, 0.7, 0.1, 1.0))
    B.assign_material_to_primitive(uu[:4], "leaf")
    B.assign_material_to_primitive(uu[4], "petiole")
    # segments appended with the material's color join it; others get an automatic material
    B.append_tube_segment(t1, np.asarray(B.tube_nodes(t1)).reshape(-1, 3)[-1] + (0, 0, 0.05), 0.01, palette[0])
    B.append_tube_segment(t2, np.asarray(B.tube_nodes(t2)).reshape(-1, 3)[-1] + (0, 0, 0.05), 0.01, palette[0])
    B.append_tube_segment(t2, np.asarray(B.tube_nodes(t2)).reshape(-1, 3)[-1] + (0, 0, 0.05), 0.01, palette[2])
    B.set_material_color("leaf", (0.9, 0.1, 0.1, 1.0))
    B.set_primitive_color(uu[0], (0.1, 0.2, 0.3))        # shared named -> __copy_ material
    B.set_primitive_color(uu[5], (0.4, 0.5, 0.6))        # automatic -> automatic of the new color
    B.set_primitive_color(uu[4], (0.7, 0.7, 0.7))        # renamed from automatic -> always copied
    B.add_material("solo")
    B.assign_material_to_primitive(uu[6], "solo")
    B.set_primitive_color(uu[6], (0.3, 0.3, 0.9))        # sole user of a named material: edited in place
    B.copy_object(t2)
    B.delete_primitive(uu[7])
    out += [-2] + B.material_labels(B.all_uuids()) + B.material_ids(B.all_uuids())
    for lbl in ("petiole", "stem", "leaf", "solo", "missing"):
        q = B.material_query(lbl)
        out += [-3] + list(q)
    out += [-4] + B.list_materials()
    s = B.snapshot()
    return _encode(out) + s.color.reshape(-1).tolist()


def _encode(xs) -> list:
    """Numbers as they are, strings as their bytes, lists flattened, all exact in float32."""
    from .ops_core_gaps import encode_data
    out = []
    for x in xs:
        if isinstance(x, str):
            out += encode_data(x, "string")
        elif isinstance(x, (list, tuple)):
            out += [-9.0] + _encode(x)
        else:
            out.append(float(x))
    return out


@case("core_gaps.materials.texture", tier="ulp", max_ulp=0, fuzz=4)
def materials_texture(B, seed):
    """setMaterialTexture on a named material re-textures its users and recomputes their solid
    fraction (from their own UVs; full image for a patch without UVs)."""
    r = np.random.default_rng(9400 + seed)
    tex = ALPHA_TEXTURES[seed % len(ALPHA_TEXTURES)]
    uu = [B.add_patch(r.uniform(-1, 1, 3), r.uniform(0.1, 1, 2), (0.0, 0.0), r.random(3)) for _ in range(3)]
    uu += [B.add_triangle_textured(r.normal(size=3), r.normal(size=3), r.normal(size=3), tex,
                                   r.random(2), r.random(2), r.random(2)) for _ in range(3)]
    B.add_material("bark")
    B.assign_material_to_primitive(uu, "bark")
    B.set_material_texture("bark", ALPHA_TEXTURES[(seed + 1) % len(ALPHA_TEXTURES)])
    return B.snapshot()


# ---- helios:: scalar helpers ------------------------------------------------------------------

def _libhelios():
    from pyhelios.wrappers import UContextWrapper as W
    lib = W.helios_lib
    fns = {}
    for name, mangled in (("deg2rad", "_ZN6helios7deg2radEf"), ("rad2deg", "_ZN6helios7rad2degEf"),
                          ("acos_safe", "_ZN6helios9acos_safeEf"), ("asin_safe", "_ZN6helios9asin_safeEf")):
        f = getattr(lib, mangled)
        f.restype, f.argtypes = ctypes.c_float, [ctypes.c_float]
        fns[name] = f
    return fns


@case("core_gaps.vec.scalar_helpers", tier="exact", fuzz=4)
def scalar_helpers(B, seed):
    """deg2rad / rad2deg / acos_safe / asin_safe against the oracle libhelios.so symbols."""
    r = np.random.default_rng(9500 + seed)
    x = np.r_[r.uniform(-400, 400, 300), r.uniform(-1.2, 1.2, 300), [-1, 1, 0, -0.0, 1.0000001, -1.0000001, 90, 180]]
    x = x.astype(np.float32)
    if B.name == "helios":
        f = _libhelios()
        return [f[k](float(v)) for k in ("deg2rad", "rad2deg", "acos_safe", "asin_safe") for v in x]
    from ....core import vec as V
    return [float(getattr(V, k)(v)) for k in ("deg2rad", "rad2deg", "acos_safe", "asin_safe") for v in x]


del math
