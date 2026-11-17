# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the leaf blade prototypes (W3-A). All at 0 float32 ulp.

Seeds: the case seed ``s`` maps to Context seed ``s + 1`` (as in W2-C), so ``fuzz=8`` covers Context seeds 1..8.

The prototypes are compared alone: on both sides the snapshot is taken with ``include_hidden=True`` and cut
down to the hidden objects and their primitives. At age 0 these are exactly the leaf blade prototypes
(objects 3..37 for tomato; the hidden inflorescence prototypes are only built with the first peduncle),
so the cut does not depend on object positions. UUIDs and object IDs are compared as they are, so the
UUID/object footprint of what comes before the cache (internode and petiole tubes) and the UUIDs consumed
by the dropped transparent facets are checked too.

* ``plantarch.leafproto.tomato``: the 35 tomato prototypes (5 x 7, ~144 primitives each): vertices, normals,
  areas, colors, UVs, solid fractions, texture basenames, hidden flags, parent IDs, object members,
  centers and transforms.
* ``plantarch.leafproto.tomato_attrs``: per prototype primitive the material label, its color, texture
  basename and two-sided flag, the ``object_label`` primitive data and the data label list; per prototype
  object its object data labels and the polymesh face set (vertices, vertex normals, UVs, faces, face
  UUIDs, normal source). Strings are encoded as their bytes.
* ``plantarch.tomato.age0_full`` / ``age0_full_attrs``: acceptance, the whole age-0 tomato
  (``buildPlantInstanceFromLibrary``) with every primitive and object, hidden or not, and the attributes of
  every polymesh object (prototypes and the 7 leaflet copies).
* ``plantarch.leafproto.fuzz`` / ``fuzz_attrs``: the mainstem leaf prototype redefined through
  ``getCurrentShootParameters``/``defineShootType`` (PyHelios JSON on Helios, the ported structs on Heliades):
  curvature (distributed, with exponents below 1), lateral curvature, roll, fold, petiole roll, wave,
  subdivisions, aspect ratio (sometimes distributed, so Ny changes from blade to blade), unique prototypes,
  leaflet count, petiolule on/off and length, leaf offset, petiole radius/taper and leaflet scales.
"""
from __future__ import annotations

import numpy as np

from ...cases import case
from ...snapshot import ObjectTable, Snapshot

f32 = np.float32


def _seed(s):
    return s + 1


def prototype_subset(snap: Snapshot) -> Snapshot:
    """The hidden objects of a snapshot and their member primitives."""
    O = snap.objects
    oids = O.oid[O.hidden]
    keep = np.isin(snap.parent, oids) & snap.hidden
    ok = O.hidden
    objs = ObjectTable(O.oid[ok], O.type[ok], [u for u, k in zip(O.uuids, ok) if k], O.center[ok], O.transform[ok],
                       O.hidden[ok])
    return Snapshot(snap.uuid[keep], snap.type[keep], snap.vertices[keep], snap.nverts[keep], snap.area[keep],
                    snap.normal[keep], snap.color[keep], snap.parent[keep], objs, snap.uv[keep], snap.nuv[keep],
                    [t for t, k in zip(snap.texture, keep) if k], snap.solid_fraction[keep], snap.hidden[keep])


def _encode_str(out, s):
    b = str(s).encode()
    out.append(float(len(b)))
    out.extend(float(c) for c in b)


def encode_attrs(attrs) -> list:
    out = []
    for a in attrs:
        out += [float(a["oid"]), float(len(a["prims"]))]
        for p in a["prims"]:
            _encode_str(out, p["material"])
            out += [float(f32(c)) for c in p["color"]]
            _encode_str(out, p["mtex"])
            out.append(float(p["twosided"]))
            _encode_str(out, p["label"])
            _encode_str(out, ",".join(p["data"]))
        _encode_str(out, ",".join(a["objdata"]))
        for name in ("vertices", "normals", "uv"):
            out.append(float(len(a[name])))
            out.extend(float(f32(x)) for v in a[name] for x in v)
        out.append(float(len(a["faces"])))
        out.extend(float(i) for f in a["faces"] for i in f)
        out.extend(float(u) for u in a["face_uuids"])
        out.append(float(a["normal_source"]))
    return out


def _build(B, s, spec=None):
    B.seed(_seed(s))
    B.plantarch_new()
    B.plantarch_load("tomato")
    if spec:
        B.plantarch_leafproto_redefine("mainstem", spec)
    B.plantarch_leafproto_build()
    return prototype_subset(B.plantarch_leafproto_snapshot())


def _attrs(B, sub):
    return encode_attrs(B.plantarch_leafproto_attrs([int(o) for o in sub.objects.oid]))


@case("plantarch.leafproto.tomato", tier="ulp", max_ulp=0, fuzz=8)
def leafproto_tomato(B, s):
    return _build(B, s)


@case("plantarch.tomato.age0_full", tier="ulp", max_ulp=0, fuzz=8)
def tomato_age0_full(B, s):
    """Acceptance: the whole age-0 tomato, every primitive and object, hidden prototypes included."""
    B.seed(_seed(s))
    B.plantarch_new()
    B.plantarch_load("tomato")
    B.plantarch_leafproto_build()
    return B.plantarch_leafproto_snapshot()


@case("plantarch.tomato.age0_full_attrs", tier="exact", fuzz=8)
def tomato_age0_full_attrs(B, s):
    """Material label / color / texture / two-sided flag and primitive data of every primitive, object data and
    polymesh face sets of every polymesh object (blade prototypes and leaflet copies)."""
    B.seed(_seed(s))
    B.plantarch_new()
    B.plantarch_load("tomato")
    B.plantarch_leafproto_build()
    snap = B.plantarch_leafproto_snapshot()
    polys = [int(o) for o, t in zip(snap.objects.oid, snap.objects.type) if t == 5]
    return encode_attrs(B.plantarch_leafproto_attrs(polys))


@case("plantarch.leafproto.tomato_attrs", tier="exact", fuzz=8)
def leafproto_tomato_attrs(B, s):
    return _attrs(B, _build(B, s))


LP = "phytomer_parameters.leaf.prototype."


def fuzz_spec(s: int) -> dict:
    """Random leaf prototype redefinition for fuzz seed ``s`` (identical on both backends)."""
    r = np.random.default_rng(9100 + s)
    u = lambda a, b: float(r.uniform(a, b))           # noqa: E731
    spec = {}
    if r.random() < 0.8:
        a = u(-0.8, 0.5)
        spec[LP + "longitudinal_curvature"] = ("uniform", (a, a + u(0.0, 0.4)))
    else:
        spec[LP + "longitudinal_curvature"] = ("constant", (0.0,))
    spec[LP + "longitudinal_curvature_exponent"] = ("constant", (u(0.5, 6.0),))
    spec[LP + "lateral_curvature"] = ("constant", (0.0 if r.random() < 0.25 else u(-0.6, 0.6),))
    if r.random() < 0.4:
        spec[LP + "roll_fraction"] = ("constant", (0.0,))
    elif r.random() < 0.5:
        spec[LP + "roll_fraction"] = ("uniform", (0.0, u(0.05, 0.6)))
    else:
        spec[LP + "roll_fraction"] = ("constant", (u(0.0, 0.7),))
    spec[LP + "midrib_fold_fraction"] = ("constant", (u(0.0, 0.4),))
    spec[LP + "petiole_roll"] = ("constant", (0.0 if r.random() < 0.5 else u(-0.05, 0.05),))
    if r.random() < 0.75:
        spec[LP + "wave_period"] = ("constant", (u(0.1, 0.6),))
        spec[LP + "wave_amplitude"] = ("uniform", (0.0, u(0.01, 0.15))) if r.random() < 0.4 else ("constant", (u(0.0, 0.15),))
    else:
        spec[LP + "wave_period"] = ("constant", (0.0,))
    if r.random() < 0.3:
        a = u(0.2, 1.0)
        spec[LP + "leaf_aspect_ratio"] = ("uniform", (a, a + u(0.05, 0.6)))
    else:
        spec[LP + "leaf_aspect_ratio"] = ("constant", (u(0.2, 1.5),))
    spec[LP + "subdivisions"] = int(r.integers(1, 11))
    spec[LP + "unique_prototypes"] = int(r.integers(1, 6))
    spec[LP + "build_petiolule"] = bool(r.random() < 0.8)
    spec[LP + "petiolule_length"] = {0: u(0.02, 0.15)}
    if r.random() < 0.3:
        spec[LP + "leaf_offset"] = (u(-0.05, 0.05), u(-0.05, 0.05), u(-0.05, 0.05))
    spec["phytomer_parameters.leaf.leaves_per_petiole"] = ("int", int(r.choice([1, 3, 5, 7, 7])))
    spec["phytomer_parameters.petiole.taper"] = ("constant", (u(0.0, 0.9),))
    spec["phytomer_parameters.petiole.radius"] = ("uniform", (0.001, u(0.0015, 0.004)))
    spec["phytomer_parameters.leaf.prototype_scale"] = ("uniform", (0.08, u(0.09, 0.2)))
    return spec


_TEX = "plugins/plantarchitecture/assets/textures/"
_FLAT = {LP + "lateral_curvature": ("constant", (0.0,)), LP + "roll_fraction": ("constant", (0.0,))}
EDGE_SPECS = [
    # pure midrib fold (closed-form branch, float*M_PI in double), no curvature, no wave
    {**_FLAT, LP + "longitudinal_curvature": ("constant", (0.0,)), LP + "wave_period": ("constant", (0.0,)),
     LP + "midrib_fold_fraction": ("constant", (0.35,))},
    # pure fold with wave and a curvature exponent below 1 (fmaxf clamp)
    {**_FLAT, LP + "longitudinal_curvature_exponent": ("constant", (0.4,)), LP + "wave_amplitude": ("constant", (0.12,))},
    # roll only (constant-curvature cross-section), long blade
    {LP + "lateral_curvature": ("constant", (0.0,)), LP + "roll_fraction": ("constant", (0.9,)),
     LP + "subdivisions": 9, LP + "leaf_aspect_ratio": ("constant", (0.8,))},
    # petiole roll, both signs, with lateral curvature
    {LP + "petiole_roll": ("constant", (0.04,)), LP + "lateral_curvature": ("constant", (0.5,))},
    {LP + "petiole_roll": ("constant", (-0.03,)), LP + "wave_period": ("constant", (0.0,))},
    # one subdivision, wide blade (odd Ny rounded up), no petiolule
    {LP + "subdivisions": 1, LP + "leaf_aspect_ratio": ("constant", (2.5,)), LP + "build_petiolule": False},
    # per-leaflet petiolule lengths and leaf offset
    {LP + "petiolule_length": {i: 0.03 + 0.01 * (i + 3) for i in range(-3, 4)}, LP + "leaf_offset": (0.01, -0.02, 0.005)},
    # every shape parameter distributed: private-stream draw order of the val()s and trailing resample()s
    {LP + "leaf_aspect_ratio": ("uniform", (0.4, 0.9)), LP + "midrib_fold_fraction": ("uniform", (0.0, 0.3)),
     LP + "longitudinal_curvature_exponent": ("uniform", (1.0, 5.0)), LP + "lateral_curvature": ("uniform", (-0.4, 0.4)),
     LP + "roll_fraction": ("uniform", (0.0, 0.3)), LP + "petiole_roll": ("uniform", (0.01, 0.03)),
     LP + "wave_period": ("uniform", (0.2, 0.5)), LP + "wave_amplitude": ("uniform", (0.02, 0.1))},
    # per-leaflet textures: one automatic material per texture, named "<base>", "<base>_1", ...
    {LP + "leaf_texture_file": {-3: _TEX + "TomatoLeaf.png", -2: _TEX + "CherryTomatoLeaf.png", -1: "TomatoLeaf_blade.png",
                                0: "TomatoLeaf_centered.png", 1: "TomatoLeaf_blade.png", 2: _TEX + "CherryTomatoLeaf.png",
                                3: _TEX + "TomatoLeaf.png"}},
    # intercalary leaflets and zero leaflet offset (stalk radius from the petiole tip for every leaflet)
    {"phytomer_parameters.leaf.intercalary_leaflet_scale": ("constant", (0.4,)),
     "phytomer_parameters.leaf.leaflet_offset": ("constant", (0.0,)), "phytomer_parameters.leaf.leaflet_scale": ("constant", (0.8,))},
]


@case("plantarch.leafproto.edges", tier="ulp", max_ulp=0, fuzz=len(EDGE_SPECS))
def leafproto_edges(B, s):
    return _build(B, s, EDGE_SPECS[s])


@case("plantarch.leafproto.edges_attrs", tier="exact", fuzz=len(EDGE_SPECS))
def leafproto_edges_attrs(B, s):
    return _attrs(B, _build(B, s, EDGE_SPECS[s]))


@case("plantarch.leafproto.fuzz", tier="ulp", max_ulp=0, fuzz=12)
def leafproto_fuzz(B, s):
    return _build(B, s, fuzz_spec(s))


@case("plantarch.leafproto.fuzz_attrs", tier="exact", fuzz=6)
def leafproto_fuzz_attrs(B, s):
    return _attrs(B, _build(B, s, fuzz_spec(s)))
