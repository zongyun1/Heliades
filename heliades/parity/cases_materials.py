# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the material table seen through loadOBJ (W2-E).

``core.materials.obj_interplay`` loads the tomato flower and fruit OBJs, other plantarchitecture
assets (textured, map_d-only texture-colour override, label collisions on ``Material.001``) and a
synthetic OBJ (faces before any ``usemtl``, unknown material names), sometimes into a Context that
already holds materials of the same names. It then runs a random sequence of material edits:
assign (primitives and objects), rename (named and automatic), addMaterial over an existing label,
set colour / texture / override / two-sided flag, setPrimitiveColor, override / use texture colour,
copy and delete primitives and objects, deleteMaterial (named and automatic), and further loads. The
comparison covers each primitive's material ID, label, override flag and texture, listMaterials in
Helios's order, getMaterialCount, existence of every label seen, and per material its colour,
texture, flags, reference count and users. It also covers the drawn appearance: RGBA colour, UVs,
solid fraction and vertices.
"""
from __future__ import annotations

import numpy as np

from .cases import case
from . import cases_fileio as CF          # module objects: the case modules import each other while loading
from . import cases_texture as CT

_OTHER_NAMES = ("TomatoLeaf.obj", "BougainvilleaFlower.obj", "AppleFruit.obj", "StrawberryFruit.obj")


def _other():
    return [CF.OBJ_DIR + f for f in _OTHER_NAMES]


def _textures():
    return [CF.LEAF_PNG] + CT.ALPHA_TEXTURES[:3]

_KD = [tuple(np.float32(x) for x in c) for c in ((0.8, 0.73, 0.04), (0.26, 0.38, 0.1), (0.4, 0.2, 0.08))]


def _pick(r, xs):
    return xs[int(r.integers(len(xs)))]


def _subset(r, xs, k_max=40):
    if not xs:
        return []
    k = int(r.integers(1, min(k_max, len(xs)) + 1))
    return sorted(int(x) for x in r.choice(np.asarray(xs), size=k, replace=False))


class _Run:
    def __init__(self, B, seed):
        self.B, self.r, self.seed = B, np.random.default_rng(7700 + seed), seed
        self.seen: list[str] = []
        self.n = 0

    def note(self, labels):
        for lbl in labels:
            if lbl not in self.seen:
                self.seen.append(lbl)

    def prims(self):
        return list(self.B.all_uuids())

    def named(self):
        return [m for m in self.B.material_list_raw() if not m.startswith("__")]

    def any_label(self):
        """A named material, or the (possibly automatic) material of a random primitive."""
        r, named, uu = self.r, self.named(), self.prims()
        if named and (r.random() < 0.6 or not uu):
            return _pick(r, named)
        if not uu:
            return None
        lbl = self.B.material_labels([_pick(r, uu)])[0]
        return None if lbl == "__default__" else lbl

    def auto_label(self):
        """The automatic material of a random primitive that has one (None if none does)."""
        uu = self.prims()
        autos = sorted({lbl for lbl in self.B.material_labels(uu) if lbl.startswith("__auto_")})
        return _pick(self.r, autos) if autos else None

    def add_colored(self):
        """A few triangles in the tomato MTL colours or a random one: automatic materials, some shared."""
        B, r = self.B, self.r
        for _ in range(int(r.integers(1, 5))):
            c = _pick(r, _KD) if r.random() < 0.6 else tuple(np.float32(x) for x in r.random(3))
            B.add_triangle(r.normal(size=3), r.normal(size=3), r.normal(size=3), c)

    def fresh(self, stem):
        self.n += 1
        return f"{stem}{self.n}"

    def load(self, f):
        CF._random_load(self.B, self.r, f)
        self.note(self.B.material_labels(self.prims()))

    def step(self):
        B, r = self.B, self.r
        k = int(r.integers(18))
        uu = self.prims()
        if k == 0 and uu:
            lbl = self.any_label()
            if lbl:
                B.assign_material_to_primitive(_subset(r, uu), lbl)
        elif k == 1:
            oids, lbl = B.object_ids(), self.any_label()
            if oids and lbl:
                B.assign_material_to_object([_pick(r, oids)], lbl)
        elif k == 2:
            lbl = self.auto_label() if r.random() < 0.5 else self.any_label()      # nameOrganMaterials-style
            if lbl:
                new = self.fresh("renamed")
                B.rename_material(lbl, new)
                self.note([new])
                users = B.material_query(new)[-1]
                if users and r.random() < 0.6:          # a renamed automatic material is copied, not edited
                    B.set_primitive_color(_subset(r, users, 3), tuple(np.float32(x) for x in r.random(3)))
        elif k == 3 and uu:
            B.set_primitive_color(_subset(r, uu), tuple(np.float32(x) for x in r.random(3)))
        elif k == 4:
            lbl = self.any_label()
            if lbl:
                B.set_material_color(lbl, (*r.random(3), float(r.choice([1.0, r.random()]))))
        elif k == 5:
            lbl = self.any_label()
            if lbl:
                B.set_material_texture(lbl, _pick(r, _textures()) if r.random() < 0.8 else "")
        elif k == 6:
            lbl = self.any_label()
            if lbl:
                B.set_material_texture_color_override(lbl, bool(r.random() < 0.5))
        elif k == 7 and uu:
            B.override_primitive_texture_color(_subset(r, uu))
        elif k == 8 and uu:
            B.use_primitive_texture_color(_subset(r, uu))
        elif k == 9 and uu:
            B.copy_primitive(_subset(r, uu, 10))
        elif k == 10:
            oids = B.object_ids()
            if oids:
                B.copy_object([_pick(r, oids)])
        elif k == 11 and uu:
            B.delete_primitive(_subset(r, uu, 20))
        elif k == 12:
            oids = B.object_ids()
            if len(oids) > 2:
                B.delete_object([_pick(r, oids)])
        elif k == 13:
            lbl = self.any_label()
            if lbl:
                B.delete_material(lbl)
        elif k == 14:
            named = self.named()
            lbl = _pick(r, named) if named and r.random() < 0.7 else self.fresh("added")
            B.add_material(lbl)                                   # an existing label is overwritten in place
            self.note([lbl])
            if uu and r.random() < 0.7:
                B.assign_material_to_primitive(_subset(r, uu), lbl)
        elif k == 15:
            lbl = self.any_label()
            if lbl:
                B.set_material_twosided_flag(lbl, int(r.integers(0, 3)))
        elif k == 16 and r.random() < 0.4:
            self.load(_pick(r, CF.TOMATO + _other()))
        elif k == 17:
            self.add_colored()


def _stream(B, run: _Run) -> list:
    run.note(B.material_labels(run.prims()))
    out = B.material_state(list(run.seen))
    s = B.snapshot_rgba()
    out += [-5.0] + s.uuid.astype(float).tolist() + s.color.reshape(-1).tolist() + s.solid_fraction.tolist()
    out += s.nuv.astype(float).tolist() + s.uv.reshape(-1).tolist() + s.vertices.reshape(-1).tolist()
    return out


def setup(B, run: _Run, on_load=None):
    """Context materials before any load, earlier automatic materials, then the OBJ loads (random order)."""
    r = run.r
    # materials already in the Context: an MTL label collides with them, or catches faces outside any usemtl
    for lbl in ("Material.003", "Material.001", "none", "unknown_mat", "red"):
        if r.random() < 0.4:
            B.add_material(lbl)
            B.set_material_color(lbl, (*r.random(3), 1.0))
            run.note([lbl])
    # earlier primitives with tomato MTL colours: their automatic materials are reused by the loads
    B.add_patch(r.uniform(-1, 1, 3), (0.3, 0.2), (0.0, 0.0), _KD[1])
    run.add_colored()
    files = CF.TOMATO + [_pick(r, _other()), _pick(r, _other()), CF.synthetic_obj(run.seed)]
    for i in r.permutation(len(files)):
        run.load(files[int(i)])
        if on_load:
            on_load(files[int(i)])


N_STEPS = 40


@case("core.materials.obj_interplay", tier="exact", fuzz=8)
def obj_interplay(B, seed):
    run = _Run(B, seed)
    setup(B, run)
    out = _stream(B, run)
    for _ in range(N_STEPS):
        run.step()
    return out + [-9.0] + _stream(B, run)
