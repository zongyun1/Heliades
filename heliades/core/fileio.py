# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core (Context_fileIO.cpp, Context.cpp polymesh accessors; Copyright (C) 2016-2026 Brian Bailey).
"""File IO: loadOBJ (and later PLY/XML), plus the polymesh topology accessors it needs. Mixed into Scene.

Materials: loadOBJ goes through the Context material table (``MaterialsMixin``, materials.py) exactly
as Helios 1.3.98 does. Each MTL material becomes a named material (``addMaterial``, a ``_k`` suffix on a
label collision, then ``setMaterialColor`` / ``setMaterialTexture`` / ``setMaterialTextureColorOverride``).
Each triangle is created with its automatic material (``setPrimitiveColor`` + ``overridePrimitiveTextureColor``
first for a map_d-only material), and is then assigned its named material when that label exists in the
Context, so the automatic materials made on the way consume material IDs as in Helios.
"""
from __future__ import annotations

import numpy as np

from . import obj as O
from . import polymesh as PM
from ..migration_registry import ports

f32 = np.float32

_OBJECT_POLYMESH = 5
MIN_TRIANGLE_AREA_THRESHOLD = f32(1e-8)
DEFAULT_IMPORT_CREASE_ANGLE_DEGREES = f32(180)
_BLUE = (0.0, 0.0, 1.0)


class FileIOMixin:

    # ---- loadOBJ ---------------------------------------------------------------------------------
    @ports("Context::loadOBJ(const char*,bool)",
           "Context::loadOBJ(const char*,const vec3&,float,const SphericalCoord&,const RGBcolor&,bool)",
           "Context::loadOBJ(const char*,const vec3&,float,const SphericalCoord&,const RGBcolor&,const char*,bool)",
           "Context::loadOBJ(const char*,const vec3&,const vec3&,const SphericalCoord&,const RGBcolor&,const char*,bool)")
    def load_obj(self, filename: str, origin=(0.0, 0.0, 0.0), height=None, rotation=(0.0, 0.0), color=None,
                 upaxis: str = "ZUP", silent: bool = True, scale=None) -> list[int]:
        """Load a Wavefront OBJ. ``height`` (a float) or ``scale`` (a vec3) select the overload; neither
        means scale (0,0,0), i.e. unscaled. ``rotation`` is (elevation, azimuth) in radians.
        ``color`` is the default for faces without a material; ``None`` is the one-argument
        overload's RGB::blue. ``height=h`` is scale (0,0,h): the model's z extent becomes h."""
        if scale is None:
            scale = (0.0, 0.0, 0.0 if height is None else height)
        return self._load_obj(filename, origin, scale, rotation, _BLUE if color is None else color, upaxis, silent)

    def _load_obj(self, filename, origin, scale, rotation, default_color, upaxis, silent) -> list[int]:
        from .texture import get_file_extension, resolve_file_path
        if not silent:
            print(f"Reading OBJ file {filename}...", end="", flush=True)
        ext = get_file_extension(filename)
        if ext not in (".obj", ".OBJ"):
            raise ValueError(f"ERROR (Context::loadOBJ): File {filename} is not OBJ format.")
        if upaxis not in ("XUP", "YUP", "ZUP"):
            raise ValueError(f"ERROR (Context::loadOBJ): Up axis of {upaxis} is not valid.  Should be one of 'XUP', 'YUP', or 'ZUP'.")
        resolved = resolve_file_path(filename)
        dc = np.asarray(default_color, f32)[:3].copy()
        F = O.parse_obj(resolved, dc)
        scl = O.obj_scale(scale, F.boxmin, F.boxmax)
        origin = np.asarray(origin, f32).reshape(3)
        rotation = (f32(rotation[0]), f32(rotation[1]))

        # register the MTL materials (std::map order), renaming on collision
        mtl_to_ctx = {}
        for name in sorted(F.materials, key=O.cpp_key):
            mat = F.materials[name]
            ctx_name = name
            if self.does_material_exist(name):
                k = 1
                while True:
                    ctx_name = f"{name}_{k}"
                    k += 1
                    if not self.does_material_exist(ctx_name):
                        break
            self.add_material(ctx_name)
            self.set_material_color(ctx_name, np.array([mat.color[0], mat.color[1], mat.color[2], 1], f32))
            if mat.texture:
                self.set_material_texture(ctx_name, mat.texture)
            self.set_material_texture_color_override(ctx_name, bool(mat.texture_color_overridden))
            if mat.normal_map_kind != "none":
                if not mat.texture:
                    raise ValueError(f"ERROR (Context::loadOBJ): Material '{name}' has the normal or bump map '{mat.normal_map}' "
                                     "but no diffuse texture (map_Kd).")
                raise NotImplementedError("loadOBJ: normal/bump maps (setMaterialNormalMap / setMaterialBumpMap) are not ported")
            mtl_to_ctx[name] = ctx_name

        tris = self._obj_triangles(F, scl, origin, rotation, dc, upaxis, mtl_to_ctx)

        # second pass: create the primitives in order. Each one first gets the automatic material of its
        # appearance (for a map_d-only material: textured black, then its color, then overridden, which
        # are three automatic materials), then the named material if that label exists in the Context.
        # A face outside any usemtl ("none") or with an unknown name keeps the raw name, so it joins a
        # Context material that happens to carry that label, as in Helios.
        uuids = []
        for td in tris:
            if td["texture"]:
                u = self.add_triangle_textured(td["v0"], td["v1"], td["v2"], td["texture"], *td["uv"])
                if td["override"]:
                    self.set_primitive_color(u, td["color"])
                    self.override_primitive_texture_color(u)
            else:
                u = self.add_triangle(td["v0"], td["v1"], td["v2"], td["color"])
            uuids.append(u)
            if td["material"] and self.does_material_exist(td["material"]):
                self.assign_material_to_primitive(u, td["material"])
            if td["object"] != "none" and u in self.primitives:
                self.set_primitive_data(u, "object_label", td["object"])

        if uuids:
            self._obj_build_polymesh(F, tris, uuids, upaxis, rotation, scl)
        if not silent:
            print("done.")
        return uuids

    def _obj_triangles(self, F, scl, origin, rotation, dc, upaxis, mtl_to_ctx) -> list[dict]:
        """First pass of loadOBJ: fan-triangulate every face (materials in std::map order, faces in file
        order), drop corner-coincident and tiny triangles, and resolve UVs / normal indices."""
        out = []
        Vf = F.vertices
        for matname in sorted(F.face_inds, key=O.cpp_key):
            texture, color, override = "", dc, False
            if matname in F.materials:
                m = F.materials[matname]
                texture, color, override = m.texture, m.color, m.texture_color_overridden
            faces = F.face_inds[matname]
            tinds = F.texture_inds.get(matname, [])
            ninds = F.normal_inds.get(matname, [])
            ctx_mat = mtl_to_ctx.get(matname, matname)
            # gather fan triangles (i, t) of this material
            fi, ti, a, b, c = [], [], [], [], []
            for i, f in enumerate(faces):
                for t in range(2, len(f)):
                    fi.append(i); ti.append(t)
                    a.append(f[0] - 1); b.append(f[t - 1] - 1); c.append(f[t] - 1)
            if not fi:
                continue
            a, b, c = np.array(a), np.array(b), np.array(c)
            v0, v1, v2 = Vf[a].T, Vf[b].T, Vf[c].T             # (3, n) columns
            degenerate = (O._mag(v0 - v1) == 0) | (O._mag(v0 - v2) == 0) | (O._mag(v1 - v2) == 0)
            v0, v1, v2 = (O.orient(v, upaxis, rotation) for v in (v0, v1, v2))
            w0, w1, w2 = (np.array([origin[k] + v[k] * scl[k] for k in range(3)], f32) for v in (v0, v1, v2))
            area = O.triangle_area(w0, w1, w2)
            keep = ~degenerate & (area > MIN_TRIANGLE_AREA_THRESHOLD)
            for k in np.nonzero(keep)[0]:
                i, t = fi[k], ti[k]
                f = faces[i]
                nrm = ninds[i] if i < len(ninds) else []
                if len(nrm) == len(f):
                    nidx = (nrm[0] - 1, nrm[t - 1] - 1, nrm[t] - 1)
                else:
                    nidx = (-1, -1, -1)
                uv = None
                if texture:
                    tu = tinds[i] if i < len(tinds) else []
                    if tu and t < len(tu):
                        idx = (tu[0] - 1, tu[t - 1] - 1, tu[t] - 1)
                        nt = len(F.texture_uv)
                        if not all(0 <= j < nt for j in idx):
                            raise ValueError(f"ERROR (Context::loadOBJ): Invalid texture coordinate indices in face for material '{matname}'.")
                        uv = [F.texture_uv[j] for j in idx]
                    else:
                        raise ValueError(f"ERROR (Context::loadOBJ): Material '{matname}' specifies texture file '{texture}' but face has no "
                                         "texture coordinates. Either remove the texture from the material or add texture coordinates (vt) "
                                         "and face texture indices (f v1/vt1 v2/vt2 v3/vt3) to the OBJ file.")
                out.append(dict(v0=w0[:, k].copy(), v1=w1[:, k].copy(), v2=w2[:, k].copy(), texture=texture, uv=uv,
                                color=np.asarray(color, f32)[:3].copy(), override=override, material=ctx_mat,
                                object=F.objects[f[0] - 1], vidx=(int(a[k]), int(b[k]), int(c[k])), nidx=nidx))
        return out

    def _obj_build_polymesh(self, F, tris, uuids, upaxis, rotation, scl):
        """Group the triangles into a polymesh and attach the de-duplicated face set (vertex index,
        normal index) -> mesh vertex; generate normals at 180 degrees if the file had none."""
        file_normals = len(F.vertex_normals) > 0
        key_to_idx: dict = {}
        mv, mn, mf = [], [], []
        pending_n = []
        for td in tris:
            corners = (td["v0"], td["v1"], td["v2"])
            fc = []
            for c in range(3):
                key = (td["vidx"][c], td["nidx"][c])
                j = key_to_idx.get(key)
                if j is None:
                    j = len(mv)
                    key_to_idx[key] = j
                    mv.append(corners[c])
                    if file_normals:
                        ni = td["nidx"][c]
                        pending_n.append(F.vertex_normals[ni] if 0 <= ni < len(F.vertex_normals) else np.array([0, 0, 1], f32))
                fc.append(j)
            mf.append(fc)
        mesh_normals = np.zeros((0, 3), f32)
        if file_normals:
            N = np.array(pending_n, f32).T
            N = O.orient(N, upaxis, rotation)
            N = np.array([N[k] / scl[k] for k in range(3)], f32).T
            mesh_normals = PM._normalize_cols(N)
        oid = self.add_polymesh_object(uuids)
        PM.set_topology(self.objects[oid], np.array(mv, f32).reshape(-1, 3), np.array(mf, np.int64).reshape(-1, 3), uuids,
                        mesh_normals, (), PM.NORMAL_SOURCE_AUTHORED if file_normals else PM.NORMAL_SOURCE_NONE)
        if not file_normals:
            PM.compute_vertex_normals(self.objects[oid], DEFAULT_IMPORT_CREASE_ANGLE_DEGREES)
        return oid

    # ---- polymesh topology accessors (Context.cpp) -----------------------------------------------
    def _polymesh(self, oid: int, who: str):
        o = self.objects.get(int(oid))
        if o is None or o.type != _OBJECT_POLYMESH:
            raise ValueError(f"ERROR (Context::{who}): Object ID {oid} is not a Polymesh object.")
        return o

    @ports("Context::setPolymeshObjectTopology")
    def set_polymesh_object_topology(self, oid: int, vertices, faces, face_uuids, vertex_normals=(), vertex_uv=(),
                                     normal_source: int = PM.NORMAL_SOURCE_NONE):
        o = self._polymesh(oid, "setPolymeshObjectTopology")
        for u in face_uuids:
            if u not in self.primitives:
                raise ValueError(f"ERROR (Context::setPolymeshObjectTopology): Primitive UUID {u} does not exist in the Context.")
            if self.primitives[u].parent_object != oid:
                raise ValueError(f"ERROR (Context::setPolymeshObjectTopology): Primitive UUID {u} does not belong to object {oid}.")
        PM.set_topology(o, vertices, faces, face_uuids, vertex_normals, vertex_uv, normal_source)

    @ports("Context::computePolymeshObjectVertexNormals")
    def compute_polymesh_object_vertex_normals(self, oid: int, crease_angle_degrees: float):
        PM.compute_vertex_normals(self._polymesh(oid, "computePolymeshObjectVertexNormals"), crease_angle_degrees)

    @ports("Context::doesPolymeshObjectHaveVertexNormals")
    def does_polymesh_object_have_vertex_normals(self, oid: int) -> bool:
        return len(PM.topology(self._polymesh(oid, "doesPolymeshObjectHaveVertexNormals"))["normals"]) > 0

    @ports("Context::getPolymeshObjectVertexNormalSource")
    def polymesh_object_vertex_normal_source(self, oid: int) -> int:
        return PM.topology(self._polymesh(oid, "getPolymeshObjectVertexNormalSource"))["normal_source"]

    @ports("Context::getPolymeshObjectVertices")
    def polymesh_object_vertices(self, oid: int) -> np.ndarray:
        return PM.vertices_global(self._polymesh(oid, "getPolymeshObjectVertices"))

    @ports("Context::getPolymeshObjectFaces")
    def polymesh_object_faces(self, oid: int) -> np.ndarray:
        return PM.topology(self._polymesh(oid, "getPolymeshObjectFaces"))["faces"].copy()

    @ports("Context::getPolymeshObjectVertexNormals")
    def polymesh_object_vertex_normals(self, oid: int) -> np.ndarray:
        return PM.vertex_normals_global(self._polymesh(oid, "getPolymeshObjectVertexNormals"))

    @ports("Context::getPolymeshObjectVertexUV")
    def polymesh_object_vertex_uv(self, oid: int) -> np.ndarray:
        return PM.topology(self._polymesh(oid, "getPolymeshObjectVertexUV"))["uv"].copy()

    @ports("Context::getPolymeshObjectVertexCount")
    def polymesh_object_vertex_count(self, oid: int) -> int:
        return len(PM.topology(self._polymesh(oid, "getPolymeshObjectVertexCount"))["vertices"])

    @ports("Context::getPolymeshObjectFaceCount")
    def polymesh_object_face_count(self, oid: int) -> int:
        return len(PM.topology(self._polymesh(oid, "getPolymeshObjectFaceCount"))["faces"])

    @ports("Context::getPolymeshObjectPrimitiveUUIDForFace")
    def polymesh_object_primitive_uuid_for_face(self, oid: int, face_index: int) -> int:
        o = self._polymesh(oid, "getPolymeshObjectPrimitiveUUIDForFace")
        t = PM.topology(o)
        if not 0 <= face_index < len(t["faces"]):
            raise IndexError(f"ERROR (Polymesh::getPrimitiveUUIDForFace): Face index {face_index} is out of range.")
        return o.uuids[int(t["face_members"][face_index])]

    @ports("Context::getPolymeshObjectFaceIndexForPrimitive")
    def polymesh_object_face_index_for_primitive(self, oid: int, uuid: int) -> int:
        o = self._polymesh(oid, "getPolymeshObjectFaceIndexForPrimitive")
        t = PM.topology(o)
        members = list(o.uuids)
        hit = -1
        for f, m in enumerate(t["face_members"]):          # UUID_to_face: the last face wins
            if members[int(m)] == uuid:
                hit = f
        if hit < 0:
            raise ValueError(f"ERROR (Polymesh::getFaceIndexForPrimitive): Primitive UUID {uuid} does not correspond to a face of polymesh object {oid}.")
        return hit
