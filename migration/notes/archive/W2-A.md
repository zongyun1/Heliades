# W2-A: `Context::loadOBJ` and polymesh topology

Baseline: Helios 1.3.98 (`core/src/Context_fileIO.cpp` loadOBJ/loadMTL, `Context_object.cpp` Polymesh,
`Context.cpp` polymesh accessors, `global.cpp` string/rotation helpers).

## What was ported

- `heliades/core/obj.py`: the OBJ/MTL readers, the libstdc++ `>>` / `getline` semantics they rely
  on (`CStream`), `parse_float` (libc `strtod`, then `(float)`, ERANGE rejected), `parse_int`
  (`strtol`), `string2vec2/3`, `string2RGBcolor`, `trim_whitespace`, `getFilePath`, `rotatePoint`,
  `calculateTriangleArea`, and the scale / up-axis rules.
- `heliades/core/polymesh.py`: `Polymesh::setTopology` (stored in the object-local frame through
  `makeNormalMatrix`), `getVertices`, `getVertexNormals`, `computeVertexNormals` (crease grouping
  and vertex splitting), `deg2rad`, `acos_safe` (libm `acosf`).
- `heliades/core/fileio.py` (`FileIOMixin`):
  - `load_obj`: all four overloads. `height=` gives scale (0,0,h), `scale=` takes a vec3, and
    neither gives the one-argument overload, whose default colour is blue.
  - the polymesh accessors: `set_polymesh_object_topology`,
    `compute_polymesh_object_vertex_normals`, `does_polymesh_object_have_vertex_normals`, and
    `polymesh_object_{vertices,faces,vertex_normals,vertex_uv,vertex_count,face_count,vertex_normal_source,primitive_uuid_for_face,face_index_for_primitive}`.
  - `primitive_material_label`, `is_primitive_texture_color_overridden`.

## Order and arithmetic

What loadOBJ reproduces exactly:
- **Face order.** Faces are grouped by material name in `std::map` order (bytewise), then kept in
  file order. Faces before any `usemtl` use material `"none"`.
- **Triangulation.** A face with k corners is fan-triangulated into triangles
  `(0, t-1, t)` for t = 2..k-1.
- **Dropped triangles.** A triangle is skipped when two of its *raw file* corners coincide
  (magnitude == 0). After the transform it is also skipped when its area is <= 1e-8f.
- **Transform, in this order:**
  1. YUP only: `rotatePointAboutLine` about +x by `float(0.5*M_PI)`, using double sin/cos;
  2. `rotatePoint(elevation, azimuth)`, using double sin/cos;
  3. `origin + v*scl`.
  - `XUP` is accepted but changes nothing (same as Helios).
- **Height mode.** `scl = h / (zmax - zmin)` over *all* `v` records. If that extent is <= 1e-6, it
  is `h` instead.
- **Mesh topology.**
  - Vertices are de-duplicated by (file vertex index, file normal index).
  - Authored normals are oriented like the vertices, divided by `scl`, then normalised with
    `vec3::operator/`, which multiplies by the reciprocal.
  - Without `vn` records the normals are computed at 180 degrees, giving `NORMAL_SOURCE_COMPUTED`.
- **Material effect on primitives.**
  - Colour becomes (Kd or default, 1).
  - The texture is the MTL path as resolved: as given if it exists relative to the cwd, otherwise
    `filebase + path`.
  - map_d alone sets texture-colour override.
  - Labels get a `_k` suffix on collision with an existing material.
  - `object_label` primitive data is the `o` name of the face's *first* vertex.

## Parity cases (all 0 ulp / exact, `python -m heliades.parity fileio`)

| case | tier | seeds | what |
|---|---|---|---|
| `fileio.obj.tomato` / `_topology` | ulp(0) / exact | 6 | seed 0 = the AssetLibrary call `(0,0,0), 0.75, nullrotation, black, "ZUP"`; other seeds random overload/origin/height/rotation/up axis |
| `fileio.obj.assets` / `_topology` | ulp(0) / exact | 6 | every plantarchitecture OBJ (51 incl. Cucumber; textured leaves, map_d-only Bougainvillea, OliveFlower vn), random overloads, a second load per seed (label collisions) |
| `fileio.obj.synthetic` / `_topology` | ulp(0) / exact | 10 | runtime OBJ+MTL+PNG: n-gons to 32 (+1 re-read), all `v/vt/vn` forms, negative indices, coincident corners, sub-1e-8 triangle, `none`/unknown material, duplicate newmtl, 4-value Kd, names with spaces, relative + absolute map_Kd, map_Kd+map_d, map_d with options, CR and trailing blanks, hex/exp/+ numbers, extra `v` columns, no final newline, loaded twice |
| `fileio.obj.errors` | exact | 1 | 21 small files: which raise, UUID counts, then the resulting Context (partial state after errors) |
| `fileio.polymesh.normals` | exact | 6 | crease 0..180 on loaded meshes, after translate/rotate/non-uniform scale and on `copyObject` copies |
| `fileio.polymesh.set_topology` | exact | 8 | `addPolymeshObject` + `setPolymeshObjectTopology` (with/without normals, UV, after transforms; unreferenced vertex; opposite faces) + recompute |

Snapshots compare vertices, area, normal, colour, UV, texture basename, solid fraction, parent, and
the object table (centre, transform). The `_topology` streams compare these exactly:
- texture-colour override, material label and `object_label` per primitive;
- per polymesh: normal source, has-normals, vertex/face counts, global vertices, faces, global
  normals, UVs, and the face -> UUID map.

Sweep outside the suite: all 51 assets x 3 overloads (ZUP height 0.75; YUP + rotation +
non-uniform scale; simple) = 189 runs, all at 0 ulp.

## Mutation check

Each mutation was applied to the source, then `python -m heliades.parity fileio` was run and the
source restored.

| mutation | caught by |
|---|---|
| fan winding `(0,t,t-1)` instead of `(0,t-1,t)` | 8 cases |
| area threshold `>= 1e-7` instead of `> 1e-8` | `obj.assets`, `obj.assets_topology` |
| normalise with true division instead of reciprocal multiply | 5 cases (every `_topology` / polymesh case) |
| materials in file order instead of `std::map` order | 8 cases |
| height scale `nextafter` by 1 ulp | 7 cases |
| `parse_float` correctly rounded (decimal -> float32) instead of `(float)stod` | `obj.synthetic`, `obj.synthetic_topology` (from the double-rounding trap numbers in the synthetic OBJ) |
| skip the trailing-whitespace re-read | none. **Equivalent mutant**: the repeated corner only ever produces a degenerate fan triangle, which is dropped |
| crease test with double `acos` instead of `acosf` | none. They differ by at most 1 ulp, and only a dot product within 1 ulp of the crease angle could tell them apart |

Also `tests/heliades/test_fileio.py::test_harness_catches_one_ulp_in_topology` (a 1-ulp normal change).

## Discrepancies / Helios and PyHelios issues

1. **PyHelios `Context.loadOBJ` drops `upaxis`** (pyhelios/Context.py:2490). The branch for
   "origin + height + rotation + color" is tested before the one that also takes upaxis, so
   `ctx.loadOBJ(f, origin=o, height=h, rotation=r, color=c, upaxis="YUP")` calls the C++ overload
   without upaxis and loads as ZUP. Repro:
   `ctx.loadOBJ(TomatoFlower, origin=vec3(0,0,0), height=0.75, rotation=SphericalCoord(1,0,0), color=RGBcolor(0,0,0), upaxis="YUP")`
   gives the same vertices as `upaxis="ZUP"` (checked). The parity ops call the ctypes wrappers directly.
2. **Two shipped assets do not load in Helios 1.3.98.** Heliades raises the same errors.
   - `RedbudLeaf.obj`: material `Material.002` has `map_Kd ../textures/RedbudLeaf.png`, but its
     faces have no `vt` references. This gives "specifies texture file ... but face has no texture
     coordinates".
   - `SugarBeetLeaf.obj`: `mtllib SugarbeetLeaf.mtl` (lowercase b), but the file on disk is
     `SugarBeetLeaf.mtl`. On a case-sensitive filesystem this gives "Could not locate asset file".
   - Neither is on the tomato path.
3. **Stream quirks kept on purpose** (they change geometry, so they are reproduced, not fixed):
   - A face line that ends in whitespace or `\r` (CRLF files) makes `stream >> tmp` fail with `tmp`
     unchanged. The last corner is pushed twice, so a 32-gon has 33 corners. The extra fan triangle
     is degenerate and gets dropped.
   - `o` as the last token with no newline: `getline` fails and the object name becomes `"o"`.
   - `f` as the last token with no newline: Helios parses `"f"` as an index and fails with "Face
     index ... must be an integer value".
   - `XUP` is validated but applies no rotation.
4. **Authored normals under non-uniform scale.** The comment in Helios says "inverse-transpose".
   The code computes `n.x / scl.x` and then normalises, which *is* the inverse-transpose for a
   diagonal scale. This is not a bug, just recorded.

## Not ported / follow-ups for other packages

- **Materials.** Primitive colour, texture and override come from the material in 1.3.98. Heliades
  has no material system, so `FileIOMixin` keeps a minimal label registry (`Scene._fileio_materials`:
  label -> colour/texture/override). It writes the material's effect onto the primitive: `color`,
  `texture_file`, `solid_fraction` when the texture changes, and the dynamic attributes
  `material_label` and `texture_color_overridden`. **Owner of `materials.py`:** please absorb the
  registry (`_mat_registry`, `_fileio_add_material`, `_fileio_assign_material`) and make
  `primitive_material_label` / `is_primitive_texture_color_overridden` read the real material.
  Labels must keep Helios's collision rule (`name_1`, `name_2`, ...) and the `__` reserved prefix,
  and `__default__` must exist.
- **Normal and bump maps** (`norm`, `map_Bump`, `bump`; `classifyBumpImage`,
  `setMaterialNormalMap`): parsed and validated, then `NotImplementedError`. No plantarchitecture
  asset uses them.
- **Polymesh member deletion** (`Polymesh::onChildPrimitiveDeleted`, `removeFacesAndCompact`) is not
  ported. Faces refer to members by position in `CompoundObject.uuids`; this survives W1-B's
  deep-copying `copy_object` with no remap hook. Once the member list changes size, the getters
  raise `NotImplementedError`. Deleting a *whole* object (the tomato's `setFloralBudState` path)
  is fine.
- **Per-primitive transforms of polymesh members** (`Polymesh::syncVerticesFromPrimitive`) and
  `setPolymeshObjectVertices` are not ported. The tomato only uses object-level transforms on its
  OBJ prototypes.
- `writeOBJ`, `loadPLY` and `loadXML` are not ported.
