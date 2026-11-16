# W2-D core gaps: notes

Baseline Helios 1.3.98. Every W2-D parity case is **0 ulp / exact** on every seed.

## What was ported

| Helios | Heliades |
|---|---|
| `Context::appendTubeSegment` (color and texture overloads), `Tube::appendTubeSegment` | `Scene.append_tube_segment`, `append_tube_segment_textured` (`core/objects.py`, `TubeEditMixin`) |
| `Context::setTubeNodes` / `Tube::recomputeCrossSections`, `Context::setTubeRadii` | `set_tube_nodes`, `set_tube_radii` |
| `Tube::bakeTransform`, `Tube::updateTriangleVertices`, `helios::vecmult` | `_bake_tube_transform`, `_update_tube_triangles`, `_vecmult` |
| `scaleTubeGirth`, `scaleTubeLength`, `pruneTubeNodes` | `scale_tube_girth`, `scale_tube_length`, `prune_tube_nodes` |
| `getTubeObjectNodeCount/SubdivisionCount/NodeColors/Volume/SegmentVolume`, `Tube::getTriangleVertices` | `tube_object_*` (W1-B already had `getTubeObjectNodes/NodeRadii`) |
| `hidePrimitive`, `showPrimitive`, `isPrimitiveHidden`, `hideObject`, `showObject`, `isObjectHidden`, `getPrimitiveCount`, `getObjectCount`, hidden-aware `getAllUUIDs` / `getAllObjectIDs` | `core/scene.py`, `all_object_ids` in `core/objects.py` |
| typed `set/get/doesPrimitiveDataExist`, `getPrimitiveDataType/Size`, `clear/listPrimitiveData`, `filterPrimitivesByData` (float/double/int/uint/string), the same for object data, `copyObjectData`, the type registry and numeric casting | `core/data.py` (`DataMixin`) |
| `getObjectArea` (`CompoundObject::getArea`, double sum), `getObjectPrimitiveCount`, `doesObjectExist`, `doesPrimitiveExist` | `object_area`, ... |
| material table: automatic labels (`generateMaterialLabel`, libstdc++ `std::hash` reproduced), IDs, reference counts, `addMaterial`, `renameMaterial`, `deleteMaterial`, `set/getMaterialColor/Texture/TextureColorOverride/TwosidedFlag`, `assignMaterialToPrimitive/Object`, `getPrimitiveMaterialLabel/ID`, `getMaterial`, `getPrimitivesUsingMaterial`, `listMaterials`, `getMaterialCount`, `setPrimitiveColor` (`setPrimitiveAppearance`, `copyMaterialForPrimitive`), `keepNamedTubeMaterial` | `core/materials.py` (`MaterialsMixin`) |
| `helios::deg2rad`, `rad2deg`, `acos_safe`, `asin_safe`, `clamp`, `sum(vector<float>)`, `asinf` (libm) | `core/vec.py` (new functions only) |
| `helios::resolvePluginAsset`, `tryResolvePluginAsset` | `core/scene.py` module functions |

## Design

- **Tube storage.** The tube now keeps Helios's full local-frame state in `CompoundObject.extra`:
  - `nodes`, `radius` (unclamped, as passed), `colors`;
  - `triangle_vertices` (nodes × (Ndiv+1) × 3).

  Every setter first bakes a non-identity object transform into that state, then rewrites the
  member triangles **by position** in the UUID list (radial slot slowest, segment fastest). Each
  triangle is rebuilt with `triangle_transform`, i.e. `setTriangleVertices`.
- **Shared tube frame.** `_tube_rings` (addTubeObject) was split into `_tube_frames`, the
  parallel-transport frame that all three Helios loops share verbatim, plus ring builders.
  - addTubeObject and appendTubeSegment compute `node + (A + B)`.
  - recomputeCrossSections computes `(node + A) + B`. The mutation check shows that this
    difference is visible.
- **Hidden flags.** They live in `Scene.hidden_primitives` and `Scene.hidden_objects`, not on the
  `Primitive`. W1-B's copy deep-copies the `Primitive`, and Helios creates copies visible.
  - `getAllUUIDs`/`getAllObjectIDs` skip hidden entries, as Helios does. `include_hidden=True` is
    a Heliades extension.
  - The snapshot follows the bulk getters by default. `snapshot(include_hidden=True)` dumps
    everything, with per-primitive and per-object `hidden` flags.
  - The Helios side enumerates hidden entries by probing `doesPrimitiveExist`/`doesObjectExist`
    until `getPrimitiveCount()`/`getObjectCount()` (which include hidden entries) are reached.
  - `compare.py` checks both kinds of `hidden` flag.
  - The Helios snapshot now reads every field through the explicit sorted UUID list, instead of
    the `getAll*` getters.
- **Data.** `primitive_data`/`object_data` stay `label -> {id: value}`, so W1-B's copy/delete code
  works unchanged. Each value is now a `TypedData(type, values)`.
  - The registry type of a label is the type of any of its entries. An empty dict means the type
    lock is released, as Helios does when the last holder goes.
  - A value of another numeric type is `static_cast` to the registered type. Any other type
    mismatch raises.
  - The C++ overload is chosen by `infer_type`: `np.float32` or Python `float` → float,
    `np.float64` → double, `np.uint32` → uint, `int` → int, length 2-4 → vecN or intN, `str`,
    `bool`. An explicit `dtype=` overrides it.
- **Materials.** `Primitive.color` and `texture_file` remain the drawn appearance that everything
  reads. `Primitive.material_id` names the material, and every material edit is written to its
  users.
  - Automatic labels match Helios exactly: `"__auto_" + std::hash` combination, with libstdc++
    `_Hash_bytes` ported.
  - Reference counts are incremental. `Scene` wraps W1-B's `_copy_primitive_one` and
    `delete_primitives` to update them, and to drop hidden flags of deleted UUIDs.
  - Not covered: normal and bump maps, material data, color and spectral textures, and revision
    counters.

## Helios / PyHelios discrepancies (each with a PyHelios repro)

1. **PyHelios's default `comparator="="` for `filterPrimitivesByData`/`filterObjectsByData` is
   rejected by Helios 1.3.98**, which only accepts `==`, `>`, `<`, `>=` and `<=` (binding bug):
   ```python
   u = c.addPatch(); c.setPrimitiveDataFloat(u, 'k', 1.0)
   c.filterPrimitivesByData([u], 'k', 1.0)   # HeliosRuntimeError: Invalid comparator
   ```
   PyHelios also exports no uint or double *primitive* filter, although the object filter has
   both. The parity ops call the ctypes wrappers with `"=="`. The uint and double primitive filters
   are unit-tested only.
2. **The filter output order is a permutation.** Helios walks the list from the back and
   swap-and-pops every rejected ID. For example, keep-values `[1,5,2,5,3,5]` filtered `== 5`
   return the UUIDs at indices `[3,1,5]`. Heliades reproduces this; the `filter_order` mutation
   shows the order is checked.
3. **A textured tube that dropped zero-area triangles cannot be edited (Helios bug).**
   `addTubeObject(texture)` deletes triangles whose textured area is 0, but every tube setter
   addresses triangles by position in the UUID list. As a result:
   - `setTubeRadii`, `setTubeNodes` and `appendTubeSegment` throw `vector::_M_range_check`;
   - they throw only *after* part of the work is done: earlier triangles are rewritten, and the
     appended node and orphan triangles remain;
   - even before the throw, positions after a dropped triangle are assigned the wrong geometry.
   ```python
   t = c.addTubeObject(6, [vec3(0,0,0), vec3(0,0,.1), vec3(0,0,.2)], [.02]*3, texturefile=".../TomatoLeaf.png")
   len(c.getObjectPrimitiveUUIDs(t))   # 23 of 24
   c.setTubeRadii(t, [.03]*3)          # Helios error 7: vector::_M_range_check: __n (which is 23) >= ... 23
   ```
   Heliades matches this, including the partial update (`core_gaps.tube.textured_append`). The
   tomato's tubes are all colored, so it is not affected.
4. **`getTubeObjectVolume` mixes frames.** It uses world radii (`getNodeRadii` applies the object
   transform) but local segment lengths. `getTubeObjectSegmentVolume` uses local values for both.
   After `scaleObject(t, (2,2,2))` on a unit-length tube of radius 0.1:
   - `getTubeObjectVolume` gives 0.1257 (true 0.2513);
   - `getTubeObjectSegmentVolume(t, 0)` gives 0.0314.

   Heliades matches both. This is a candidate upstream bug. It matters for plantarchitecture only
   if a tube's object transform is not the identity when its volume is read; the tube setters
   bake the transform away.
5. **`appendTubeSegment` rebuilds the previous last ring with the stored, unclamped radius.**
   `addTubeObject` used the radius clamped to ≥1e-5. So a tube created with a final radius of 0
   has that ring collapse onto the node when a segment is appended. In the repro, the area of the
   first segment changes from 0.0057159 to 0.0057131. The new node's own radius is clamped to
   `std::max(1e-5f, r)`, and NaN becomes 1e-5. Heliades matches.
6. **`setTubeRadii` is path dependent and lossy.** It rescales each ring vertex along its current
   offset from the node and does not clamp. A radius of 0 therefore collapses the ring, and later
   radius-only updates cannot re-inflate it: `normalize(0)` stays 0. Only `setTubeNodes`
   recomputes the rings from scratch, with radii clamped to 1e-5. See
   `core_gaps.tube.radius_only_repeat` and the unit test `test_set_tube_radii_is_path_dependent`.
7. **`Context::setPrimitiveData<bool>` / `setObjectData<bool>` leave `data_type` uninitialized
   (UB).** There is no `bool` branch in the type chain (`Context.h` scalar `setPrimitiveData`).
   PyHelios has no bool setter, so this is not oracle-testable. Heliades registers `TYPE_BOOL`.
8. **Copies of hidden objects are visible.** `copyObject` and `copyPrimitive` build new primitives,
   and `ishidden` defaults to false. The 35 hidden blade prototypes and 3 inflorescence prototypes
   of the tomato are copied into visible organs this way. `getObjectCount()` and
   `getPrimitiveCount()` include hidden entries, while `getAllObjectIDs()` and `getAllUUIDs()` do
   not.
9. **`RGBAcolor::operator==` uses a 1e-6 tolerance**, so `setMaterialColor` with a color within
   1e-6 of the current one is a no-op. Heliades matches.

## Mutation checks (all caught)

Each one is applied with a monkeypatch, and the listed cases must fail. Two of them are in
`tests/heliades/test_core_gaps.py`.

| mutation | caught by |
|---|---|
| setTubeRadii recomputes from scratch (no path dependence) | `tube.radius_only_repeat`, `tube.edit_fuzz` |
| recomputeCrossSections computes `node + (A+B)` | `tube.edit_fuzz`, `tube.append_chain` |
| skip bakeTransform | `tube.edit_fuzz` |
| appended triangles at the end instead of per radial block | `tube.append_chain` |
| filter keeps the input order | `data.primitive`, `data.object` |
| no numeric cast to the registered type | `data.primitive`, `data.object` |
| getAllUUIDs lists hidden primitives | `hidden.visible_snapshot`, `hidden.queries` |
| copies of hidden primitives stay hidden | all `hidden.*` |
| no copy-on-write for a shared named material | `materials` |
| getObjectArea accumulated in float | `object_area` |

## Left for later

- Normal and bump maps, material data, color and spectral textures. `setPrimitiveTextureFile`.
- `setPolymeshObjectTopology` and `computePolymeshObjectVertexNormals` belong to the OBJ/polymesh
  package (W2-A), not here.
- `AxisRotation` / `make_AxisRotation` are plantarchitecture types (W2-C).
- Helios clamps `RGBcolor` components to [0,1] at construction. `Scene._new` (pre-existing) does
  not; the material setters do.
- `listMaterials`, `listAllPrimitiveDataLabels` and `getPrimitivesUsingMaterial` iterate hash maps
  in Helios. Heliades returns them sorted.
