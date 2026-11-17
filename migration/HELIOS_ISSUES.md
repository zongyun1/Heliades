# Discrepancies found by differential testing

Each entry: what differs, where it is in the Helios source (pinned 1.3.98 tree, `third_party/pyhelios_src`), the
evidence (a minimal reproduction or the parity case that pins it), and what Heliades does about it. Candidates to
report upstream are marked **[upstream?]**.

Source abbreviations: `core/` = `helios-core/core/`; `PA` = `plugins/plantarchitecture/src/PlantArchitecture.cpp`,
`PL` = `.../src/PlantLibrary.cpp`, `H` = `.../include/PlantArchitecture.h`; `pyhelios/` = the PyHelios package.

Entries 3 onwards were folded in from the work-package notes (`migration/notes/archive/`, named in brackets); the
notes keep the longer write-ups. New findings go straight into this file.

## 1. PyHelios `addPatch` default color is white; C++ default is black (binding inconsistency)

- C++ `Context::addPatch(center, size[, rotation])` uses `make_RGBAcolor(0,0,0,1)`
  (`core/src/Context_primitive.cpp:23-33`).
- PyHelios `Context.addPatch(..., color=None)` substitutes `RGBcolor(1,1,1)` (`pyhelios/Context.py:475`).
- PyHelios `addTriangle(..., color=None)` calls the colorless C++ overload, which gives black. So the
  two primitive types disagree within PyHelios.
- Reproduction: `Context().addPatch()` and then `getPrimitiveColor`, which returns white (C++ gives black).
- Heliades follows C++. The parity backend passes an explicit black.

## 2. The pip wheel and the source tree are different Helios versions

- `pyhelios3d==0.1.39` (pip) bundles Helios commit `f97ef3c` (1.3.94), where `randn` comes from
  libstdc++ `std::normal_distribution<float>`. Helios 1.3.98 replaced it with portable distributions,
  so the same seed now gives different normal draws (and therefore different plants).
- Heliades pins 1.3.98 and builds the oracle from source (`migration/PORTING.md`).

## Core: textures and solid fraction

### 3. The solid fraction depends on creation / hash order (cache keyed by rounded pixel UVs) [upstream?] [W1-A, W2-E]

- What: `Texture::getSolidFraction` caches by `round(uv * (res-1))` per vertex. Two polygons whose UVs round to the
  same pixels share the value of whichever was computed first, although their exact coverage differs. Through
  `setMaterialTexture` the "first" is the material's users in `unordered_map<uint, Primitive*>` order, i.e. hash
  order, not UUID order.
- Where: `core/src/Context.cpp:172` (`Texture::getSolidFraction`); the user walk in `setMaterialTexture`.
- Evidence: two `addTriangleTextured` calls with UVs `(0.3,0.3),(0.6,0.31),(0.45,0.62)` and
  `(0.3001,0.3),(0.6,0.3101),(0.45,0.62)`: order (0,1) gives 0.985505819 for both, order (1,0) 0.98549813 for both.
  Via a shared material (W2-E repro, `CherryTomatoLeaf` then `TomatoLeaf`): both get 0.8942139 (= B alone); swap the
  UV sets and both get 0.8880575 (= A alone).
- Heliades: one cache per texture with the same key, and the Helios hash order of primitives is reproduced
  (it holds while UUIDs are consumed and erased at the same points as Helios). `texture.solid_fraction_cache`
  fails if the cache is removed.

### 4. PyHelios makes patch/triangle texture paths absolute but passes tile/tube paths verbatim [W1-A]

- What: `addPatchTextured` / `addTriangleTextured` call `_validate_file_path` (`os.path.abspath`);
  `addTileObject` / `addTubeObject` pass `texturefile` unchanged. Helios stores the string it receives as
  `Material::texture_file`, so the same file gives two registry entries (and two solid-fraction caches).
- Where: `pyhelios/Context.py` (`_validate_file_path` and the four add methods).
- Evidence: after `os.chdir(textures)`, `getPrimitiveTextureFile` reports `/…/textures/TomatoLeaf.png` for a patch
  and `TomatoLeaf.png` for a tile.
- Heliades: stores the string verbatim, like C++; snapshots compare textures by basename (`compare.texture_key`).
  `test_relative_paths_match_helios` checks relative-path resolution against the oracle.

### 5. The batch texture-file getter is wrong for non-ASCII paths (PyHelios) [W1-A]

- What: `Context.getPrimitiveTextureFile(list)` decodes the concatenated buffer to `str` and slices it with the C
  **byte** offsets.
- Where: `pyhelios/Context.py` (`getPrimitiveTextureFile`, list branch).
- Evidence: three patches textured from `.../blätter/TomatoLeaf.png` give
  `['/…/blätter/TomatoLeaf.png/', 'tmp/…/blätter/TomatoLeaf.png/t', 'mp/…/blätter/TomatoLeaf.png']`.
- Heliades: not affected; `helios_snapshot` reads texture files one UUID at a time. Fix: slice bytes before decoding.

### 6. Textured `addPatch` without UVs skips the size check [W1-A]

- What: the colored overload and the `uv_center`/`uv_size` overload reject `size < 1e-6`; the plain textured
  overload only `assert`s `size > 0`, which is compiled out in release builds. Size exactly 0 is still rejected
  (by `Primitive::scale`).
- Where: `core/src/Context_primitive.cpp` (`addPatch` overloads).
- Evidence: `addPatchTextured(vec3(0,0,0), vec2(1e-8,1e-8), TomatoLeaf)` is accepted (area 5.7e-17); with
  `uv_center`/`uv_size` it is rejected.
- Heliades: same behaviour.

### 7. UVs are clamped only from above [W1-A]

- What: the Patch/Triangle constructors apply `std::min(uv, 1.f)`; negative UVs are kept, and the solid-fraction
  bounding box clamps them to pixel 0. `(-0.5,0), (1.5,0), (0.5,2)` is stored as `(-0.5,0), (1,0), (0.5,1)`.
- Where: Patch / Triangle constructors (`core/src/Context_primitive.cpp`).
- Heliades: same behaviour.

### 8. One-pixel-wide or -tall images give inf/NaN edge tests [W1-A]

- What: for `W == 1` or `H == 1`, `1/(W-1)` is `inf`; an edge coefficient of 0 makes `0*inf = NaN`, and
  `if (L < 0)` is false for NaN, so such pixels count as inside. Zero-width UV triangles on a 1-row image get a
  nonzero solid fraction.
- Where: `Texture::getSolidFraction` (`core/src/Context.cpp:172`).
- Evidence: `texture.png_formats` fails if Heliades tests `L >= 0` instead.
- Heliades: uses `~(L < 0)` to match.

### 9. Smaller texture notes [W1-A]

- PyHelios accepts `.tga`/`.bmp` in `_validate_file_path`, but Helios `addTexture` rejects anything but PNG/JPEG.
- A textured primitive's color is the material color, black `(0,0,0,1)`; `getPrimitiveColor` returns black.
- The textured `addTileObject` stores the requested `texture_repeat`, not the corrected one, and does not call
  `setColor` on the object.
- Heliades: all reproduced.

## Core: objects, transforms and shapes

### 10. `copyObject` resets `object_origin` to (0,0,0) [upstream?] [W1-B]

- What: `Context::copyObject` builds the new object with its constructor and never sets `object_origin`, so
  `rotateObjectAboutOrigin` / `scaleObjectAboutOrigin` on a copy pivot about the world origin, while on the original
  they pivot about the creation centre.
- Where: `core/src/Context.cpp:1191` (`copyObject`); the pivots at `Context.cpp:1450` / `1510`.
- Evidence: a tile at (5,0,0) and its copy, both `rotateObject(·, pi/2, z, about_origin=True)`: the original stays at
  (5,0,0), the copy moves to (0,5,0). Pinned by `core.object_transforms.origin`.
- Heliades: matches. plantarchitecture never calls the about-origin variants.

### 11. `translateObject` does not move `object_origin` [W1-B]

- What: after a translation, about-origin rotation and scaling still pivot at the creation centre (same for rotate
  and scale).
- Evidence: a tile created at (5,0,0), translated by (0,3,0), ends at (5,-3,0) after `rotateObject(pi, z,
  about_origin=True)`.
- Heliades: matches.

### 12. PyHelios transform docstrings disagree with C++; PyHelios rejects a zero axis [W1-B]

- What: `rotateObject(obj, a, axis)` / `rotatePrimitive` say "about the object/primitive centre"; C++ rotates about
  the axis through the world origin. `scalePrimitive(uuid, s)` says "about primitive centre"; C++ scales about the
  world origin (`makeScaleMatrix`). The C++ `about_origin` branch comment says "global origin" (its docstring is
  right). PyHelios raises on a zero axis vector; C++ accepts it (guarded `normalize` leaves it at zero, giving a
  `cos`-diagonal scale matrix).
- Evidence: a tile at (5,0,0) rotated pi/2 about z moves to (0,5,0); a patch at (1,1,0) scaled by 2 ends at (2,2,0).
- Heliades: follows C++. The parity cases never pass a zero axis.

### 13. Compound-object members are transform-locked [W1-B]

- What: `translatePrimitive` / `rotatePrimitive` / `scalePrimitive` on a tile or tube member warn and do nothing;
  polymesh members do move.
- Heliades: the W1-B object-transform methods implement the lock (`_transform_locked`); the older `Scene` methods do
  not.

### 14. Disk object transform ignores rotation; `object_origin` differs by type [W1-C]

- What: `addDiskObject` rotates each triangle (+elevation about y, then azimuth about z, unlike patch/tile which use
  -elevation about x), but the object transform is only `scale(size.x, size.y, 1)` + translation, so
  `addDiskObject(4, (0,0,0), (1,1), SphericalCoord(1, 0.5, 0))` reports an identity transform while its primitives are
  tilted. `object_origin` is (0,0,0) for spheres (never set), the centre argument for disks and boxes, and the
  bounding-box centre for cones. `getDiskObjectSubdivisionCount` returns only `Ndivs.x`. `addBoxObject`'s `subsize`
  is computed and unused in the colored overload.
- Where: `core/src/Context.cpp` (`addDiskObject`, `addSphereObject`, `addConeObject`, `addBoxObject`).
- Heliades: reproduced (`CompoundObject.origin` stores the same values). Degenerate inputs Helios accepts (disk size
  (0,0)/(1,0), cone radii <= 0, coincident cone nodes with NaN axis, sphere/cone Ndivs 1-2) are bit-exact.

### 15. PyHelios shape-binding quirks [W1-C]

- `addSphereObject` rejects `ndivs < 3`; C++ accepts 1 and 2.
- `addDiskObject` with `int2` ndivs and no color substitutes gray (0.5,0.5,0.5); C++ has no colorless `int2`
  overload. With `uint` ndivs and no color, C++ uses red (1,0,0,1).
- `addBoxObject(reverse_normals=True)` with no color raises; C++ has only color+bool / texture+bool overloads.
- `addDiskObject` builds `SphericalCoord(radius, elevation, azimuth)` from the Python object (elevation, not zenith).
- Heliades: `ops_shapes.py` calls `pyhelios.wrappers.UContextWrapper` directly to choose each C++ overload.

## Core: OBJ loading

### 16. PyHelios `Context.loadOBJ` drops `upaxis` [W2-A]

- What: the "origin + height + rotation + color" branch is tested before the one that also takes `upaxis`, so
  `loadOBJ(f, origin=o, height=h, rotation=r, color=c, upaxis="YUP")` calls the C++ overload without `upaxis` and
  loads as ZUP.
- Where: `pyhelios/Context.py:2490` (`loadOBJ`, def at 2445).
- Evidence: `loadOBJ(TomatoFlower, origin=vec3(0,0,0), height=0.75, rotation=SphericalCoord(1,0,0),
  color=RGBcolor(0,0,0), upaxis="YUP")` gives the same vertices as `upaxis="ZUP"`.
- Heliades: implements `upaxis`; the parity ops call the ctypes wrappers directly.

### 17. Two shipped assets do not load; OBJ stream quirks [W2-A]

- `RedbudLeaf.obj`: material `Material.002` has `map_Kd ../textures/RedbudLeaf.png` but its faces have no `vt`
  ("face has no texture coordinates"). `SugarBeetLeaf.obj`: `mtllib SugarbeetLeaf.mtl`, file on disk is
  `SugarBeetLeaf.mtl` ("Could not locate asset file" on a case-sensitive filesystem). Neither is on a ported path.
- A face line ending in whitespace or `\r` (CRLF) makes `stream >> tmp` fail with `tmp` unchanged, so the last corner
  is pushed twice (a 32-gon gets 33 corners; the extra fan triangle is degenerate and dropped). `o` as the last token
  with no newline names the object `"o"`; `f` as the last token fails with "Face index ... must be an integer value".
  `XUP` is validated but applies no rotation.
- Heliades: raises the same errors and reproduces the stream quirks (they change geometry).

## Core: data, filters and materials

### 18. `filterPrimitivesByData`: PyHelios default comparator is rejected; output is a permutation [W2-D, W3-E]

- What: PyHelios defaults `comparator="="` for `filterPrimitivesByData` / `filterObjectsByData`; Helios 1.3.98 accepts
  only `==`, `>`, `<`, `>=`, `<=` (binding bug). PyHelios also exports no uint or double primitive filter. Helios walks
  the list from the back and swap-and-pops every rejected ID, so the result is not a stable filter.
- Where: `core/src/Context_data.cpp:1768`; `pyhelios/Context.py:5543, 5861`.
- Evidence: `filterPrimitivesByData([u], 'k', 1.0)` raises "Invalid comparator". Keep-values `[1,5,2,5,3,5]` filtered
  `== 5` return the UUIDs at indices `[3,1,5]` (`filter_order` mutation; `object_label` selections in W3-E).
- Heliades: reproduces the permutation; the parity ops pass `"=="` through the ctypes wrappers.

### 19. `setPrimitiveData<bool>` / `setObjectData<bool>` leave `data_type` uninitialised (UB) [W2-D]

- Where: scalar `setPrimitiveData` type chain in `core/include/Context.h` (no `bool` branch).
- Evidence: not oracle-testable (PyHelios has no bool setter).
- Heliades: registers `TYPE_BOOL`.

### 20. Copies of hidden objects are visible; counts include hidden entries [W2-D]

- What: `copyObject` / `copyPrimitive` build new primitives with `ishidden = false`, so the tomato's 35 hidden blade
  prototypes and 3 inflorescence prototypes are copied into visible organs. `getObjectCount()` /
  `getPrimitiveCount()` include hidden entries; `getAllObjectIDs()` / `getAllUUIDs()` do not.
- Heliades: matches.

### 21. `RGBAcolor::operator==` uses a 1e-6 tolerance [W2-D]

- What: `setMaterialColor` with a colour within 1e-6 of the current one is a no-op.
- Heliades: matches.

### 22. `copyPrimitive` of a texture-colour-overridden primitive leaves its named material [W2-E]

- What: `copyPrimitive` shares the source's material and then calls `overrideTextureColor()` on the copy; for a named
  (shared) material, `setPrimitiveAppearance` makes a `__copy_` material, so the copy stops following
  `setMaterialColor` on the label.
- Where: `core/src/Context_primitive.cpp:648` (`copyPrimitive`, `if (textureoverride)`).
- Evidence: in `BougainvilleaFlower.obj`, copying an overridden primitive gives material `'__copy_7_520'` while the
  source stays `'Material.001'`.
- Heliades: matches (the tomato OBJs have no map_d-only material).

### 23. A map_d-only MTL material leaves two unused automatic materials per distinct colour [W2-E]

- What: they are textured black and colour+texture and stay in the table with reference count 0; they only cost IDs.
- Heliades: consumes the IDs identically. (Correction to the W2-D note: `listMaterials` is `std::map` order, sorted.)

### 24. `getAllUUIDs()` and the `getAll*` getters are in hash order [W1-A, W1-D]

- What: `Context::getAllUUIDs()` iterates an `std::unordered_map` (`core/include/Context.h:3548`), so its order is
  libstdc++ bucket order, not UUID order.
- Heliades: snapshots pass an explicit sorted UUID list to the batch getters and compare by sorted UUID.

## Core: tubes

### 25. The textured tube deletes zero-area triangles, and then cannot be edited [upstream?] [W1-A, W2-D]

- What: `addTubeObject(…, texture)` deletes each triangle whose textured area (`getPrimitiveArea`, including the solid
  fraction) is not `> 0`, after its UUID was consumed (UUID gaps; the colored overload keeps every triangle). Every
  tube setter addresses triangles by position in the UUID list, so on such a tube `setTubeRadii`, `setTubeNodes` and
  `appendTubeSegment` throw `vector::_M_range_check` *after* part of the work is done (earlier triangles rewritten,
  appended node and orphan triangles remain), and positions after a dropped triangle get the wrong geometry.
- Where: `core/src/Context.cpp` (`addTubeObject` textured overload; `appendTubeSegment` 5589/5593, `setTubeRadii` 5601).
- Evidence: `addTubeObject(6, [(0,0,0),(0,0,.1),(0,0,.2)], [.02]*3, texturefile=TomatoLeaf)` has 23 of 24 triangles;
  `setTubeRadii(t, [.03]*3)` raises "Helios error 7: vector::_M_range_check: __n (which is 23) >= ... 23".
  `texture.tube` shows 56 UUID gaps on seed 0; `core_gaps.tube.textured_append` pins the partial update.
- Heliades: matches, including the partial update. The plantarchitecture tubes are colored.

### 26. `getTubeObjectVolume` mixes frames [upstream?] [W2-D]

- What: it uses world radii (`getNodeRadii` applies the object transform) but local segment lengths;
  `getTubeObjectSegmentVolume` uses local values for both.
- Where: `core/src/Context.cpp:5581`.
- Evidence: after `scaleObject(t, (2,2,2))` on a unit-length tube of radius 0.1, `getTubeObjectVolume` = 0.1257 (true
  0.2513), `getTubeObjectSegmentVolume(t, 0)` = 0.0314.
- Heliades: matches both. The plantarchitecture tube setters bake the transform away before volumes are read.

### 27. `appendTubeSegment` and `setTubeRadii` are lossy for small radii [W2-D]

- What: `appendTubeSegment` rebuilds the previous last ring with the stored, unclamped radius (`addTubeObject` used the
  radius clamped to >= 1e-5), so a final radius of 0 collapses that ring on append (first-segment area 0.0057159 ->
  0.0057131); the new node's radius is `std::max(1e-5f, r)` (NaN -> 1e-5). `setTubeRadii` rescales each ring vertex
  along its current offset and does not clamp, so a radius of 0 collapses the ring for good (`normalize(0)` stays 0);
  only `setTubeNodes` rebuilds rings, with radii clamped to 1e-5.
- Evidence: `core_gaps.tube.radius_only_repeat`, `test_set_tube_radii_is_path_dependent`.
- Heliades: matches.

## RNG and seeding

### 28. Seeds 0 and 1 give the same plant (by design of `std::minstd_rand0`) [W1-D]

- What: `Context::seedRandomGenerator(s)` calls `std::minstd_rand0::seed(s)`; for an LCG with increment 0, a seed with
  `s mod (2^31-1) == 0` sets the state to 1, so seed 0 == seed 1 (== 2147483647).
- Where: `core/src/Context.cpp:57-62`.
- Evidence: oracle tomato age 20, seeds 0 and 1 give the identical vertex hash (also seen in the B2 sim tests).
- Heliades: `MinstdRand0.seed` reproduces it; parity grids use Context seeds 1..8.

### 29. Seeding after `loadPlantModelFromLibrary()` is not reproducible [W1-D]

- What: loading draws the `LeafPrototype::unique_prototype_identifier` values (10 raw engine calls for the tomato, seed
  5) from the Context generator, still wall-clock seeded unless the caller seeded first; the last identifier seeds the
  private `prototype_generator` that builds the leaflet shapes (PA:3475). The plugin keeps `&Context::generator`
  (PA:383), so seeding before or after constructing `PlantArchitecture` makes no difference.
- Heliades: `Scene()` starts at seed 1 instead of the wall clock; every parity case seeds before loading.

### 30. Results depend on the compiler's argument evaluation order [W1-D, W3-C, W4-D]

- What: calls with several `randu()` arguments are evaluated right to left by GCC:
  `make_AxisRotation(0, randu(0,2pi), randu(0,2pi))` in `buildTomatoPlant` (PL:6214), the lateral
  `make_AxisRotation(deg2rad(insertion), deg2rad(base_yaw.val()), deg2rad(base_roll.val()))`, and the grapevine
  `addComponent(randu(amp), randu(wl), randu(phase))` (phase drawn first). A Clang or MSVC build of Helios gives a
  different plant for the same seed.
- Evidence: tomato XML `<base_rotation>` is (0, draw2, draw1) for seeds 2, 3, 5 (`scripts/oracle/argorder.py`); the
  grapevine age-0 snapshot and its mutation check.
- Heliades: follows GCC, as the pinned oracle does.

## plantarchitecture

### 31. `AxisRotation::operator-` is reversed [upstream?] [W2-C]

- What: `x - a` returns `{a.pitch - pitch, ...}`, i.e. `a - x`.
- Where: `H:479`.
- Heliades: ported as written.

### 32. Stale `ShootParameters::operator=` comment; constructor defects in `PhytomerParameters` [W1-D, W2-C]

- What: the `ShootParameters::operator=` comment (H:1585) says the nested `PhytomerParameters` assignment resamples;
  it does not (every sub-struct copies `RandomParameter`s member-wise, cache included; resampling is the explicit
  `phytomer_parameters.resample()` in the `Phytomer` constructor, PA:2700). `PhytomerParameters()` assigns
  `petiole.color` twice in the peduncle block and never sets `peduncle.color`, which stays black (PA:992/996).
  `PhytomerParameters(gen)` leaves `leaf.leaflet_scale` / `leaf.intercalary_leaflet_scale` (set by `operator=(float)`)
  and `internode.max_vegetative_buds_per_petiole` / `max_floral_buds_per_petiole` (never initialised) without a
  generator, so giving them a distribution through the C++ API throws on the first `val()`.
- Evidence: `load.tomato` plus unit tests (no resample); the tomato sets `peduncle.color` explicitly. PyHelios's
  `defineShootType` works because its wrapper rebuilds every parameter with `initialize(gen)` (and makes no engine
  draws in 1.3.98).
- Heliades: same behaviour; the Heliades `defineShootType` op mirrors the PyHelios wrapper.

### 33. `LeafPrototype::operator=` re-draws the prototype identifier [W1-D]

- What: `operator=` draws a new `unique_prototype_identifier` whenever the prototype has a generator; the copy
  constructor and `LeafParameters::operator=` (`duplicate()`) do not. An innocuous assignment consumes 2-3 Context
  draws.
- Heliades: reproduced; the draw count is load-bearing for parity (`plantarch.load.*`).

### 34. Shoot constructor reads `gravitropic_curvature` from the argument, not the member [W3-B]

- What: `gravitropic_curvature = shoot_params.gravitropic_curvature.val()` reads the local copy passed in from
  `addBaseStemShoot`, so a distributed value is drawn and cached on that temporary.
- Where: PA:6958.
- Heliades: ported literally (it matters for the strawberry, whose `gravitropic_curvature` is distributed).

### 35. `removeLeaf` does not reduce `downstream_leaf_area` (shed leaves keep thickening the stem) [upstream?] [W4-A, W4-D]

- What: the `propagateDownstreamLeafArea(..., -getLeafArea())` call is commented out (PA:6767). The capsicum creation
  function removes the leaf after `appendPhytomer` has added its area, so the first six main-stem internodes are
  thickened by leaves that never existed; on the grapevine, girth after leaf pulling still counts the pulled leaves.
- Heliades: ported as is.

### 36. Capsicum `secondary` shoots have no phytomer creation function [upstream?] [W4-A]

- What: `phytomer_parameters_secondary` is copied (PL:3267) before `CapsicumPhytomerCreationFunction` is assigned,
  and `shoot_parameters_secondary.phytomer_parameters = phytomer_parameters_secondary` (PL:3307) overwrites the copied
  pointer with null. Secondaries get no leaf-size ramp, and their rank >= 2 children would not be killed (they never
  break before day 80: p = 0.06). Looks unintended.
- Heliades: ported as is.

### 37. Other capsicum / cherry-tomato oddities [W4-A]

- Capsicum builds 10 closed + 10 open `AlmondFlowerPrototype` flower prototypes with the fruit cache although its
  flowers never appear (initiation / opening thresholds are -1: buds go straight to fruit set).
- `CapsicumFruitPrototype` makes a Context `randn()` every time it builds a prototype (10 per cache fill).
- Cherry tomato `lateral.vegetative_bud_break_probability_max = 0` has no effect: with decay rate 0,
  `sampleVegetativeBudBreak` uses `probability_min`; each lateral bud still makes one `randu()`.
- Heliades: all reproduced (`plantarch.w4a.capsicum.*`, `plantarch.w4a.cherrytomato.*`).

### 38. PyHelios vs C++ plant-query return shapes [W3-E]

- What: `getShootIDsByRank` returns a dict in PyHelios and a vector in C++. `getPlantLeafObjectIDs` includes IDs of
  leaves that no longer exist (C++ does not filter); `getPlantLeafAreas` skips them. `getAllPrototypeObjectIDs` is not
  exposed by PyHelios.
- Heliades: follows PyHelios for the return shape and C++ for the contents; prototypes are checked indirectly (the
  hidden objects of the snapshot are exactly the prototypes).

### 39. The blade builder and the maize tassel consume large, invisible parts of the streams [W3-A, W3-C, W4-C]

- What: every blade consumes 48 facet UUIDs of which 8 fully transparent ones are deleted again (40 survive per tomato
  blade; 152 UUIDs for 144 primitives per leaf cache). `MaizeTasselPrototype` draws ~10^5 numbers from the Context
  stream (day 57 onward: 549 vs 104121 engine calls for seed 1).
- Heliades: reproduces the UUID consumption; the tassel prototype is not ported yet (a maize build past day 56
  raises).

### 40. PyHelios passes build parameters as fixed values only [W4-D]

- What: PyHelios 1.3.98 cannot pass a distributed build parameter, so `BuildParameterDistribution::sample`
  (uniform / normal / discrete) is not reachable from the oracle.
- Heliades: ported (fresh `helios::` distributions, like C++) but not oracle-checked.
