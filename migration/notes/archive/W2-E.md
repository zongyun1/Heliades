# W2-E: one material table for loadOBJ and the material API

Baseline: Helios 1.3.98. Sources used:
- `Context_fileIO.cpp`: loadOBJ material registration and the second pass;
- `Context.cpp`: the material management methods, `overridePrimitiveTextureColor`;
- `Context_primitive.cpp`: `copyPrimitive`, `Primitive::overrideTextureColor` / `useTextureColor`;
- `/usr/include/c++/13/bits/hashtable.h`: the oracle was built with GCC 13.3.

All parity cases pass at 0 ulp / exact (72/72), including the new `core.materials.obj_interplay`.

## What changed

- **One registry.** The W2-A mini-registry is deleted:
  - `Scene._fileio_materials`, `_mat_registry`, `_fileio_add_material`, `_fileio_assign_material`,
    `_fileio_does_material_exist`;
  - `FileIOMixin.primitive_material_label`, which competed with
    `MaterialsMixin.get_primitive_material_label`;
  - `FileIOMixin.is_primitive_texture_color_overridden`;
  - the `material_label` / `texture_color_overridden` attributes on `Primitive`.

  No method name is now defined by more than one Scene mixin (checked over the MRO).
- **`loadOBJ` follows Helios call for call through `MaterialsMixin`.**
  - Each MTL entry, in `std::map` order: `does_material_exist`, the `_k` suffix, `add_material`,
    `set_material_color(Kd, 1)`, `set_material_texture` (when there is a texture),
    `set_material_texture_color_override`.
  - Each triangle: `add_triangle` / `add_triangle_textured`, which gives it its automatic material.
    For a map_d-only material it then calls `set_primitive_color(Kd)` and
    `override_primitive_texture_color`. The triangle is then given `assign_material_to_primitive(name)`
    if that label exists.
  - So the transient automatic materials consume material IDs exactly as in Helios. The faces of
    the "none" and unknown materials keep the raw name, so they join a Context material that
    happens to have that label.
- **New in `MaterialsMixin`:**
  - `override_primitive_texture_color`, `use_primitive_texture_color` and
    `is_primitive_texture_color_overridden`, which reads the material;
  - the `copyPrimitive` override rule (Discrepancy 1);
  - `UnorderedUIntMapOrder`, a model of libstdc++'s `std::unordered_map<uint>` iteration order, kept in
    `Scene.primitive_order`. `Scene._new`, the copy wrapper and the delete wrapper insert into it and
    erase from it.
- **Ordering, the cosmetic note from W2-D.**
  - `listMaterials` / `getMaterialCount`: `material_label_to_id` is a `std::map`, not a hash map.
    Bytewise order is Python's `str` order, so Heliades already matched. The oracle op no longer
    sorts, so the order is now checked.
  - `getPrimitivesUsingMaterial`: Helios walks `std::unordered_map<uint, Primitive*>`. Heliades now
    reproduces that order exactly (rehash policy, prime table, bucket-begin insertion, relinking on
    rehash, erase), so the list is no longer sorted. This is not only cosmetic: see Discrepancy 2.
    `markMaterialPrimitivesModified` (`_refresh_material_users`) uses the same order.
  - `listAllPrimitiveDataLabels` (data.py, outside this package) is still sorted.
- **Parity ops.**
  - `obj_queries` (fileio) now compares each primitive's material **ID** and its full label,
    automatic labels included. Before, both sides masked automatic labels as `(auto)`.
  - `material_query` and `list_materials` (core_gaps) compare Helios's own order.
  - New `ops_materials.py`: `delete_material`, `set_material_texture_color_override`,
    `set_material_twosided_flag`, `override_/use_primitive_texture_color`, `material_list_raw` and
    `material_state`.
  - **Reference counts.** PyHelios exposes none, so `material_state` reads
    `Material::reference_count` through ctypes from `Context::getMaterial(uint)`
    (`_ZNK6helios7Context11getMaterialEj`), at offset 140. Each read is checked against the ID at
    offset 0 and the color at offset 40.

## Parity case `core.materials.obj_interplay` (exact, 8 seeds; 20 checked by hand)

**Setup.** The Context may hold `Material.003`, `Material.001`, `none`, `unknown_mat` and `red`
before any load. There is also a patch and a few triangles in the tomato Kd colours, whose
automatic materials the loads reuse. Then these are loaded in random order, through random loadOBJ
overloads:
- TomatoFlower and TomatoFruit (`Material.003` collides);
- two of TomatoLeaf, BougainvilleaFlower (map_d only, so override), AppleFruit and StrawberryFruit
  (`Material.001` collides);
- the W2-A synthetic OBJ: faces before any `usemtl`, `unknown_mat`, map_Kd+map_d, map_d only, and
  `my mat`.

**Steps.** Then 40 random steps:
- assign to primitives or objects;
- rename a named or automatic material, then sometimes `setPrimitiveColor` on its users (a material
  renamed from an automatic one is copied, not edited);
- `setPrimitiveColor`;
- `setMaterialColor`, `setMaterialTexture` (including ""), `setMaterialTextureColorOverride`,
  `setMaterialTwosidedFlag`;
- `override/usePrimitiveTextureColor`;
- `copyPrimitive`, `copyObject`, `deletePrimitive`, `deleteObject`;
- `deleteMaterial` (named or automatic);
- `addMaterial`, on an existing label (overwritten in place) or a new one, then assign;
- add more coloured triangles;
- load another OBJ.

**Compared.** The stream is taken after the loads and again at the end. It holds:
- per primitive: material ID, label, override flag and texture;
- `listMaterials` in Helios's order, and `getMaterialCount`;
- `doesMaterialExist` for every label seen during the run;
- per reachable material: ID, label, RGBA, texture, override, two-sided flag, **reference count**,
  and its users in Helios's order;
- the appearance: RGBA, solid fraction, UVs and vertices.

## Mutation checks (all caught)

| mutation | caught by |
|---|---|
| `copyPrimitive` ignores the source's texture-colour override, as W2-D's version did | `obj_interplay` seeds 0,2,4,5,6,7 (unit test) |
| `setMaterialTexture` refreshes users in UUID order instead of hash order | `obj_interplay` seeds 0,2 (unit test) |
| `getPrimitivesUsingMaterial` sorted | `obj_interplay` all seeds, `core_gaps.materials` all seeds |
| loadOBJ map_d-only path skips the intermediate `setPrimitiveColor` (one automatic material fewer) | `obj_interplay` 7/8 seeds, `fileio.obj.synthetic_topology` 8/10, `fileio.obj.assets_topology` seed 1 |

The first two mutations are pytest tests in `tests/heliades/test_materials.py`. That file also checks
the hash-order model directly against Helios: a random sequence of 150 add, delete and copy steps,
and a small sequence whose order was read from Helios (`[23..14, 4..13]`).

## Helios behaviour found (PyHelios repros)

1. **A copy of a primitive whose material has the texture-colour override leaves that material.**
   `copyPrimitive` shares the source's material and then calls `overrideTextureColor()` on the copy
   (`Context_primitive.cpp`, `if (textureoverride)`). For a named material, which the source and
   the copy now share, `setPrimitiveAppearance` makes a `__copy_` material. The copy therefore stops
   following `setMaterialColor` on the named label.
   ```python
   c = Context(); uu = c.loadOBJ(".../assets/obj/BougainvilleaFlower.obj")
   u = [x for x in uu if c.isPrimitiveTextureColorOverridden(x)][0]
   c.getPrimitiveMaterialLabel(c.copyPrimitive(u))     # '__copy_7_520', source stays 'Material.001'
   ```
   Heliades matches. The tomato OBJs have no map_d-only material, so the tomato is not affected.
2. **Solid fraction depends on the hash order of primitives.** `Texture::getSolidFraction` caches by
   UVs rounded to pixels. `setMaterialTexture` visits the material's users in
   `unordered_map<uint, Primitive*>` order. So when two users' UVs round to the same pixels, both
   get the value of the user visited first. In this repro that is UUID 1, not 0.
   ```python
   uvA = [vec2(.625,.125), vec2(.75,.125), vec2(.7495790123939514,.25)]
   uvB = [vec2(.625,.1235240027308464), vec2(.75,.125), vec2(.75,.25)]
   a = c.addTriangleTextured(v0,v1,v2, CherryTomatoLeaf, *uvA); b = c.addTriangleTextured(v0,v1,v2, CherryTomatoLeaf, *uvB)
   c.addMaterial("m"); c.assignMaterialToPrimitive([a,b], "m"); c.setMaterialTexture("m", TomatoLeaf)
   # both 0.8942139 (= B alone); swap the UV sets and both are 0.8880575 (= A alone)
   ```
   Heliades reproduces the hash order, and with it these values. The model holds only while
   Heliades inserts and erases primitives at the same points as Helios. That is true today: UUIDs
   are only consumed by `Scene._new` / copy, and erasures go through `delete_primitives`.
3. **The W2-D note was wrong about `listMaterials`.** It is `std::map` order (sorted), not hash
   order.
4. **A map_d-only MTL material leaves two unused automatic materials per distinct colour.** They are
   textured black and colour+texture, and stay in the table with reference count 0. They only cost
   IDs, which Heliades consumes identically.

## Not covered / follow-ups

- `getAllUUIDs` in Helios also walks the hash map. Heliades and the oracle ops sort it, which is fine
  as long as no Helios algorithm depends on that order. `Scene.primitive_order` is available if one
  turns out to.
- Normal and bump maps in MTL files are still `NotImplementedError` (W2-A).
- `migration/inventory.py` could not run in the worktree: there is no `third_party/` checkout there.
