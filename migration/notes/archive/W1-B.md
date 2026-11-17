# W1-B object transforms: notes

Baseline Helios 1.3.98. All W1-B parity cases are **bit-exact** (max 0 ulp on every check, every
seed). They run at `max_ulp=1`, or `exact` for the integer and data cases.

## What was ported (`heliades/core/object_transforms.py`)

| Helios | Heliades |
|---|---|
| `helios::makeRotationMatrix(float, vec3)` / `(float, vec3 origin, vec3 axis)` | `make_rotation_matrix_axis`, `make_rotation_matrix_about` |
| `helios::makeScaleMatrix(vec3, vec3 point)` | `make_scale_matrix_about_point` |
| `applyTranslationToMatrix` / `translateObjectPrimitives_private` (adds the shift to the translation column for affine matrices) | `_apply_translation` |
| `Context::copyPrimitive` (both overloads), `copyPrimitiveData` | `copy_primitive`, `copy_primitive_data` |
| `Context::copyObject` (both overloads) | `copy_object` |
| `Context::deletePrimitive` (both overloads) | `delete_primitives` (see "Needed in scene.py") |
| `Context::deleteObject` (both overloads) | `delete_object` |
| `Context::translateObject`, `CompoundObject::translate` | `translate_object` |
| `Context::rotateObject` (axis string / vec3 axis / origin + axis), `CompoundObject::rotate` x3 | `rotate_object(oids, rad, axis, origin=None)` |
| `Context::rotateObjectAboutOrigin` | `rotate_object_about_origin` |
| `Context::scaleObject`, `scaleObjectAboutCenter`, `scaleObjectAboutPoint`, `scaleObjectAboutOrigin`, `CompoundObject::scale*` | `scale_object*` |
| `Context::rotatePrimitive` vec3 overloads (axis; origin + axis), `Patch/Triangle::rotate(vec3[, vec3])` | `rotate_primitive_about_axis(uuids, rad, axis, origin=None)` |
| `Context::scalePrimitiveAboutPoint`, `Primitive::scale(S, point)` | `scale_primitive_about_point` |
| `Tube::getNodes`, `Tube::getNodeRadii`, `Tile::getVertices`, `Tile::getCenter` | `tube_object_nodes`, `tube_object_node_radii`, `tile_object_vertices`, `tile_object_center` |

All of them accept one ID or a list of IDs.

The object-local data model follows Helios. An object transform left-multiplies the same matrix onto
`CompoundObject.transform` and onto every member's `Primitive.transform`. Translation is the
exception: it adds the shift to the translation column, as Helios does. Type-specific data stays in
the local frame and is never rewritten: tube `nodes` and `radius`, and tile `subdiv`. The getters
apply the object transform when they read. `getObjectCenter` is the bounding-box center of the
current member vertices, as it was before this package.

`copyObject` deep-copies the whole `CompoundObject`, including `extra`, and copies the transform
verbatim. `copyPrimitive` deep-copies the whole `Primitive`, so it keeps fields that other
packages add later. It also copies the primitive's data. The copy gets `parent_object = 0`.
Helios 1.3.98 deliberately no longer inherits the parent.

## Helios / PyHelios discrepancies (each with a minimal PyHelios repro)

1. **A copied object's `object_origin` is reset to (0,0,0) (Helios quirk; Heliades matches it).**
   `Context::copyObject` builds the new object with its constructor and never sets `object_origin`,
   so `vec3()` gives (0,0,0). As a result, `rotateObjectAboutOrigin` and `scaleObjectAboutOrigin`
   on a copy pivot about the *world* origin, while on the original they pivot about the creation
   center.
   ```python
   t = ctx.addTileObject(vec3(5,0,0), vec2(1,1), SphericalCoord(1,0,0), int2(1,1), RGBcolor(0,1,0))
   k = ctx.copyObject(t)
   ctx.rotateObject(t, pi/2, vec3(0,0,1), about_origin=True)  # center stays (5,0,0)
   ctx.rotateObject(k, pi/2, vec3(0,0,1), about_origin=True)  # center -> (0,5,0)
   ```
   plantarchitecture never calls the about-origin variants, so the tomato is not affected. This is
   a candidate upstream bug: the copy should probably carry the source origin. It is covered by
   `core.object_transforms.origin`.
2. **`translateObject` does not move `object_origin` (Helios behaviour; Heliades matches it).**
   After a translation, about-origin rotation and scaling still pivot at the creation center. For
   example, a tile created at (5,0,0) and translated by (0,3,0) ends up at (5,-3,0) after
   `rotateObject(pi, z, about_origin=True)`. The same holds for rotate and scale.
3. **PyHelios docstrings disagree with C++ (binding documentation only).**
   - `rotateObject(obj, a, vec3_axis)` and `rotatePrimitive(uuid, a, vec3_axis)` say they rotate
     "about the object/primitive center". C++ rotates about the axis through the **world origin**:
     a tile at (5,0,0) rotated by pi/2 about z moves to (0,5,0).
   - `scalePrimitive(uuid, s)` with no point says it scales "about primitive center". C++ uses
     `makeScaleMatrix(S)`, which scales about the world origin: a patch at (1,1,0) scaled by 2 ends
     up at (2,2,0).
   - In the code, the `about_origin` branch of `rotateObject` is commented "Rotate about global
     origin". Its docstring correctly says `object_origin`.
4. **PyHelios rejects a zero axis vector** (`ValueError: axis vector cannot be zero`). C++ accepts
   one: the guarded `vec3::normalize` leaves it at zero, which gives a matrix with `cos` on the
   diagonal (a scale, not a rotation). Heliades follows C++. The parity cases never pass a zero
   axis.
5. **Compound-object members are transform-locked in Helios.** `translatePrimitive`,
   `rotatePrimitive` and `scalePrimitive` on a tile or tube member print a warning and do nothing.
   Polymesh is deformable, so its members do move. The W1-B methods implement the lock
   (`_transform_locked`), but the pre-existing `Scene` methods do not. See below.

## Needed in files W1-B does not own (for the coordinator)

- **`scene.py`, `Scene.delete_primitive`** only removes the primitive. Helios also removes it from
  its parent object's member list, and deletes the object once it is empty. `Scene` defines the
  method itself, so it shadows the mixin. Please replace its body with
  `self.delete_primitives(uuid)`, or delete it and rename `delete_primitives` to
  `delete_primitive` in `object_transforms.py`.
- **`scene.py`, `Scene.translate_primitive` / `rotate_primitive` / `scale_primitive`** ignore the
  compound-object lock (item 5). For a tile member, Helios leaves the transform unchanged (repro:
  translate, rotate and scale a 1x1 tile's patch, and `getPrimitiveTransformationMatrix` is still
  the identity), while Heliades moves the patch. Fix: return early when
  `self._transform_locked(self.primitives[uuid])` (the helper is in the mixin). The existing
  `core.primitive.transforms` case only uses free primitives, so it does not see this.
- **W1-A (textures):** `copy_primitive` already reproduces Helios's
  `solid_fraction = getArea() / (size.x * size.y)` for textured patches, gated on
  `Primitive.texture_file`. If W1-A uses a different field to mark a patch as textured, update that
  gate. Copying is otherwise generic (deepcopy), so new UV or texture fields carry over.
- **Object data:** `copy_object` and `delete_object` copy or drop `self.object_data[label][oid]` if
  such a dict exists. Nothing defines it yet.
- **Tile/tube getters:** the four getters live in the mixin only because they are needed to check
  local-frame data. They can move to `objects.py` together with the other object getters.

## Not done

- Polymesh internal topology (`vertices`, `faces`, `face_UUIDs`, `syncVerticesFromPrimitive`,
  `removeFacesAndCompact`). Heliades' polymesh does not store topology yet, so there is nothing to
  remap on copy or compact on delete.
- Sphere, box, disk, cone and adaptive-tile copies. These are handled generically, because `extra`
  is deep-copied, but they have no parity case until W1-C lands. Cone's `scaleConeObjectLength`
  and similar are out of scope.
- Voxels: rotating about a vec3 axis skips them, as Helios does. No voxel creation exists yet.

## Parity cases (`heliades/parity/cases_object_transforms.py`)

| case | tier | seeds | max ulp |
|---|---|---|---|
| `core.object_transforms.fuzz`: 70 random create (patch/tri/tile/tube/polymesh), copy, transform and delete ops | ulp (1) | 12 | 0 |
| `core.object_transforms.stored_data`: tube nodes/radii, tile corners/center, and every primitive matrix after fuzzing (stream, no abs floor) | ulp (1) | 8 | 0 |
| `core.object_transforms.origin`: about-origin semantics on originals, copies and copies of copies | ulp (1) | 4 | 0 |
| `core.object_transforms.primitive_axis`: vec3-axis rotation and scale-about-point on free, locked and polymesh members | ulp (1) | 6 | 0 |
| `core.object_transforms.delete`: membership and IDs after partial, emptying and object deletes | exact | 4 | 0 |
| `core.object_transforms.leaf_pattern`: the plantarchitecture copy, scale, rotate x/y/z, rotate about base, translate, rescale-about-point sequence | ulp (1) | 6 | 0 |
| `core.object_transforms.primitive_data`: data carried by copies and dropped by deletes | exact | 3 | 0 |

**Mutation check.** Each mutation was monkeypatched in, one at a time:

| mutation | caught by |
|---|---|
| reassociated product in one Rodrigues entry | fuzz, stored_data, origin |
| 1e-7 offset in the scale-about-point shift | fuzz, stored_data, origin, leaf_pattern |
| 1 ulp in the translate shift | fuzz, stored_data, origin, leaf_pattern |
| copy keeps the source origin | fuzz, stored_data, origin |
| member lock disabled | fuzz, stored_data, primitive_axis |
| emptied objects not deleted | fuzz, stored_data, delete |
| copy inherits the parent ID | fuzz, stored_data |

The snapshot compare has a 1e-6 absolute floor, which hides ulp-level noise at unit scale. That is
why `stored_data`, a stream case without the floor, is the case that catches pure-rounding mutations.
