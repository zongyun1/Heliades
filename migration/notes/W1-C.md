# W1-C shapes: notes

Ported from Helios 1.3.98 (`core/src/Context_object.cpp`, `Context.cpp`, `global.cpp`) into
`heliades/core/shapes.py`. Every parity case is **bit-exact** (0 ulp) against the oracle.

## Ported

- `Context::addSphereObject`: scalar and vec3 (ellipsoid) radius, RGB or default green.
- `Context::addDiskObject`: `uint` and `int2` Ndivs, rotation, RGB or RGBA, default opaque red.
- `Context::addConeObject`: RGB or default green.
- `Context::addBoxObject`: RGB or default green, `reverse_normals`.
- `Context::addTile`: loose patches, used by the box.
- `helios::vecmult`.
- Getters:
  - sphere: `getSphereObjectCenter/Radius/SubdivisionCount/Volume`;
  - disk: `getDiskObjectCenter/Size/SubdivisionCount`;
  - cone: `getConeObjectSubdivisionCount/Nodes/NodeRadii/Node/NodeRadius/AxisUnitVector/Length/Volume`;
  - box: `getBoxObjectCenter/Size/SubdivisionCount/Volume`.

  They read the object transform the same way the C++ classes do, so they stay correct once W1-B
  object transforms update `CompoundObject.transform`.

`getObjectCenter` and `getObjectTransformationMatrix` are compared for every object in every
snapshot case.

## Findings for other packages (action needed outside W1-C)

1. **glibc `sinf`/`cosf` are not correctly rounded, and `vec.sinf`/`vec.cosf` assume they are.**
   Of the arguments `2*PI_F*j/N` with N < 64, 42 give a result that differs by 1 ulp from
   `float(sin(double(x)))`. The smallest N affected is 13 (j=7 sin, j=11 cos). `shapes.py` calls libm
   `sinf`/`cosf` through ctypes, which makes the cones and disks bit-exact.
   - Repro: `addTubeObject(13, [(0,0,0),(0.1,0.2,1),(0.3,0.1,2)], [0.5,0.4,0.3])` gives a vertex
     difference of up to 128 ulp against Helios today.
   - The existing tube fuzz uses Ndivs 3..11, so it never reaches N = 13.
   - Suggestion: the owner of `vec.py` (and `objects.py` for the tube) switches `sinf`/`cosf` to the
     libm calls in `shapes._sinf`/`_cosf`.
   - The same question applies to `make_rotation_matrix` in `primitives.py`, which uses double
     `sin`/`cos`. Disk rotations are bit-exact with it in every case here, which is consistent with
     Helios calling double `sin()` in `makeRotationMatrix`. Leave it as is.
2. **`helios_snapshot` drops alpha.** `getAllPrimitiveColors` returns RGB only and alpha is padded
   with 1, so an RGBA mismatch is invisible. `ops_shapes.py` adds a `snapshot_rgba` op that refills
   alpha from `getPrimitiveColorRGBA`. W1-A owns `snapshot.py`/`backends.py` and could fold this in.
3. **`compare_snapshots` has a `1e-6` absolute floor, which hides 1-ulp errors on O(1) geometry.**
   With only the snapshot cases, a deliberate 1-ulp `sinf` mutation in the cone went undetected.
   `core.shape.raw_stream` compares the same snapshot data as a stream with no floor, and it does
   catch that mutation.

## Helios behaviour reproduced (quirks, not fixed)

- **The disk object transform ignores rotation.** `addDiskObject` rotates each triangle (by
  +elevation about **y**, then azimuth about z, unlike patch/tile, which use -elevation about x),
  but the object transform is only `scale(size.x, size.y, 1)` followed by a translation.
  - Repro: `addDiskObject(4, (0,0,0), (1,1), SphericalCoord(1, 0.5, 0))` gives an identity
    transform while the primitives are tilted.
  - `getDiskObjectSize` and `getDiskObjectCenter` are still right. Any later transform logic that
    rebuilds the geometry from the object transform would lose the tilt.
- **`object_origin` differs by type.**
  - Sphere: Helios never sets it, so it stays `vec3()` = (0,0,0).
  - Disk and box: the center argument.
  - Cone: `getObjectCenter` (the bounding-box center).

  It is stored the same way in `CompoundObject.origin`. PyHelios can't observe it, but it matters
  to W1-B if any transform uses `object_origin`.
- **`getDiskObjectSubdivisionCount` returns only `Ndivs.x`.**
- **Degenerate inputs that Helios accepts (all bit-exact, including the NaN normals):**
  - disk size (0,0) or (1,0);
  - cone radii 0 or negative;
  - coincident cone nodes, where the axis falls back to +z. `getConeObjectAxisUnitVector` is then
    0/0 = NaN; PyHelios's `vec3` refuses NaN, so the ops call the raw ctypes wrapper.
  - sphere Ndivs 1 and 2 (caps only);
  - cone Ndivs 1 and 2.
- `addBoxObject`'s `subsize` is computed and never used in the colored overload.

## Deliberate deviations

These are inputs where Helios has undefined or empty behaviour. Heliades raises `ValueError`.

- Sphere `Ndivs = 0`: Helios builds an object with no primitives.
- Cone `Ndivs = 0`: Helios computes 0/0 and builds an empty object.
- Disk `Ndivs.x < 1` or `Ndivs.y < 1`: with `Ndivs.y = 0` Helios sizes a `std::vector` with a
  negative count, which is UB / `length_error`.

## PyHelios binding quirks (seen while choosing oracle calls)

- `Context.addSphereObject` rejects `ndivs < 3`, but C++ accepts 1 and 2.
- `Context.addDiskObject` with `int2` ndivs and no color substitutes gray (0.5,0.5,0.5). C++ has no
  colorless `int2` overload. With `uint` ndivs and no color, C++ uses red (1,0,0,1).
- `Context.addBoxObject(reverse_normals=True)` with no color raises. C++ has only a color+bool (or
  texture+bool) overload for it.
- `Context.addDiskObject` builds `SphericalCoord(radius, elevation, azimuth)` from the Python
  object, so it receives elevation, not zenith.

`ops_shapes.py` calls `pyhelios.wrappers.UContextWrapper` directly so that each C++ overload is
chosen explicitly.

## Cases (`heliades/parity/cases_shapes.py`)

All cases are tier `exact` with `max_ulp=0`. The observed maximum is 0 ulp on every check.

| case | seeds | covers |
|---|---|---|
| `core.shape.{sphere,disk,cone,box}` | 8 each | random fuzz, snapshot with RGBA |
| `core.shape.{sphere,disk,cone,box}_edges` | 1 each | minimum subdivisions, Helios `core/tests` inputs, degenerate sizes/radii, vertical/horizontal/down cones, axis parallel to the seed vector, tiny axis, reverse normals |
| `core.shape.{sphere,disk,cone,box}_getters`, `cone_edge_getters` | 6 each (1 for `cone_edge_getters`) | every getter PyHelios exposes (stream, no floor) |
| `core.shape.mixed` | 6 | interleaved shapes, patches and tiles (UUID and object-ID order) |
| `core.shape.raw_stream` | 8 | geometry, colors, object center and transform as a floorless stream |

## Mutation check

One small mutation at a time in `shapes.py`. Every mutation was caught by at least one case.

| mutation | caught by |
|---|---|
| sphere top-cap elevation moved 1 ulp | `sphere`, `sphere_edges`, `mixed`, `raw_stream` |
| sphere volume product reassociated | `sphere_getters` |
| cone `sinf` replaced by double-rounded `sin` | `raw_stream` (missed by the floored snapshot cases) |
| disk ring radius `size*r/ny` instead of `size/ny*r` | `disk`, `raw_stream` |
| box reverse-normal right face, azimuth 0 instead of PI | `box`, `box_edges`, `mixed`, `raw_stream` |
| cone length via double `hypot` instead of `powf` | `cone_getters` |

## Left for later

- Textured overloads of all four shapes (W2-B).
- Shape setters and modifiers: `scaleConeObjectLength/Girth`, `setSphere...`, and the other
  cone/sphere modifiers. These belong to W1-B or a later package.
- `getObjectArea` and the analytic vertex normals (`Sphere/Cone::getPrimitiveVertexNormals`).
- The non-object helpers `addSphere`, `addDisk`, `addCone` and `addBox`. They return UUID lists
  and are not in W1-C scope.
