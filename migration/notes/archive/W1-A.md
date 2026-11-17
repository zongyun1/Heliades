# W1-A textures: notes and Helios/PyHelios discrepancies

Baseline Helios 1.3.98. The repros use PyHelios built from `third_party/pyhelios_src`, with
`T = .../plugins/plantarchitecture/assets/textures/`. Heliades matches Helios in every item below.
None of them is fixed.

## What was ported

- `Context::addTexture`, the texture registry `Context::textures`, and `helios::Texture`
  (constructor, `getSolidFraction` with its cache, `computeSolidFraction`).
- `helios::readPNGAlpha`, `PNGHasAlpha`, `resolveFilePath`/`tryResolveFilePath` (+ `getBuildDirectory`),
  `getFileExtension`, `getImageResolutionJPEG`.
- `Context::addPatch(…, texture)`, `addPatch(…, texture, uv_center, uv_size)`, `addTriangle(…, texture, uv0..2)`.
- `addTileObject(…, texture[, texture_repeat])` (+ `correctTextureRepeat`) and `addTubeObject(…, texture[, textureuv_ufrac])`.
- Queries: `getPrimitiveTextureFile`, `getPrimitiveTextureUV`, `getPrimitiveSolidFraction`, `Primitive::hasTexture`,
  `getPrimitiveTextureSize`, `primitiveTextureHasTransparencyChannel`, `getPrimitiveTextureTransparencyData`.

## How Helios computes the solid fraction

These are the rules that decide whether a match is exact:

- **Alpha mask:** a pixel is solid iff its 8-bit alpha is `>= 250`, after libpng's read transforms:
  - `png_set_strip_16` keeps the high byte, so 16-bit alpha is solid iff `>= 64000`;
  - palette alpha comes from tRNS, and indices past the end of the table are opaque;
  - gray/RGB tRNS gives alpha 0 where the raw sample equals the key.
  - A PNG without an alpha channel or tRNS, and any JPEG, has solid fraction 1.
- **Per primitive:** the solid fraction is not the image-wide fraction. It is the fraction of opaque
  pixels among the pixel centres `((i+0.5)/(W-1), (j+0.5)/(H-1))` that pass all the half-space edge
  tests of the UV polygon, in float32. The mask is flipped vertically, so v = 0 is the bottom row.
  - The scan covers only the polygon's bounding box: `floor`/`ceil` of `uv*(W-1)`, clamped to the image.
- **Area:** a patch is `size.x*size.y*sf` and a triangle is `triangle_area*sf`. Scaling a primitive
  later does not recompute the solid fraction.

Pillow decodes the pixels. IHDR and tRNS are read from the raw chunks rather than from `im.info`,
because Pillow drops or rescales some tRNS forms. The `texture.png_formats` case covers all 104
libpng `contrib/testpngs` images plus synthetic edge cases. It matches at 0 ulp.

## Discrepancies and oddities (Helios behaviour is reproduced unless said otherwise)

### 1. The solid fraction depends on creation history (cache keyed by rounded pixel coordinates)

`Texture::getSolidFraction` caches by `round(uv * (res-1))` for each vertex. Two polygons whose UVs
round to the same pixels share the value of whichever was created first, even though their exact
coverage differs.

```python
for order in ((0, 1), (1, 0)):
    c = Context(); uvs = [[(0.3,0.3),(0.6,0.31),(0.45,0.62)], [(0.3001,0.3),(0.6,0.3101),(0.45,0.62)]]
    for k in order: c.addTriangleTextured(vec3(0,0,0), vec3(1,0,0), vec3(0,1,0), T+"TomatoLeaf.png", *map(lambda x: vec2(*x), uvs[k]))
# order (0,1): both 0.985505819     order (1,0): both 0.98549813
```

Heliades keeps a cache for each texture with the same key. The `texture.solid_fraction_cache` case
breaks if the cache is removed (mutation check). This is a candidate upstream issue: the result
should not depend on order.

### 2. PyHelios makes patch/triangle texture paths absolute but passes tile/tube paths through verbatim

`addPatchTextured` and `addTriangleTextured` call `_validate_file_path` (`os.path.abspath`).
`addTileObject` and `addTubeObject` pass `texturefile` unchanged. Helios stores the string it
receives as `Material::texture_file`, so `getPrimitiveTextureFile` reports:

```python
os.chdir(T); c = Context()
c.addPatchTextured(vec3(0,0,0), vec2(1,1), "TomatoLeaf.png")   # -> '/…/textures/TomatoLeaf.png'
c.addTileObject(subdiv=int2(1,1), texturefile="TomatoLeaf.png") # -> 'TomatoLeaf.png'
```

The registry is keyed by the string as given, so these two calls create two registry entries with
separate caches. Heliades follows C++ and stores the string verbatim. **Snapshot comparison uses the
basename** (`compare.texture_key`) because the full string depends on the binding and the cwd, not on
Helios. A relative string resolves against the cwd first, then against the build dir (`$HELIOS_BUILD`,
or a cwd/parent that contains `plugins/`). `test_relative_paths_match_helios` checks this against
the oracle.

### 3. The batch texture-file getter is wrong for non-ASCII paths (PyHelios)

`Context.getPrimitiveTextureFile(list)` decodes the concatenated buffer to `str` and slices it with
the C **byte** offsets:

```python
d = ".../blätter/"; [c.addPatchTextured(vec3(0,0,0), vec2(1,1), d+"TomatoLeaf.png") for _ in range(3)]
c.getPrimitiveTextureFile(c.getAllUUIDs())
# ['/…/blätter/TomatoLeaf.png/', 'tmp/…/blätter/TomatoLeaf.png/t', 'mp/…/blätter/TomatoLeaf.png']
```

The fix is to slice the bytes before decoding. `helios_snapshot` avoids the problem by reading
texture files one UUID at a time.

### 4. The textured `addPatch` without UVs skips the size check

The colored overload and the `uv_center`/`uv_size` overload reject `size < 1e-6`. The plain textured
overload only `assert`s `size > 0`, and the assert is compiled out in release builds:

```python
c.addPatchTextured(vec3(0,0,0), vec2(1e-8,1e-8), T+"TomatoLeaf.png")                   # accepted, area 5.7e-17
c.addPatchTextured(vec3(0,0,0), vec2(1e-8,1e-8), T+"TomatoLeaf.png", uv_center=…, uv_size=…)  # rejected
```

A size of exactly 0 is still rejected, by `Primitive::scale`. Heliades does the same.

### 5. UVs are clamped only from above

The Patch/Triangle constructors apply `std::min(uv, 1.f)`. Negative UVs are kept, and the
solid-fraction bounding box clamps them to pixel 0. For example, the UVs `(-0.5,0), (1.5,0), (0.5,2)`
are stored as `(-0.5,0), (1,0), (0.5,1)`.

### 6. The textured tube deletes zero-area triangles; the colored tube does not

`addTubeObject(…, texture)` deletes each triangle whose `getPrimitiveArea` (which includes the solid
fraction) is not `> 0`. The deleted triangle's UUID has already been consumed, so fully transparent
texture regions leave gaps in the UUIDs. The colored overload keeps every triangle. Heliades does the
same: `texture.tube` produces 56 gaps on seed 0.

### 7. Images one pixel wide or tall give inf/NaN edge tests

When `W == 1` or `H == 1`, `1/(W-1)` is `inf`. Each pixel-centre coordinate becomes `inf`, and an edge
coefficient of 0 turns `0*inf` into `NaN`. `if (L < 0)` is false for NaN, so such pixels count as
*inside*. In practice most polygons on a 1-row image get solid fraction 0 (some edge gives `-inf`),
but zero-width UV triangles get a nonzero value. Heliades uses `~(L < 0)` to match. Comparing
`L >= 0` instead fails `texture.png_formats`.

### 8. Smaller notes

- PyHelios accepts `.tga`/`.bmp` in `_validate_file_path`, but Helios rejects anything other than
  PNG or JPEG in `addTexture`.
- A textured primitive's color is the material color, black `(0,0,0,1)`. `getPrimitiveColor` returns
  black for textured primitives.
- `getAllUUIDs()` (and therefore every `getAll*` batch getter) is not in UUID order.
  `helios_snapshot` passes an explicit sorted UUID list to the batch getters.
- The textured `addTileObject` stores the requested `texture_repeat` on the Tile, not the corrected
  one, and does not call `setColor` on the object.

## Known limitation (Heliades)

A 16-bit RGB PNG with a tRNS key: Pillow exposes only the high bytes, so a pixel whose high bytes
equal the key's is ambiguous. Heliades raises `NotImplementedError` in that case. When no pixel is
ambiguous, the result is exact (`rgb16_trns_unambiguous.png` in the cases). No Helios asset uses this
format. Gray tRNS on 1-, 2- and 4-bit images compares the raw sample with the key; a key outside the
bit-depth range is untested.
