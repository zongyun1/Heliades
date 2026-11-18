# Core performance: data layout, batched transforms, bit-exactness

Heliades must build the same plants as Helios 1.3.98 to the last bit (`python -m heliades.parity`, 0 ulp)
while staying fast enough for dataset generation. This note describes how the core stores geometry, how
transforms are applied in bulk, why that stays bit-exact, and the rules for adding new bulk operations.

## Data layout (structure of arrays)

`heliades/core/store.py`

* **`PrimitiveStore`**: one growable numpy array per primitive field, **indexed by UUID**. Helios hands out
  UUIDs in increasing order and never reuses them, so the UUID is the row; deleted (and skipped) UUIDs are
  dead rows (`alive[u] == False`). Capacity doubles on demand (`reserve`).

  | field | dtype / shape | Helios |
  |---|---|---|
  | `T` | float32 (cap, 16) | `Primitive::transform` (row-major 4x4) |
  | `ptype` | int8 | `PrimitiveType` (0 patch, 1 triangle, 2 voxel) |
  | `color` | float32 (cap, 4) | `Primitive::color` (RGBA) |
  | `parent` | int64 | `parent_object_ID` |
  | `sf` | float32 | `solid_fraction` |
  | `mat` | int64 (-1 = none yet) | `materialID` |
  | `tex` | object | texture file string or `None` |
  | `uv`, `nuv` | float32 (cap, 4, 2), int8 | texture (u,v) (`nuv == 0`: Helios's empty vector) |
  | `alive` | bool | primitive exists |
  | `extra` | dict UUID -> dict | any other per-primitive attribute (rare; deep-copied on copy) |

* **`Scene.primitives`** is a `PrimitiveMap`: a dict-like `UUID -> Primitive` view (`[]`, `get`, `in`,
  `len`, iteration in increasing UUID order, `del`). It hands out **`Primitive` proxies**
  (`heliades/core/primitives.py`): `p.transform`, `p.color`, `p.uv` return copies, assignments write the
  row, `p.vertices()/area()/normal()/scale()/rotate()/translate()` are the scalar op-for-op ports. Use
  proxies for one-off, per-primitive logic; use rows for anything in bulk.
* **`CompoundObject`** keeps its member UUID list (`obj.uuids`, assignment invalidates the cache) and
  caches `member_index()`: a `slice(a, b)` when the members are consecutive UUIDs (almost always: an
  object's primitives are created or copied together), else an int64 index array. Slices make row
  operations views instead of gathers.
* Object data, primitive data (`label -> {id: TypedData}`), materials, textures, the libstdc++ hash order
  of `Context::primitives` (`primitive_order`) and hidden flags are unchanged dict/set structures.
  `TypedData` entries are never modified in place (setters replace them, getters copy), so a copied
  primitive shares an entry that holds only immutable values.

## Transforms: eager and batched across primitives

Helios composes nothing lazily: `translateObject` adds the shift to the translation column of the object
matrix and of every member primitive's matrix (`applyTranslationToMatrix`, with a `T*M` fallback for a
non-affine matrix); `rotateObject` / `scaleObject*` build one 4x4 `T` and left-multiply it
(`helios::matmult`) onto the object matrix and onto each member matrix; vertices, areas and normals are
computed from the matrix on read. Heliades does exactly the same, but **one numpy operation per call over
all member rows** instead of one Python call per primitive:

* `PrimitiveStore.translate_rows(rows, shift)`: `T[rows, 3:12:4] += shift` for affine rows (the common
  case is one in-place op on a slice), `T(shift) * M` for the others.
* `PrimitiveStore.left_multiply_rows(rows, T)` -> `matmult_rows`: `M[4i+j] = 0; M[4i+j] += L[4i+k] * R[4k+j]`
  for k = 0..3, every entry the same sequence, vectorised over rows (contiguous columns for large batches).
* A call on **several objects** (`translate_object([o1, o2, ...])`, `rotate_object`, `scale_object*`) is one
  row operation over the concatenated members when the objects exist, are distinct and share no member
  (`ObjectTransformsMixin._batch`); otherwise it falls back to object by object, which is what Helios does.
* `copy_object` / `copy_primitive(list)` copy rows (`copy_rows`) instead of `deepcopy`ing Python objects,
  then redo, in order, the per-copy bookkeeping Helios does (textured-patch solid fraction, primitive data,
  hash-order insert, material reference / texture-colour override).
* Bulk getters: `snapshot()`, `object_area`, `object_bounding_box`, `primitive_areas(uuids)`,
  `primitive_vertices_rows(uuids)`, `get_primitive_material_labels(uuids)`; bulk creation:
  `add_triangles(v0, v1, v2, color)`; tube triangle rebuilds (`setTubeNodes/Radii`, `appendTubeSegment`)
  rebuild all member triangles in one `triangle_transform_rows`.

**Why not lazy (deferred) transforms?** Deferring would only pay if consecutive operations could be
merged, and they cannot: `(M + s1) + s2 != M + (s1 + s2)` in float32, and `T2*(T1*M) != (T2*T1)*M`. Every
primitive matrix must see every operation, in order, with its own rounding, so a deferred queue performs
the same arithmetic later. The only saving would be fewer numpy calls by batching queued operations across
objects at flush time, against flush hazards on every read path (vertices, areas, copies, bounding boxes,
tube data, XML). After the eager batching, the remaining object-transform cost is a minor share of a build
(see the profile below), so the complexity was not worth it. If it becomes worth it, the natural place is a
per-object op queue flushed per object on read and globally before `snapshot()`; the row kernels above are
the flush primitives.

## Why the batched code is bit-exact

1. Every row sees **the same float32 operation sequence** as the scalar C++ code, in the same order.
   Vectorisation is only across rows / vertices / stations, never inside one expression.
2. numpy's elementwise float32 `+ - * /` and `sqrt` are IEEE-754 correctly rounded, as is the x86-64 SSE
   code of the Helios build; no FMA contraction happens in numpy.
3. Running sums use `np.add.accumulate` / `np.subtract.accumulate` along an axis, which is a strictly
   sequential loop (`out[i] = out[i-1] op a[i]`). Never `sum`, `np.add.reduce`, `@`, `dot` or `einsum`:
   they use pairwise or BLAS summation orders (and BLAS may use FMA).
4. Masked terms of a running sum are made exact no-ops: subtracting `+0` and adding `-0` leave every float,
   including the sign of a zero, unchanged (`x - (+0) == x`, `x + (-0) == x`), so a dense accumulation over a
   masked pair matrix equals the sparse C++ loop (grapevine cluster overlap pass).
5. All operands stay float32: Python scalars are weak under NEP 50, and float64 only appears where Helios
   uses double (e.g. object area accumulation, `acos` in tube frames), done element by element.
6. libm calls stay glibc scalar calls through ctypes (`sinf`, `cosf`, `acosf`, `powf`, double `sin/cos/acos`
   via `math`), one call per element. numpy's `np.sin`/`np.cos` are different implementations.
7. Data-dependent control flow (clamps, branches, `std::min`/`max` NaN behaviour) is reproduced with
   `np.where` using the same comparison (`min(1, x)` keeps 1 for NaN: `np.where(x < 1, x, 1)`), or kept scalar.
8. Fast paths that rely on a property of the data check it first and fall back on the scalar code otherwise
   (non-affine matrices, missing or duplicate members, the vertex weld's conflict cliques, ...).

Each batched kernel has a test against the scalar port it replaced (`tests/core/test_store.py`,
`tests/plantarch/test_vectorised_kernels.py`), and the parity suite checks whole plants against Helios.

## Adding a new bulk operation

* Write (or keep) the scalar, statement-by-statement port first; tag it with `@ports`.
* Vectorise across independent elements only; keep each element's op order (write the expression with the
  same parenthesisation as the C++; `a + b + c` is `(a + b) + c`).
* Read rows through `scene._store` (`T`, `ptype`, `alive`, ...) or the batch getters above; write rows
  with slices when the members are consecutive (`obj.member_index()`), fancy indices otherwise. Check
  `alive` and bounds, and fall back on the scalar path when an error would be raised mid-way (Helios's
  partial-update semantics).
* Anything order dependent in Helios (texture solid-fraction cache, material creation, RNG draws,
  `primitive_order` inserts) stays sequential, in Helios's order.
* Add a test comparing the batched kernel with the scalar one bit for bit (`view(np.uint32)`), including
  zeros of both signs, NaN-free degenerate inputs and the fallback cases.

## GPU (Warp) spike

`scripts/bench/gpu_fma_spike.py`: Warp 1.19 compiles CUDA kernels with NVRTC `--fmad=true` by default
(module option `fuse_fp=True`), so `a*b + c` may be contracted into one FMA. With
`module_options={"fuse_fp": False}` (or `wp.set_module_options`) Warp passes `--fmad=false`. On 1e6 random
matrices / vertices (RTX 5060 Ti):

| kernel | `fuse_fp=True` | `fuse_fp=False` |
|---|---|---|
| batched `helios::matmult` (16 entries each) | 3,656,682 entries differ | 0 differ |
| vertex transform (`T[0]x + T[1]y + T[2]z + T[3]`) | 657,337 coordinates differ | 0 differ |

So a bit-exact GPU path for the batched transforms is viable (with `fuse_fp=False`, `fast_math=False`,
and the same expression order). It is not used: the eager batches are small (one object, typically 10^2 to
10^3 rows), where a launch plus transfer costs more than the numpy op, and whole-scene reads (`snapshot`)
already take well under a second on the CPU. A GPU flush becomes interesting with a deferred-transform
design or for many plants at once (dataset generation of canopies), with the store kept on the device.
