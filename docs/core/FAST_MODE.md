# Fast mode: `Scene(mode="fast")`

Exact mode (the default, `Scene()`) builds the same plants as Helios 1.3.98 to the last bit. Fast mode builds the
same plants with the same random numbers, but lets the member geometry differ from Helios by float32 rounding, so
that work Helios does eagerly on every growth sub-step can be merged or deferred.

```python
from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture

sc = Scene(mode="fast")                 # a heliades.core.fast.FastScene; the whole Scene API
sc.seed_random_generator(1)
pa = PlantArchitecture(sc)
pa.loadPlantModelFromLibrary("tomato")
pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 120.0)
snap = sc.snapshot()                    # every read sees current geometry; sc.flush() forces it

# many plants across processes
from heliades.plantarchitecture.batch import PlantJob, build_plants
res = build_plants([PlantJob("tomato", 120, seed=s) for s in range(1, 65)], workers=16, out_dir="out/")
```

The plant-architecture code does not know about the mode, except three model-level fast paths that test
`ctx.mode == "fast"` (see below). Command line: `scripts/bench/plant_build.py heliades tomato 120 --mode fast`,
`scripts/bench/batch_throughput.py`, `scripts/bench/fast_survey.py`.

## What fast mode guarantees

* **The same random draws** as exact mode (and so Helios): same number, same order, same final generator state.
  Fast mode itself never draws and every draw-dependent branch reads plant-side state, object matrices or tube
  nodes/radii, which are exact. The only way the streams part is a growth decision that reads geometry flipping and
  changing the plant (then later draws follow the other plant): this happened once in all the runs (pistachio seed 19
  of 32: 352 instead of 336 fruits, 15516 instead of 15449 draws), and exact mode with every primitive area moved one
  float32 ulp up builds exactly the fast-mode plant, draws included. All 32 models x seeds 1..8: same draws.
* **Bit-identical object matrices and tube nodes/radii.** Object transforms update the object matrix with exact
  mode's float32 operations; tube bakes use exact mode's float32 node/radius arithmetic. Only member primitive
  matrices (and the tube ring vertices) are approximated.
* **Every read sees current geometry**, through any API: object/primitive getters flush what they read, and any
  raw read of the primitive matrix array (`scene._store.T`, snapshots, OBJ/XML/USD export, code that does not know
  about fast mode) flushes everything first. Copying an object carries its pending state.
* **The same topology** (primitives, UUIDs, types, parents, organ labels, materials, colors, textures, objects and
  their members, plant-structure dict up to its float values) for every model whose growth decisions do not read
  Context geometry. Geometry then differs from exact mode by float32 rounding plus what the model itself amplifies
  (next section).

## What it does not guarantee

* **Bit-exactness.** Member matrices are `float32(P @ T)` with `P` the float64 product of the pending operations,
  tube rings are rebuilt from float64 frames, triangles from one float64 product, any summation order is allowed.
* **Identical plants where growth reads geometry.** Leaf areas are read back from geometry (`getObjectArea`,
  the season bookkeeping) and feed internode girth, which moves child-shoot origins (`0.9 * parent radius`), which
  feeds every downstream position and angle. One float32 rounding of a leaf area already changes exact mode's own
  output by up to ~1e-4 of the plant size (the *sensitivity floor* below). Where such a value meets a threshold or a
  discrete choice, the plants diverge:

  | model | what reads geometry | effect |
  |---|---|---|
  | cowpea, pistachio, easternredbud | leaf inclination tracking (facet normals and areas pick target bins every sub-step) | same topology, leaf poses differ (up to a few % of the plant size) |
  | bougainvillea, capsicum | petiole tube creation thresholds (`makePetioleTube` minimum length / radius) crossed one sub-step apart after the girth feedback | a tube (and its UUIDs) appears one sub-step earlier or later: different UUID/object numbering |

  For these, fast mode is checked statistically (32 seeds, `compare_distributions` on primitive / object / shoot /
  leaf / fruit counts, leaf area, height, bounding box).
* **The sign of zero.** Fast mode skips some no-op work (a zero petiole-base shift, re-setting a data value to the
  same bits), which in Helios turns `-0` coordinates into `+0`.
* **Helios's error-path partial updates** for rotations and scales: a call on a list with a missing object raises
  before changing anything (exact mode, like Helios, changes the objects before it first). Fast mode also assumes
  affine object matrices for its pending member transforms (every Context object is).
* Model-level fast paths: grape cluster packing sums overlap pushes in any order; the fruit vertex weld welds near
  vertex chains to one vertex instead of falling back on the sequential greedy weld; both change the result by
  rounding only on the library models (same materials and topology).

## How it works (`heliades/core/fast.py`)

* **Deferred, composed object transforms.** `translateObject` / `rotateObject` / `scaleObject*` keep the object
  matrix exact and compose the operation into a pending float64 4x4 `P` per object. On a read the members get
  `T <- float32(P @ T)`, batched over every pending object in one product (translation-only `P`: one add).
* **Deferred tube rebuilds.** `setTubeNodes`, `setTubeRadii`, `appendTubeSegment` keep the nodes and a per-node ring
  radius `c` (the "recipe"); ring vertices are always `node + c * (cos * radial + sin * orthogonal)` on the parallel
  transport frame of the current nodes, which is what Helios's eager sequence produces up to rounding
  (`setTubeRadii` rescales along the current ring offsets: `c <- sign(c) * r`, so a collapsed ring stays collapsed;
  `appendTubeSegment` recomputes only the last two rings and earlier rings depend only on earlier nodes). Rings and
  member triangles are rebuilt once, on read. A tube whose rings stop being frame rings (an object rotation baked into
  it, `scaleTubeGirth`, `scaleTubeLength`, `pruneTubeNodes`) drops its recipe and is edited like in exact mode.
* **Flushing.** `FastPrimitiveStore.T` is a property that flushes all pending work before handing out the array;
  store methods that only append rows use the raw array. Object and primitive getters (areas, vertices, bounding boxes,
  centers, proxies `scene.primitives[u]`) flush only the objects they read, which is what keeps the deferral
  effective during growth (leaf areas are read every sub-step).
* **Cheaper bookkeeping.** The libstdc++ hash order of `Context::primitives` (needed only by material operations
  that iterate it) is logged and replayed on first iteration; whole-object deletions and copies are row operations;
  copied primitive data entries are shared (entries are never modified in place).

## Accuracy (32 models, Context seeds 1..8)

ACCURACY_TABLE

## Speed

SPEED_TABLE
