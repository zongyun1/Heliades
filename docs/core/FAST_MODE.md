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
  | cowpea, pistachio, easternredbud | leaf inclination tracking (facet normals and areas pick target bins every sub-step) | same topology for seeds 1..8, leaf poses differ (up to 2-3 % of the plant size); pistachio seed 19: a different plant (more fruits) and draw stream |
  | bougainvillea (6 of 8 seeds), capsicum (3/8), olive 548 d (1/8) | tube creation thresholds (`makePetioleTube` minimum length / radius) reached one sub-step apart after the girth feedback | a tube (and its UUIDs) appears one sub-step earlier or later: different UUID / object numbering, same draws |
  | rice | same mechanism | same topology in fast mode for seeds 1..8, but exact mode changes topology under a one-ulp area change (sensitivity floor), so it can diverge |

  Every one of these is a property of the model, not of fast mode: exact mode with every primitive area moved by
  one float32 ulp shows the same kind of change (topology changes for bougainvillea, capsicum, olive and rice;
  pistachio seed 19 then builds exactly the fast-mode plant). For them fast mode is checked statistically: 32 seeds,
  `compare_distributions` (KS test p >= 1e-3, mean within 5 %) on primitive / object / shoot / leaf / fruit counts,
  total leaf area, height and bounding box; all pass (smallest KS p 0.16, largest mean difference 0.19 %).
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

`scripts/bench/fast_survey.py` (`heliades.parity.fastmode.check`): each model at the age in its `AGES`, Context seeds
1..8, built in both modes. *rel dev* is the largest vertex deviation over all primitives divided by the plant's
bounding-box diagonal (max over the seeds with the same topology, and median). *floor* is exact mode against exact
mode with every primitive area moved one float32 ulp up / down (seed 1): the deviation the model itself produces
from one rounding of its leaf areas. "inf" = that perturbation already changes the topology in exact mode.

| model | age (d) | same draws | same topology | max abs dev (m) | max rel dev | median rel dev | floor (exact, 1 ulp area) | exact / fast build (s, mean; 6-10 builds in parallel, before the last optimisations) |
|---|---|---|---|---|---|---|---|---|
| almond | 548 | 8/8 | 8/8 | 3.2e-04 | 6.2e-05 | 6.4e-06 | 2.4e-06 | 24.4 / 13.8 |
| almond_aldrich | 548 | 8/8 | 8/8 | 1.7e-03 | 3.4e-04 | 1.1e-04 | 1.2e-04 | 23.9 / 22.3 |
| almond_independence | 548 | 8/8 | 8/8 | 7.3e-04 | 1.6e-04 | 6.1e-05 | 4.1e-05 | 12.7 / 13.4 |
| almond_wood_colony | 548 | 8/8 | 8/8 | 3.1e-04 | 8.5e-05 | 2.9e-05 | 1.0e-04 | 13.3 / 9.8 |
| apple | 548 | 8/8 | 8/8 | 5.2e-04 | 1.3e-04 | 3.9e-05 | 6.7e-05 | 14.1 / 12.6 |
| apple_fruitingwall | 548 | 8/8 | 8/8 | 2.0e-04 | 4.8e-05 | 2.4e-05 | 3.1e-05 | 25.0 / 13.1 |
| asparagus | 120 | 8/8 | 8/8 | 1.5e-04 | 1.4e-04 | 1.9e-05 | 2.3e-05 | 0.6 / 0.4 |
| bean | 60 | 8/8 | 8/8 | 3.4e-05 | 1.9e-05 | 4.4e-07 | 1.6e-05 | 1.2 / 0.6 |
| bindweed | 60 | 8/8 | 8/8 | 3.0e-06 | 1.7e-06 | 1.5e-06 | 0.0e+00 | 2.3 / 1.4 |
| bougainvillea | 180 | 8/8 | 2/8 | 1.2e-05 | 3.3e-06 | 2.6e-06 | inf (topology changes) | 5.5 / 3.2 |
| butterlettuce | 50 | 8/8 | 8/8 | 5.1e-07 | 3.4e-07 | 2.5e-07 | 0.0e+00 | 2.1 / 2.4 |
| capsicum | 160 | 8/8 | 5/8 | 1.2e-04 | 4.0e-05 | 1.9e-06 | inf (topology changes) | 2.4 / 1.9 |
| cheeseweed | 50 | 8/8 | 8/8 | 3.9e-06 | 2.9e-06 | 3.2e-07 | 1.8e-10 | 0.4 / 0.1 |
| cherrytomato | 110 | 8/8 | 8/8 | 1.2e-05 | 4.1e-06 | 2.1e-06 | 1.3e-09 | 2.1 / 1.3 |
| cowpea | 100 | 8/8 | 8/8 | 6.7e-02 | 2.1e-02 | 1.4e-02 | 1.5e-02 | 10.0 / 8.0 |
| cucumber | 165 | 8/8 | 8/8 | 5.1e-05 | 4.9e-06 | 2.7e-06 | 4.9e-08 | 13.2 / 8.4 |
| easternredbud | 548 | 8/8 | 8/8 | 6.8e-02 | 2.6e-02 | 1.7e-04 | 2.1e-06 | 7.7 / 6.0 |
| grapevine_VSP | 120 | 8/8 | 8/8 | 1.1e-04 | 2.7e-05 | 1.4e-05 | 4.1e-05 | 14.4 / 8.3 |
| grapevine_Wye | 120 | 8/8 | 8/8 | 2.5e-05 | 5.3e-06 | 4.9e-06 | 2.3e-08 | 21.7 / 13.3 |
| groundcherryweed | 90 | 8/8 | 8/8 | 7.3e-04 | 3.4e-04 | 1.0e-04 | 1.1e-04 | 10.8 / 5.6 |
| maize | 120 | 8/8 | 8/8 | 2.4e-06 | 7.2e-07 | 6.6e-07 | 1.4e-09 | 9.1 / 6.7 |
| olive | 548 | 8/8 | 7/8 | 6.1e-05 | 2.1e-05 | 2.0e-06 | inf (topology changes) | 6.5 / 3.5 |
| pistachio | 548 | 8/8 | 8/8 | 7.7e-02 | 1.9e-02 | 1.1e-05 | 9.2e-06 | 17.4 / 15.3 |
| puncturevine | 45 | 8/8 | 8/8 | 3.0e-06 | 1.0e-06 | 1.0e-06 | 0.0e+00 | 74.5 / 66.5 |
| rice | 90 | 8/8 | 8/8 | 1.5e-04 | 8.7e-05 | 1.2e-06 | inf (topology changes) | 4.7 / 3.1 |
| sorghum | 100 | 8/8 | 8/8 | 1.8e-06 | 7.9e-07 | 7.0e-07 | 5.2e-08 | 53.5 / 43.2 |
| soybean | 70 | 8/8 | 8/8 | 8.3e-05 | 4.1e-05 | 2.3e-06 | 3.0e-05 | 1.4 / 0.9 |
| strawberry | 100 | 8/8 | 8/8 | 6.0e-07 | 2.1e-07 | 1.5e-07 | 6.4e-10 | 1.1 / 0.7 |
| sugarbeet | 120 | 8/8 | 8/8 | 4.5e-07 | 2.8e-07 | 2.0e-07 | 1.8e-09 | 1.1 / 0.5 |
| tomato | 120 | 8/8 | 8/8 | 2.0e-04 | 6.8e-05 | 5.4e-05 | 9.3e-05 | 4.0 / 2.0 |
| walnut | 548 | 8/8 | 8/8 | 1.1e-04 | 2.3e-05 | 1.5e-05 | 4.6e-05 | 6.6 / 3.9 |
| wheat | 90 | 8/8 | 8/8 | 7.2e-07 | 4.5e-07 | 2.9e-07 | 8.7e-10 | 1.3 / 0.9 |

Reading the table:

* Draws: identical for all 256 builds. Topology: identical except bougainvillea / capsicum / olive (thresholds, above).
* The target of 1e-5 relative is met by 13 of the 32 models (bindweed, bougainvillea, butterlettuce, cheeseweed,
  cherrytomato, cucumber, grapevine_Wye, maize, puncturevine, sorghum, strawberry, sugarbeet, wheat); the others have
  1.9e-5 .. 3.4e-4 (plus the three leaf-tracking models at ~2e-2). For most of those the floor (one seed, two perturbations) is of
  the same order, and where it is lower (almond 2.4e-6 against 6.2e-5 over 8 seeds) a single perturbation simply
  does not hit the seed's most sensitive spot: the deviation is the model's amplification of the leaf-area rounding (girth -> child-shoot origins -> downstream
  positions and angles), which exact mode shows too. Models without geometry feedback (bindweed, maize, sorghum,
  wheat, strawberry, sugarbeet, puncturevine) sit at 1e-7 .. 2e-6, the float32 rounding of composed transforms.
* Leaf areas agree to ~1e-7 relative; object matrices, tube nodes and radii are bit-identical by construction.

## Speed

Single process, seed 1, `scripts/bench/plant_build.py` build time (min of 2 fast runs; machine otherwise idle
except one other single-threaded job). Sorghum, maize and tomato remeasured after the `nameOrganMaterials` index
(a plant-code change that helps both modes; sorghum exact was 32.3 s before it).

| model, age | Helios 1.3.98 | Heliades exact | Heliades fast | fast vs exact | fast vs Helios |
|---|---|---|---|---|---|
| tomato 120 d | 1.28 s | 3.68 s | 1.99 s | 1.85x faster | 1.6x slower |
| grapevine_VSP 120 d | 1.33 s | 10.0 s | 6.44 s | 1.56x | 4.8x slower |
| grapevine_Wye 120 d | 2.37 s | 18.9 s | 10.5 s | 1.81x | 4.4x slower |
| maize 365 d | 1.49 s | 8.24 s | 6.00 s | 1.37x | 4.0x slower |
| sorghum 365 d | 5.14 s | 15.0 s | 13.9 s | 1.08x | 2.7x slower |
| cucumber 165 d | 0.95 s | 9.17 s | 6.07 s | 1.51x | 6.4x slower |
| puncturevine 45 d | 3.11 s | 59.0 s | 42.4 s | 1.39x | 13.6x slower |
| almond 913 d | 2.25 s | 57.9 s | 34.4 s | 1.68x | 15x slower |
| olive 1825 d | 15.9 s, 1.7 GB | 303 s, 1.3 GB | 172 s, 1.4 GB | 1.76x | 10.8x slower |

**Fast mode is not faster than Helios for a single plant on any of these models.** It removes most of the core's
share (the deferred object transforms and tube rebuilds; after it the core is 20-45 % of a build, mostly primitive
creation and material bookkeeping); what is left is the plant-architecture code itself, which is Python executing
Helios's per-phytomer, per-sub-step logic statement by statement (shoot node updates, petiole and leaf re-seating,
girth, tube frames, leaf-lattice deformation, season logic, C++-value-semantics parameter structs). Getting under
Helios per plant needs that code restructured (vectorised across phytomers / shoots) or compiled, which is beyond a
core package. The trees and puncturevine (many small organs, hundreds of sub-steps) are furthest away.

**Throughput** (`scripts/bench/batch_throughput.py`, `build_plants` with 14 spawned workers on the 16-core machine,
seeds 1..N, against Helios building the same model sequentially in one process):

| model, age | plants | Heliades fast, 14 workers | Helios, 1 process | ratio |
|---|---|---|---|---|
| tomato 120 d | 42 | 10,700 plants/h (14.1 s wall; 3.8 s per build in a loaded worker) | 2,880 plants/h | 3.7x |
| grapevine_VSP 120 d | 28 | 3,380 plants/h | 2,650 plants/h | 1.3x |
| almond 913 d | 14 | 500 plants/h | 1,420 plants/h | 0.35x |

A build in a busy worker is about 2x slower than alone (memory bandwidth, shared caches), so N workers give ~N/2
speed-up. Helios would of course scale the same way across processes; the comparison is the one asked for (a batch
of Heliades workers against one Helios process).
