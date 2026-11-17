# Heliades porting plan: road to a Heliades-grown tomato

Goal: `heliades` builds the Helios tomato from scratch. The same seed and age must give the same
plant as Helios 1.3.98 at **ulp** parity. The plant is then exported to USD, simulated, and
evaluated against TomatoWUR. Scope decisions are in `BUILD_VS_BUY.md`; porting rules are in
`PORTING.md`.

## How the work is parallelised

- **One agent per work package, each in its own git worktree and branch.** The coordinator
  reviews each diff, reruns `pytest -q tests` and `python -m heliades.parity` on main after
  merging, and pushes.
- **File ownership is strict.** An agent edits only the files its package lists. New backend
  operations go in `heliades/parity/suites/<group>/ops_<area>.py` (`@op` registry). New cases go in
  `heliades/parity/suites/<group>/cases_<area>.py`, with `<group>` = `core`, `plantarch`, or a new
  subpackage per Helios plugin, and `<area>` named by content (e.g. `grapevine`, not the package id).
  Everything under `suites/` is auto-loaded, so no shared adapter is edited. Tests go in `tests/<group>/`.
- **Shared-file changes belong to a single package per wave.** That package is named below.
- **Notes go in `migration/notes/<package>.md`; Helios discrepancies go straight into
  `migration/HELIOS_ISSUES.md`.** Notes of finished packages are in `migration/notes/archive/`.
- **Acceptance for every package:**
  - its new parity cases pass at the stated tier with fuzzing over several seeds;
  - the full suite stays green;
  - every ported function carries `@ports("Class::method")`;
  - a mutation check shows the new cases catch a small deliberate error.

## Wave 1 (parallel)

| pkg | scope | owns | depends |
|---|---|---|---|
| **W1-A textures** | textured patch/triangle (`addPatch(..., texture[, uv_center, uv_size])`, `addTriangle(..., texture, uv0..2)`). UVs, the texture registry, solid fraction from the PNG alpha mask, area × solid fraction. Textured tile and tube overloads. Snapshot and compare carry UV, texture file and solid fraction | `core/primitives.py`, `core/scene.py` (primitive add/query methods), `core/objects.py` (textured overloads only), `core/texture.py` (new), `parity/snapshot.py`, `parity/compare.py`, `parity/backends.py` (`helios_snapshot` only) | — |
| **W1-B object transforms** | `copyPrimitive`, `copyObject`, `translateObject`, `rotateObject` (axis string / axis vector / about origin), `scaleObject` (and about center/point), `deletePrimitive`, `deleteObject`, rotation of a primitive about an arbitrary axis. Prioritised by what `plantarchitecture` calls | `core/object_transforms.py` (new methods; arbitrary-axis matrix helpers live here) | — |
| **W1-C shapes** | sphere, disk (int and int2 subdivisions), cone, box: colored overloads plus their object getters | `core/shapes.py` | — |
| **W1-D tomato spec** | read-only analysis of `plantarchitecture` for the tomato build path. Output: `migration/modules/plantarchitecture/SPEC.md` (call graph, Context/global APIs used, RNG draw order and generator ownership, data structures, organ prototypes and assets) and the Wave 3 work breakdown | `migration/modules/plantarchitecture/*` | — |

## Wave 2 (running: W2-A OBJ + polymesh topology, W2-C plantarchitecture data layer, W2-D core gaps)

- **W2-A OBJ loader:** `loadOBJ` with Helios's face, UV and material ordering. Depends on W1-A.
- **W2-B textured shapes:** textured overloads of the W1-C shapes. Depends on W1-A and W1-C.
- **W2-C plantarchitecture data layer:** parameter structs, `RandomParameter` distributions, shoot
  and phytomer data structures, following the W1-D spec.

## Wave 3: the tomato

Work is split according to the W1-D breakdown. Expected packages:
- leaf prototype geometry;
- internode and petiole tubes;
- phytomer and shoot growth with `advanceTime`;
- inflorescence and fruit;
- `buildPlantInstanceFromLibrary("tomato")`.

The acceptance case is `plantarchitecture.tomato`: ages {10, 30, 50, 75, 100, 120} × seeds {1..8}, at
0-ulp tier. Seed 0 is the same as seed 1 in minstd_rand0, and the first flowers and fruit appear
between days 75 and 100 (W1-D). Packages W3-A…E follow SPEC.md §8.

## Wave 4 (last porting wave)

- Remaining species from the 8-crop gallery: cherry tomato, capsicum, strawberry, bean, soybean,
  grapevine_VSP and maize.
- `planttwin` (`legacy/planttwin`) conversion reads Heliades directly, with no PyHelios at runtime.
- USD export and simulation.
- The first TomatoWUR structural evaluation.

**Porting ends here** (target: mid-January 2027). Per the scope decision in `BUILD_VS_BUY.md`,
radiation, camera, physiology, LiDAR, soil and the other plugins are not ported.

## Parallel tracks (not porting)

| track | packages | status |
|---|---|---|
| **B: deformable simulation** | B1 tree direct solver (CPU) → B2 GPU (Warp) batched → B3 leaf shells + contact + pedicel breakage → B4 inverse statics → B5 validation against Newton/IPC reference | B1 running |
| **C: agentic generation + dataset** | minimal calibration loop on TomatoWUR → generator-code self-improvement → dataset generation + benchmarks + datasheet | starts when the Heliades tomato passes parity |
| **D: large-scale RL** | robot coupling (Newton/MuJoCo) → task envs (leaf pushing, harvesting, pruning) with domain randomization → training at scale | starts after B2; needs cluster GPUs |

Target venues:
- dataset: NeurIPS 2027 D&B (~May 2027);
- solver: SIGGRAPH Asia 2027 (~May 2027);
- RL benchmark: CoRL 2027 / ICRA 2028.
