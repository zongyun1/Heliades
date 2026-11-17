# W1-D notes: tomato spec (plantarchitecture), Helios 1.3.98

Findings for the coordinator to fold into `HELIOS_ISSUES.md` / `PORTING.md`. Details and line
references are in `migration/modules/plantarchitecture/SPEC.md`.

## 1. Seeds 0 and 1 give the same plant (by design of `std::minstd_rand0`, not a PyHelios bug)

- `Context::seedRandomGenerator(s)` calls `std::minstd_rand0::seed(s)` (`core/src/Context.cpp:57-62`).
  For a linear congruential engine with increment 0, a seed `s` with `s mod (2^31-1) == 0` sets the
  state to 1 ([rand.eng.lcong]). So seed 0 is seed 1 (and so is `2147483647`).
- Oracle, 1.3.98, tomato age 20: seeds 0 and 1 give the identical vertex hash; seed 2 differs.
- Heliades's `MinstdRand0.seed` already reproduces this (`heliades/core/random.py:30-32`).
- Recommendation: the acceptance grid `seeds {0..7}` contains only 7 distinct plants. Use seeds
  {1..8} (or document that 0 == 1).

## 2. Seeding after `loadPlantModelFromLibrary()` is not reproducible

- `loadPlantModelFromLibrary("tomato")` draws 4 `LeafPrototype::unique_prototype_identifier`
  values (10 raw engine calls for seed 5) from the Context generator, which at that time is still
  seeded from the wall clock unless the caller seeded it earlier. The last identifier seeds the
  private `std::minstd_rand0 prototype_generator` that builds the 5 unique leaflet shapes
  (`PlantArchitecture.cpp:3475`). Seeding after load therefore gives the same topology but
  different leaf blades on every run (oracle: same seed after load -> different hashes run to run).
- Seeding before or after constructing `PlantArchitecture` makes no difference: the plugin keeps
  the pointer `&Context::generator` (`PlantArchitecture.cpp:383`, `Context.cpp:64`), and `seed()`
  re-seeds that same object.
- Rule for parity cases: seed the Context before `loadPlantModelFromLibrary()`.

## 3. The proposed acceptance ages {10, 30, 50, 75} never build a flower or fruit

- Oracle, seeds 0 and 3: no peduncle, flower or fruit primitive at age 75. At age 100 (seed 3):
  800 peduncle, 18 186 flower and 30 276 fruit triangles. At age 150: fruit only.
- Recommendation: add ages 100 and 120 (or 110, the age Helios's own selfTest uses for "bearing
  fruit", `tests/selfTest.cpp:9138`) so that the inflorescence package is covered.

## 4. Stale comments / small defects (no effect on tomato)

- `ShootParameters::operator=` (`include/PlantArchitecture.h:1585`) says the nested `PhytomerParameters` assignment
  "resamples deliberately". It does not: every sub-struct `operator=` copies `RandomParameter`s
  member-wise (cached `sampled`/`constval` included). Resampling is the explicit
  `phytomer_parameters.resample()` call in the `Phytomer` constructor (`PlantArchitecture.cpp:2700`).
- `PhytomerParameters::PhytomerParameters(gen)` assigns `petiole.color` in the peduncle block and
  never initialises `peduncle.color` (`PlantArchitecture.cpp:992`). Tomato sets `peduncle.color`
  explicitly, so it has no effect here.
- `LeafPrototype::operator=` re-draws `unique_prototype_identifier` whenever the prototype has a
  generator, while the copy constructor and `LeafParameters::operator=` (`duplicate()`) do not. An
  innocuous-looking assignment therefore consumes 2-3 Context draws. The draw count is load-bearing
  for parity.

## 5. Oracle facts useful for parity tooling

- `Context.randu()` called right after a stage fingerprints the generator state: find `n` such
  that the (n+1)-th raw `minstd_rand0` output reproduces it. This gives exact engine-call counts
  per stage with no changes to Helios (script: `migration/modules/plantarchitecture/oracle/draw_count.py`). Seed 5: load = 10, build(age 0)
  = 36, age 2 = 66, age 4 = 101, age 10 = 208.
- `advanceTime(20)` == `advanceTime(10)` twice, but differs from `advanceTime(5)` four times: the
  call is split into `floor(dt/phyllochron_min)` sub-steps of 2 days plus a remainder sub-step. A
  parity case must reproduce the caller's exact `advanceTime` call sequence.
- `Context::getAllUUIDs()` iterates an `std::unordered_map` (`core/include/Context.h:3548`), so its
  order is libstdc++ bucket order. Compare by sorted UUID, never by list position.

## 6. C++ argument evaluation order matters

- `buildTomatoPlant` calls `make_AxisRotation(0, randu(0,2pi), randu(0,2pi))` (`PlantLibrary.cpp:6214`).
- GCC evaluates the two `randu` arguments right to left. On the oracle, the XML `<base_rotation>`
  is (0, draw2, draw1) for seeds 2, 3 and 5 (`oracle/argorder.py`).
- A Clang or MSVC build of Helios would therefore give a different plant for the same seed.
- Heliades follows GCC, which is what the pinned oracle uses.
