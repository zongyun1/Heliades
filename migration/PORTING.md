# Porting Helios to Heliades

The rules for anyone, human or coding agent, who ports a Helios module. Heliades is LGPL-2.1-or-later
because it is a derivative work of Helios (Copyright (C) 2016-2026 Brian Bailey).

## The baseline is pinned

- Port **only** from `third_party/pyhelios_src/helios-core` at the commit recorded in `STATUS.md`
  (currently Helios 1.3.98, `14860044f`). Do not read Helios master, a pip wheel, or memory of Helios.
- The oracle is PyHelios built from that same tree (`third_party/pyhelios_src`, see below). The
  pip wheel bundles a different Helios. Its `randn` uses libstdc++'s `std::normal_distribution`,
  while 1.3.98 uses its own portable distributions.
- Moving the baseline is its own change: bump the submodule, rebuild the oracle, rerun all parity
  cases, and fix what breaks before porting anything new.

## Workflow for one module

1. **Spec.** Read the module's headers, sources and `tests/`. Write `migration/modules/<module>/SPEC.md`
   covering:
   - the public API you will port;
   - the data layout;
   - every random draw (in which order, and from which distribution);
   - float vs double at each step;
   - the error conditions.
2. **Backend API.** Add the operations a parity case needs to *both* `HeliosBackend` and
   `ForgeBackend` (`heliades/parity/backends.py`).
3. **Cases.** Add parity cases in `heliades/parity/cases.py` (or `cases_<module>.py`):
   - fuzz cases with many seeds;
   - edge cases taken from Helios's own `tests/`.
   Run them before porting. They must fail on Heliades and must run on Helios.
4. **Port.** Implement in `heliades/`, tagging each function with `@ports("Class::method")`.
   - Keep Helios's operation order and float32 rounding wherever bit-level parity is the goal.
   - Add a comment explaining any deliberate deviation (a bug fix, or vectorisation that changes
     rounding).
5. **Parity.** Run `python -m heliades.parity <module>` until it passes. Then run
   `pytest tests/heliades` and `python migration/inventory.py`.
6. **Bugs found in Helios.** If the oracle looks wrong, do not copy the bug silently. Record it in
   `migration/HELIOS_ISSUES.md` with a minimal reproduction. Then either match Helios behind a flag,
   or fix it and mark that case `expected_divergence`.

## Parity tiers (`heliades/parity/compare.py`)

| tier | meaning | used for |
|---|---|---|
| `exact` | bitwise equal | RNG streams, integer topology, UUID assignment |
| `ulp` | float32 within N ulp, or below 1e-6 absolute | geometry from the same op sequence (libm `sin`/`log` may differ by 1 ulp) |
| `dist` | KS test plus relative mean, over many seeds | stochastic models whose draw order is not yet reproduced |

The target for plant generation is **`ulp`**: the same seed gives the same plant. Use `dist` only as
a stepping stone, and say so in the case name.

## Not ported

- `projectbuilder` and `visualizer` (GUI and OpenGL) are replaced by USD export
  (`heliades.io.usd`) and external viewers.
- Collision and voxel queries are replaced by Warp BVH queries (`heliades.core.device`).
  They are still parity-tested through their observable results.

## Design rules for the new platform

- **Host generation, device bulk.** Branchy, sequential code (growth rules, RNG draws) runs on the
  host in numpy. Anything per-primitive or per-ray goes through `Scene.snapshot()` to Warp arrays.
- **Batch-first APIs on the device side.** Many plants or many parameter sets are handled in one
  launch.
- **USD is the interchange format.** UUIDs survive the round trip as the `heliosUUID` primvar.

## Building the oracle

```bash
cd third_party/pyhelios_src
../../.venv/bin/python build_scripts/build_helios.py --nogpu --novis --exclude lidar
../../.venv/bin/python -m pip install -e . --no-build-isolation   # or: uv pip install -e .
```

LiDAR needs the X11 dev headers (`libxrandr-dev`, etc.) because it pulls in the visualizer. GPU
plugins (radiation) need CUDA/OptiX or Vulkan and are not built yet.
