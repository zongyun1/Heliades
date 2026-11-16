# Discrepancies found by differential testing

Each entry: what differs, a minimal reproduction, and what Heliades does about it. Candidates to
report upstream.

## 1. PyHelios `addPatch` default color is white; C++ default is black (binding inconsistency)

- C++ `Context::addPatch(center, size[, rotation])` uses `make_RGBAcolor(0,0,0,1)`
  (`core/src/Context_primitive.cpp:23-33`).
- PyHelios `Context.addPatch(..., color=None)` substitutes `RGBcolor(1,1,1)` (`pyhelios/Context.py:475`).
- PyHelios `addTriangle(..., color=None)` calls the colorless C++ overload, which gives black. So the
  two primitive types disagree within PyHelios.
- Reproduction: `Context().addPatch()` and then `getPrimitiveColor`, which returns white (C++ gives black).
- Heliades follows C++. The parity backend passes an explicit black.

## 2. The pip wheel and the source tree are different Helios versions

- `pyhelios3d==0.1.39` (pip) bundles Helios commit `f97ef3c` (1.3.94), where `randn` comes from
  libstdc++ `std::normal_distribution<float>`. Helios 1.3.98 replaced it with portable distributions,
  so the same seed now gives different normal draws (and therefore different plants).
- Heliades pins 1.3.98 and builds the oracle from source (`migration/PORTING.md`).
