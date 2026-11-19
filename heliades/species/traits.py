# SPDX-License-Identifier: LGPL-2.1-or-later
"""Plausibility harness for species with no oracle: measured traits against the ranges a species file expects.

A species file lists ``traits`` (``{trait, age, range, source, seeds}``). :func:`check` builds ``seeds`` plants
(Context seeds 1..n, fast mode by default) for each listed age, measures each trait and compares the **mean over the
seeds** with the expected range. Static traits are measured on a plant built directly at ``age``; *timing* traits
(``first_flower_day``, ``first_fruit_day``) grow one plant per seed from age 0 in one-day steps up to ``age`` (the
horizon) and report the first day the organ exists.

    python -m heliades.species.traits eggplant barley          # table of measured vs expected

Units: metres, square metres, days, counts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _pca_extent(V: np.ndarray) -> float:
    """Extent of a point cloud along its principal axis (a leaf blade's or a fruit's length)."""
    V = np.asarray(V, np.float64).reshape(-1, 3)
    if len(V) < 2:
        return 0.0
    c = V - V.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    s = c @ vt[0]
    return float(s.max() - s.min())


def _object_vertices(sc, objID) -> np.ndarray:
    uu = sc.object_primitive_uuids(objID)
    if not uu:
        return np.zeros((0, 3))
    V, n = sc.primitive_vertices_rows(uu)
    return np.concatenate([V[i, :n[i]] for i in range(len(n))])


def _main_shoot_internodes(pa, pid):
    shoot = pa.plant_instances[pid].shoot_tree[0]
    return [float(ph.getInternodeLength()) for ph in shoot.phytomers]


def plant_height(sc, pa, pid):
    """Height of the plant's bounding box (m), ``getPlantHeight``."""
    return float(pa.getPlantHeight(pid))


def leaf_count(sc, pa, pid):
    """Leaf objects (blades / leaflets) on the plant, ``getPlantLeafCount``."""
    return float(pa.getPlantLeafCount(pid))


def leaf_area(sc, pa, pid):
    """One-sided blade area (m^2), ``sumPlantLeafArea``."""
    return float(pa.sumPlantLeafArea(pid))


def main_stem_nodes(sc, pa, pid):
    """Phytomers on the main stem (shoot 0)."""
    return float(len(pa.plant_instances[pid].shoot_tree[0].phytomers))


def internode_length_mean(sc, pa, pid):
    """Mean internode length of the main stem (m)."""
    L = _main_shoot_internodes(pa, pid)
    return float(np.mean(L)) if L else 0.0


def leaf_length_max(sc, pa, pid):
    """Longest leaf blade (leaflet for a compound leaf), along its principal axis (m)."""
    ids = pa.getPlantLeafObjectIDs(pid)
    return max((_pca_extent(_object_vertices(sc, o)) for o in ids), default=0.0)


def fruit_count(sc, pa, pid):
    """Fruit objects on the plant."""
    return float(len(pa.getPlantFruitObjectIDs(pid)))


def flower_count(sc, pa, pid):
    """Flower objects (open or closed) on the plant."""
    return float(len(pa.getPlantFlowerObjectIDs(pid)))


def fruit_length_max(sc, pa, pid):
    """Longest fruit along its principal axis (m); 0 without fruit."""
    return max((_pca_extent(_object_vertices(sc, o)) for o in pa.getPlantFruitObjectIDs(pid)), default=0.0)


def shoot_count(sc, pa, pid):
    """Shoots of the plant (main stem included): 1 + branches / tillers."""
    return float(len(pa.plant_instances[pid].shoot_tree))


def _first_day(organ_fn):
    def timing(build, age):
        sc, pa, pid = build(0.0)
        for day in range(1, int(math.ceil(age)) + 1):
            pa.advanceTime(pid, 1.0)
            if organ_fn(sc, pa, pid) > 0:
                return float(day)
        return float("nan")
    return timing


TRAITS = {
    "plant_height": ("m", plant_height),
    "leaf_count": ("count", leaf_count),
    "leaf_area": ("m2", leaf_area),
    "main_stem_nodes": ("count", main_stem_nodes),
    "internode_length_mean": ("m", internode_length_mean),
    "leaf_length_max": ("m", leaf_length_max),
    "fruit_count": ("count", fruit_count),
    "flower_count": ("count", flower_count),
    "fruit_length_max": ("m", fruit_length_max),
    "shoot_count": ("count", shoot_count),
    "first_flower_day": ("day", None),
    "first_fruit_day": ("day", None),
}
TIMING = {"first_flower_day": _first_day(flower_count), "first_fruit_day": _first_day(fruit_count)}


@dataclass
class TraitResult:
    trait: str
    age: float
    units: str
    expected: tuple
    values: list
    source: str

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def ok(self) -> bool:
        lo, hi = self.expected
        return bool(np.all(np.isfinite(self.values))) and lo <= self.mean <= hi

    def row(self) -> str:
        v = np.asarray(self.values, float)
        return (f"{self.trait:<22} {self.age:>6g} d  {self.mean:>9.4g} [{np.nanmin(v):.4g}, {np.nanmax(v):.4g}]  "
                f"expected [{self.expected[0]:g}, {self.expected[1]:g}] {self.units:<5} "
                f"{'ok  ' if self.ok else 'FAIL'} ({self.source})")


def _builder(name, seed, mode):
    from ..core.scene import Scene
    from ..plantarchitecture import PlantArchitecture

    def build(age):
        sc = Scene(mode=mode)
        sc.seed_random_generator(int(seed))
        pa = PlantArchitecture(sc)
        pa.loadPlantModelFromLibrary(name)
        pid = pa.buildPlantInstanceFromLibrary((0.0, 0.0, 0.0), float(age))
        return sc, pa, pid
    return build


def measure(name: str, traits: list, mode: str = "fast") -> list[TraitResult]:
    """Measure the listed traits (``SpeciesSpec.traits`` entries)."""
    static, results = {}, []
    for t in traits:
        units, fn = TRAITS[t["trait"]]
        vals = []
        for seed in range(1, t["seeds"] + 1):
            build = _builder(name, seed, mode)
            if fn is None:
                vals.append(TIMING[t["trait"]](build, t["age"]))
                continue
            key = (t["age"], seed)
            if key not in static:
                static[key] = build(t["age"])
            vals.append(fn(*static[key]))
        results.append(TraitResult(t["trait"], t["age"], units, t["range"], vals, t["source"]))
    return results


def check(name: str, mode: str = "fast") -> list[TraitResult]:
    from .loader import load_spec
    return measure(name, load_spec(name).traits, mode)


def _main(argv):
    from .loader import available_species, load_spec
    names = [a for a in argv if not a.startswith("--")] or available_species()
    bad = 0
    for name in names:
        spec = load_spec(name)
        if not spec.traits:
            continue
        print(f"== {name} ({spec.metadata['latin_name']}, template {spec.template})")
        for r in measure(name, spec.traits):
            print("  " + r.row())
            bad += not r.ok
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
