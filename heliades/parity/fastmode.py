# SPDX-License-Identifier: LGPL-2.1-or-later
"""Fast mode against exact mode: one library plant built in both modes and compared.

``record(model, age, seed, mode)`` builds a plant and keeps what the comparison needs: the random draw count and
final generator state, the whole-scene snapshot, the organ label and material label of every primitive, the object
types and members, the plant-structure dict (``writePlantStructureXML``, parsed) and a few scalar metrics.

``compare(exact, fast)`` reports

* ``draws``: same number of random draws and same final generator state (always required);
* ``topology``: same primitives (UUIDs, types, parents, organ labels, materials, colors, texture files), same objects
  (IDs, types, members), same plant-structure dict up to its float values (keys, list lengths, integers, strings
  and booleans equal; floats compared separately: ``structure_rel``);
* ``max_dev`` / ``rel_dev``: the largest vertex deviation, absolute and relative to the plant's bounding-box
  diagonal.

``metrics(rec)`` gives the scalars used by the distribution comparison (``compare_distributions``) for models whose
growth decisions read geometry (see ``docs/core/FAST_MODE.md``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Record:
    model: str
    age: float
    seed: int
    mode: str
    seconds: float
    draws: int
    rng_state: int
    snap: object
    labels: list
    materials: list
    objects: dict                  # oid -> (type, tuple(uuids))
    structure: dict
    metrics: dict = field(default_factory=dict)


def _counting(gen):
    """Count draws of the Scene's engine without changing them (class swap on the instance)."""
    base = type(gen)
    box = [0]

    class Counting(base):
        def __call__(self):
            box[0] += 1
            return base.__call__(self)
    gen.__class__ = Counting
    return box


def record(model: str, age: float, seed: int = 1, mode: str = "exact", build_parameters=None) -> Record:
    from ..core.scene import Scene
    from ..plantarchitecture import PlantArchitecture
    sc = Scene(mode=mode)
    sc.seed_random_generator(int(seed))
    box = _counting(sc.rng.generator)
    pa = PlantArchitecture(sc)
    pa.loadPlantModelFromLibrary(model)
    t0 = time.perf_counter()
    if build_parameters:
        pid = pa.buildPlantInstanceFromLibrary((0.0, 0.0, 0.0), float(age), dict(build_parameters))
    else:
        pid = pa.buildPlantInstanceFromLibrary((0.0, 0.0, 0.0), float(age))
    dt = time.perf_counter() - t0
    draws = box[0]
    snap = sc.snapshot(include_hidden=True)
    lab = sc.primitive_data.get("object_label", {})
    labels = [(lab[u].values[0] if u in lab else None) for u in snap.uuid.tolist()]
    materials = sc.get_primitive_material_labels(snap.uuid.tolist()) if len(snap.uuid) else []
    objects = {o: (sc.objects[o].type, tuple(sc.objects[o].uuids)) for o in sorted(sc.objects)}
    structure = pa.plant_structure_dict(pid)
    m = {
        "primitives": float(len(snap.uuid)),
        "objects": float(len(objects)),
        "shoots": float(len(pa.plant_instances[pid].shoot_tree)),
        "leaves": float(pa.getPlantLeafCount(pid)),
        "fruits": float(len(pa.getPlantFruitObjectIDs(pid))),
        "leaf_area": float(pa.sumPlantLeafArea(pid)),
        "height": float(pa.getPlantHeight(pid)) if objects else 0.0,
    }
    vis = snap.nverts[:, None] > np.arange(4)[None]
    if vis.any():
        V = snap.vertices[vis]
        ext = V.max(0) - V.min(0)
        m.update(bbox_x=float(ext[0]), bbox_y=float(ext[1]), bbox_z=float(ext[2]))
    return Record(model, float(age), int(seed), mode, dt, draws, int(sc.rng.generator.x), snap, labels, materials,
                  objects, structure, m)


def metrics(rec: Record) -> dict:
    return dict(rec.metrics)


def _struct_cmp(a, b, path, out):
    """Walk two structure dicts: discrete values must agree; float values contribute to the relative deviation."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            out["struct_ok"] = False
            out.setdefault("struct_first", path + " keys")
            return
        for k in a:
            _struct_cmp(a[k], b[k], f"{path}/{k}", out)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out["struct_ok"] = False
            out.setdefault("struct_first", path + " length")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _struct_cmp(x, y, f"{path}[{i}]", out)
        return
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if np.isnan(a) and np.isnan(b):
            return
        d = abs(float(a) - float(b))
        out["struct_abs"] = max(out.get("struct_abs", 0.0), d)
        out["struct_rel"] = max(out.get("struct_rel", 0.0), d / max(abs(float(a)), 1e-3))
        return
    if type(a) is not type(b) and not (isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer))):
        out["struct_ok"] = False
        out.setdefault("struct_first", f"{path} type")
        return
    if a != b:
        out["struct_ok"] = False
        out.setdefault("struct_first", f"{path}: {a!r} != {b!r}")


def compare(a: Record, b: Record) -> dict:
    """Exact ``a`` against fast ``b`` (see the module docstring)."""
    r = {"model": a.model, "age": a.age, "seed": a.seed, "draws": a.draws == b.draws and a.rng_state == b.rng_state,
         "n_draws": a.draws, "t_exact": a.seconds, "t_fast": b.seconds}
    sa, sb = a.snap, b.snap
    same = len(sa.uuid) == len(sb.uuid) and bool((sa.uuid == sb.uuid).all())
    if same:
        same = bool((sa.type == sb.type).all() and (sa.parent == sb.parent).all() and (sa.nverts == sb.nverts).all()
                    and (sa.hidden == sb.hidden).all() and np.array_equal(sa.color, sb.color)
                    and list(sa.texture) == list(sb.texture))
    r["primitives_same"] = same
    r["labels_same"] = same and a.labels == b.labels
    r["materials_same"] = same and list(a.materials) == list(b.materials)
    r["objects_same"] = a.objects == b.objects
    s = {"struct_ok": True}
    _struct_cmp(a.structure, b.structure, "", s)
    r.update(s)
    r["topology"] = bool(r["primitives_same"] and r["labels_same"] and r["materials_same"] and r["objects_same"]
                         and s["struct_ok"])
    if same and len(sa.uuid):
        vis = sa.nverts[:, None] > np.arange(4)[None]
        V = sa.vertices[vis]
        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        d = np.abs(sa.vertices.astype(np.float64) - sb.vertices)[vis]
        r["max_dev"] = float(d.max())
        r["diag"] = diag
        r["rel_dev"] = float(d.max()) / diag if diag > 0 else 0.0
        r["area_rel"] = float(np.abs(sa.area.astype(np.float64) - sb.area).max() / max(float(np.abs(sa.area).max()), 1e-30))
    return r


def check(model: str, age: float, seed: int = 1, build_parameters=None) -> dict:
    """Build in both modes and compare."""
    return compare(record(model, age, seed, "exact", build_parameters), record(model, age, seed, "fast", build_parameters))


__all__ = ["Record", "record", "compare", "check", "metrics"]
