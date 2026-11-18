# SPDX-License-Identifier: LGPL-2.1-or-later
"""Build many library plants, one Scene per plant, across worker processes.

    from heliades.plantarchitecture.batch import PlantJob, build_plant, build_plants

    sc, pa, pid = build_plant("tomato", 120, seed=3, mode="fast")          # one plant, in this process
    jobs = [PlantJob("tomato", 120, seed=s) for s in range(1, 65)]
    results = build_plants(jobs, workers=8, out_dir="out/")                 # snapshots saved as .npz

Every job builds a fresh ``Scene(mode=...)`` seeded with ``job.seed`` (``Context::seedRandomGenerator``), loads
``job.model`` and calls ``buildPlantInstanceFromLibrary(job.position, job.age, job.build_parameters)``, so a job's
plant does not depend on which worker built it or on the other jobs. Results come back in job order.

What a job returns is a :class:`PlantResult` (timings, counts, the saved file) or, with ``collect``, whatever
``collect(scene, plantarchitecture, plantID, job)`` returns (it runs in the worker and must be picklable, e.g. a
module-level function). Large plants are better saved (``out_dir``) than returned: the snapshot of a mature tree is
hundreds of MB and would be pickled through a pipe.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PlantJob:
    model: str
    age: float
    seed: int = 1
    build_parameters: dict | None = None
    position: tuple = (0.0, 0.0, 0.0)
    mode: str = "fast"

    def name(self) -> str:
        bp = "" if not self.build_parameters else "_" + "_".join(f"{k}{v:g}" for k, v in sorted(self.build_parameters.items()))
        return f"{self.model}_{self.age:g}d_seed{self.seed}{bp}_{self.mode}"


@dataclass
class PlantResult:
    job: PlantJob
    seconds: float                 # build time (buildPlantInstanceFromLibrary)
    primitives: int
    objects: int
    path: str | None = None        # saved snapshot (.npz), when out_dir was given
    pid: int = field(default_factory=os.getpid)


def build_plant(model: str, age: float, seed: int = 1, mode: str = "fast", build_parameters: dict | None = None,
                position=(0.0, 0.0, 0.0)):
    """One library plant in a new Scene: (scene, plantarchitecture, plantID)."""
    from ..core.scene import Scene
    from .architecture import PlantArchitecture
    sc = Scene(mode=mode)
    sc.seed_random_generator(int(seed))
    pa = PlantArchitecture(sc)
    pa.loadPlantModelFromLibrary(model)
    if build_parameters:
        pid = pa.buildPlantInstanceFromLibrary(tuple(position), float(age), dict(build_parameters))
    else:
        pid = pa.buildPlantInstanceFromLibrary(tuple(position), float(age))
    return sc, pa, pid


def save_snapshot(scene, path: str, include_hidden: bool = False) -> str:
    """The whole-scene snapshot (``Scene.snapshot``) as a compressed .npz (same layout as
    ``scripts/bench/plant_build.py --save``)."""
    s = scene.snapshot(include_hidden)
    o = s.objects
    np.savez_compressed(path, uuid=s.uuid, type=s.type, vertices=s.vertices, nverts=s.nverts, area=s.area,
                        normal=s.normal, color=s.color, parent=s.parent, uv=s.uv, nuv=s.nuv,
                        solid_fraction=s.solid_fraction, hidden=s.hidden,
                        texture=np.array([t.rsplit("/", 1)[-1] for t in s.texture]), oid=o.oid, otype=o.type,
                        ocenter=o.center, otransform=o.transform, ohidden=o.hidden,
                        omembers=np.concatenate(o.uuids) if o.uuids else np.zeros(0, np.int64),
                        olens=np.array([len(u) for u in o.uuids], np.int64))
    return path


def _run(args):
    job, out_dir, collect = args
    t0 = time.perf_counter()
    sc, pa, pid = build_plant(job.model, job.age, job.seed, job.mode, job.build_parameters, job.position)
    dt = time.perf_counter() - t0
    if collect is not None:
        return collect(sc, pa, pid, job)
    path = None
    if out_dir is not None:
        path = save_snapshot(sc, os.path.join(out_dir, job.name() + ".npz"))
    return PlantResult(job, dt, sc.primitive_count(), sc.object_count(), path)


def build_plants(jobs, workers: int | None = None, out_dir: str | None = None, collect=None,
                 start_method: str | None = None, maxtasksperchild: int | None = 8) -> list:
    """Build every job (a :class:`PlantJob` or a (model, age[, seed]) tuple) on ``workers`` processes
    (default: ``os.cpu_count()``; 1 or 0 builds in this process; processes are spawned unless ``start_method`` says
    otherwise, so ``collect`` must be importable). Returns the results in job order."""
    jobs = [j if isinstance(j, PlantJob) else PlantJob(*j) for j in jobs]
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    args = [(j, out_dir, collect) for j in jobs]
    workers = os.cpu_count() if workers is None else int(workers)
    if workers <= 1 or len(jobs) <= 1:
        return [_run(a) for a in args]
    import multiprocessing as mp
    ctx = mp.get_context(start_method or "spawn")     # no fork of a process that may hold threads (BLAS, Warp)
    with ctx.Pool(min(workers, len(jobs)), maxtasksperchild=maxtasksperchild) as pool:
        return pool.map(_run, args, chunksize=1)


__all__ = ["PlantJob", "PlantResult", "build_plant", "build_plants", "save_snapshot"]
