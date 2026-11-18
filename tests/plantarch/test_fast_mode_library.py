# SPDX-License-Identifier: LGPL-2.1-or-later
"""Fast mode against exact mode over the whole plant library (heavy: ``--runslow``).

* every model, Context seeds 1..8, at a representative mature age (``scripts/bench/fast_survey.py`` AGES): the same
  random draws; for the models whose growth decisions do not read geometry, the same topology (primitives, organ
  labels, materials, objects, plant-structure dict up to its floats) and a vertex deviation below ``REL_LIMIT``
  of the plant's bounding-box diagonal;
* the models whose decisions read geometry (``DIVERGENT``): the plant metrics of seeds 1..32 have the same
  distribution in both modes (``compare_distributions``).

``docs/core/FAST_MODE.md`` explains the limits.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "fast_survey", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "bench", "fast_survey.py"))
_survey = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_survey)
AGES = _survey.AGES

#: leaf inclination tracking (facet normals and areas pick discrete target bins): same topology, poses diverge
ORIENTATION_DIVERGENT = {"cowpea", "pistachio", "easternredbud"}
#: tube-creation thresholds (petiole length / radius) reached one sub-step apart after the leaf-area feedback
TOPOLOGY_DIVERGENT = {"bougainvillea", "capsicum", "olive"}
DIVERGENT = ORIENTATION_DIVERGENT | TOPOLOGY_DIVERGENT
#: relative vertex deviation allowed for the others: the leaf-area -> girth -> position feedback amplifies one
#: float32 rounding to ~1e-4 of the plant size in exact mode itself (see FAST_MODE.md, sensitivity floor)
REL_LIMIT = 1e-3


def _check_job(job):
    from heliades.parity.fastmode import check
    return check(*job)


def _pool():
    import multiprocessing as mp
    return mp.get_context("spawn").Pool(min(6, os.cpu_count() or 1), maxtasksperchild=1)


@pytest.mark.slow
@pytest.mark.parametrize("model", sorted(AGES))
def test_fast_mode_against_exact(model):
    """Seeds 1..8 (built on up to 6 processes)."""
    with _pool() as pool:
        res = pool.map(_check_job, [(model, AGES[model], s) for s in range(1, 9)])
    for r in res:
        assert r["draws"], r
        if model in TOPOLOGY_DIVERGENT:
            continue
        assert r["topology"], r
        if model not in ORIENTATION_DIVERGENT:
            assert r["rel_dev"] < REL_LIMIT, r


def _metrics_job(job):
    model, age, seed = job
    from heliades.parity.fastmode import record
    out = {}
    for mode in ("exact", "fast"):
        rec = record(model, age, seed, mode)
        out[mode] = (rec.metrics, rec.draws)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("model", sorted(DIVERGENT))
def test_divergent_model_distributions(model):
    from heliades.parity.compare import compare_distributions
    jobs = [(model, AGES[model], s) for s in range(1, 33)]
    with _pool() as pool:
        res = pool.map(_metrics_job, jobs)
    for r in res:
        # the draw stream only parts when a geometry-reading decision flipped and changed the plant (pistachio seed 19:
        # a pruning decision; exact mode with every area one ulp up makes the same plant and the same draws)
        if r["exact"][1] != r["fast"][1]:
            assert r["exact"][0]["primitives"] != r["fast"][0]["primitives"], r
    keys = res[0]["exact"][0].keys()
    a = {k: np.array([r["exact"][0][k] for r in res]) for k in keys}
    b = {k: np.array([r["fast"][0][k] for r in res]) for k in keys}
    rep = compare_distributions(b, a)
    assert rep.ok, rep.checks
