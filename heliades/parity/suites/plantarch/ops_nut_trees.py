# SPDX-License-Identifier: LGPL-2.1-or-later
"""Nut-tree (W5-D) operations for both backends.

* ``tree_build(age, build_parameters)``: ``buildPlantInstanceFromLibrary((0,0,0), age, build_parameters)`` (PyHelios
  passes build parameters as fixed values);
* ``tree_build_error(age, build_parameters)``: 1 if that build is rejected (PlantArchitectureError / ValueError /
  RuntimeError on either side), else 0;
* ``tree_build_parameter_names(model)``: ``getPlantModelBuildParameterNames(model)``;
* ``tree_shoot_summary``: per shoot (pruned ones included, as empty shells): the node count and the stored
  internode nodes and radii at full precision.
"""
from __future__ import annotations

import numpy as np

from ...ops import op

f32 = np.float32


@op("tree_build", "helios")
def _(B, age=0.0, build_parameters=None):
    from pyhelios.types import vec3
    if build_parameters:
        B.pid = B.pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), float(age), build_parameters=dict(build_parameters))
    else:
        B.pid = B.pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), float(age))
    return B.pid


@op("tree_build", "heliades")
def _(B, age=0.0, build_parameters=None):
    B.pid = B.pa.buildPlantInstanceFromLibrary((0.0, 0.0, 0.0), float(age), dict(build_parameters or {}))
    return B.pid


def _rejected(fn):
    try:
        fn()
    except Exception:     # PyHelios: PlantArchitectureError / ValueError; Heliades: PlantArchitectureError
        return 1.0
    return 0.0


@op("tree_build_error", "helios")
def _(B, age, build_parameters):
    from pyhelios.types import vec3
    return _rejected(lambda: B.pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), float(age),
                                                                build_parameters=dict(build_parameters)))


@op("tree_build_error", "heliades")
def _(B, age, build_parameters):
    return _rejected(lambda: B.pa.buildPlantInstanceFromLibrary((0.0, 0.0, 0.0), float(age), dict(build_parameters)))


@op("tree_build_parameter_names", "helios")
def _(B, model):
    return list(B.pa.getPlantModelBuildParameterNames(model))


@op("tree_build_parameter_names", "heliades")
def _(B, model):
    from ....plantarchitecture.library.build_parameters import getPlantModelBuildParameterNames
    return getPlantModelBuildParameterNames(B.pa, model)


@op("tree_shoot_summary", "helios")
def _(B):
    out = []
    for sid in [int(s) for s in B.pa.getAllShootIDs(B.pid)]:
        verts = B.pa.getShootInternodeVertices(B.pid, sid)
        radii = B.pa.getShootInternodeRadii(B.pid, sid)
        out.append((sid, int(B.pa.getShootNodeCount(B.pid, sid)),
                    [float(f32(c)) for v in verts for c in (v[0], v[1], v[2])], [float(f32(r)) for r in radii]))
    return out


@op("tree_shoot_summary", "heliades")
def _(B):
    out = []
    for sid in [int(s) for s in B.pa.getAllShootIDs(B.pid)]:
        s = B.pa.plant_instances[B.pid].shoot_tree[sid]
        out.append((sid, len(s.phytomers), [float(c) for row in s.shoot_internode_vertices for v in row for c in v],
                    [float(r) for row in s.shoot_internode_radii for r in row]))
    return out
