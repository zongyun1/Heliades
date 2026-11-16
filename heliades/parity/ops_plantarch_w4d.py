# SPDX-License-Identifier: LGPL-2.1-or-later
"""Grapevine (W4-D) operations for both backends.

* ``gv_build(age, build_parameters)``: ``buildPlantInstanceFromLibrary((0,0,0), age, build_parameters)``
  (PyHelios passes build parameters as fixed values);
* ``gv_build_error(build_parameters)``: 1 if that build is rejected (PlantArchitectureError / ValueError /
  RuntimeError on either side), else 0;
* ``gv_build_parameter_names``: ``getPlantModelBuildParameterNames("grapevine_VSP")``;
* ``gv_shoot_summary``: per shoot, the node count and the stored internode nodes and radii (full precision).
"""
from __future__ import annotations

import numpy as np

from .ops import op

f32 = np.float32


@op("gv_build", "helios")
def _(B, age=0.0, build_parameters=None):
    from pyhelios.types import vec3
    if build_parameters:
        B.pid = B.pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), float(age), build_parameters=dict(build_parameters))
    else:
        B.pid = B.pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), float(age))
    return B.pid


@op("gv_build", "heliades")
def _(B, age=0.0, build_parameters=None):
    B.pid = B.pa.buildPlantInstanceFromLibrary((0.0, 0.0, 0.0), float(age), dict(build_parameters or {}))
    return B.pid


def _rejected(fn):
    try:
        fn()
    except Exception:     # PyHelios: PlantArchitectureError / ValueError; Heliades: PlantArchitectureError
        return 1.0
    return 0.0


@op("gv_build_error", "helios")
def _(B, build_parameters):
    from pyhelios.types import vec3
    return _rejected(lambda: B.pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), 0.0,
                                                                build_parameters=dict(build_parameters)))


@op("gv_build_error", "heliades")
def _(B, build_parameters):
    return _rejected(lambda: B.pa.buildPlantInstanceFromLibrary((0.0, 0.0, 0.0), 0.0, dict(build_parameters)))


@op("gv_build_parameter_names", "helios")
def _(B):
    return list(B.pa.getPlantModelBuildParameterNames("grapevine_VSP"))


@op("gv_build_parameter_names", "heliades")
def _(B):
    from ..plantarchitecture.library.build_parameters import getPlantModelBuildParameterNames
    return getPlantModelBuildParameterNames(B.pa, "grapevine_VSP")


@op("gv_shoot_summary", "helios")
def _(B):
    out = []
    for sid in [int(s) for s in B.pa.getAllShootIDs(B.pid)]:
        verts = B.pa.getShootInternodeVertices(B.pid, sid)
        radii = B.pa.getShootInternodeRadii(B.pid, sid)
        out.append((int(B.pa.getShootNodeCount(B.pid, sid)),
                    [float(f32(c)) for v in verts for c in (v[0], v[1], v[2])], [float(f32(r)) for r in radii]))
    return out


@op("gv_shoot_summary", "heliades")
def _(B):
    out = []
    for s in B.pa.plant_instances[B.pid].shoot_tree:
        out.append((len(s.phytomers), [float(c) for row in s.shoot_internode_vertices for v in row for c in v],
                    [float(r) for row in s.shoot_internode_radii for r in row]))
    return out
