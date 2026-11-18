# SPDX-License-Identifier: LGPL-2.1-or-later
"""Grapevine Wye (W5-C) operations for both backends.

The W4-D grape ops (``gv_build``, ``gv_build_error``, ``gv_shoot_summary``) are model-agnostic and reused. New:

* ``gvw_build_parameter_names(model)``: ``getPlantModelBuildParameterNames(model)``;
The gable wires (attraction points) have no PyHelios getter; they are checked through the growth they steer.
"""
from __future__ import annotations

from ...ops import op


@op("gvw_build_parameter_names", "helios")
def _(B, model):
    return list(B.pa.getPlantModelBuildParameterNames(model))


@op("gvw_build_parameter_names", "heliades")
def _(B, model):
    from ....plantarchitecture.library.build_parameters import getPlantModelBuildParameterNames
    return getPlantModelBuildParameterNames(B.pa, model)
