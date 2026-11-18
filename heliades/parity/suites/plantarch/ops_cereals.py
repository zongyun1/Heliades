# SPDX-License-Identifier: LGPL-2.1-or-later
"""Operations for the cereal parity cases (W5-A: maize tassel, wheat, rice, sorghum), both backends.

* ``cereal_build_parameter_names(model)``: ``getPlantModelBuildParameterNames(model)``;
* ``cereal_build_error_text(build_parameters, age)``: the message of the error a build raises, reduced to the
  C++ ``helios_runtime_error`` text (PyHelios wraps it; Heliades raises it as is), or "" if the build succeeds;
* ``cereal_counts``: (flower objects, fruit objects, leaves, primitives) of the current plant.

Builds use W4-D's ``gv_build`` (``buildPlantInstanceFromLibrary`` with fixed build parameters on both sides).
"""
from __future__ import annotations

from ...ops import op
from ....plantarchitecture import leaf_prototype  # noqa: F401  (binds the W3-A GenericLeafPrototype hook)


@op("cereal_build_parameter_names", "helios")
def _(B, model):
    return list(B.pa.getPlantModelBuildParameterNames(model))


@op("cereal_build_parameter_names", "heliades")
def _(B, model):
    from ....plantarchitecture.library.build_parameters import getPlantModelBuildParameterNames
    return getPlantModelBuildParameterNames(B.pa, model)


def _error_text(fn) -> str:
    """The innermost ``ERROR (...)`` message. PyHelios prefixes the C++ text with its own ``ERROR
    (PlantArchitecture::buildPlantInstanceFromLibrary):`` and rejects an undeclared key itself, before the C++ call,
    with a message of its own; both sides report that case as ``UNKNOWN_KEY``."""
    try:
        fn()
    except Exception as e:
        text = str(e)
        if "Unknown build parameter" in text or "is not accepted by plant model" in text:
            return "UNKNOWN_KEY"
        i = text.rfind("ERROR (")
        return (text[i:] if i >= 0 else text).strip().splitlines()[0]
    return ""


@op("cereal_build_error_text", "helios")
def _(B, build_parameters, age=0.0):
    from pyhelios.types import vec3
    return _error_text(lambda: B.pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), float(age),
                                                                  build_parameters=dict(build_parameters)))


@op("cereal_build_error_text", "heliades")
def _(B, build_parameters, age=0.0):
    return _error_text(lambda: B.pa.buildPlantInstanceFromLibrary((0.0, 0.0, 0.0), float(age), dict(build_parameters)))


@op("cereal_counts", "helios")
def _(B):
    return [len(B.pa.getPlantFlowerObjectIDs(B.pid)), len(B.pa.getPlantFruitObjectIDs(B.pid)),
            int(B.pa.getPlantLeafCount(B.pid)), int(B.ctx.getPrimitiveCount())]


@op("cereal_counts", "heliades")
def _(B):
    return [len(B.pa.getPlantFlowerObjectIDs(B.pid)), len(B.pa.getPlantFruitObjectIDs(B.pid)),
            int(B.pa.getPlantLeafCount(B.pid)), int(B.scene.primitive_count())]
