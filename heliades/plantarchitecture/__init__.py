# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture (Copyright (C) 2016-2026 Brian Bailey).
"""Heliades port of the Helios PlantArchitecture plugin.

W2-C (this layer): parameter structs with C++ value semantics and RNG behaviour, the plant-library
registry and the tomato shoot parameters. Geometry and growth arrive with W3 (see
``migration/modules/plantarchitecture/SPEC.md`` §8).
"""
from __future__ import annotations

from .architecture import PlantArchitecture, context_generator
from .params import (BUD_ACTIVE, BUD_DEAD, BUD_DORMANT, BUD_FLOWER_CLOSED, BUD_FLOWER_OPEN, BUD_FRUITING,
                     AxisRotation, BudState, FloralBud, LeafPrototype, LeafShapeValues, PhytomerParameters,
                     ShootParameters, VegetativeBud, make_AxisRotation)
from .random_parameter import (PlantArchitectureError, RandomParameter_float, RandomParameter_int,
                               RandomParameterFloat, RandomParameterInt)

__all__ = ["PlantArchitecture", "context_generator", "AxisRotation", "make_AxisRotation", "BudState", "VegetativeBud",
           "FloralBud", "LeafPrototype", "LeafShapeValues", "PhytomerParameters", "ShootParameters",
           "RandomParameterFloat", "RandomParameterInt", "RandomParameter_float", "RandomParameter_int",
           "PlantArchitectureError", "BUD_DORMANT", "BUD_ACTIVE", "BUD_FLOWER_CLOSED", "BUD_FLOWER_OPEN",
           "BUD_FRUITING", "BUD_DEAD"]
