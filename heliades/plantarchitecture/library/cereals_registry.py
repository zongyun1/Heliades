# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:525-535 (Copyright (C) 2016-2026 Brian Bailey).
"""The ``registerPlantModelBuildParameters`` calls of the library grasses (W5-A), in the C++ order:
sorghum, maize, rice, wheat."""
from __future__ import annotations

import importlib


def register_build_parameters(pa):
    for name in ("sorghum", "maize", "rice", "wheat"):
        try:
            module = importlib.import_module(f".{name}", __package__)
        except ModuleNotFoundError:
            continue
        module.register_build_parameters(pa)
