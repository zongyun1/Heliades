# SPDX-License-Identifier: LGPL-2.1-or-later
"""``setPlantBasePosition`` for both backends.

* ``pa_set_base_position(x, y, z)``: ``setPlantBasePosition(plantID, (x, y, z))``;
* ``pa_set_base_position_error(plantID)``: 1 if moving that plant is rejected, else 0.
"""
from __future__ import annotations

from ...ops import op


@op("pa_set_base_position", "helios")
def _(B, x, y, z):
    from pyhelios.types import vec3
    B.pa.setPlantBasePosition(B.pid, vec3(float(x), float(y), float(z)))


@op("pa_set_base_position", "heliades")
def _(B, x, y, z):
    B.pa.setPlantBasePosition(B.pid, (float(x), float(y), float(z)))


@op("pa_set_base_position_error", "helios")
def _(B, plantID):
    from pyhelios.types import vec3
    try:
        B.pa.setPlantBasePosition(int(plantID), vec3(1, 0, 0))
    except Exception:
        return 1.0
    return 0.0


@op("pa_set_base_position_error", "heliades")
def _(B, plantID):
    try:
        B.pa.setPlantBasePosition(int(plantID), (1.0, 0.0, 0.0))
    except Exception:
        return 1.0
    return 0.0
