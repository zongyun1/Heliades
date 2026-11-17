# SPDX-License-Identifier: LGPL-2.1-or-later
"""Registry self-test: one op registered for both backends (also a template for ops_* modules)."""
from ...ops import op


@op("primitive_count", "helios")
def _(B):
    return len(B.ctx.getAllUUIDs())


@op("primitive_count", "heliades")
def _(B):
    return len(B.scene.all_uuids())
