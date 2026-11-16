# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/Assets.cpp:4080-4090 (Copyright (C) 2016-2026 Brian Bailey).
"""Tomato flower and fruit prototypes (``TomatoFlowerPrototype`` / ``TomatoFruitPrototype``), bound to the
``hooks`` slots that ``initializeTomatoShoots`` stores in ``InflorescenceParameters`` (W3-D).

Both load an OBJ asset with ``loadOBJ(file, (0,0,0), 0.75, nullrotation, RGB::black, "ZUP", silent)`` and wrap
the UUIDs in a polymesh. ``flower_is_open`` and the build parameters are ignored, as in Helios: open and closed
flowers share the mesh. Importing this module binds the hooks (``heliades.plantarchitecture.inflorescence``
imports it lazily on first use).
"""
from __future__ import annotations

from ...migration_registry import ports
from .. import hooks


def _plugin_asset(asset_path: str) -> str:
    """``helios::resolvePluginAsset("plantarchitecture", asset_path)``, with the pinned Helios tree as the
    Heliades fallback (as for leaf textures, ``leaf_prototype._try_plugin_asset``)."""
    from ...core.scene import try_resolve_plugin_asset
    p = try_resolve_plugin_asset("plantarchitecture", asset_path)
    if p:
        return p
    from ..leaf_prototype import _try_plugin_asset
    p = _try_plugin_asset(asset_path)
    if p:
        return p
    from ...core.scene import resolve_plugin_asset
    return resolve_plugin_asset("plantarchitecture", asset_path)     # raises with Helios's message


def _load(context, asset_path: str) -> int:
    UUIDs = context.load_obj(_plugin_asset(asset_path), (0.0, 0.0, 0.0), height=0.75, rotation=(0.0, 0.0),
                             color=(0.0, 0.0, 0.0), upaxis="ZUP", silent=True)
    return context.add_polymesh_object(UUIDs)


@hooks.TomatoFruitPrototype.implement
@ports("TomatoFruitPrototype")
def TomatoFruitPrototype(context, subdivisions, build_parameters=None) -> int:
    """AS:4080-4084."""
    return _load(context, "assets/obj/TomatoFruit.obj")


@hooks.TomatoFlowerPrototype.implement
@ports("TomatoFlowerPrototype")
def TomatoFlowerPrototype(context, subdivisions, flower_is_open, build_parameters=None) -> int:
    """AS:4086-4090."""
    return _load(context, "assets/obj/TomatoFlower.obj")


__all__ = ["TomatoFlowerPrototype", "TomatoFruitPrototype"]
