# SPDX-License-Identifier: LGPL-2.1-or-later
"""Where species assets live and how a species file names them.

Generated assets (leaf textures from :mod:`.leaf_texture`) are written to
``heliades/species/assets/<species>/<name>.png`` and committed, so a build never depends on an image of unclear
licence: every image there is produced by this package from the description in the species file.

:func:`resolve_asset` turns a name used in a species file into a file: ``<name>`` of the species' own asset folder
first, then an absolute or working-directory path, then a Helios plant-architecture asset (``assets/textures/``,
``assets/obj/``; LGPL, from the pinned Helios tree).
"""
from __future__ import annotations

import os

ASSET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def species_asset_dir(species: str) -> str:
    return os.path.join(ASSET_ROOT, species)


def leaf_texture_path(species: str, name: str) -> str:
    return os.path.join(species_asset_dir(species), name + ".png")


def ensure_leaf_texture(species: str, name: str, desc: dict) -> str:
    """The generated texture's path; written if it is missing (it is normally committed)."""
    path = leaf_texture_path(species, name)
    if not os.path.exists(path):
        from .leaf_texture import render, write_png
        write_png(path, render(desc))
    return path


def resolve_asset(name: str, species: str | None = None) -> str:
    if species:
        p = os.path.join(species_asset_dir(species), name)
        if os.path.exists(p):
            return p
    if name.startswith("assets/"):
        # helios::resolvePluginAsset("plantarchitecture", name), as the library code resolves it
        from ..plantarchitecture.library.tomato_inflorescence import _plugin_asset
        return _plugin_asset(name)
    if os.path.isabs(name) and os.path.exists(name):
        return name
    if os.path.exists(name):
        return os.path.abspath(name)
    from ..plantarchitecture.leaf_prototype import _try_plugin_asset
    for sub in ("", "assets/textures/", "assets/obj/"):
        p = _try_plugin_asset(sub + name)
        if p:
            return p
    raise FileNotFoundError(f"species asset '{name}' not found (species folder, path, or Helios plantarchitecture "
                            "assets)")
