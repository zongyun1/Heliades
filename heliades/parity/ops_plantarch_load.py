# SPDX-License-Identifier: LGPL-2.1-or-later
"""PlantArchitecture data-layer operations for both backends (W2-C).

``plantarch_new`` / ``plantarch_load`` / ``plantarch_shoot_params`` exist on both. The Helios side also
exposes oracle-only observations of an age-0 build (``plantarch_build``, ``plantarch_xml``,
``plantarch_leaf_shape``). On Heliades the age-0 observables are read from a real build: the W3-B
Phytomer constructor and the minimal ``buildTomatoPlant`` path replaced W2-C's ``age0_draw_plan`` replay.
"""
from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .ops import op

f32 = np.float32


# ---- Helios (PyHelios) ---------------------------------------------------------------------------

@op("plantarch_new", "helios")
def _(B):
    from pyhelios import PlantArchitecture
    B.pa = PlantArchitecture(B.ctx)
    try:
        B.pa.disableMessages()
    except Exception:
        pass
    return B.pa


@op("plantarch_load", "helios")
def _(B, model="tomato"):
    B.pa.loadPlantModelFromLibrary(model)


@op("plantarch_shoot_params", "helios")
def _(B, label):
    return B.pa.getCurrentShootParameters(label)


@op("plantarch_available_models", "helios")
def _(B):
    return list(B.pa.getAvailablePlantModels())


@op("plantarch_build", "helios")
def _(B, position=(0.0, 0.0, 0.0), age=0.0):
    from pyhelios.types import vec3
    B.pid = B.pa.buildPlantInstanceFromLibrary(vec3(*map(float, position)), float(age))
    return B.pid


@op("plantarch_xml", "helios")
def _(B):
    """writePlantStructureXML of the last built plant, parsed."""
    d = tempfile.mkdtemp(prefix="heliades_pa_")
    path = os.path.join(d, "plant.xml")
    B.pa.writePlantStructureXML(B.pid, path)
    root = ET.parse(path).getroot()
    os.remove(path)
    os.rmdir(d)
    return root


@op("plantarch_leaf_shape", "helios")
def _(B, shoot, node, petiole, leaf):
    return B.pa.getLeafShape(B.pid, shoot, node, petiole, leaf)


# ---- Heliades ------------------------------------------------------------------------------------

@op("plantarch_new", "heliades")
def _(B):
    from ..plantarchitecture import PlantArchitecture
    B.pa = PlantArchitecture(B.scene)
    return B.pa


@op("plantarch_load", "heliades")
def _(B, model="tomato"):
    B.pa.loadPlantModelFromLibrary(model)


@op("plantarch_shoot_params", "heliades")
def _(B, label):
    from ..plantarchitecture.structure import shoot_parameters_to_dict
    return shoot_parameters_to_dict(B.pa.getCurrentShootParameters(label))


@op("plantarch_available_models", "heliades")
def _(B):
    return B.pa.getAvailablePlantModels()


@op("plantarch_build", "heliades")
def _(B, position=(0.0, 0.0, 0.0), age=0.0):
    B.pid = B.pa.buildPlantInstanceFromLibrary(position, age)
    return B.pid


@op("plantarch_age0_observables", "heliades")
def _(B):
    """The same observables as the Helios op, read from the built plant (shoot 0, phytomer 0, petiole 0)."""
    from ..plantarchitecture.params import rad2deg
    B.plantarch_build((0.0, 0.0, 0.0), 0.0)
    shoot = B.pa.plant_instances[B.pid].shoot_tree[0]
    ph = shoot.phytomers[0]
    rot = shoot.base_rotation
    nleaf = len(ph.leaf_objIDs[0])
    return {"base_rotation_deg": [rad2deg(rot.pitch), rad2deg(rot.yaw), rad2deg(rot.roll)],
            "internode_pitch_deg": rad2deg(ph.internode_pitch),
            "internode_phyllotactic_angle_deg": rad2deg(ph.internode_phyllotactic_angle),
            "petiole_pitch_deg": rad2deg(ph.petiole_pitch[0]),
            "petiole_curvature": f32(ph.petiole_curvature[0]),
            "leaf_prototype": list(ph.leaf_prototype_index[0]),
            "leaf_longitudinal_curvature": [f32(ph.getLeafShape(0, l).longitudinal_curvature) for l in range(nleaf)]}


@op("plantarch_age0_observables", "helios")
def _(B):
    B.plantarch_build((0.0, 0.0, 0.0), 0.0)
    root = B.plantarch_xml()
    shoot = root.find("plant_instance/shoot")
    ph = shoot.find("phytomer")
    pet = ph.find("internode/petiole")
    leaves = pet.findall("leaf")
    return {"base_rotation_deg": [float(x) for x in shoot.find("base_rotation").text.split()],
            "internode_pitch_deg": float(ph.find("internode/internode_pitch").text),
            "internode_phyllotactic_angle_deg": float(ph.find("internode/internode_phyllotactic_angle").text),
            "petiole_pitch_deg": float(pet.find("petiole_pitch").text),
            "petiole_curvature": float(pet.find("petiole_curvature").text),
            "leaf_prototype": [int(lf.find("leaf_prototype").text) for lf in leaves],
            "leaf_longitudinal_curvature": [f32(B.plantarch_leaf_shape(0, 0, 0, l).longitudinal_curvature)
                                            for l in range(len(leaves))]}


# ---- redefining a shoot type with other distributions --------------------------------------------
# spec: list of (dotted path under ShootParameters, distribution, parameters)

_JSON_DIST = {"normal": "normal", "weibull": "weibull", "uniform": "uniform", "discretevalues": "discretevalues",
              "constant": "constant"}


@op("plantarch_redefine", "helios")
def _(B, label, spec):
    """getCurrentShootParameters -> edit the dict -> defineShootType (the PyHelios JSON round trip)."""
    d = B.pa.getCurrentShootParameters(label)
    for path, dist, params in spec:
        node = d
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = {"distribution": _JSON_DIST[dist], "parameters": [float(x) for x in params]}
    B.pa.defineShootType(label, d)


@op("plantarch_redefine", "heliades")
def _(B, label, spec):
    """Mirrors the wrapper: each edited parameter is rebuilt as ``RandomParameter_*; initialize(gen)``
    (jsonToRandomParameterFloat/Int), so even a parameter Helios leaves without a generator
    (``internode.max_floral_buds_per_petiole``) can be distributed."""
    from ..plantarchitecture import RandomParameterFloat, RandomParameterInt
    sp = B.pa.getCurrentShootParameters(label)
    for path, dist, params in spec:
        keys = path.split(".")
        parent = sp
        for k in keys[:-1]:
            parent = getattr(parent, k)
        old = getattr(parent, keys[-1])
        new = RandomParameterInt() if isinstance(old, RandomParameterInt) else RandomParameterFloat()
        new.initialize(B.pa.generator)
        if dist == "constant":
            new.set(params[0])
        elif dist == "uniform":
            new.uniformDistribution(*params)
        elif dist == "normal":
            new.normalDistribution(*params)
        elif dist == "weibull":
            new.weibullDistribution(*params)
        elif dist == "discretevalues":
            new.discreteValues(params)
        else:
            raise ValueError(dist)
        setattr(parent, keys[-1], new)          # C++ copy assignment
    B.pa.defineShootType(label, sp)
