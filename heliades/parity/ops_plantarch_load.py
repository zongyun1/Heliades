# SPDX-License-Identifier: LGPL-2.1-or-later
"""PlantArchitecture data-layer operations for both backends (W2-C).

``plantarch_new`` / ``plantarch_load`` / ``plantarch_shoot_params`` exist on both. The Helios side also
exposes oracle-only observations of an age-0 build (``plantarch_build``, ``plantarch_xml``,
``plantarch_leaf_shape``); the Heliades counterparts of those that W2-C can answer are reconstructed
from the ported parameter layer plus an explicit replay of the age-0 Context draw plan
(``age0_draw_plan``), until W3-B/W3-E port the Phytomer constructor and the builder.
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
    return B.pa.buildPlantInstanceFromLibrary(position, age)     # NotImplementedError until W3-E


def age0_draw_plan(pa, shoot_label="mainstem"):
    """Replay the Context draws of ``buildTomatoPlant`` at age 0 up to the last leaflet (SPEC §3.3 step 2),
    using the ported parameter objects, and the private blade stream (§3.1).

    This is a W2-C *stand-in* for the parts of the Phytomer constructor that only consume parameters;
    it is not the geometry port. Returns a dict with the sampled values the XML / getLeafShape expose.
    """
    from ..core.random import MinstdRand0
    from ..plantarchitecture.library.tomato import base_rotation_draws
    from ..plantarchitecture.params import deg2rad, rad2deg
    rot = base_rotation_draws(pa)                       # 2 randu, right to left
    sp = pa.shoot_types[shoot_label].copy()             # snapshot -> Shoot copy (no draws)
    pp = sp.phytomer_parameters.copy()                  # Phytomer's own copy
    pp.resample()                                       # (a)-(f)
    phyl = deg2rad(pp.internode.phyllotactic_angle.val())
    petiole_pitch = deg2rad(pp.petiole.pitch.val())
    pp.petiole.pitch.resample()
    petiole_curvature = pp.petiole.curvature.val()
    pp.petiole.curvature.resample()
    pp.petiole.taper.val()
    pp.petiole.taper.resample()                         # PA:3406 (constant for tomato)
    pp.internode.max_vegetative_buds_per_petiole.val()
    pp.internode.max_vegetative_buds_per_petiole.resample()   # PA:3439
    pp.internode.max_floral_buds_per_petiole.val()
    pp.internode.max_floral_buds_per_petiole.resample()       # createPetioleFloralBuds, PA:2678
    # blade cache from the private stream: blade b = p*L + l uses draw b of longitudinal_curvature
    nleaf = pp.leaf.leaves_per_petiole.val()
    proto = pp.leaf.prototype
    private = MinstdRand0(proto.unique_prototype_identifier)
    ctx_gen = proto.setRandomGenerator(private)
    curv = []
    for _p in range(proto.unique_prototypes):
        for _l in range(nleaf):
            curv.append(proto.longitudinal_curvature.val())
            proto.longitudinal_curvature.resample()     # GenericLeafPrototype's trailing resample (AS:868)
    proto.setRandomGenerator(ctx_gen)
    choice = []
    for _l in range(nleaf):
        choice.append(pa.context_ptr.randu(0, int(proto.unique_prototypes) - 1))   # PA:3503
        pp.leaf.pitch.resample()                        # PA:3538
    return {"base_rotation_deg": [rad2deg(rot.pitch), rad2deg(rot.yaw), rad2deg(rot.roll)],
            "internode_pitch_deg": rad2deg(deg2rad(pp.internode.pitch.val())),
            "internode_phyllotactic_angle_deg": rad2deg(phyl),
            "petiole_pitch_deg": rad2deg(petiole_pitch),
            "petiole_curvature": f32(petiole_curvature),
            "leaf_prototype": choice,
            "leaf_longitudinal_curvature": [curv[p * nleaf + l] for l, p in enumerate(choice)]}


@op("plantarch_age0_observables", "heliades")
def _(B):
    return age0_draw_plan(B.pa)


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
