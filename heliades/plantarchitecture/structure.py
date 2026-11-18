# SPDX-License-Identifier: LGPL-2.1-or-later
"""Intermediate-state dumps for parity (SPEC §8 "Common intermediate-state oracle").

``shoot_parameters_to_dict`` mirrors the JSON PyHelios returns from ``getCurrentShootParameters``
(``native/src/pyhelios_wrapper_plantarchitecture.cpp``: ``randomParameterFloatToJSON`` and friends), so a
Heliades ``ShootParameters`` can be compared with the oracle's field by field. Like the wrapper, a
constant parameter is reported as ``[val()]`` *of a copy* (the wrapper takes it by value), so dumping
never changes the cache state or draws.

``canonical`` turns either side's dict into a string with every number rounded to float32, which is
the precision both sides store.

``plant_structure_dict`` (W3-E) is the structured form of ``writePlantStructureXML``: it parses the XML text
of the port (``plant.plant_structure_xml``), so every value is exactly what Helios writes (``ostream << float``,
6 significant digits), and the same parser reads an oracle XML file (:func:`xml_to_structure_dict`). The live
plant then adds what the XML does not carry: the organ object IDs (``organs``) per shoot and phytomer and per plant.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np

from .hooks import hook_name
from .params import PhytomerParameters, ShootParameters
from .random_parameter import RandomParameterFloat, RandomParameterInt

f32 = np.float32


def random_parameter_to_dict(p) -> dict:
    if isinstance(p, RandomParameterFloat):
        if p.distribution == "constant":
            return {"distribution": p.distribution, "parameters": [float(p.copy().val())]}
        return {"distribution": p.distribution, "parameters": [float(x) for x in p.distribution_parameters]}
    if isinstance(p, RandomParameterInt):
        if p.distribution == "constant":
            return {"distribution": p.distribution, "parameters": [float(f32(p.copy().val()))]}
        return {"distribution": p.distribution, "parameters": [float(f32(x)) for x in p.distribution_parameters]}
    raise TypeError(type(p))


def _rgb(c):
    return {"r": float(c[0]), "g": float(c[1]), "b": float(c[2])}


def _vec3(v):
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def leaf_prototype_to_dict(p) -> dict:
    R = random_parameter_to_dict
    d = {k: R(getattr(p, k)) for k in (
        "leaf_aspect_ratio", "midrib_fold_fraction", "longitudinal_curvature", "longitudinal_curvature_exponent",
        "lateral_curvature", "petiole_roll", "roll_fraction", "wave_period", "wave_amplitude", "flexibility",
        "flexibility_taper", "flexibility_aging", "flexibility_aging_max", "leaf_buckle_length", "leaf_buckle_angle")}
    d.update({
        "OBJ_model_file": p.OBJ_model_file,
        "build_petiolule": bool(p.build_petiolule),
        "leaf_class_map_file": {str(k): v for k, v in sorted(p.leaf_class_map_file.items())},
        "leaf_offset": _vec3(p.leaf_offset),
        "leaf_texture_file": {str(k): v for k, v in sorted(p.leaf_texture_file.items())},
        "leaf_texture_series": {str(k): v for k, v in sorted(p.leaf_texture_series.items())},
        "leaf_texture_stage_ages": [float(f32(a)) for a in p.leaf_texture_stage_ages],
        "petiolule_length": {str(k): float(f32(v)) for k, v in sorted(p.petiolule_length.items())},
        "prototype_function": hook_name(p.prototype_function),
        "subdivisions": int(p.subdivisions),
        "unique_prototypes": int(p.unique_prototypes),
    })
    return d


def phytomer_parameters_to_dict(pp: PhytomerParameters) -> dict:
    R = random_parameter_to_dict
    it, pe, lf, pd, fl = pp.internode, pp.petiole, pp.leaf, pp.peduncle, pp.inflorescence
    return {
        "internode": {
            "pitch": R(it.pitch), "phyllotactic_angle": R(it.phyllotactic_angle), "radius_initial": R(it.radius_initial),
            "max_vegetative_buds_per_petiole": R(it.max_vegetative_buds_per_petiole),
            "max_floral_buds_per_petiole": R(it.max_floral_buds_per_petiole), "color": _rgb(it.color),
            "image_texture": it.image_texture, "image_class_map_file": it.image_class_map_file,
            "length_segments": int(it.length_segments), "radial_subdivisions": int(it.radial_subdivisions)},
        "petiole": {
            "petioles_per_internode": int(pe.petioles_per_internode), "pitch": R(pe.pitch), "radius": R(pe.radius),
            "length": R(pe.length), "curvature": R(pe.curvature), "taper": R(pe.taper), "color": _rgb(pe.color),
            "length_segments": int(pe.length_segments), "radial_subdivisions": int(pe.radial_subdivisions),
            "flexibility": R(pe.flexibility), "flexibility_aging": R(pe.flexibility_aging)},
        "leaf": {
            "leaves_per_petiole": R(lf.leaves_per_petiole), "pitch": R(lf.pitch), "yaw": R(lf.yaw), "roll": R(lf.roll),
            "leaflet_offset": R(lf.leaflet_offset), "leaflet_scale": R(lf.leaflet_scale),
            "intercalary_leaflet_scale": R(lf.intercalary_leaflet_scale), "prototype_scale": R(lf.prototype_scale),
            "prototype": leaf_prototype_to_dict(lf.prototype)},
        "peduncle": {
            "length": R(pd.length), "radius": R(pd.radius), "pitch": R(pd.pitch), "roll": R(pd.roll), "yaw": R(pd.yaw),
            "curvature": R(pd.curvature), "color": _rgb(pd.color), "length_segments": int(pd.length_segments),
            "radial_subdivisions": int(pd.radial_subdivisions)},
        "inflorescence": {
            "flowers_per_peduncle": R(fl.flowers_per_peduncle), "flower_offset": R(fl.flower_offset),
            "pitch": R(fl.pitch), "roll": R(fl.roll), "flower_prototype_scale": R(fl.flower_prototype_scale),
            "flower_prototype_function": hook_name(fl.flower_prototype_function),
            "fruit_prototype_scale": R(fl.fruit_prototype_scale),
            "fruit_prototype_function": hook_name(fl.fruit_prototype_function),
            "fruit_class_map_file": fl.fruit_class_map_file,
            "fruit_gravity_factor_fraction": R(fl.fruit_gravity_factor_fraction),
            "unique_prototypes": int(fl.unique_prototypes),
            "inflorescence_maturity_period": R(fl.inflorescence_maturity_period)},
    }


def shoot_parameters_to_dict(sp: ShootParameters) -> dict:
    from .params import _SHOOT_RP
    d = {k: random_parameter_to_dict(getattr(sp, k)) for k in _SHOOT_RP}
    d.update({
        "phytomer_parameters": phytomer_parameters_to_dict(sp.phytomer_parameters),
        "flowers_require_dormancy": bool(sp.flowers_require_dormancy),
        "growth_requires_dormancy": bool(sp.growth_requires_dormancy),
        "determinate_shoot_growth": bool(sp.determinate_shoot_growth),
        "child_shoot_types": {"labels": list(sp.child_shoot_type_labels),
                              "probabilities": [float(p) for p in sp.child_shoot_type_probabilities]},
    })
    return d


def _canon(v):
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, (bool, np.bool_)) or v is None or isinstance(v, str):
        return v
    return repr(float(f32(v)))


def canonical(d) -> str:
    """Key-sorted JSON with every number written as its float32 value."""
    return json.dumps(_canon(d), sort_keys=True, separators=(",", ":"))


# ---- plant structure (W3-E) -------------------------------------------------------------------------

#: XML tags that repeat inside their parent -> the list key they are collected under
_REPEATED = {"plant_instance": "plant_instances", "shoot": "shoots", "phytomer": "phytomers", "petiole": "petioles",
             "leaf": "leaves", "floral_bud": "floral_buds", "flower": "flowers",
             "internode_floral_buds": "internode_floral_buds"}
#: tags whose text is a string (everything else is numeric)
_STRING_TAGS = {"shoot_type_label", "growth_shoot_type_label", "plant_model"}
#: tags that are always lists of numbers, even with one value
_VECTOR_TAGS = {"base_position", "base_rotation", "petiole_axis_initial", "petiole_rotation_axis",
                "curvature_perturbations", "yaw_perturbations", "azimuthal_restoring_angles", "gravitropic_angles"}
_INT_TAGS = {"parent_shoot_ID", "parent_node_index", "parent_petiole_index", "is_dormant", "dormancy_cycles",
             "meristem_is_alive", "nodes_this_season", "is_axis_continuation", "max_nodes_growth_cap",
             "max_nodes_per_season_growth_cap", "is_evergreen", "minimum_leaf_texture_stage", "phytomer_is_dormant",
             "internode_length_segments", "petiole_length_segments", "petiole_radial_subdivisions", "leaf_is_sessile",
             "leaf_prototype", "bud_state", "parent_index", "bud_index", "is_terminal"}


def _num(tok: str, integer: bool):
    if integer:
        return int(tok)
    return float(f32(float(tok)))


def _value(tag: str, text: str):
    text = (text or "").strip()
    if tag in _STRING_TAGS:
        return text
    if tag == "vegetative_bud_states":                       # " 1 1 ; 5 ; " -> [[1, 1], [5]]
        return [[int(t) for t in grp.split()] for grp in text.split(";")[:-1]] if text else []
    toks = text.replace(";", " ").split()
    vals = [_num(t, tag in _INT_TAGS) for t in toks]
    if tag in _VECTOR_TAGS:
        return vals
    return vals[0] if len(vals) == 1 else vals


def _element(el) -> dict:
    d: dict = {k: (int(v) if v.lstrip("-").isdigit() else v) for k, v in el.attrib.items()}
    for child in el:
        if child.tag in _REPEATED:
            d.setdefault(_REPEATED[child.tag], []).append(_element(child))
        elif len(child):
            d[child.tag] = _element(child)
        else:
            d[child.tag] = _value(child.tag, child.text)
    return d


def xml_to_structure_dict(text: str) -> dict:
    """``writePlantStructureXML`` text (Helios or Heliades) -> nested dict.

    ``{"ID", "base_position", "plant_age", ..., "shoots": [{"ID", "shoot_type_label", "base_rotation", ...,
    "phytomers": [{"phytomer_age", "vegetative_bud_states", "internode": {"internode_length", ...,
    "petioles": [{"petiole_length", ..., "leaves": [...], "floral_buds": [{"bud_state", "peduncle": {...},
    "inflorescence": {"flower_offset", "flowers": [...]}}]}], "internode_floral_buds": [{"floral_buds": [...]}]}}]}]}``.
    Angles are in degrees as in the XML; every number is the float32 value of the printed text."""
    root = ET.fromstring(text)
    plants = [_element(p) for p in root.findall("plant_instance")]
    if len(plants) != 1:
        raise ValueError(f"expected one <plant_instance>, found {len(plants)}")
    return plants[0]


def plant_structure_dict(pa, plantID: int, organs: bool = True) -> dict:
    """The structured plant of ``writePlantStructureXML`` (shoots, phytomers, petioles, leaves, floral buds and
    their sampled parameters), for the simulator converter and the calibration loop.

    With ``organs`` the live object IDs are added (``-1`` = no geometry):

    * per shoot: ``organs = {"internode": objID}``;
    * per phytomer: ``organs = {"petioles": [objID per existing petiole tube], "leaves": [[objID per leaflet] per petiole],
      "peduncles": [[objIDs] per floral bud], "inflorescences": [[objIDs] per floral bud]}`` (floral buds in
      ``floral_buds`` order: per petiole, then the internode buds);
    * per plant: ``organs`` = the plant-level getters (``internode``, ``petiole``, ``leaf``, ``peduncle``,
      ``flower``, ``fruit`` object IDs, in their C++ order).
    """
    from .xml_io import plant_structure_xml
    d = xml_to_structure_dict(plant_structure_xml(pa, plantID))
    if not organs:
        return d
    ctx = pa.context_ptr
    live = [s for s in pa.plant_instances[plantID].shoot_tree if not s.isPruned()]
    for sd, shoot in zip(d.get("shoots", []), live):
        assert sd["ID"] == shoot.ID
        oid = shoot.internode_tube_objID
        sd["organs"] = {"internode": int(oid) if ctx.does_object_exist(oid) else -1}
        for pd, ph in zip(sd.get("phytomers", []), shoot.phytomers):
            pd["organs"] = {
                "petioles": [int(o) for o in ph.getExistingPetioleObjIDs()],
                "leaves": [[int(o) for o in pet] for pet in ph.leaf_objIDs],
                "peduncles": [[int(o) for o in fb.peduncle_objIDs] for pet in ph.floral_buds for fb in pet],
                "inflorescences": [[int(o) for o in fb.inflorescence_objIDs] for pet in ph.floral_buds for fb in pet],
            }
    d["organs"] = {"internode": pa.getPlantInternodeObjectIDs(plantID), "petiole": pa.getPlantPetioleObjectIDs(plantID),
                   "leaf": pa.getPlantLeafObjectIDs(plantID), "peduncle": pa.getPlantPeduncleObjectIDs(plantID),
                   "flower": pa.getPlantFlowerObjectIDs(plantID), "fruit": pa.getPlantFruitObjectIDs(plantID)}
    return d
