# SPDX-License-Identifier: LGPL-2.1-or-later
"""The overridable fields of a species file: where each lives in the parameter structs, its kind and its valid range.

A species file overrides values in seven sections of a shoot type (``shoot``, ``internode``, ``petiole``, ``leaf``,
``peduncle``, ``inflorescence``, ``leaf_prototype``). Each field has a *kind*:

``float`` / ``int``
    a ``RandomParameter_float`` / ``RandomParameter_int``: a fixed ``value`` or a distribution (``uniform``,
    ``normal``, ``weibull`` for floats; ``uniform``, ``discrete`` for ints);
``uint``, ``real``, ``bool``, ``rgb``, ``vec3``, ``str``
    a plain member (``uint`` counts, ``real`` a plain float, ``rgb`` three numbers in [0, 1], ``vec3`` three
    numbers, ``str`` a file name);
``texture_map`` / ``length_map``
    ``std::map<int, std::string>`` / ``std::map<int, float>`` keyed by compound-leaf index; a single value means
    ``{0: value}``.

The ranges are plausibility bounds (a typo guard), not biology: a value outside them is almost certainly a unit or
sign mistake. Units: metres, degrees, days (the plant-architecture clock runs in degree-days, one per day at the
default temperature), degrees per metre for curvatures.
"""
from __future__ import annotations

import math

INF = math.inf

# (kind, min, max)
_ANGLE = (-360.0, 360.0)
_LEN = (0.0, 5.0)
_PROB = (0.0, 1.0)

SHOOT = {
    "max_nodes": ("int", 1, 10000),
    "max_nodes_per_season": ("int", 1, 100000),
    "girth_area_factor": ("float", 0.0, 100.0),
    "insertion_angle_tip": ("float",) + _ANGLE,
    "insertion_angle_decay_rate": ("float", -1000.0, 1000.0),
    "internode_length_max": ("float",) + _LEN,
    "internode_length_min": ("float",) + _LEN,
    "internode_length_decay_rate": ("float", -10.0, 10.0),
    "base_roll": ("float",) + _ANGLE,
    "base_yaw": ("float",) + _ANGLE,
    "gravitropic_curvature": ("float", -5000.0, 5000.0),
    "tortuosity": ("float", 0.0, 1000.0),
    "tortuosity_persistence_length": ("float", 0.0, 10.0),
    "phyllochron_min": ("float", 0.01, 365.0),
    "elongation_rate_max": ("float", 0.0, 10.0),
    "leaf_expansion_rate_max": ("float", -1.0, 10.0),        # -1: unset (leaves follow elongation_rate_max)
    "vegetative_bud_break_probability_min": ("float",) + _PROB,
    "vegetative_bud_break_probability_max": ("float",) + _PROB,
    "vegetative_bud_break_probability_decay_rate": ("float", -100.0, 100.0),
    "max_terminal_floral_buds": ("int", 0, 100),
    "flower_bud_break_probability": ("float",) + _PROB,
    "fruit_set_probability": ("float",) + _PROB,
    "vegetative_bud_break_time": ("float", 0.0, 3650.0),
    "flowers_require_dormancy": ("bool", None, None),
    "growth_requires_dormancy": ("bool", None, None),
    "determinate_shoot_growth": ("bool", None, None),
}

INTERNODE = {
    "pitch": ("float",) + _ANGLE,
    "phyllotactic_angle": ("float",) + _ANGLE,
    "radius_initial": ("float", 0.0, 1.0),
    "max_vegetative_buds_per_petiole": ("int", 0, 20),
    "max_floral_buds_per_petiole": ("int", 0, 20),
    "color": ("rgb", None, None),
    "image_texture": ("str", None, None),
    "image_class_map_file": ("str", None, None),
    "length_segments": ("uint", 1, 100),
    "radial_subdivisions": ("uint", 3, 64),
}

PETIOLE = {
    "petioles_per_internode": ("uint", 1, 10),
    "pitch": ("float",) + _ANGLE,
    "radius": ("float", 0.0, 1.0),
    "length": ("float",) + _LEN,
    "curvature": ("float", -100000.0, 100000.0),
    "taper": ("float", 0.0, 1.0),
    "color": ("rgb", None, None),
    "length_segments": ("uint", 1, 100),
    "radial_subdivisions": ("uint", 3, 64),
    "flexibility": ("float", 0.0, 10000.0),
    "flexibility_aging": ("float", 0.0, 10000.0),
}

LEAF = {
    "leaves_per_petiole": ("int", 1, 50),
    "pitch": ("float",) + _ANGLE,
    "yaw": ("float",) + _ANGLE,
    "roll": ("float",) + _ANGLE,
    "leaflet_offset": ("float", 0.0, 1.0),
    "leaflet_scale": ("float", 0.0, 2.0),
    "intercalary_leaflet_scale": ("float", 0.0, 2.0),
    "prototype_scale": ("float", 0.0, 5.0),
}

PEDUNCLE = {
    "length": ("float",) + _LEN,
    "radius": ("float", 0.0, 1.0),
    "pitch": ("float",) + _ANGLE,
    "roll": ("float",) + _ANGLE,
    "yaw": ("float",) + _ANGLE,
    "curvature": ("float", -100000.0, 100000.0),
    "color": ("rgb", None, None),
    "length_segments": ("uint", 1, 100),
    "radial_subdivisions": ("uint", 3, 64),
}

INFLORESCENCE = {
    "flowers_per_peduncle": ("int", 1, 500),
    "flower_offset": ("float", 0.0, 1.0),
    "pitch": ("float",) + _ANGLE,
    "roll": ("float",) + _ANGLE,
    "flower_prototype_scale": ("float", 0.0, 5.0),
    "fruit_prototype_scale": ("float", 0.0, 5.0),
    "fruit_class_map_file": ("str", None, None),
    "fruit_gravity_factor_fraction": ("float",) + _PROB,
    "unique_prototypes": ("uint", 1, 1000),
    "inflorescence_maturity_period": ("float", -1.0, 3650.0),   # -1: the plant's fruit-maturity threshold
}

LEAF_PROTOTYPE = {
    "leaf_aspect_ratio": ("float", 0.001, 20.0),
    "midrib_fold_fraction": ("float",) + _PROB,
    "longitudinal_curvature": ("float", -10.0, 10.0),
    "longitudinal_curvature_exponent": ("float", 0.0, 20.0),
    "lateral_curvature": ("float", -10.0, 10.0),
    "roll_fraction": ("float",) + _PROB,
    "petiole_roll": ("float",) + _ANGLE,
    "wave_period": ("float", 0.0, 10.0),
    "wave_amplitude": ("float", 0.0, 10.0),
    "flexibility": ("float", 0.0, 10000.0),
    "flexibility_taper": ("float", 0.0, 10000.0),
    "flexibility_aging": ("float", 0.0, 10000.0),
    "flexibility_aging_max": ("float", 0.0, 10000.0),
    "leaf_buckle_length": ("float",) + _PROB,
    "leaf_buckle_angle": ("float", 0.0, 180.0),
    "leaf_offset": ("vec3", -10.0, 10.0),
    "subdivisions": ("uint", 1, 200),
    "unique_prototypes": ("uint", 1, 1000),
    "build_petiolule": ("bool", None, None),
    "petiolule_length": ("length_map", 0.0, 5.0),
    "petiolule_radius_fraction": ("real", 0.0, 1.0),
    "leaf_texture_file": ("texture_map", None, None),
    "leaf_class_map_file": ("texture_map", None, None),
    "OBJ_model_file": ("str", None, None),
}

#: section name -> (attribute path from ShootParameters, field table)
SECTIONS = {
    "shoot": ((), SHOOT),
    "internode": (("phytomer_parameters", "internode"), INTERNODE),
    "petiole": (("phytomer_parameters", "petiole"), PETIOLE),
    "leaf": (("phytomer_parameters", "leaf"), LEAF),
    "peduncle": (("phytomer_parameters", "peduncle"), PEDUNCLE),
    "inflorescence": (("phytomer_parameters", "inflorescence"), INFLORESCENCE),
    "leaf_prototype": (("phytomer_parameters", "leaf", "prototype"), LEAF_PROTOTYPE),
}

FLOAT_DISTRIBUTIONS = ("value", "uniform", "normal", "weibull")
INT_DISTRIBUTIONS = ("value", "uniform", "discrete")


def target(sp, section: str):
    """The struct of ``sp`` (a ShootParameters) that a section's fields live in."""
    obj = sp
    for a in SECTIONS[section][0]:
        obj = getattr(obj, a)
    return obj
