# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture (PlantArchitecture.h:430-1646, PlantArchitecture.cpp:798-1150;
# Copyright (C) 2016-2026 Brian Bailey).
"""Plant-architecture parameter structs with C++ value semantics.

Every struct here is a value type in C++. Python objects are references, so the structs implement the
C++ operations explicitly and the attribute syntax is routed through them:

``copy()``
    the copy constructor (implicit unless noted);
``_assign_from(other)``
    ``operator=`` (custom where Helios declares one, otherwise memberwise);
``s.field = value``
    if ``field`` currently holds a value type (a ``RandomParameter``, a sub-struct, a
    ``LeafPrototype``) this calls ``s.field._assign_from(value)``, i.e. the C++ assignment, so
    ``pp.internode.pitch = 10`` is ``operator=(float)`` and ``pp.leaf.prototype = proto`` is
    ``LeafPrototype::operator=`` (which redraws the identifier). Plain fields are coerced to their
    C++ type (``uint``, ``float``, RGB color).

RNG traps reproduced (SPEC §9.8):

* ``LeafPrototype(gen)`` and ``LeafPrototype::operator=`` draw an identifier when the generator is
  non-null; the copy constructor and ``duplicate()`` do not;
* ``LeafParameters::operator=`` uses ``prototype.duplicate()`` (no draw), so neither
  ``PhytomerParameters`` nor ``ShootParameters`` assignment draws;
* ``PhytomerParameters(gen)`` draws twice: ``LeafPrototype(gen)`` then ``operator=`` (PA:983).
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np

from ..core.random import uniform_integer
from ..core.vec import sinf
from ..migration_registry import ports
from . import hooks
from .random_parameter import PlantArchitectureError, RandomParameterFloat, RandomParameterInt

f32 = np.float32
_M_PI_F = f32(np.pi)
UINT_MAX = 2**32 - 1


# ---- C++ scalar helpers ---------------------------------------------------------------------------

def c_uint(v) -> int:
    """Implicit conversion to ``uint`` (wraps modulo 2^32; floats truncate)."""
    return int(v) % 2**32


@ports("helios::deg2rad")
def deg2rad(deg) -> np.float32:
    """``deg * float(M_PI) / 180.f`` (global.cpp:855)."""
    return f32(f32(f32(deg) * _M_PI_F) / f32(180.0))


@ports("helios::rad2deg")
def rad2deg(rad) -> np.float32:
    """``rad * 180.f / float(M_PI)`` (global.cpp:859)."""
    return f32(f32(f32(rad) * f32(180.0)) / _M_PI_F)


def clampf(v, lo, hi) -> np.float32:
    v, lo, hi = f32(v), f32(lo), f32(hi)
    return lo if v < lo else hi if v > hi else v


@ports("helios::sum")
def sum_f(values) -> np.float32:
    """helios::sum(vector<float>): float accumulation from 0."""
    if len(values) == 0:
        raise PlantArchitectureError("ERROR (sum): Vector is empty.")
    m = f32(0)
    for v in values:
        m = f32(m + f32(v))
    return m


class RGBcolor(tuple):
    """helios::RGBcolor (immutable float32 triple; value semantics come for free)."""

    def __new__(cls, r=0.0, g=0.0, b=0.0, _clamp=True):
        c = (lambda x: clampf(x, 0, 1)) if _clamp else f32
        return super().__new__(cls, (c(r), c(g), c(b)))

    r = property(lambda s: s[0])
    g = property(lambda s: s[1])
    b = property(lambda s: s[2])


def make_RGBcolor(r, g, b) -> RGBcolor:
    """``make_RGBcolor(float, float, float)``: arguments narrowed to float, then clamped to [0,1]."""
    return RGBcolor(r, g, b)


class RGB:
    forestgreen = make_RGBcolor(0.133, 0.545, 0.133)   # global.cpp:108
    black = make_RGBcolor(0, 0, 0)


def vec3(x=0.0, y=0.0, z=0.0) -> tuple:
    return (f32(x), f32(y), f32(z))


# ---- value-semantics base ------------------------------------------------------------------------

_MISSING = object()


def _is_value_type(v) -> bool:
    return hasattr(v, "_assign_from") and hasattr(v, "copy")


def _copy_value(v):
    if _is_value_type(v):
        return v.copy()
    if isinstance(v, dict):
        return {k: _copy_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_copy_value(x) for x in v]
    return v   # immutable (number, str, tuple, RGBcolor, Hook, None) or a pointer (generator)


class _Struct:
    """Base for C++ structs: ``_FIELDS`` in declaration order, ``_TYPES`` for scalar coercion."""

    _FIELDS: tuple = ()
    _TYPES: dict = {}

    def __setattr__(self, name, value):
        cur = self.__dict__.get(name, _MISSING)
        if cur is not _MISSING and _is_value_type(cur) and cur is not value:
            cur._assign_from(value)
            return
        t = self._TYPES.get(name)
        if t == "uint":
            value = c_uint(value)
        elif t == "int":
            value = int(value)
        elif t == "float":
            value = f32(value)
        elif t == "bool":
            value = bool(value)
        elif t == "rgb":
            value = value if isinstance(value, RGBcolor) else RGBcolor(*value)
        elif t == "vec3":
            value = vec3(*value)
        object.__setattr__(self, name, value)

    def _raw_set(self, name, value):
        object.__setattr__(self, name, value)

    def _memberwise_copy(self):
        c = type(self).__new__(type(self))
        for f in self._FIELDS:
            c._raw_set(f, _copy_value(self.__dict__[f]))
        return c

    def _memberwise_assign(self, a, fields=None):
        for f in (self._FIELDS if fields is None else fields):
            src = a.__dict__[f]
            cur = self.__dict__.get(f, _MISSING)
            if cur is not _MISSING and _is_value_type(cur):
                cur._assign_from(src)
            else:
                self._raw_set(f, _copy_value(src))

    def copy(self):
        """Implicit copy constructor."""
        return self._memberwise_copy()

    def _assign_from(self, a):
        """Implicit copy assignment (overridden where Helios declares a custom operator=)."""
        if a is not self:
            if not isinstance(a, type(self)):
                raise TypeError(f"cannot assign {type(a).__name__} to {type(self).__name__}")
            self._memberwise_assign(a)
        return self

    def __repr__(self):
        return f"{type(self).__name__}(" + ", ".join(f"{f}={self.__dict__.get(f)!r}" for f in self._FIELDS) + ")"


# ---- AxisRotation, BudState, buds ----------------------------------------------------------------

class AxisRotation(_Struct):
    """AxisRotation (H:430-481)."""

    _FIELDS = ("pitch", "yaw", "roll", "azimuth", "peduncle_axis", "gravity_fraction", "contact_tilt_axis",
               "contact_tilt_angle")
    _TYPES = {"pitch": "float", "yaw": "float", "roll": "float", "azimuth": "float", "peduncle_axis": "vec3",
              "gravity_fraction": "float", "contact_tilt_axis": "vec3", "contact_tilt_angle": "float"}

    @ports("AxisRotation::AxisRotation")
    def __init__(self, a_pitch=0.0, a_yaw=0.0, a_roll=0.0):
        self.pitch = a_pitch
        self.yaw = a_yaw
        self.roll = a_roll
        self.azimuth = 0
        self.peduncle_axis = (0, 0, 1)
        self.gravity_fraction = 0
        self.contact_tilt_axis = (0, 0, 1)
        self.contact_tilt_angle = 0

    @ports("AxisRotation::operator+")
    def __add__(self, a):
        # {a.pitch + pitch, ...}: three-argument constructor, so the other members are reset
        return AxisRotation(f32(a.pitch + self.pitch), f32(a.yaw + self.yaw), f32(a.roll + self.roll))

    @ports("AxisRotation::operator-")
    def __sub__(self, a):
        # Helios H:479 returns {a.pitch - pitch, ...}: x - a computes a - x. Ported as written.
        return AxisRotation(f32(a.pitch - self.pitch), f32(a.yaw - self.yaw), f32(a.roll - self.roll))

    def __str__(self):
        return f"AxisRotation<{self.pitch}, {self.yaw}, {self.roll}>"


@ports("make_AxisRotation")
def make_AxisRotation(a_pitch, a_yaw, a_roll) -> AxisRotation:
    """Note the C++ caller evaluates draw arguments right to left (GCC); see SPEC §3.3 / §9.5."""
    return AxisRotation(a_pitch, a_yaw, a_roll)


class BudState(IntEnum):
    BUD_DORMANT = 0
    BUD_ACTIVE = 1
    BUD_FLOWER_CLOSED = 2
    BUD_FLOWER_OPEN = 3
    BUD_FRUITING = 4
    BUD_DEAD = 5


BUD_DORMANT, BUD_ACTIVE, BUD_FLOWER_CLOSED, BUD_FLOWER_OPEN, BUD_FRUITING, BUD_DEAD = tuple(BudState)


class VegetativeBud(_Struct):
    """VegetativeBud (H:616-624)."""

    _FIELDS = ("state", "shoot_type_label", "shoot_ID")
    _TYPES = {"shoot_ID": "uint"}

    def __init__(self):
        self.state = BudState.BUD_DORMANT
        self.shoot_type_label = ""
        self.shoot_ID = -1          # uint: 4294967295


class FloralBud(_Struct):
    """FloralBud (H:626-672)."""

    _FIELDS = ("state", "time_counter", "age", "isterminal", "parent_index", "bud_index",
               "current_fruit_scale_factor", "previous_fruit_scale_factor", "dd_to_inflorescence_maturity",
               "base_position", "base_rotation", "bending_axis", "inflorescence_bases", "inflorescence_rotation",
               "inflorescence_base_scales", "peduncle_objIDs", "inflorescence_objIDs", "contact_swing_axis",
               "contact_swing_angle")
    _TYPES = {"time_counter": "float", "age": "float", "isterminal": "bool", "parent_index": "uint",
              "bud_index": "uint", "current_fruit_scale_factor": "float", "previous_fruit_scale_factor": "float",
              "dd_to_inflorescence_maturity": "float", "base_position": "vec3", "bending_axis": "vec3",
              "contact_swing_axis": "vec3", "contact_swing_angle": "float"}

    def __init__(self):
        self.state = BudState.BUD_DORMANT
        self.time_counter = 0
        self.age = 0
        self.isterminal = False
        self.parent_index = 0
        self.bud_index = 0
        self.current_fruit_scale_factor = 1
        self.previous_fruit_scale_factor = 0
        self.dd_to_inflorescence_maturity = -1
        self.base_position = (0, 0, 0)
        self.base_rotation = AxisRotation()
        self.bending_axis = (0, 0, 0)
        self.inflorescence_bases = []
        self.inflorescence_rotation = []     # list[AxisRotation]
        self.inflorescence_base_scales = []
        self.peduncle_objIDs = []
        self.inflorescence_objIDs = []
        self.contact_swing_axis = (0, 0, 1)
        self.contact_swing_angle = 0



# ---- LeafPrototype -------------------------------------------------------------------------------

class LeafShapeValues(_Struct):
    """LeafShapeValues (H:674-699)."""

    _FIELDS = ("aspect_ratio", "midrib_fold_fraction", "longitudinal_curvature", "longitudinal_curvature_exponent",
               "lateral_curvature", "petiole_roll", "wave_period", "wave_amplitude", "roll_fraction",
               "subdivisions_x", "subdivisions_y", "lattice_offset")
    _TYPES = {**{f: "float" for f in _FIELDS[:9]}, "subdivisions_x": "uint", "subdivisions_y": "uint",
              "lattice_offset": "vec3"}

    def __init__(self):
        self.aspect_ratio = 1
        self.midrib_fold_fraction = 0
        self.longitudinal_curvature = 0
        self.longitudinal_curvature_exponent = 4
        self.lateral_curvature = 0
        self.petiole_roll = 0
        self.wave_period = 0
        self.wave_amplitude = 0
        self.roll_fraction = 0
        self.subdivisions_x = 1
        self.subdivisions_y = 2
        self.lattice_offset = (0, 0, 0)


# Prototype shape parameters, in LeafPrototype::duplicate order (H:917-951).
_LEAF_RP_FIELDS = ("leaf_aspect_ratio", "midrib_fold_fraction", "longitudinal_curvature",
                   "longitudinal_curvature_exponent", "lateral_curvature", "roll_fraction", "petiole_roll",
                   "wave_period", "wave_amplitude", "flexibility", "flexibility_taper", "flexibility_aging",
                   "flexibility_aging_max", "leaf_buckle_length", "leaf_buckle_angle")

# Members copied by duplicate()/operator= (everything except last_built_shape[_valid]).
_LEAF_COPIED = ("leaf_texture_file", "leaf_texture_series", "leaf_class_map_file", "leaf_texture_stage_ages",
                "texture_series_selector", "OBJ_model_file") + _LEAF_RP_FIELDS + (
                "leaf_offset", "subdivisions", "unique_prototypes", "unique_prototype_identifier", "build_petiolule",
                "petiolule_length", "petiolule_radius_fraction", "prototype_function", "generator")

_DEFAULT_CTOR = object()


class LeafPrototype(_Struct):
    """LeafPrototype (H:701-1042, PA:798-930).

    ``LeafPrototype()`` is the defaulted constructor (no initialisation, null generator, null prototype
    function). ``LeafPrototype(generator)`` is the explicit one, which initialises every shape parameter
    with that generator and, if it is not None, draws ``unique_prototype_identifier``.
    """

    _FIELDS = _LEAF_COPIED + ("last_built_shape", "last_built_shape_valid")
    _TYPES = {"texture_series_selector": "uint", "subdivisions": "uint", "unique_prototypes": "uint",
              "unique_prototype_identifier": "uint", "build_petiolule": "bool", "petiolule_radius_fraction": "float",
              "leaf_offset": "vec3", "last_built_shape_valid": "bool"}

    @ports("LeafPrototype::LeafPrototype")
    def __init__(self, generator=_DEFAULT_CTOR):
        self._default_members()
        if generator is _DEFAULT_CTOR:
            return
        self._raw_set("generator", generator)
        self.leaf_aspect_ratio.initialize(1.0, generator)
        self.midrib_fold_fraction.initialize(0.0, generator)
        self.longitudinal_curvature.initialize(0.0, generator)
        self.longitudinal_curvature_exponent.initialize(4.0, generator)
        self.lateral_curvature.initialize(0.0, generator)
        self.roll_fraction.initialize(0.0, generator)
        self.petiole_roll.initialize(0.0, generator)
        self.wave_period.initialize(0.0, generator)
        self.wave_amplitude.initialize(0.0, generator)
        self.flexibility.initialize(0.0, generator)
        self.flexibility_taper.initialize(1.0, generator)
        self.flexibility_aging.initialize(0.0, generator)
        self.flexibility_aging_max.initialize(4.0, generator)
        self.leaf_buckle_length.initialize(0.0, generator)
        self.leaf_buckle_angle.initialize(0.0, generator)
        self.subdivisions = 1
        self.unique_prototypes = 1
        self.leaf_offset = (0, 0, 0)
        self.prototype_function = hooks.GenericLeafPrototype
        self.build_petiolule = False
        if generator is not None:
            self.sampleIdentifier()

    def _default_members(self):
        """Default member initialisers (``LeafPrototype() = default``)."""
        r = self._raw_set
        r("prototype_function", None)
        r("OBJ_model_file", "")
        r("leaf_texture_file", {})          # std::map<int, std::string>
        r("leaf_texture_series", {})        # std::map<int, vector<vector<string>>>
        r("leaf_class_map_file", {})
        r("leaf_texture_stage_ages", [])
        r("texture_series_selector", 0)
        for f in _LEAF_RP_FIELDS:
            r(f, RandomParameterFloat())
        r("leaf_offset", vec3(0, 0, 0))
        r("subdivisions", 1)
        r("unique_prototypes", 1)
        r("build_petiolule", False)
        r("petiolule_length", {0: f32(0.05)})
        r("unique_prototype_identifier", 0)
        r("generator", None)
        r("petiolule_radius_fraction", f32(0))
        r("last_built_shape", LeafShapeValues())
        r("last_built_shape_valid", False)

    @ports("LeafPrototype::duplicate")
    def duplicate(self, a: "LeafPrototype"):
        """Copy every member except the last-built shape; never draws."""
        self._memberwise_assign(a, _LEAF_COPIED)

    @ports("LeafPrototype::LeafPrototype(const LeafPrototype&)")
    def copy(self) -> "LeafPrototype":
        c = LeafPrototype()
        c.duplicate(self)
        return c

    @ports("LeafPrototype::operator=")
    def _assign_from(self, a):
        if not isinstance(a, LeafPrototype):
            raise TypeError(f"cannot assign {type(a).__name__} to LeafPrototype")
        if a is not self:
            self._memberwise_assign(a, _LEAF_COPIED)
            if self.generator is not None:
                self.sampleIdentifier()
        return self

    @ports("LeafPrototype::sampleIdentifier")
    def sampleIdentifier(self):
        """``helios::uniform_int_distribution<uint>`` over [0, UINT_MAX]: 2 engine calls per attempt."""
        assert self.generator is not None
        self.unique_prototype_identifier = uniform_integer(self.generator, UINT_MAX)

    @ports("LeafPrototype::setRandomGenerator")
    def setRandomGenerator(self, rand_generator):
        previous = self.generator
        self._raw_set("generator", rand_generator)
        for f in _LEAF_RP_FIELDS:
            self.__dict__[f].setRandomGenerator(rand_generator)
        return previous

    def getRandomGenerator(self):
        return self.generator

    @ports("LeafPrototype::resolveFlexibility")
    def resolveFlexibility(self) -> np.float32:
        flexibility_value = self.flexibility.val()
        buckle_angle_degrees = self.leaf_buckle_angle.val()
        buckle_length_fraction = self.leaf_buckle_length.val()
        if flexibility_value > f32(0) or buckle_angle_degrees <= f32(0):
            return flexibility_value
        aspect_ratio = max(self.leaf_aspect_ratio.val(), f32(1e-3))
        hinge = f32(f32(f32(1) - clampf(buckle_length_fraction, 0, 1)) * sinf(deg2rad(buckle_angle_degrees)))
        return max(f32(f32(f32(8) * hinge) / aspect_ratio), f32(0))

    @ports("LeafPrototype::validateTextureSeries")
    def validateTextureSeries(self, shoot_type_label: str):
        series, ages = self.leaf_texture_series, self.leaf_texture_stage_ages
        if not series:
            if ages:
                raise PlantArchitectureError(
                    f"ERROR (LeafPrototype::validateTextureSeries): The leaf prototype of shoot type '{shoot_type_label}' "
                    "sets leaf_texture_stage_ages but not leaf_texture_series, so there are no images for the ages to "
                    "apply to. Set leaf_texture_series.")
            return
        prefix = (f"ERROR (LeafPrototype::validateTextureSeries): The leaf prototype of shoot type "
                  f"'{shoot_type_label}' sets leaf_texture_series, ")
        first = series[min(series)]
        stage_count = 0 if not first else len(first[0])
        if self.prototype_function is not hooks.GenericLeafPrototype or self.OBJ_model_file:
            raise PlantArchitectureError(prefix + "which is only available with the default prototype function and "
                                         "without an OBJ model.")
        if self.unique_prototypes == 0:
            raise PlantArchitectureError(prefix + "but unique_prototypes is 0.")
        for idx in sorted(series):
            variants = series[idx]
            if not variants:
                raise PlantArchitectureError(prefix + f"but leaflet {idx} has no variants.")
            if len(variants) > self.unique_prototypes:
                raise PlantArchitectureError(prefix + f"but leaflet {idx} has {len(variants)} variants and "
                                             f"unique_prototypes is only {self.unique_prototypes}.")
            for v in variants:
                if not v:
                    raise PlantArchitectureError(prefix + f"but a variant of leaflet {idx} has no images.")
                if len(v) != stage_count:
                    raise PlantArchitectureError(prefix + f"but a variant of leaflet {idx} has {len(v)} images and "
                                                 f"another has {stage_count}.")
        if stage_count > 1 and len(ages) != stage_count:
            raise PlantArchitectureError(prefix + f"with {stage_count} images per variant, but "
                                         f"leaf_texture_stage_ages has {len(ages)} values.")
        if stage_count == 1 and len(ages) > 1:
            raise PlantArchitectureError(prefix + "with one image per variant, but leaf_texture_stage_ages has "
                                         f"{len(ages)} values.")
        if ages and f32(ages[0]) != f32(0):
            raise PlantArchitectureError(prefix + "but the first value of leaf_texture_stage_ages is not 0.")
        for s in range(1, len(ages)):
            if f32(ages[s]) <= f32(ages[s - 1]):
                raise PlantArchitectureError(prefix + f"but leaf_texture_stage_ages is not strictly increasing at "
                                             f"index {s}.")

    @ports("LeafPrototype::getTextureSeriesVariant")
    def getTextureSeriesVariant(self, compound_leaf_index: int, prototype_index: int):
        series = self.leaf_texture_series
        if len(series) > 1:
            if compound_leaf_index not in series:
                raise PlantArchitectureError("ERROR (LeafPrototype::getTextureSeriesVariant): Leaf prototype texture "
                                             f"series for compound leaf index {compound_leaf_index} was not found.")
            variants = series[compound_leaf_index]
        else:
            variants = series[min(series)]
        return variants[prototype_index % len(variants)]

    @ports("LeafPrototype::getTextureStageAtAge")
    def getTextureStageAtAge(self, age) -> int:
        stage = 0
        for c in range(1, len(self.leaf_texture_stage_ages)):
            if f32(age) >= f32(self.leaf_texture_stage_ages[c]):
                stage = c
        return stage


# ---- PhytomerParameters --------------------------------------------------------------------------

class InternodeParameters(_Struct):
    _FIELDS = ("pitch", "phyllotactic_angle", "radius_initial", "max_vegetative_buds_per_petiole",
               "max_floral_buds_per_petiole", "color", "image_texture", "image_class_map_file", "length_segments",
               "radial_subdivisions")
    _TYPES = {"color": "rgb", "length_segments": "uint", "radial_subdivisions": "uint"}

    def __init__(self):
        r = self._raw_set
        r("pitch", RandomParameterFloat())
        r("phyllotactic_angle", RandomParameterFloat())
        r("radius_initial", RandomParameterFloat())
        r("max_vegetative_buds_per_petiole", RandomParameterInt())
        r("max_floral_buds_per_petiole", RandomParameterInt())
        r("color", RGBcolor())
        r("image_texture", "")
        r("image_class_map_file", "")
        r("length_segments", 0)     # indeterminate in C++; always set by PhytomerParameters()
        r("radial_subdivisions", 0)


class PetioleParameters(_Struct):
    _FIELDS = ("petioles_per_internode", "pitch", "radius", "length", "curvature", "taper", "color",
               "length_segments", "radial_subdivisions", "flexibility", "flexibility_aging")
    _TYPES = {"petioles_per_internode": "uint", "color": "rgb", "length_segments": "uint",
              "radial_subdivisions": "uint"}

    def __init__(self):
        r = self._raw_set
        r("petioles_per_internode", 0)
        for f in ("pitch", "radius", "length", "curvature", "taper"):
            r(f, RandomParameterFloat())
        r("color", RGBcolor())
        r("length_segments", 0)
        r("radial_subdivisions", 0)
        r("flexibility", RandomParameterFloat())
        r("flexibility_aging", RandomParameterFloat())


class LeafParameters(_Struct):
    _FIELDS = ("leaves_per_petiole", "pitch", "yaw", "roll", "leaflet_offset", "leaflet_scale",
               "intercalary_leaflet_scale", "prototype_scale", "prototype")

    def __init__(self):
        r = self._raw_set
        r("leaves_per_petiole", RandomParameterInt())
        for f in self._FIELDS[1:-1]:
            r(f, RandomParameterFloat())
        r("prototype", LeafPrototype())

    def _assign_from(self, a):
        """Custom operator= (H:1199-1213): the prototype is copied with duplicate(), so no identifier draw."""
        if a is not self:
            self._memberwise_assign(a, self._FIELDS[:-1])
            self.prototype.duplicate(a.prototype)
        return self


class PeduncleParameters(_Struct):
    _FIELDS = ("length", "radius", "pitch", "roll", "yaw", "curvature", "color", "length_segments",
               "radial_subdivisions")
    _TYPES = {"color": "rgb", "length_segments": "uint", "radial_subdivisions": "uint"}

    def __init__(self):
        r = self._raw_set
        for f in self._FIELDS[:6]:
            r(f, RandomParameterFloat())
        r("color", RGBcolor())       # never set by PhytomerParameters(): RGBcolor() = (0,0,0)
        r("length_segments", 0)
        r("radial_subdivisions", 0)


class InflorescenceParameters(_Struct):
    _FIELDS = ("flowers_per_peduncle", "flower_offset", "pitch", "roll", "flower_prototype_scale",
               "flower_prototype_function", "fruit_prototype_scale", "fruit_prototype_function", "fruit_class_map_file",
               "fruit_gravity_factor_fraction", "unique_prototypes", "inflorescence_maturity_period")
    _TYPES = {"unique_prototypes": "uint"}

    def __init__(self):
        r = self._raw_set
        r("flowers_per_peduncle", RandomParameterInt())
        for f in ("flower_offset", "pitch", "roll", "flower_prototype_scale"):
            r(f, RandomParameterFloat())
        r("flower_prototype_function", None)
        r("fruit_prototype_scale", RandomParameterFloat())
        r("fruit_prototype_function", None)
        r("fruit_class_map_file", "")
        r("fruit_gravity_factor_fraction", RandomParameterFloat())
        r("unique_prototypes", 0)
        r("inflorescence_maturity_period", RandomParameterFloat())


def _resample_if_distributed(p):
    if p.distribution != "constant":
        p.resample()


class PhytomerParameters(_Struct):
    """PhytomerParameters (H:1043-1375, PA:947-1078). No user-declared copy operations: the implicit ones
    call each sub-struct's (custom) operator= / implicit copy constructor."""

    _FIELDS = ("internode", "petiole", "leaf", "peduncle", "inflorescence", "phytomer_creation_function",
               "phytomer_callback_function")

    @ports("PhytomerParameters::PhytomerParameters")
    def __init__(self, generator=None):
        r = self._raw_set
        r("internode", InternodeParameters())
        r("petiole", PetioleParameters())
        r("leaf", LeafParameters())
        r("peduncle", PeduncleParameters())
        r("inflorescence", InflorescenceParameters())
        r("phytomer_creation_function", None)
        r("phytomer_callback_function", None)
        g = generator
        it, pe, lf, pd, fl = self.internode, self.petiole, self.leaf, self.peduncle, self.inflorescence
        # --- internode ---
        it.pitch.initialize(20, g)
        it.phyllotactic_angle.initialize(137.5, g)
        it.radius_initial.initialize(0.001, g)
        it.color = RGB.forestgreen
        it.length_segments = 1
        it.radial_subdivisions = 7
        # --- petiole ---
        pe.petioles_per_internode = 1
        pe.pitch.initialize(90, g)
        pe.radius.initialize(0.001, g)
        pe.length.initialize(0.05, g)
        pe.curvature.initialize(0, g)
        pe.taper.initialize(0, g)
        pe.color = RGB.forestgreen
        pe.length_segments = 1
        pe.radial_subdivisions = 7
        pe.flexibility.initialize(0, g)
        pe.flexibility_aging.initialize(0, g)
        # --- leaf ---
        lf.leaves_per_petiole.initialize(1, g)
        lf.pitch.initialize(0, g)
        lf.yaw.initialize(0, g)
        lf.roll.initialize(0, g)
        lf.leaflet_offset.initialize(0, g)
        lf.leaflet_scale = 1                    # operator=(float): generator stays null
        lf.intercalary_leaflet_scale = 0        # idem
        lf.prototype_scale.initialize(0.05, g)
        lf.prototype = LeafPrototype(g)         # PA:983: ctor draws one identifier, operator= another
        # --- peduncle ---
        pd.length.initialize(0.05, g)
        pd.radius.initialize(0.001, g)
        pd.pitch.initialize(0, g)
        pd.roll.initialize(0, g)
        pd.yaw.initialize(0, g)
        pd.curvature.initialize(0, g)
        pe.color = RGB.forestgreen              # PA:996 sets petiole.color again; peduncle.color stays black
        pd.length_segments = 3
        pd.radial_subdivisions = 7
        # --- inflorescence ---
        fl.flowers_per_peduncle.initialize(1, g)
        fl.flower_offset.initialize(0, g)
        fl.pitch.initialize(0, g)
        fl.roll.initialize(0, g)
        fl.flower_prototype_scale.initialize(0.0075, g)
        fl.fruit_prototype_scale.initialize(0.0075, g)
        fl.fruit_gravity_factor_fraction.initialize(0, g)
        fl.inflorescence_maturity_period.initialize(-1, g)
        fl.unique_prototypes = 1

    # Fixed order of PhytomerParameters::resample (PA:1009-1078).
    RESAMPLE_ORDER = (
        "internode.pitch", "internode.phyllotactic_angle", "internode.radius_initial",
        "internode.max_vegetative_buds_per_petiole", "internode.max_floral_buds_per_petiole",
        "petiole.pitch", "petiole.radius", "petiole.length", "petiole.curvature", "petiole.taper",
        "petiole.flexibility", "petiole.flexibility_aging",
        "leaf.leaves_per_petiole", "leaf.pitch", "leaf.yaw", "leaf.roll", "leaf.leaflet_offset", "leaf.leaflet_scale",
        "leaf.intercalary_leaflet_scale", "leaf.prototype_scale",
        "leaf.prototype.leaf_aspect_ratio", "leaf.prototype.midrib_fold_fraction",
        "leaf.prototype.longitudinal_curvature_exponent", "leaf.prototype.longitudinal_curvature",
        "leaf.prototype.lateral_curvature", "leaf.prototype.roll_fraction", "leaf.prototype.petiole_roll",
        "leaf.prototype.wave_period", "leaf.prototype.wave_amplitude", "leaf.prototype.flexibility",
        "leaf.prototype.flexibility_taper", "leaf.prototype.flexibility_aging", "leaf.prototype.flexibility_aging_max",
        "leaf.prototype.leaf_buckle_length", "leaf.prototype.leaf_buckle_angle",
        "peduncle.length", "peduncle.radius", "peduncle.pitch", "peduncle.roll", "peduncle.yaw", "peduncle.curvature",
        "inflorescence.flowers_per_peduncle", "inflorescence.flower_offset", "inflorescence.pitch",
        "inflorescence.roll", "inflorescence.flower_prototype_scale", "inflorescence.inflorescence_maturity_period",
        "inflorescence.fruit_prototype_scale", "inflorescence.fruit_gravity_factor_fraction",
    )

    def get(self, path: str):
        o = self
        for part in path.split("."):
            o = getattr(o, part)
        return o

    @ports("PhytomerParameters::resample")
    def resample(self):
        """Resample every distributed parameter in the fixed C++ order; constants draw nothing."""
        for path in self.RESAMPLE_ORDER:
            _resample_if_distributed(self.get(path))


# ---- ShootParameters -----------------------------------------------------------------------------

_SHOOT_RP = ("max_nodes", "max_nodes_per_season", "girth_area_factor", "insertion_angle_tip",
             "insertion_angle_decay_rate", "internode_length_max", "internode_length_min",
             "internode_length_decay_rate", "base_roll", "base_yaw", "gravitropic_curvature", "tortuosity",
             "tortuosity_persistence_length", "phyllochron_min", "elongation_rate_max", "leaf_expansion_rate_max",
             "vegetative_bud_break_probability_min", "vegetative_bud_break_probability_max",
             "vegetative_bud_break_probability_decay_rate", "max_terminal_floral_buds", "flower_bud_break_probability",
             "fruit_set_probability", "vegetative_bud_break_time")
_SHOOT_INT = ("max_nodes", "max_nodes_per_season", "max_terminal_floral_buds")


class ShootParameters(_Struct):
    """ShootParameters (H:1377-1646, PA:1081-1150)."""

    leaf_expansion_rate_unset = f32(-1.0)

    _FIELDS = ("phytomer_parameters",) + _SHOOT_RP + ("flowers_require_dormancy", "growth_requires_dormancy",
                                                      "determinate_shoot_growth", "child_shoot_type_labels",
                                                      "child_shoot_type_probabilities")
    _TYPES = {"flowers_require_dormancy": "bool", "growth_requires_dormancy": "bool", "determinate_shoot_growth": "bool"}

    @ports("ShootParameters::ShootParameters")
    def __init__(self, generator=None):
        r = self._raw_set
        r("phytomer_parameters", PhytomerParameters())     # default member: PhytomerParameters(nullptr), no draw
        for f in _SHOOT_RP:
            r(f, RandomParameterInt() if f in _SHOOT_INT else RandomParameterFloat())
        r("flowers_require_dormancy", False)
        r("growth_requires_dormancy", False)
        r("determinate_shoot_growth", False)
        r("child_shoot_type_labels", [])
        r("child_shoot_type_probabilities", [])
        g = generator
        self.max_nodes.initialize(10, g)
        self.max_nodes_per_season.initialize(9999, g)
        self.insertion_angle_tip.initialize(20, g)
        self.insertion_angle_decay_rate.initialize(0, g)
        self.internode_length_max.initialize(0.02, g)
        self.internode_length_min.initialize(0.002, g)
        self.internode_length_decay_rate.initialize(0, g)
        self.base_roll.initialize(0, g)
        self.base_yaw.initialize(0, g)
        self.gravitropic_curvature.initialize(0, g)
        self.tortuosity.initialize(0, g)
        self.tortuosity_persistence_length.initialize(0.5, g)
        self.phyllochron_min.initialize(2, g)
        self.elongation_rate_max.initialize(0.2, g)
        self.leaf_expansion_rate_max.initialize(ShootParameters.leaf_expansion_rate_unset, g)
        self.girth_area_factor.initialize(0, g)
        self.vegetative_bud_break_time.initialize(5, g)
        self.vegetative_bud_break_probability_min.initialize(0, g)
        self.vegetative_bud_break_probability_max.initialize(1.0, g)
        self.vegetative_bud_break_probability_decay_rate.initialize(-0.5, g)
        self.max_terminal_floral_buds.initialize(0, g)
        self.flower_bud_break_probability.initialize(0, g)
        self.fruit_set_probability.initialize(0, g)
        self.flowers_require_dormancy = False
        self.growth_requires_dormancy = False
        self.determinate_shoot_growth = True

    @ports("ShootParameters::operator=")
    def _assign_from(self, a):
        """Custom operator= (H:1577-1615). It copies every member (some twice) and resamples nothing,
        despite the comment above it in the header (notes/W1-D.md). No self-assignment guard in C++; the
        result is the same."""
        if not isinstance(a, ShootParameters):
            raise TypeError(f"cannot assign {type(a).__name__} to ShootParameters")
        if a is not self:
            self._memberwise_assign(a)
        return self

    @ports("ShootParameters::defineChildShootTypes")
    def defineChildShootTypes(self, labels, probabilities):
        if len(labels) != len(probabilities):
            raise PlantArchitectureError("ERROR (ShootParameters::defineChildShootTypes): Child shoot type labels and "
                                         "probabilities must be the same size.")
        if len(labels) == 0:
            raise PlantArchitectureError("ERROR (ShootParameters::defineChildShootTypes): Input argument vectors were "
                                         "empty.")
        probs = [f32(p) for p in probabilities]
        if sum_f(probs) != f32(1):
            raise PlantArchitectureError("ERROR (ShootParameters::defineChildShootTypes): Child shoot type "
                                         "probabilities must sum to 1.")
        self._raw_set("child_shoot_type_labels", [str(s) for s in labels])
        self._raw_set("child_shoot_type_probabilities", probs)

    def getChildShootTypeLabels(self):
        return list(self.child_shoot_type_labels)

    def getChildShootTypeProbabilities(self):
        return list(self.child_shoot_type_probabilities)

    @ports("ShootParameters::inheritCustomFunctionsFrom")
    def inheritCustomFunctionsFrom(self, source: "ShootParameters"):
        if source is self:
            return
        pp, sp = self.phytomer_parameters, source.phytomer_parameters
        pp.phytomer_creation_function = sp.phytomer_creation_function
        pp.phytomer_callback_function = sp.phytomer_callback_function
        pp.leaf.prototype.prototype_function = sp.leaf.prototype.prototype_function
        pp.inflorescence.flower_prototype_function = sp.inflorescence.flower_prototype_function
        pp.inflorescence.fruit_prototype_function = sp.inflorescence.fruit_prototype_function


__all__ = ["AxisRotation", "make_AxisRotation", "BudState", "VegetativeBud", "FloralBud", "LeafShapeValues",
           "LeafPrototype", "InternodeParameters", "PetioleParameters", "LeafParameters", "PeduncleParameters",
           "InflorescenceParameters", "PhytomerParameters", "ShootParameters", "RGBcolor", "make_RGBcolor", "RGB",
           "deg2rad", "rad2deg", "c_uint", "sum_f", "vec3", "UINT_MAX"]
