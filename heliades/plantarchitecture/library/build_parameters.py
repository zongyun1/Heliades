# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:182-397, 538-629, 649-700
# (Copyright (C) 2016-2026 Brian Bailey).
"""Library build parameters (W4-D).

* ``BuildParameterDeclaration`` (H) and the ``getBuildParameterValue`` readers (PL:182-226);
* ``BuildParameterDistribution`` (PL:230-395): ``constant``, ``uniform``, ``normal``, ``discrete`` and
  ``sample(generator)``;
* ``registerPlantModelBuildParameters``, ``getPlantModelBuildParameterNames``, ``validateBuildParameters``
  (PL:538-629);
* :func:`prepare_build_parameters`: the part of ``buildPlantInstanceFromLibrary`` (PL:667-696) that
  validates the map and samples each structure parameter once from the Context engine. A constant is taken
  without a draw, so a build with fixed values draws exactly what a build without parameters draws.

Only the declarations of the models ported to Heliades are registered (see ``register_build_parameters``);
for any other model the map is empty, exactly as in Helios for a model that declares none.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...core.random import Discrete, Normal, UniformInt, UniformReal
from ...migration_registry import ports
from ..random_parameter import PlantArchitectureError

f32 = np.float32


def _std_to_string(v) -> str:
    """``std::to_string(float)``: ``%f``."""
    return "%f" % float(f32(v))


@dataclass(frozen=True)
class BuildParameterDeclaration:
    """``BuildParameterDeclaration{name, min_value, max_value, is_integer, description}`` (constexpr)."""
    name: str
    min_value: np.float32
    max_value: np.float32
    is_integer: bool
    description: str

    def __post_init__(self):
        object.__setattr__(self, "min_value", f32(self.min_value))
        object.__setattr__(self, "max_value", f32(self.max_value))


@ports("getBuildParameterValue")
def getBuildParameterValueRaw(build_parameters: dict, parameter_name: str, default_value, min_value, max_value,
                              parameter_description: str) -> np.float32:
    """PL:182-201."""
    if parameter_name not in build_parameters:
        return f32(default_value)
    value = f32(build_parameters[parameter_name])
    if not (value >= f32(min_value) and value <= f32(max_value)):
        raise PlantArchitectureError(
            f"ERROR (getBuildParameterValue): build parameter '{parameter_name}' ({parameter_description}) has value "
            f"{_std_to_string(value)} which is outside the valid range [{_std_to_string(min_value)}, "
            f"{_std_to_string(max_value)}].")
    return value


def getBuildParameterValue(build_parameters: dict, declaration: BuildParameterDeclaration, default_value):
    """PL:211-216 (the declaration overload)."""
    if declaration.is_integer:
        raise PlantArchitectureError(f"ERROR (getBuildParameterValue): build parameter '{declaration.name}' is declared as "
                                     "a whole number. Read it with getBuildParameterIntegerValue().")
    return getBuildParameterValueRaw(build_parameters, declaration.name, default_value, declaration.min_value,
                                     declaration.max_value, declaration.description)


def getBuildParameterIntegerValue(build_parameters: dict, declaration: BuildParameterDeclaration, default_value) -> int:
    """PL:203-209, 218-223."""
    if not declaration.is_integer:
        raise PlantArchitectureError(f"ERROR (getBuildParameterIntegerValue): build parameter '{declaration.name}' is not "
                                     "declared as a whole number. Read it with getBuildParameterValue().")
    value = getBuildParameterValueRaw(build_parameters, declaration.name, f32(int(default_value)),
                                      f32(int(declaration.min_value)), f32(int(declaration.max_value)),
                                      declaration.description)
    if value != f32(round(float(value))):
        raise PlantArchitectureError(f"ERROR (getBuildParameterIntegerValue): build parameter '{declaration.name}' "
                                     f"({declaration.description}) has value {_std_to_string(value)}, but it must be a "
                                     "whole number.")
    return int(value)


class BuildParameterDistribution:
    """PL:230-395. ``parameters`` and ``weights`` are float32 lists."""

    CONSTANT, UNIFORM, NORMAL, DISCRETE = "constant", "uniform", "normal", "discrete"

    def __init__(self, type_, parameters, weights=()):
        self.type = type_
        self.parameters = [f32(p) for p in parameters]
        self.weights = [f32(w) for w in weights]

    def getType(self):
        return self.type

    def getParameters(self):
        return list(self.parameters)

    @classmethod
    def constant(cls, value):
        return cls(cls.CONSTANT, [value])

    @classmethod
    def uniform(cls, min_value, max_value):
        lo, hi = f32(min_value), f32(max_value)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise PlantArchitectureError("ERROR (BuildParameterDistribution::uniform): The interval bounds must be finite.")
        if lo > hi:
            raise PlantArchitectureError(f"ERROR (BuildParameterDistribution::uniform): min_value ({_std_to_string(lo)}) "
                                         f"must not be greater than max_value ({_std_to_string(hi)}).")
        return cls(cls.UNIFORM, [lo, hi])

    @classmethod
    def normal(cls, mean, standard_deviation, min_value, max_value):
        m, sd, lo, hi = f32(mean), f32(standard_deviation), f32(min_value), f32(max_value)
        if not all(math.isfinite(x) for x in (m, sd, lo, hi)):
            raise PlantArchitectureError("ERROR (BuildParameterDistribution::normal): The mean, standard deviation and "
                                         "interval bounds must be finite.")
        if sd <= 0:
            raise PlantArchitectureError("ERROR (BuildParameterDistribution::normal): The standard deviation must be "
                                         "positive. Use BuildParameterDistribution::constant() for a fixed value.")
        if lo >= hi:
            raise PlantArchitectureError(f"ERROR (BuildParameterDistribution::normal): min_value ({_std_to_string(lo)}) "
                                         f"must be less than max_value ({_std_to_string(hi)}).")
        erf_scale = float(sd) * math.sqrt(2.0)
        p_in = 0.5 * (math.erf((float(hi) - float(m)) / erf_scale) - math.erf((float(lo) - float(m)) / erf_scale))
        if p_in < 1e-3:
            raise PlantArchitectureError("ERROR (BuildParameterDistribution::normal): The interval holds less than 0.1% of "
                                         "the distribution.")
        return cls(cls.NORMAL, [m, sd, lo, hi])

    @classmethod
    def discrete(cls, values, weights=()):
        if len(values) == 0:
            raise PlantArchitectureError("ERROR (BuildParameterDistribution::discrete): At least one value must be given.")
        if not all(math.isfinite(f32(v)) for v in values):
            raise PlantArchitectureError("ERROR (BuildParameterDistribution::discrete): Every value must be finite.")
        if weights:
            if len(weights) != len(values):
                raise PlantArchitectureError(f"ERROR (BuildParameterDistribution::discrete): {len(weights)} weights were "
                                             f"given for {len(values)} values.")
            s = f32(0)
            for w in weights:
                w = f32(w)
                if not math.isfinite(w) or w < 0:
                    raise PlantArchitectureError("ERROR (BuildParameterDistribution::discrete): Every weight must be finite "
                                                 "and non-negative.")
                s = f32(s + w)
            if s <= 0:
                raise PlantArchitectureError("ERROR (BuildParameterDistribution::discrete): The weights must not all be "
                                             "zero.")
        return cls(cls.DISCRETE, values, weights)

    def getMinimum(self):
        if self.type == self.NORMAL:
            return self.parameters[2]
        if self.type == self.DISCRETE:
            return min(self.parameters)
        return self.parameters[0]

    def getMaximum(self):
        if self.type == self.CONSTANT:
            return self.parameters[0]
        if self.type == self.NORMAL:
            return self.parameters[3]
        if self.type == self.DISCRETE:
            return max(self.parameters)
        return self.parameters[1]

    @ports("BuildParameterDistribution::sample")
    def sample(self, generator) -> np.float32:
        """PL:359-395. Not oracle-checked: PyHelios 1.3.98 passes build parameters as fixed values only."""
        p = self.parameters
        if self.type == self.CONSTANT:
            return p[0]
        if self.type == self.UNIFORM:
            u = UniformReal(0.0, 1.0, "float")(generator)
            return f32(p[0] + f32(u * f32(p[1] - p[0])))
        if self.type == self.NORMAL:
            dist = Normal(p[0], p[1], "float")
            for _ in range(100000):
                value = dist(generator)
                if p[2] <= value <= p[3]:
                    return value
            raise PlantArchitectureError("ERROR (BuildParameterDistribution::sample): No value of the truncated normal "
                                         "distribution fell inside its interval in 100000 draws.")
        if len(p) == 1:
            return p[0]
        if not self.weights:
            return p[UniformInt(0, len(p) - 1)(generator)]
        return p[Discrete(self.weights)(generator)]


@dataclass
class PlantModelBuildParameters:
    structure_parameters: list
    organ_parameters: list
    validate_organ_parameters: object


@ports("PlantArchitecture::registerPlantModelBuildParameters")
def registerPlantModelBuildParameters(pa, name, structure_parameters, organ_parameters=(), validate_organ_parameters=None):
    if name not in pa.plant_builders:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::registerPlantModelBuildParameters): Plant model '{name}' "
                                     "has not been registered. Call registerPlantModel() first.")
    if organ_parameters and validate_organ_parameters is None:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::registerPlantModelBuildParameters): Plant model '{name}' "
                                     "declares organ parameters but no function to validate them.")
    if not hasattr(pa, "plant_model_build_parameters"):
        pa.plant_model_build_parameters = {}
    pa.plant_model_build_parameters[name] = PlantModelBuildParameters(list(structure_parameters), list(organ_parameters),
                                                                      validate_organ_parameters)


@ports("PlantArchitecture::getPlantModelBuildParameterNames")
def getPlantModelBuildParameterNames(pa, plant_model_name) -> list:
    if plant_model_name not in pa.plant_builders:
        raise PlantArchitectureError("ERROR (PlantArchitecture::getPlantModelBuildParameterNames): plant label of "
                                     f"{plant_model_name} does not exist in the library.")
    decl = getattr(pa, "plant_model_build_parameters", {}).get(plant_model_name)
    if decl is None:
        return []
    return [d.name for d in decl.structure_parameters] + [d.name for d in decl.organ_parameters]


@ports("PlantArchitecture::validateBuildParameters")
def validateBuildParameters(pa, plant_model_name, build_parameters: dict, calling_function: str) -> dict:
    """PL:567-629. ``build_parameters``: name -> BuildParameterDistribution. Returns the organ parameters."""
    organ_parameters: dict = {}
    if not build_parameters:
        return organ_parameters
    decl = getattr(pa, "plant_model_build_parameters", {}).get(plant_model_name)
    structure = decl.structure_parameters if decl else []
    organ = decl.organ_parameters if decl else []
    accepted = ", ".join(getPlantModelBuildParameterNames(pa, plant_model_name))
    fixed_organ: dict = {}
    for name in sorted(build_parameters):                       # std::map order
        dist = build_parameters[name]
        declaration = next((d for d in organ if d.name == name), None)
        if declaration is not None:
            organ_parameters[name] = dist
            if dist.getType() == BuildParameterDistribution.CONSTANT:
                fixed_organ[name] = dist.getParameters()[0]
        else:
            declaration = next((d for d in structure if d.name == name), None)
            if declaration is None:
                raise PlantArchitectureError(
                    f"ERROR ({calling_function}): Build parameter '{name}' is not accepted by plant model "
                    f"'{plant_model_name}'. " + ("This model accepts no build parameters." if not accepted
                                                 else f"Accepted parameters are: {accepted}."))
        if dist.getType() == BuildParameterDistribution.CONSTANT:
            continue
        label = f"Build parameter '{name}' ({declaration.description})"
        if declaration.is_integer:
            if dist.getType() != BuildParameterDistribution.DISCRETE:
                raise PlantArchitectureError(f"ERROR ({calling_function}): {label} must be a whole number, so its "
                                             "distribution must be BuildParameterDistribution::discrete() with "
                                             "whole-number values.")
            for v in dist.getParameters():
                if v != f32(round(float(v))):
                    raise PlantArchitectureError(f"ERROR ({calling_function}): {label} must be a whole number, but its "
                                                 f"discrete distribution includes the value {_std_to_string(v)}.")
        if dist.getMinimum() < declaration.min_value or dist.getMaximum() > declaration.max_value:
            raise PlantArchitectureError(
                f"ERROR ({calling_function}): {label} is given a distribution spanning [{_std_to_string(dist.getMinimum())}, "
                f"{_std_to_string(dist.getMaximum())}], which reaches outside the valid range "
                f"[{_std_to_string(declaration.min_value)}, {_std_to_string(declaration.max_value)}].")
    if organ_parameters:
        decl.validate_organ_parameters(fixed_organ)
    return organ_parameters


def as_distributions(build_parameters) -> dict:
    """``constantBuildParameterDistributions`` for a plain {name: float} map; distributions pass through."""
    out = {}
    for k, v in (build_parameters or {}).items():
        out[str(k)] = v if isinstance(v, BuildParameterDistribution) else BuildParameterDistribution.constant(v)
    return out


@ports("PlantArchitecture::buildPlantInstanceFromLibrary (build parameters)")
def prepare_build_parameters(pa, plant_model_name, build_parameters) -> tuple[dict, dict]:
    """PL:667-679: validate, then sample each structure parameter once (in std::map order) from the Context
    engine; organ parameters given as fixed values are kept. Returns (plant_build_parameters,
    organ_build_parameters)."""
    dists = as_distributions(build_parameters)
    organ = validateBuildParameters(pa, plant_model_name, dists, "PlantArchitecture::buildPlantInstanceFromLibrary")
    plant: dict = {}
    for name in sorted(dists):
        dist = dists[name]
        if name not in organ:
            plant[name] = dist.sample(pa.generator)
        elif dist.getType() == BuildParameterDistribution.CONSTANT:
            plant[name] = dist.getParameters()[0]
    return plant, organ


@ports("PlantArchitecture::sampleOrganBuildParameters")
def sampleOrganBuildParameters(pa, organ_build_parameters: dict) -> dict:
    """PA:3775: one sample of each organ parameter (std::map order) for a flower / fruit prototype."""
    return {name: organ_build_parameters[name].sample(pa.generator) for name in sorted(organ_build_parameters)}


def register_build_parameters(pa):
    """The ``registerPlantModelBuildParameters`` calls of PL:511-535 for the models ported to Heliades."""
    from . import grapevine
    grapevine.register_build_parameters(pa)
    from . import cereals_registry           # W5-A: maize, rice, sorghum, wheat
    cereals_registry.register_build_parameters(pa)
