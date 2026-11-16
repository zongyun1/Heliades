# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/include/PlantArchitecture.h:57-306
# (Copyright (C) 2016-2026 Brian Bailey).
"""``RandomParameter_float`` / ``RandomParameter_int``: architecture parameters with a lazy sample cache.

C++ semantics reproduced literally (SPEC §3.2, §4.1):

* ``val()`` draws only when ``sampled`` is false; ``resample()`` always marks the parameter sampled and
  draws only if the distribution is not ``"constant"``. A constant parameter never touches the engine.
* ``generator`` is a *pointer*: here a reference to a shared :class:`~heliades.core.random.MinstdRand0`
  (the Scene's, or a private one). Reseeding the Scene mutates that object in place, as in Helios.
* Copy construction and copy assignment copy everything, including ``sampled``, ``constval`` and the
  generator pointer (the C++ copy operations are the implicit ones). ``set(x)`` is ``operator=(float)``:
  it makes the parameter constant, clears ``sampled`` and leaves the generator alone.
* Every distribution object is built fresh per draw (``helios::uniform_real_distribution<float>`` etc.),
  so a ``normal`` parameter never keeps a polar spare between draws.

Python has no value semantics, so the containing structs route attribute assignment through
``_assign_from`` (see ``params._Struct``): ``pp.internode.pitch = 10`` behaves like the C++ statement.
"""
from __future__ import annotations

import numpy as np

from ..core.random import Normal, UniformInt, UniformReal, Weibull, unit_uniform
from ..migration_registry import ports

f32 = np.float32

FLOAT_DISTRIBUTIONS = ("constant", "uniform", "normal", "weibull")
INT_DISTRIBUTIONS = ("constant", "uniform", "discretevalues")


def _c_int(v) -> int:
    """C++ implicit conversion to ``int`` (truncation toward zero), wrapped to 32 bits."""
    i = int(v)
    return ((i + 2**31) % 2**32) - 2**31


def _initialize_args(args):
    """The two C++ ``initialize`` overloads: ``(value, generator*)`` and ``(generator*)``."""
    if len(args) == 2:
        return args[0], args[1]
    if len(args) == 1:
        return None, args[0]
    raise TypeError("initialize(value, generator) or initialize(generator)")


class PlantArchitectureError(RuntimeError):
    """helios_runtime_error / std::runtime_error raised by the plugin."""


class RandomParameterFloat:
    """RandomParameter_float (H:57-205)."""

    __slots__ = ("distribution", "distribution_parameters", "sampled", "constval", "generator")

    @ports("RandomParameter_float::RandomParameter_float")
    def __init__(self, value=None, generator=None):
        """``RandomParameterFloat()`` / ``(float val)`` / ``(std::minstd_rand0*)``.

        The C++ has three explicit constructors: ``()`` -> 0, ``(float)`` -> that constant, and
        ``(generator*)`` -> 0 with the generator. ``value`` and ``generator`` may not both be given.
        """
        if value is not None and generator is not None:
            raise TypeError("RandomParameter_float has no (value, generator) constructor; use initialize()")
        self.constval = f32(0.0) if value is None else f32(value)
        self.distribution = "constant"
        self.distribution_parameters: list = []
        self.generator = generator
        self.sampled = False

    # ---- initialisation ----

    @ports("RandomParameter_float::initialize")
    def initialize(self, *args):
        """``initialize(float a_val, gen*)`` or ``initialize(gen*)`` (the latter sets constval = 1)."""
        a_val, rand_generator = _initialize_args(args)
        self.constval = f32(1.0) if a_val is None else f32(a_val)
        self.distribution = "constant"
        self.generator = rand_generator
        self.sampled = False

    @ports("RandomParameter_float::setRandomGenerator")
    def setRandomGenerator(self, rand_generator):
        self.generator = rand_generator
        self.sampled = False

    @ports("RandomParameter_float::operator=")
    def set(self, a):
        """``operator=(float)``: constant, unsampled, generator unchanged."""
        self.distribution = "constant"
        self.constval = f32(a)
        self.sampled = False
        return self

    @ports("RandomParameter_float::uniformDistribution")
    def uniformDistribution(self, minval, maxval):
        minval, maxval = f32(minval), f32(maxval)
        if minval > maxval:
            raise PlantArchitectureError("ERROR (PlantArchitecture): RandomParameter_float::uniformDistribution() - "
                                         "minval must be less than or equal to maxval.")
        self.distribution = "uniform"
        self.distribution_parameters = [minval, maxval]
        self.sampled = False

    @ports("RandomParameter_float::normalDistribution")
    def normalDistribution(self, mean, std_dev):
        self.distribution = "normal"
        self.distribution_parameters = [f32(mean), f32(std_dev)]
        self.sampled = False

    @ports("RandomParameter_float::weibullDistribution")
    def weibullDistribution(self, shape, scale):
        self.distribution = "weibull"
        self.distribution_parameters = [f32(shape), f32(scale)]
        self.sampled = False

    # ---- sampling ----

    @ports("RandomParameter_float::val")
    def val(self) -> np.float32:
        if not self.sampled:
            self.constval = self.resample()
        return self.constval

    @ports("RandomParameter_float::resample")
    def resample(self) -> np.float32:
        self.sampled = True
        if self.distribution != "constant":
            if self.generator is None:
                raise PlantArchitectureError("ERROR (PlantArchitecture): Random parameter was not properly initialized "
                                             "with random number generator.")
            p = self.distribution_parameters
            if self.distribution == "uniform":
                # p0 + unif(*gen) * (p1 - p0), all float (H:180)
                u = unit_uniform(self.generator, "float")
                self.constval = f32(p[0] + f32(u * f32(p[1] - p[0])))
            elif self.distribution == "normal":
                self.constval = f32(Normal(p[0], p[1], "float")(self.generator))
            elif self.distribution == "weibull":
                self.constval = f32(Weibull(p[0], p[1], "float")(self.generator))
            # any other string: no draw, constval unchanged (as in C++)
        return self.constval

    # ---- value semantics ----

    @ports("RandomParameter_float::RandomParameter_float(const RandomParameter_float&)")
    def copy(self) -> "RandomParameterFloat":
        """Implicit copy constructor: every member, including the cache and generator pointer."""
        c = RandomParameterFloat.__new__(RandomParameterFloat)
        c.distribution = self.distribution
        c.distribution_parameters = list(self.distribution_parameters)
        c.sampled = self.sampled
        c.constval = self.constval
        c.generator = self.generator
        return c

    def _assign_from(self, a):
        """C++ assignment: implicit copy assignment from a parameter, ``operator=(float)`` from a number."""
        if isinstance(a, RandomParameterFloat):
            if a is not self:
                self.distribution = a.distribution
                self.distribution_parameters = list(a.distribution_parameters)
                self.sampled = a.sampled
                self.constval = a.constval
                self.generator = a.generator
        elif isinstance(a, RandomParameterInt):
            raise TypeError("cannot assign RandomParameter_int to RandomParameter_float")
        else:
            self.set(a)
        return self

    # helpers (not in C++)
    def is_distributed(self) -> bool:
        return self.distribution != "constant"

    def __repr__(self):
        d = self.distribution
        return (f"RandomParameterFloat({d}, {[float(x) for x in self.distribution_parameters]}, "
                f"constval={float(self.constval)!r}, sampled={self.sampled})")


class RandomParameterInt:
    """RandomParameter_int (H:207-306)."""

    __slots__ = ("distribution", "distribution_parameters", "sampled", "constval", "generator")

    @ports("RandomParameter_int::RandomParameter_int")
    def __init__(self):
        self.constval = 1
        self.distribution = "constant"
        self.distribution_parameters: list = []
        self.generator = None
        self.sampled = False

    @ports("RandomParameter_int::initialize")
    def initialize(self, *args):
        """``initialize(int a_val, gen*)`` or ``initialize(gen*)`` (the latter sets constval = 1)."""
        a_val, rand_generator = _initialize_args(args)
        self.constval = 1 if a_val is None else _c_int(a_val)
        self.distribution = "constant"
        self.generator = rand_generator
        self.sampled = False

    @ports("RandomParameter_int::operator=")
    def set(self, a):
        self.distribution = "constant"
        self.constval = _c_int(a)
        self.sampled = False
        return self

    @ports("RandomParameter_int::uniformDistribution")
    def uniformDistribution(self, minval, maxval):
        minval, maxval = _c_int(minval), _c_int(maxval)
        if minval > maxval:
            raise PlantArchitectureError("ERROR (PlantArchitecture): RandomParameter_int::uniformDistribution() - "
                                         "minval must be less than or equal to maxval.")
        self.distribution = "uniform"
        self.distribution_parameters = [minval, maxval]
        self.sampled = False

    @ports("RandomParameter_int::discreteValues")
    def discreteValues(self, values):
        self.distribution = "discretevalues"
        self.distribution_parameters = [_c_int(v) for v in values]
        self.sampled = False

    @ports("RandomParameter_int::val")
    def val(self) -> int:
        if not self.sampled:
            self.constval = self.resample()
        return self.constval

    @ports("RandomParameter_int::resample")
    def resample(self) -> int:
        self.sampled = True
        if self.distribution != "constant":
            if self.generator is None:
                raise PlantArchitectureError("ERROR (PlantArchitecture): Random parameter was not properly initialized "
                                             "with random number generator.")
            p = self.distribution_parameters
            if self.distribution == "uniform":
                # helios::uniform_int_distribution<>(a, b): one draw even when a == b (no Context shortcut)
                self.constval = int(UniformInt(p[0], p[1])(self.generator))
            elif self.distribution == "discretevalues":
                # uniform_int_distribution<>(0, size()-1); size()-1 is size_t, narrowed to int. An empty list
                # gives -1 -> range 2^64-1 in the C++ (then .at() throws); we raise directly.
                if not p:
                    raise PlantArchitectureError("RandomParameter_int: discretevalues with no values")
                self.constval = p[int(UniformInt(0, len(p) - 1)(self.generator))]
        return self.constval

    def copy(self) -> "RandomParameterInt":
        c = RandomParameterInt.__new__(RandomParameterInt)
        c.distribution = self.distribution
        c.distribution_parameters = list(self.distribution_parameters)
        c.sampled = self.sampled
        c.constval = self.constval
        c.generator = self.generator
        return c

    def _assign_from(self, a):
        if isinstance(a, RandomParameterInt):
            if a is not self:
                self.distribution = a.distribution
                self.distribution_parameters = list(a.distribution_parameters)
                self.sampled = a.sampled
                self.constval = a.constval
                self.generator = a.generator
        elif isinstance(a, RandomParameterFloat):
            raise TypeError("cannot assign RandomParameter_float to RandomParameter_int")
        else:
            self.set(a)
        return self

    def is_distributed(self) -> bool:
        return self.distribution != "constant"

    def __repr__(self):
        return (f"RandomParameterInt({self.distribution}, {self.distribution_parameters}, "
                f"constval={self.constval}, sampled={self.sampled})")


# C++ spellings
RandomParameter_float = RandomParameterFloat
RandomParameter_int = RandomParameterInt

__all__ = ["RandomParameterFloat", "RandomParameterInt", "RandomParameter_float", "RandomParameter_int",
           "PlantArchitectureError", "UniformReal"]
