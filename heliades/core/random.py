# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core/include/portable_random.h and Context.cpp (Copyright (C) 2016-2026 Brian Bailey).
"""Bit-exact port of the Helios random number stack.

Helios uses ``std::minstd_rand0`` (Lehmer, a=16807, m=2^31-1) plus its own platform-independent
distributions. Reproducing both exactly means a Heliades model seeded like a Helios model draws the
same sequence, so differential tests can compare plants value by value instead of only statistically.

Float semantics: ``Real=float`` results are rounded to float32 exactly where the C++ code converts.
"""
from __future__ import annotations

import bisect
import math

import numpy as np

_f32 = np.float32


class MinstdRand0:
    """std::minstd_rand0: x <- 16807 x mod (2^31 - 1). min()=1, max()=2^31-2."""

    A, M = 16807, 2147483647
    MIN, MAX = 1, 2147483646

    def __init__(self, seed: int = 1):
        self.seed(seed)

    def seed(self, s: int = 1):
        s = int(s) % self.M
        self.x = 1 if s == 0 else s

    def __call__(self) -> int:
        self.x = (self.A * self.x) % self.M
        return self.x

    def copy(self) -> "MinstdRand0":
        g = MinstdRand0.__new__(MinstdRand0)
        g.x = self.x
        return g


# ---- random_detail --------------------------------------------------------------------------

def _span(g) -> int:
    return g.MAX - g.MIN


def _draw(g) -> int:
    return g() - g.MIN


def uniform_integer(g, rng: int) -> int:
    """Uniform integer in [0, rng] by rejection (random_detail::uniformInteger)."""
    span = _span(g)
    if span > rng:
        values = rng + 1
        group = span // values
        limit = values * group
        while True:
            v = _draw(g)
            if v < limit:
                return v // group
    if span < rng:
        per_draw = span + 1
        while True:
            high = per_draw * uniform_integer(g, rng // per_draw)
            v = high + _draw(g)
            if not (v > rng or v < high):
                return v
    return _draw(g)


def _bits_per_draw(g) -> int:
    n, values = 0, _span(g) + 1
    while values > 1:
        values >>= 1
        n += 1
    return n


def unit_uniform(g, real: str = "float"):
    """Uniform in [0,1) (random_detail::unitUniform). real in {'float','double'}."""
    bits_needed = 24 if real == "float" else 53
    bpd = _bits_per_draw(g)
    if bpd >= bits_needed:
        r = float(_draw(g)) / (float(_span(g)) + 1.0)
        if real == "float":
            r = _f32(r)
            if r >= _f32(1):
                r = np.nextafter(_f32(1), _f32(0))
        return r
    chunk_range = (1 << bpd) - 1
    bits, got = 0, 0
    while got < bits_needed:
        chunk, cb = uniform_integer(g, chunk_range), bpd
        if got + cb > bits_needed:
            surplus = got + cb - bits_needed
            chunk >>= surplus
            cb -= surplus
        bits = (bits << cb) | chunk
        got += cb
    r = math.ldexp(float(bits), -bits_needed)
    return _f32(r) if real == "float" else r


# ---- distributions --------------------------------------------------------------------------

class UniformReal:
    def __init__(self, a=0.0, b=1.0, real="double"):
        self.real = real
        cast = _f32 if real == "float" else float
        self.a, self.b = cast(a), cast(b)

    def __call__(self, g):
        u = unit_uniform(g, self.real)
        if self.real == "float":
            return _f32(self.a + _f32(_f32(self.b - self.a) * u))
        return self.a + (self.b - self.a) * u

    def reset(self):
        pass


class UniformInt:
    def __init__(self, a=0, b=2**31 - 1):
        self.a, self.b = int(a), int(b)

    def __call__(self, g):
        return self.a + uniform_integer(g, self.b - self.a)

    def reset(self):
        pass


class Normal:
    """Marsaglia polar method; keeps the spare value (normal_distribution)."""

    def __init__(self, mean=0.0, stddev=1.0, real="double"):
        self.mean, self.stddev, self.real = mean, stddev, real
        if real == "float":
            self.mean, self.stddev = _f32(mean), _f32(stddev)
        self.reset()

    def reset(self):
        self.has_spare, self.spare = False, 0.0

    def standard(self, g) -> float:
        if self.has_spare:
            self.has_spare = False
            return self.spare
        while True:
            x = 2.0 * unit_uniform(g, "double") - 1.0
            y = 2.0 * unit_uniform(g, "double") - 1.0
            r2 = x * x + y * y
            if not (r2 > 1.0 or r2 == 0.0):
                break
        m = math.sqrt(-2.0 * math.log(r2) / r2)
        self.spare, self.has_spare = x * m, True
        return y * m

    def __call__(self, g):
        z = self.standard(g) * float(self.stddev) + float(self.mean)
        return _f32(z) if self.real == "float" else z


class Weibull:
    def __init__(self, a=1.0, b=1.0, real="double"):
        self.a, self.b, self.real = a, b, real

    def __call__(self, g):
        u = unit_uniform(g, "double")
        v = float(self.b) * math.pow(-math.log(1.0 - u), 1.0 / float(self.a))
        return _f32(v) if self.real == "float" else v

    def reset(self):
        pass


class Gamma:
    """Marsaglia & Tsang (2000); shape < 1 boosted and scaled back."""

    def __init__(self, alpha=1.0, beta=1.0, real="double"):
        self.alpha, self.beta, self.real = alpha, beta, real
        self._n = Normal(0.0, 1.0, "double")

    def reset(self):
        self._n.reset()

    def __call__(self, g):
        alpha = float(self.alpha)
        ba = alpha + 1.0 if alpha < 1.0 else alpha
        d = ba - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = self._n(g)
            v = 1.0 + c * x
            if v <= 0.0:
                continue
            v = v * v * v
            u = unit_uniform(g, "double")
            x2 = x * x
            if u < 1.0 - 0.0331 * x2 * x2 or (u > 0.0 and math.log(u) < 0.5 * x2 + d * (1.0 - v + math.log(v))):
                s = d * v
                break
        if alpha < 1.0:
            while True:
                u = unit_uniform(g, "double")
                if u != 0.0:
                    break
            s *= math.pow(u, 1.0 / alpha)
        s *= float(self.beta)
        return _f32(s) if self.real == "float" else s


class Cauchy:
    def __init__(self, a=0.0, b=1.0, real="double"):
        self.a, self.b, self.real = a, b, real

    def __call__(self, g):
        while True:
            u = unit_uniform(g, "double")
            if u != 0.0:
                break
        v = float(self.a) + float(self.b) * math.tan(3.14159265358979323846 * (u - 0.5))
        return _f32(v) if self.real == "float" else v

    def reset(self):
        pass


class Discrete:
    def __init__(self, weights=None):
        self.cum = []
        total = 0.0
        for w in weights or []:
            if not (float(w) >= 0.0):
                raise ValueError("discrete_distribution: weight negative or NaN")
            total += float(w)
            self.cum.append(total)
        if not self.cum:
            self.cum = [1.0]
        elif not total > 0.0:
            raise ValueError("discrete_distribution: weights sum to zero")

    def __call__(self, g):
        t = unit_uniform(g, "double") * self.cum[-1]
        return min(bisect.bisect_right(self.cum, t), len(self.cum) - 1)

    def reset(self):
        pass


def shuffle(seq: list, g) -> None:
    """Fisher-Yates from the last element down (helios::shuffle), in place."""
    for i in range(len(seq) - 1, 0, -1):
        j = uniform_integer(g, i)
        if i != j:
            seq[i], seq[j] = seq[j], seq[i]


class ContextRandom:
    """The Context-level RNG API: seedRandomGenerator / randu / randn (Context.cpp)."""

    def __init__(self, seed: int | None = None):
        self.generator = MinstdRand0()
        self.unif = UniformReal(0.0, 1.0, "float")
        self.norm = Normal(0.0, 1.0, "float")
        if seed is not None:
            self.seed(seed)

    def seed(self, s: int):
        self.generator.seed(s)
        self.unif.reset()
        self.norm.reset()

    def randu(self, lo=None, hi=None):
        if lo is None:
            return self.unif(self.generator)
        if isinstance(lo, (int, np.integer)) and isinstance(hi, (int, np.integer)):
            if hi < lo:
                raise ValueError("randu: max < min")
            return int(lo) if hi == lo else int(UniformInt(lo, hi)(self.generator))
        lo, hi = _f32(lo), _f32(hi)
        if hi < lo:
            raise ValueError("randu: max < min")
        if hi == lo:
            return lo
        return _f32(lo + _f32(self.unif(self.generator) * _f32(hi - lo)))

    def randn(self, mean=None, stddev=None):
        if mean is None:
            return self.norm(self.generator)
        return _f32(_f32(mean) + _f32(self.norm(self.generator) * _f32(abs(_f32(stddev)))))
