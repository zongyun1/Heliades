"""Count raw minstd_rand0 engine calls consumed by each tomato build stage.

Seed the Context, run a stage, then call ctx.randu() once and find n such that the
(n+1)-th raw engine output reproduces that randu value (heliades.core.random).
"""
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "../../../.."))
import numpy as np
from heliades.core.random import MinstdRand0, UniformReal
from pyhelios import Context, PlantArchitecture
from pyhelios.types import vec3


def table(seed, nmax=2_000_000):
    g = MinstdRand0(seed)
    vals = {}
    u = UniformReal(0.0, 1.0, "float")
    out = []
    for n in range(nmax):
        h = g.copy()
        out.append(u(h))
        g()
    return out


def find(tab, v):
    v = np.float32(v)
    hits = [i for i, x in enumerate(tab) if x == v]
    return hits[:3]


seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ages = [float(a) for a in sys.argv[2:]] or [0.0, 2.0, 10.0]
tab = table(seed, 400_000)


def stage(n_stage, age=None):
    ctx = Context()
    ctx.seedRandomGenerator(seed)
    if n_stage >= 1:
        pa = PlantArchitecture(ctx)
    if n_stage >= 2:
        pa.loadPlantModelFromLibrary("tomato")
    if n_stage >= 3:
        pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), age)
    r = ctx.randu()
    return find(tab, r), len(ctx.getAllUUIDs())


print("ctx only", stage(0), flush=True)
print("after PlantArchitecture()", stage(1), flush=True)
print("after load('tomato')", stage(2), flush=True)
for a in ages:
    print("after build age", a, stage(3, a), flush=True)
