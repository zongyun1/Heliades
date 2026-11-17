import hashlib, sys, numpy as np
from pyhelios import Context, PlantArchitecture
from pyhelios.types import vec3


def digest(ctx):
    uu = ctx.getAllUUIDs()
    h = hashlib.sha1()
    for u in uu:
        v = ctx.getPrimitiveVertices(u)
        h.update(np.array([[p.x, p.y, p.z] for p in v], dtype=np.float32).tobytes())
    return len(uu), h.hexdigest()[:12]


def build(seed, when, age):
    ctx = Context()
    if when == "before":
        ctx.seedRandomGenerator(seed)
    pa = PlantArchitecture(ctx)
    if when == "after_ctor":
        ctx.seedRandomGenerator(seed)
    pa.loadPlantModelFromLibrary("tomato")
    if when == "after_load":
        ctx.seedRandomGenerator(seed)
    pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), age)
    return digest(ctx)


age = float(sys.argv[1]) if len(sys.argv) > 1 else 20
for when in ["before", "after_ctor", "after_load"]:
    for s in [0, 1, 2]:
        print(when, s, build(s, when, age), build(s, when, age), flush=True)
print("noseed", build(0, "none", age), build(0, "none", age))
