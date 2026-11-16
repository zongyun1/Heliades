import hashlib, sys, time, numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "../../../.."))
from pyhelios import Context, PlantArchitecture
from pyhelios.types import vec3


def digest(ctx):
    uu = ctx.getAllUUIDs()
    h = hashlib.sha1()
    for u in uu:
        v = ctx.getPrimitiveVertices(u)
        h.update(np.array([[p.x, p.y, p.z] for p in v], dtype=np.float32).tobytes())
    return len(uu), len(ctx.getAllObjectIDs()), h.hexdigest()[:10]


def build(seed, age, split=None):
    ctx = Context()
    ctx.seedRandomGenerator(seed)
    pa = PlantArchitecture(ctx)
    pa.loadPlantModelFromLibrary("tomato")
    t = time.time()
    if split is None:
        pid = pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), age)
    else:
        pid = pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), 0)
        for dt in split:
            pa.advanceTime(dt, pid) if False else pa.advanceTime(dt)
    dt = time.time() - t
    return digest(ctx), round(dt, 1), len(pa.getAllShootIDs(pid)), len(pa.getPlantFruitObjectIDs(pid)), len(pa.getPlantFlowerObjectIDs(pid)), len(pa.getPlantLeafObjectIDs(pid))


mode = sys.argv[1]
if mode == "grid":
    for age in [10, 30, 50, 75]:
        for seed in [0, 3]:
            print("age", age, "seed", seed, build(seed, age), flush=True)
elif mode == "split":
    print("build(20)       ", build(3, 20), flush=True)
    print("build(0)+adv(20)", build(3, 0, [20]), flush=True)
    print("build(0)+adv(10)x2", build(3, 0, [10, 10]), flush=True)
    print("build(0)+adv(5)x4", build(3, 0, [5, 5, 5, 5]), flush=True)
