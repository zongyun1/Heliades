import sys, collections
from pyhelios import Context, PlantArchitecture
from pyhelios.types import vec3

seed = int(sys.argv[1]); age = float(sys.argv[2])
ctx = Context()
ctx.seedRandomGenerator(seed)
pa = PlantArchitecture(ctx)
pa.loadPlantModelFromLibrary("tomato")
pid = pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), age)
print("uuids", len(ctx.getAllUUIDs()), "objs", len(ctx.getAllObjectIDs()))
for name in ["getAllLeafUUIDs", "getAllInternodeUUIDs", "getAllPetioleUUIDs", "getAllPeduncleUUIDs", "getAllFlowerUUIDs", "getAllFruitUUIDs"]:
    print(name, len(getattr(pa, name)()))
print("plant objs", len(pa.getAllPlantObjectIDs(pid)))
ot = collections.Counter(ctx.getObjectType(o) for o in ctx.getAllObjectIDs())
print("object types", dict(ot))
pt = collections.Counter(ctx.getPrimitiveType(u) for u in ctx.getAllUUIDs())
print("prim types", dict(pt))
labels = collections.Counter()
for u in ctx.getAllUUIDs():
    labels[tuple(ctx.listPrimitiveData(u))] += 1
print("primdata label sets", dict(labels))
tex = collections.Counter(ctx.getPrimitiveTextureFile(u).split('/')[-1] for u in ctx.getAllUUIDs())
print("textures", dict(tex))
try:
    print("age", pa.getPlantAge(pid))
except Exception as e:
    print(e)
for sid in pa.getAllShootIDs(pid):
    print("shoot", sid, pa.getShoot(pid, sid) if hasattr(pa, "getShoot") else "")
