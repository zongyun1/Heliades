import sys, re, math
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "../../../.."))
import numpy as np
from heliades.core.random import ContextRandom, MinstdRand0, uniform_integer
from pyhelios import Context, PlantArchitecture
from pyhelios.types import vec3

out = "/tmp/tomato_struct.xml"
for seed in [5, 2, 3]:
    ctx = Context(); ctx.seedRandomGenerator(seed)
    pa = PlantArchitecture(ctx); pa.loadPlantModelFromLibrary("tomato")
    pid = pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), 0)
    pa.writePlantStructureXML(pid, out)
    txt = open(out).read()
    br = re.search(r"<base_rotation>([^<]*)</base_rotation>", txt).group(1)
    r = ContextRandom(seed)
    for _ in range(4):
        uniform_integer(r.generator, 2**32 - 1)
    hi = np.float32(2 * math.pi)
    a = r.randu(np.float32(0), hi); b = r.randu(np.float32(0), hi)
    print(seed, "xml base_rotation:", br, "| draw1", a, "draw2", b,
          "deg", float(a) * 180 / math.pi, float(b) * 180 / math.pi)
    m = re.search(r"<internode_phyllotactic_angle>([^<]*)<", txt)
    print("   phyllotactic", m.group(1) if m else None, " next draws:", [float(r.randu()) for _ in range(6)])
