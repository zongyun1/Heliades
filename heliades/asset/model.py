"""Structured Plant Asset: the intermediate layer between asset sources and simulation.

Sources (procedural generators such as Helios, or reconstruction) *produce* this;
the simulator only ever *consumes* this. Units are SI (metres, kilograms, seconds).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class StemGraph:
    nodes: np.ndarray   # (N, 3) node positions
    edges: np.ndarray   # (E, 2) int, directed parent -> child (a forest rooted at the pot)
    radius: np.ndarray  # (N,) stem radius at each node
    fixed: np.ndarray   # (N,) bool, nodes clamped to the pot
    # optional per-node extras
    point_mass: np.ndarray | None = None  # (N,) kg lumped at the node (fruit, flowers)
    organ: np.ndarray | None = None       # (N,) int, see ORGANS

    def parent(self) -> np.ndarray:
        par = -np.ones(len(self.nodes), dtype=int)
        par[self.edges[:, 1]] = self.edges[:, 0]
        return par


ORGANS = {0: "stem", 1: "petiole", 2: "peduncle", 3: "pedicel"}


@dataclass
class Fruit:
    node: int        # stem node at the fruit centre (end of its pedicel); its mass is in point_mass
    radius: float
    mass: float


@dataclass
class Leaf:
    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray     # (F, 3) int
    attach_node: int      # stem node the leaf hangs from (petiole tip)
    attach_vertex: int    # leaf vertex coincident with that stem node (welded in sim)
    keypoints: dict = field(default_factory=dict)  # name -> vertex index, used for tracking


@dataclass
class PhysicsParams:
    # Mass priors. Only stiffness/mass ratios are identifiable from video without a
    # force measurement, so density is held fixed and stiffness is what we fit.
    stem_density: float = 1000.0        # kg/m^3 (fresh tissue is mostly water)
    leaf_areal_density: float = 0.25    # kg/m^2 (~0.25 mm thick blade)
    # Stem/petiole: Cosserat rod. Bending stiffness per joint is E*I with I = pi r^4 / 4
    # taken from the reconstructed radius, so one physical modulus covers the whole tree.
    stem_youngs_modulus: float = 1e8    # Pa (herbaceous stems ~ 1e7 .. 1e9)
    stem_poisson: float = 0.4           # sets twist stiffness G*J
    stem_stretch_compliance: float = 1e-9
    # Leaf blade: XPBD compliances on the leaf mesh. Effective (discretisation-dependent)
    # values; mapping to plate stiffness E t^3 / 12(1-nu^2) is a calibration step.
    leaf_stretch_compliance: float = 1e-7
    leaf_bend_compliance: float = 2e-2
    damping: float = 1.5                # 1/s, linear (and angular) velocity damping

    def copy(self, **kw) -> "PhysicsParams":
        d = asdict(self)
        d.update(kw)
        return PhysicsParams(**d)


@dataclass
class PlantAsset:
    stem: StemGraph
    leaves: list[Leaf]
    physics: PhysicsParams = field(default_factory=PhysicsParams)
    meta: dict = field(default_factory=dict)  # provenance, scale reference, species...
    fruits: list[Fruit] = field(default_factory=list)

    # ---- serialisation -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "stem": {k: np.asarray(v).tolist() for k, v in asdict(self.stem).items() if v is not None},
            "fruits": [asdict(f) for f in self.fruits],
            "leaves": [
                {
                    "vertices": np.asarray(l.vertices).tolist(),
                    "faces": np.asarray(l.faces).tolist(),
                    "attach_node": int(l.attach_node),
                    "attach_vertex": int(l.attach_vertex),
                    "keypoints": {k: int(v) for k, v in l.keypoints.items()},
                }
                for l in self.leaves
            ],
            "physics": asdict(self.physics),
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "PlantAsset":
        s = d["stem"]
        stem = StemGraph(
            nodes=np.array(s["nodes"], float),
            edges=np.array(s["edges"], int).reshape(-1, 2),
            radius=np.array(s["radius"], float),
            fixed=np.array(s["fixed"], bool),
            point_mass=np.array(s["point_mass"], float) if "point_mass" in s else None,
            organ=np.array(s["organ"], int) if "organ" in s else None,
        )
        leaves = [
            Leaf(
                vertices=np.array(l["vertices"], float),
                faces=np.array(l["faces"], int).reshape(-1, 3),
                attach_node=l["attach_node"],
                attach_vertex=l["attach_vertex"],
                keypoints=l.get("keypoints", {}),
            )
            for l in d["leaves"]
        ]
        fruits = [Fruit(**f) for f in d.get("fruits", [])]
        return PlantAsset(stem, leaves, PhysicsParams(**d["physics"]), d.get("meta", {}), fruits)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @staticmethod
    def load(path: str | Path) -> "PlantAsset":
        return PlantAsset.from_dict(json.loads(Path(path).read_text()))
