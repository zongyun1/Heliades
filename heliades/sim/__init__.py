"""Heliades plant-specific deformable simulator.

B1: forward tree direct solver -- batched, implicit, reduced-coordinate articulated rods
with an O(n) articulated-body (Featherstone) sweep. See docs/sim/DESIGN.md.
"""
from .tree import GRAVITY, PointForce, SimState, SphereCollider, TreeSim
from .topology import TreeTopology

__all__ = ["TreeSim", "TreeTopology", "PointForce", "SphereCollider", "SimState", "GRAVITY"]
