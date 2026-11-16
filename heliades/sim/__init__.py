"""Heliades plant-specific deformable simulator.

B1: forward tree direct solver -- batched, implicit, reduced-coordinate articulated rods
with an O(n) articulated-body (Featherstone) sweep. See docs/sim/DESIGN.md.
B2: inverse statics + stability (inverse.py, hessian.py; docs/sim/INVERSE.md).
B3: contact and ground (contact.py; implicit Baraff Schur complement + PGS, Coulomb cone),
floating bases, segment refinement (refine.py). See docs/sim/CONTACT.md.
"""
from .tree import GRAVITY, BodyForce, PointForce, SimState, SphereCollider, TreeSim
from .topology import TreeTopology
from .contact import CapsuleObstacle, ContactParams, GroundPlane, SphereObstacle, stake
from .refine import refine_asset

__all__ = ["TreeSim", "TreeTopology", "PointForce", "BodyForce", "SphereCollider", "SimState", "GRAVITY",
           "GroundPlane", "SphereObstacle", "CapsuleObstacle", "ContactParams", "stake", "refine_asset"]
