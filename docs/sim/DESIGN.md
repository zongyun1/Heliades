# heliades.sim B1: a forward tree direct solver

Status: prototype (CPU, numpy, batched). Code: `heliades/sim/`. Tests: `tests/heliades/test_sim_tree.py`.
Results: `migration/notes/B1.md`.

## 1. Problem

`planttwin.sim` (small-step XPBD, graph-coloured Gauss–Seidel) cannot hold a full-size crop.
Plant rods are long, thin, nearly inextensible and stiff, they form deep trees, and they carry
heavy distal loads (leaves, 11 g fruit on near-massless pedicels). Gauss–Seidel propagates
information one constraint per iteration. Its error contracts slowly when stiffness ratios
and mass ratios are both large (Macklin 2020: primal methods fail on stiffness ratios, dual
methods on mass ratios, and plants have both). The result is creep under gravity even at
E = 1e11 Pa and 256 substeps.

A rooted tree without contact has an acyclic constraint graph, so the linear system of an
implicit step can be solved **exactly in O(n)** by one leaf-to-root and one root-to-leaf sweep
(Baraff 1996; Featherstone's articulated-body algorithm, ABA). B1 builds that solver.

## 2. Formulation chosen: reduced coordinates, implicit joint springs, ABA

Choice (a) from the brief. Reasons:

| | (a) reduced coordinates + ABA (chosen) | (b) maximal coordinates + direct KKT (Deul 2018 / Baraff 1996) |
|---|---|---|
| inextensibility | exact by construction (no stretch DOF) | a stiff or hard constraint; needs Newton iterations |
| mass ratios | irrelevant: ABA is an exact factorisation of M(q) | exact too, but every constraint row is solved |
| unknowns per segment | 3 (spherical joint) | 6 + 6 multipliers |
| per-step work | 1 direct solve (linearly-implicit) | Newton loop of direct solves (Deul: several per step) |
| statics | the equilibrium is the root of τ(q); no constraint drift | constraint residual plus drift |
| contact / loops | harder (needs Schur complement, see §8) | natural (Baraff's O(kn) extension) |

Plants are acyclic and nearly inextensible, and the failures we have to fix are mass ratio and
stiffness, so (a) gives the most exactness for the least work. The loop/contact weakness is
handled later with the standard O(kn) Schur complement on top of the same factorisation (§8).

### 2.1 Bodies, joints, payloads

* Each rod edge (asset `StemGraph.edges`) is a rigid **segment body**: a cylinder of length l,
  radius r = mean of its node radii, and mass ρπr²l, with the exact cylinder inertia.
  Segments whose two nodes are both `fixed` are welded to the ground and removed.
* Each body has **one 3-DOF spherical joint at its proximal node**. The joint connects it to the
  segment ending at that node, or to the ground. Child frame: z = segment tangent; x, y come from
  parallel transport of the parent frame (rest frames `R0`).
* **Leaves** are rigid payloads welded to the body that ends at the attachment node. Mass,
  first moment and second moment are integrated exactly over the triangulated mesh
  (∫ x xᵀ dA = A/12 · (Σ vᵢvᵢᵀ + s sᵀ) per triangle) times the areal density.
* **Fruit / point masses** are payloads at their node: m and a solid-sphere inertia (2/5) m r².
* Mass properties are stored per unit density (segments, leaves, point masses separately), so
  a batch of densities is a linear combination and no per-batch geometry is recomputed.

### 2.2 Coordinates and elastic energy

Joint i carries a unit quaternion Qᵢ, the deviation from the rest relative rotation:

    R_rel,i = R0rel,i · R(Qᵢ),     R_i = R_parent · R_rel,i

The generalized velocity uᵢ ∈ ℝ³ is the relative angular velocity in the child frame
(Q̇ = ½ Q ⊗ (0, u)). With v = vec(Q) = sin(θ/2)·axis and w = scalar(Q):

    U_i = 2 vᵀ K v,   K = diag(k_b, k_b, k_t)
    k_b = E / c_i,  k_t = G·2 / c_i,  c_i = Σ_{half segments at the joint} (l/2)/I,  I = πr⁴/4,
    J = 2I,  G = E / (2(1+ν))

* U = ½ φᵀKφ + O(θ⁴), so it is the standard discrete Cosserat/Kirchhoff bending-twisting energy.
  The joint compliance is the series sum of the two half segments meeting at the joint. That is
  exact for tapering rods, and at a junction the thin branch dominates automatically.
* U is **smooth on all of SO(3)** (it is even in Q). A first version used rotation-vector
  coordinates φ with U = ½φᵀKφ, and it blew up on the tomato. Helios petiolules have r = 0.18 mm,
  so the leaflets hang like pendulums on strings, joints wind past π, and Jr⁻¹(φ) is singular at
  |φ| = 2π. The quaternion energy has no winding singularity. It softens at large per-joint
  angles (torque ∝ sin θ), which is a discretisation effect. Refine the segments if a joint
  regularly exceeds ~0.5 rad.
* Generalized torque: τ = −∂U/∂u = −2 Γ K v with Γ = wI − [v]×.

### 2.3 Time integration: linearly-implicit Euler

The equations of motion are M(q) u̇ + C(q, u) = τ_el(q) + τ_damp + Jᵀ f_ext. One step of size h:

    (M + h·D + h²·K̃) a = τ_el(Q_n) − (h K̃ + D) u_n − C̃(u_n) + Jᵀ f_ext(q_n)
    u_{n+1} = u_n + h a,      Q_{n+1} = Q_n ⊗ exp(h u_{n+1})

Everything that is stiff is evaluated at the end of the step through its linearisation:

| term | treatment | enters the ABA as |
|---|---|---|
| joint springs | K̃ = Γ K Γᵀ (SPD, = K near rest) | joint "armature" h²K̃ in Dᵢ (3×3 per joint) |
| joint (Rayleigh-β) damping | β K̃ | armature h β K̃ |
| mass-proportional damping α (`PhysicsParams.damping`, 1/s) | −α I_b v_{n+1} on every body | body inertia × (1 + hα), force −α I_b v_n |
| gyroscopic and centrifugal body bias b(v) | b(v_{n+1}) ≈ b + h (∂b/∂v) a (Catto 2015) | non-symmetric 6×6 added to body inertia |
| Coriolis/centripetal propagation cᵢ(ω_p, uᵢ) | cᵢ ≈ cᵢ + h ∂c/∂a_p · a_p + h ∂c/∂u̇ · u̇ | modified motion map X' = X + hC_p and joint map S' = S + hC_u |
| gravity geometric stiffness | PSD part of K_g = (g·mc) I − sym(g mcᵀ) per body | h²K_g⁺ added to body inertia, −hK_g⁺ω_n |
| penalty contact springs/dampers | k Gᵀ n nᵀ G, c Gᵀ n nᵀ G per contact point | h²k + h c on body inertia |

All of these are **joint-local or body-local** matrices, so the ABA solves the whole implicit
system exactly in O(n). None of them changes the fixed point. At u = 0 the update is zero iff
τ_el(q) + Jᵀ f(q) = 0, which is the static equilibrium. So **the equilibrium does not depend on
h** (tested: h = 1/60, 1/240, 1/1000 agree to 1e-5). XPBD cannot have this property, because
its stiffness floor scales with h².

### 2.4 The O(n) algorithm (one step)

Spatial vectors are [angular; linear], expressed in world orientation about each body's origin
oᵢ (its proximal joint node), so parent→child transforms are pure shifts X = [[I,0],[−[r]×,I]],
r = oᵢ − o_p. Accelerations are classical (a = [ω̇; ä_O]); the Newton–Euler body equation is
f = I_b a + b(v), with I_b = [[I_O, [mc]×],[[mc]×ᵀ, m]] and b = [ω×I_Oω; ω×(ω×mc)].

```
pass 0  (bulk)          R_rel = R0rel R(Q); Γ, K̃, τ, armature A = (h² + hβ) K̃
pass 1  root → leaf     R_i = R_p R_rel,i;  o_i = tip_p;  ω_i = ω_p + R_i u_i;  v_Oi = v_Op + ω_p × r
                        c_i = [ω_p × R_i u_i ; ω_p × (ω_p × r)]
pass 1b (bulk)          I_b, bias, gravity, damping, K_g⁺, contacts → I_eff,i, p_i = b_i − f_ext,i
                        X'_i, S'_i (linearised c_i)
pass 2  leaf → root     U = IA S',  W = Sᵀ IA,  D = W S' + A,  ũ = τ − Sᵀ pA
                        Ia = IA − U D⁻¹ W,   pa = pA + Ia c + U D⁻¹ ũ
                        IA_p += Xᵀ Ia X',    pA_p += Xᵀ pa            (segmented sum over children)
pass 3  root → leaf     a' = X' a_p + c;   u̇ = D⁻¹(ũ − W a');   a = a' + S' u̇
update                  u += h u̇;  Q ← normalise(Q ⊗ exp(h u))
```

Cost: three sweeps over depth levels plus bulk work. Per body that is a constant number of
6×6/3×3 products and one 3×3 inverse. **Exact, non-iterative**, O(n) in bodies and O(B·n) in
the batch.

### 2.5 Stability argument

* **Stiff linear parts are L-stable.** Springs, damping and contact penalties are solved with
  backward Euler on their exact linearisation. Their frequencies, however high (twist of a 0.2 mm
  petiolule is ~10⁴ rad/s), are damped, not amplified. Neither stiffness nor mass ratio enters a
  stability bound, because ABA factorises M exactly for any mass distribution (test 5: 1000:1
  tip mass, h = 1/60).
* **Velocity-product terms.** Explicit Coriolis/gyroscopic terms gain energy at a rate of about
  h·ω² (measured: dE/E ∝ h, O(h) consistent, sign positive). On the tomato, whipping leaflets
  reach ~100 rad/s, so at h = 1/60 the explicit version diverged in 0.5 s. The linearised-implicit
  forms (body bias, and X', S' for the propagation terms) flip the sign: the measured energy error
  is now dissipative O(h), and on the tomato with zero damping and no gravity **energy is
  non-increasing at every step** (test 8).
* **Gravity.** Its geometric stiffness is stabilising for hanging parts (pendulum leaves, ω² ~ g/l),
  and those parts limit an explicit treatment to hω < 2. The PSD part goes into the implicit
  matrix, so hanging leaves are stable at any h. The negative (buckling) part stays explicit. It is
  physical, and treating it implicitly could make D indefinite.
* **Remaining explicit parts:** the negative gravity stiffness, plus the dropped non-symmetric
  parts of the spring Hessian (O(θ)·k). Both are bounded and small compared with the implicit
  terms. In practice h = 1/15 … 1/1000 are stable on the tomato.
* **Cost of stability: numerical damping.** Backward Euler damps a mode of frequency ω with
  ζ_num ≈ ωh/2. At h = 1/60 a 3 Hz mode loses ~18%/cycle, which is fine for settling and
  quasi-static manipulation and visible for free ring-down. For dynamics fidelity use h ≤ 1/500
  (test 4 uses 1/1000: first frequency within 0.25%), or see §7 (BDF2 / θ-method).

## 3. API

```python
from heliades.sim import TreeSim, PointForce, SphereCollider
sim = TreeSim(asset, [phys_1, ..., phys_B], gravity=(0, 0, -9.81), joint_damping=0.0)
sim.step(h, external=[PointForce(node, f), SphereCollider(center, radius, stiffness)], damping=None)
sim.settle(h=1/30, max_time=3, tol=1e-3, damping=5.0)   # dynamic settle, quiet-window criterion
sim.solve_static(h=0.2, iters=200)                       # pseudo-time modified Newton (same fixed point)
sim.state  # .x (B,N,3) rod nodes, .v_nodes, .R (B,nb,3,3), .quat, .Q, .phi, .u, .t
sim.leaf_vertices(); sim.energy(); sim.point_speeds(); sim.run(duration, h, external=callable(t))
```

Per batch element: `stem_youngs_modulus`, `stem_poisson`, `stem_density`, `leaf_areal_density`,
`damping`. Topology and geometry are shared. `external` may be a list of actions or an (B,N,3)
array of node forces. Sphere contacts sample the distal rod nodes and all leaf vertices, with an
implicit penalty spring and damper per contact (stable at 60 Hz with k = 5e3 N/m on a 2 g rod).

## 4. Data layout (Warp-portable)

`TreeTopology` flattens everything into arrays, and no Python object per node is touched in the
hot loop:

* bodies renumbered in **level order** (depth = number of joints to the ground). Level d is the
  contiguous slice `lvl[d]:lvl[d+1]`; inside a level, bodies are sorted by parent, so the children
  of one parent are contiguous;
* `parent[nb]` (−1 = ground), `node_a/node_b`, `l`, `r`, `R0`, `R0rel`, joint stiffness geometry;
* the child→parent accumulation in pass 2 is a **segmented sum** (`np.add.reduceat` over
  precomputed segment starts). There are no atomics and no scatter conflicts, so the GPU port is a
  plain segmented reduction;
* every array carries the batch dimension B in front: (B, nb, 6, 6) and so on.

The tomato (476 bodies, 497 leaves) has only **27 levels**, so the sequential depth of each sweep
is 27, not 476.

## 5. Limitations (prototype)

1. **No loops, no self-contact, no leaf–leaf contact.** Sphere contact is penalty-based (implicit,
   but a penalty) with no friction.
2. **Leaves are rigid** payloads. Leaf bending is the next package.
3. **Inextensible, unshearable** rods (correct for stems; prevents modelling stretch).
4. **Numerical damping** of backward Euler (§2.5).
5. **Large per-joint angles soften** (sin θ torque). Helios petiolules at E = 1e8 reach ~1 rad per
   joint. Segment refinement (subdividing asset edges) is not implemented yet.
6. Settling time is physical. A near-buckling structure (tomato at E = 3e8) has a 0.26 Hz soft
   mode, and no damping value can settle it in 3 s. `solve_static` is the tool for statics.
7. The gravity geometric stiffness is body-level (no commutator terms), so it serves as a
   preconditioner. Exact Newton statics would need the full generalized Hessian.
8. CPU numpy: per-level Python overhead dominates at small B. Large B is compute-bound on stacked
   6×6 products (see the B1 notes for steps/s).

## 6. GPU port plan (Warp)

* One `wp.struct`-free SoA layout, as above; B × nb threads per bulk kernel.
* Pass 0, 1b and the update are embarrassingly parallel kernels.
* Passes 1, 2, 3: one kernel launch per level (27 launches per sweep for the tomato), or one
  persistent kernel per tree with a level barrier. Pass 2's child→parent sum is a segmented
  reduction over the contiguous child ranges (one thread per parent loops over its children; no
  atomics).
* 6×6 and 3×3 math as `wp.mat66`/`wp.mat33` value types, and a 3×3 inverse by closed form (D is
  SPD near rest).
* Batch = population or RL envs: launch dimension B × level_width. For B ≥ 64 that keeps a GPU
  busy even for the 2–30-wide levels of a single plant.
* Differentiability: the step is a fixed sequence of dense small ops. `wp.Tape` gives gradients
  for Real2Sim without unrolled Gauss–Seidel.

## 7. How later packages plug in

* **Leaf shells (condensed).** The leaf attaches at one node, so a leaf shell keeps the global
  graph a tree. Replace the rigid payload with a child *body* of the host segment that has a
  6-DOF (or 3-DOF rotational) compliant joint whose 6×6 stiffness and inertia are the shell's
  Schur complement (static condensation) onto the attachment frame, plus interior shell DOFs
  solved locally per leaf after pass 3. The coupling point is `TreeTopology.leaf_part` / the
  `LeafPayload` list: a leaf currently adds (m, h, I_O) to its host body; a shell instead adds its
  condensed articulated inertia (IA, pA) to the host in pass 2 and reads back its host
  acceleration in pass 3. This is exactly what a child body does in the ABA.
* **Inverse statics.** The rooted tree without contact is statically determinate. The internal
  force and moment at every joint follow from one leaf-to-root sum of distal loads (the same
  segmented reduction as pass 2 with gravity only). The rest relative rotation then follows
  locally: given the observed shape Q_obs and joint moment M_i, solve 2ΓKv = M_i for the deviation
  quaternion, and set R0rel ← R_rel,obs · R(Q_dev)ᵀ. This is O(n), exact, and consistent with
  *this* discrete forward model, so `settle(rest_from_inverse) == observed` to round-off. Open
  point: buckling. Upright stems are in compression, and the recovered rest shape must keep the
  Hessian PD. `solve_static` plus the buckling test above provide the check.
* **Contact and loops.** For k contact or loop constraints, use Baraff's O(kn) extension: keep the
  tree factorisation (pass 2), solve the k×k Schur complement (Delassus operator) for the contact
  impulses with a projected Gauss–Seidel / LCP on that small system, then apply them in pass 3.
  The current penalty spheres become this.
* **Higher-order integration.** Use BDF2 or a θ-method on the spring term (armature (θh)²K̃) for
  less numerical damping. The ABA structure is unchanged.
