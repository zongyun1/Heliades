# heliades.sim B3: contact and ground on the tree solver

Status: prototype (CPU, numpy + one Warp kernel, batched). Code: `heliades/sim/contact.py` (colliders,
detection, Delassus operator, PGS, step hook), `contact_wp.py` (Warp PGS kernel), `refine.py`
(segment refinement), `inverse_contact.py` (contact-aware inverse statics), `scenarios.py` (shared
test/demo set-ups); additive edits to `tree.py` (floating bases, stored factorisation, bias-only
response sweep, `BodyForce`), `topology.py` (ancestor table, floating bases) and `hessian.py`
(`BodyForce` dead loads). Tests: `tests/heliades/test_sim_contact.py`. Results: `migration/notes/B3.md`.
Demo: `scripts/contact_demo.py` → `outputs/sim/tomato_ground_rest.mp4`, `outputs/sim/tomato_finger_push.mp4`,
`outputs/sim/contact_summary.json`.

## 1. Problem

B1's tree solver is exact and O(n) for a clamped tree under smooth forces. Contact breaks the tree:
a leaf touching the ground or a finger closes a loop through the environment. B1 only had an
implicit *penalty* sphere. Penalty contact needs a stiffness that is high enough to keep the
penetration small, which makes the contact the stiffest element in the system. Its error is
proportional to the load over the stiffness, and it has no friction. B2 showed that below
E_cr ≈ 7.3e8 Pa the Helios tomato folds over and passes through the ground. Robot fingers and
stakes are the use cases.

## 2. Formulation: velocity-level contact on the factorised implicit step

B1's step is linearly implicit: `M~ a = rhs`, `u+ = u + h a`, where `M~ = M + hD + h²K~ + h(∂b/∂v) + h²K_g⁺`
(armature, damping, implicit gyroscopic/Coriolis terms, gravity stiffness), and the articulated-body
sweep factorises it exactly in O(n). Contact is added at the **velocity level** (Stewart–Trinkle /
Anitescu time stepping) on **the same `M~`**:

    M~ a = rhs + Jᵀ λ / h            λ: contact impulses (normal, 2 tangential) per contact
    u+   = u* + M~⁻¹ Jᵀ λ            u* = contact-free update (B1 step)
    v    = J u+ = v_free + W λ,      W = J M~⁻¹ Jᵀ   (Delassus operator, 3k × 3k)

with, per contact c, `0 ≤ λ_n ⊥ v_n − b_n ≥ 0` and Coulomb friction `|λ_t| ≤ μ λ_n`, with `λ_t`
opposing the slip. One step does the following:

1. B1 factorises the tree and solves the contact-free step (`TreeSim.step`, unchanged apart from
   storing `UD = IA S' D⁻¹`, `W = Sᵀ IA`, `D⁻¹`, `X'`, `S'` and the floating-base `IA⁻¹`);
2. it detects contacts at q_n;
3. it forms `W` from the stored factorisation (§3);
4. it runs projected Gauss–Seidel on the 3k impulses only (§5);
5. it back-substitutes: the impulses enter as body forces in **one bias-only O(n) sweep of the same
   factorisation** (`TreeSim.response`), and `u+ = u* + Δu`.

Why this and not penalty:

* **It is consistent with the implicit step.** The contact sees exactly the implicit stiffness and
  damping of the plant (`M~` includes `h²K~`). A finger pressing a 1.5 N/m cantilever and a stake
  holding a 0.5 kg plant use the same solver with no stiffness to tune. A hard contact on a stiff
  plant at h = 1/60 is no harder than a soft one.
* **It is exact at rest.** At equilibrium the impulses are the static reactions. The cantilever
  force matches beam theory to 0.07% and the chain reaction matches the weight to 1e-6 (§8).
  Penalty contact instead carries an error of load/k_pen. That error is small for a stiff spring,
  but it is not zero, and its stiffness then dominates the system's conditioning.
* **It adds no stability limit.** Impulses are solved for the end-of-step velocity, so there is no
  h²k_pen/m condition and no contact oscillation.
* **It costs O(k·depth + k²) + PGS(k) on top of B1's O(n).** The factorisation is reused, and the
  only new work is in k (§3, §9).

## 3. Delassus operator from the tree factorisation, O(k·depth + k²)

Baraff (1996) forms `J M⁻¹ Jᵀ` column by column with one O(n) solve per constraint, which costs
O(kn). We exploit the tree further. The response to a test force on body b is non-zero along its
root path in the up-sweep, and it affects other bodies only through the parent acceleration where
their path leaves b's path. For contact c on body b_c, with path t_0 … t_d (t_d = b_c), the
computation is:

* **Up-sweep along the path** (unit forces along n, t1, t2 at p_c; bias `pA = −[(p−o)×d; d]`):
  `ul_s = −R_sᵀ pA[:3]`, `pa = pA + UD_s ul_s`, `pA ← Xᵀ(r_s) pa`. At a floating root,
  `a_0 = −IA⁻¹ pA`.
* **Down-sweep along the path:** `a'_s = X'_s a_{s−1}`, `u̇_s = D⁻¹(ul_s − W_s a'_s)`,
  `a_s = a'_s + S'_s u̇_s` (the modified accelerations of B1's implicit maps). In parallel we
  accumulate the **plain velocity about the world origin**,
  `Vp_s = Vp_{s−1} + [w_s; o_s × w_s]` with `w_s = R_s u̇_s`. That prefix sum is exactly the
  contribution of path joints ≤ s to any point velocity: `d·v(p) = [(p×d)ᵀ, dᵀ] · Vp`.
* **Pulled-back row:** off the path of the column's contact, the bias is zero, so
  `u̇_s = −D⁻¹W X'_s a_{s−1}` and `a_s = Ψ_s a_{s−1}`, with `Ψ = (I − S'D⁻¹W)X'`. For contact c's
  rows `Gw_c = [(p_c×d)ᵀ, dᵀ]` this gives the backward recursion
  `ρ_c(e) = −Gw_c Λ_{t_{e+1}} + ρ_c(e+1) Ψ_{t_{e+1}}`, `ρ_c(d_c) = 0`, `Λ = Ŝ D⁻¹W X'`, `Ŝ = [R; [o]×R]`.
* **Block (c, c′):** `W_cc′ = Gw_c Vp_c′(e) + ρ_c(e) a_c′(e)`, with e the depth of the lowest common
  ancestor of b_c and b_c′. The LCA comes from the new ancestor table `TreeTopology.anc`: e is the
  number of equal leading entries minus 1. Two contacts in different root subtrees do not couple.

Cost: O(k·depth) 6×6/6×3 products, plus O(k²) 3×6·6×3 blocks (assembled in row chunks, so memory
is O(B·chunk·k)), plus O(nb) for Ψ, Λ on path bodies.
It is **exact**: it equals 3k brute-force columns (each one O(n) `response` sweep) to 1e-15
relative, for clamped and floating trees, with all of B1's implicit terms (test).

`W` is not symmetric in motion. B1's linearised-implicit velocity-product terms (`X'`, `S'`, the
gyroscopic body Jacobian) make `M~` non-symmetric by O(hω). For whipping leaflets (hω ≈ 1) the
measured asymmetry is 40–100%, and it vanishes at rest. We keep the exact operator. Using its
symmetric part would make the predicted contact velocity inconsistent with the dynamics that is
actually integrated.

## 4. Contact geometry

| plant feature | geometry | ground | spheres / capsules |
|---|---|---|---|
| rod segment (body) | capsule o→tip, radius r (mean of node radii) | distal end sphere (+ proximal end of a floating base) | segment–segment closest points (mid-segment contact: stem on a stake) |
| leaf vertex | point, radius `leaf_thickness/2` (0.25 mm) | ✓ | point–segment |
| fruit | sphere at the fruit node, fruit radius | ✓ | ✓ |

Colliders (`contact.py`):
* `GroundPlane(height, normal, mu)`, a half-space.
* `SphereObstacle(center, radius, velocity, mu)` and `CapsuleObstacle(a, b, radius, velocity, mu)`:
  kinematic, translating with the given velocity. They stand in for robot fingertips and gripper jaws.
* `stake(x, y, radius, height, mu)`: a static vertical capsule. By default it touches the main
  stem and fruit only (`organs=(0,)`, `leaves=False`), because a vertical line through a bushy
  Helios canopy always intersects some petioles initially, and real leaves pass around a stake.
* Every collider takes a feature filter (`organs`, `leaves`, `fruit`, `bodies`, e.g.
  `contact.main_axis(sim)`) and `mode="implicit" | "penalty"`.
* `TreeSim(colliders=[...])` makes colliders persistent. `step(h, external=[...])` takes per-step
  kinematic colliders, which are moved by the caller.

Contact rules:
* **Speculative contacts.** Candidates are features with gap φ < `min(margin + 3h|v_point|, 2 cm)`
  (2 mm + 3 steps of travel, capped so that whipping leaflets do not flood the set). At most
  `max_contacts` = 512 per member are kept, smallest gaps first. The normal target is `v_n ≥ −φ/h` for φ ≥ 0: an approaching point
  stops exactly at the surface, nothing tunnels, and a speculative contact that is not approaching
  stays inactive.
* **Penetration recovery.** For φ < 0 the target is `v_n ≥ min(β(−φ)/h, 0.2 m/s)`, with β = 0.5.
* **Deep penetration in a curved collider** (φ < −5 mm, e.g. a sphere spawned inside the canopy)
  recovers at only 2 cm/s. A body inside a sphere has contacts with opposing normals, and
  separating all of them at once is infeasible. Half-spaces always recover fully, since all
  their normals are identical.
* **Contact reduction is off by default** (`max_per_body=0`). It is optional: per feature group
  (one rod capsule, one leaflet, one fruit) and collider it keeps the deepest point plus the
  extremes along ±t1, ±t2 and the diagonals (hull corners of a rigid patch). On the tomato it saves
  little (61 vs 53 contacts at rest) and costs accuracy. Helios leaflets are curved, so with the
  hull corners supported an interior vertex can still sink while the leaflet rotates: late
  penetration was 1.3 mm with 5 per leaflet, 0.3 mm with 9, and 0.005 mm without reduction. Use
  it only for scenes with very many contacts. Group per feature, not per host body: 2–4 leaflets
  share a petiole body.

## 5. Friction model and the PGS solve

**Choice: an isotropic Coulomb cone, non-associated, with an exact local solve per contact.**
Alternatives considered:

| | pyramid (n facets) | CCP cone (Anitescu–Tasora, convex) | **disk, non-associated (chosen)** |
|---|---|---|---|
| isotropy | capacity varies with slip direction, by up to 1/cos(π/n) (4 facets: √2) | ✓ | ✓ |
| stick/slip threshold at tan φ = μ | only along a facet normal | ✓ | ✓ |
| normal velocity while sliding | 0 | **μ\|v_t\| separation** (a sliding block hops) | 0 |
| convex | LCP | yes (cone QP) | no (NCP; PGS fixed point) |

The incline test needs the threshold within 2% in an arbitrary direction, and leaves slide on the
ground and on fingers. That rules out the pyramid (anisotropy) and the CCP cone (dilation).

**The local tangential step must be solved in the W_tt metric.** The common shortcut
`λ_t ← proj_disk(λ_t − W_tt⁻¹ v_t)` (unconstrained 2×2 solve, then radial clamp) is a Euclidean
projection. At the friction limit its fixed point gives λ_t antiparallel to W_tt⁻¹v_t, not to v_t.
For a rigid body's anisotropic local W_tt that is not Coulomb. In the block-on-incline test PGS then
converged (to 1e-14) to a **spurious sliding solution at tan φ = 0.98μ**: the block yawed and slid
at 0.46 m/s² with Ft/Fn = 0.44 < μ. We now solve the local problem

    min ½ λᵀ W_tt λ + bᵀ λ   s.t.  |λ| ≤ μ λ_n        (b = v_t − W_tt λ_t,old)

exactly. Interior gives λ = −W_tt⁻¹b (stick, v_t = 0). On the boundary the KKT condition
`v_t = −γλ`, γ ≥ 0, is maximal dissipation, with λ = −(W_s + γI)⁻¹b and γ from the secular
equation |λ(γ)| = μλ_n (Moré–Sorensen: Newton on 1/|λ| − 1/μλ_n, 12 iterations). With this, the
block sticks at 0.98μ (2e-11 m in 2 s) and slides at 1.02μ with a = g(sin φ − μ cos φ) to 1e-14.

**PGS.** For each contact, in turn: a normal projection
`λ_n ← clamp(λ_n + (b_n − v_n)/W_nn, 0, λ_cap)`, then the local disk QP, with `v` updated
incrementally (v += W[:, c] Δλ_c). It is batched in lock-step over B with per-member convergence.
Details:
* **Stopping rule:** a sweep changes no contact velocity by more than 1e-6 m/s (|Δλ|·W_ii). This
  is the physical criterion: a relative impulse tolerance never triggers on redundant contacts
  (stiff chain: 11 contacts, rank 6).
* **Warm start** from the previous step by a stable key (collider, feature). The impulse is stored
  as a world vector and re-projected onto the new frame. At rest PGS needs 1–2 sweeps.
* **Safeguards against inconsistent or non-converged problems.** There are two:
  * a per-contact cap `λ_n ≤ 100 (|v_free,n| + |b_n| + 1 mm/s)/W_nn`;
  * a step-level velocity guard: all impulses are scaled by α ∈ (0, 1] so that no contact
    velocity exceeds `max(|v_free,i|, 2(max|v_free| + max b) + 1 cm/s)`. v is linear in α, so α
    has a closed form per component.

  A converged, consistent solve has α = 1 and never meets the cap: every acceptance case does.
  Without the guards, a sphere spawned 4 cm deep inside the canopy diverged within 3–12 steps. The
  impulses spun the leaflets up, which made W non-symmetric and indefinite. The per-contact cap
  alone is not enough, because one contact's impulse reaches other contacts through
  off-diagonal, non-symmetric W.
* **Backend:** Gauss–Seidel is sequential over contacts, so numpy pays Python overhead per contact
  per sweep (≈ 0.8 s/step for 45 tomato contacts). `contact_wp.py` runs the identical algorithm as
  a Warp kernel, one thread per batch member, in float64. It matches numpy to 4e-17 and is about
  400× faster. `ContactParams(backend="auto"|"warp"|"numpy")` selects it, and numpy is the
  reference and fallback.

Convergence in practice: at rest and in quasi-static manipulation (finger, stake) PGS converges in
1–200 sweeps. In violent transients (leaflets at 1–2 m/s hitting the ground, hω ≈ 1, non-symmetric
W, redundant contacts on one leaflet) it can cycle and hit the cap (200 sweeps). Each step still
stays bounded, and the residual is removed by the next steps' targets. The measured impact
penetration on the E = 1e8 tomato fall is 1.1 mm (2 m/s leaflets), and < 0.001 mm once landed.

## 6. Contact-aware inverse statics (stretch goal)

A clamped tree under dead loads is statically determinate (B2). A tree resting on supports is not.
`inverse_contact.supported_inverse` proceeds as follows:

1. **Candidates:** features within `tol` of the colliders in the target shape, with their frames.
2. **Admissible forces:** non-negative combinations of the edges of an *inscribed* friction pyramid
   (8 edges), so every admissible force lies inside the Coulomb cone.
3. **Choice of forces:** the joint moments are linear in the edge weights β,
   `m = m_gravity + Gβ`. We choose β ≥ 0 minimising the **rest-shape prestress**
   `Σ_j m_jᵀ K_j⁻¹ m_j` (the first-order elastic energy of the rest deviation), which is an NNLS
   problem. This is the regularisation that makes the indeterminate problem well posed, in the
   spirit of Derouet-Jourdan 2013's choice of the "most natural" rest shape.
4. **Rest shape:** B2's exact inverse with gravity plus the chosen forces as dead `BodyForce`
   loads. The target is then an exact equilibrium with these reactions.
5. **Stability under the support:** the dense Hessian, including the load stiffness of the support
   forces, is projected on the null space of the active contact rows (normal; with friction also
   the tangential rows of sticking contacts). The collider's surface curvature is neglected.

**Known support forces** (the minimum the brief asks for) are B2 unchanged, with `BodyForce` dead
loads: `inverse_statics(sim, target, external=[BodyForce(body, local_point, force), ...])`.
`hessian.subtree_loads`/`potential` and the forward step support `BodyForce`.

## 7. Floating bases (needed for "dropped" bodies)

An unclamped root node with exactly one child edge is a **floating base**: a 6-DOF root body.
There is no joint spring (K = 0) and Q is its absolute orientation deviation. Its state adds
`p_float`, `v_float` (B, nf, 3). The ABA root solve is `a = −IA⁻¹ pA`, with `u̇ = Rᵀω̇`, and the
update is `v += h a_lin`, `p += h v`. All of B1's implicit terms apply unchanged. Statics/inverse
(B2) refuse floating trees (`NotImplementedError`), since a free body has no clamped
equilibrium to invert. This gives the dropped chain and the block on the incline. Later it will
give detached fruit.

## 8. Segment refinement

`refine.refine_asset(asset, max_length=…, min_per_organ=…, factor=…)` subdivides edges. Original
nodes keep their indices, and new nodes are appended on the straight edge, with radius
interpolated, organ copied and `fixed` only if both ends are fixed. So leaves, fruit, point masses
and every caller-held node index stay attached at the same node (tested). `organ_runs` defines a
"per organ" chain (same organ, between branch points).

Convergence of the inverse critical modulus (B2's `critical_modulus`, exact Sturm bisection):

| test structure | observed order |
|---|---|
| kinked chain (trunk + one inclined branch + tip mass) | **2.00** (O(1/m²), like B1's beams) |
| Y junction (two branches) | 0.85 → 0.93 → 1 (**O(1/m)**) |
| Helios tomato | ≈ 1.1 (dominated by its ~190 junctions) |

Finding: B1's joint compliance at a junction counts the parent's last half segment once per child
(series sum `l_p/2/EI_p + l_c/2/EI_c`). That is exact on chains and O(l) wrong at branch points,
where the parent half segment is physically shared. Refinement removes it at O(1/m). Tomato
numbers: see `migration/notes/B3.md`. E_cr rises from 7.32e8 (unrefined) to 8.57e8 (×8), with an
extrapolated 8.7e8.

## 9. Data layout notes for the Warp port (B4)

* The per-step factorisation should stay resident: `UD (6×3)`, `W (3×6)`, `D⁻¹ (3×3)`, `X' (6×6)`,
  `S' (6×3)` per body (B, nb). B1 discarded them after the step, and contact needs them twice
  (Delassus, back-substitution).
* `X'` is a pure shift plus O(h) blocks. Storing `r` plus the two 3×3 h-blocks instead of the full
  6×6 saves 60% of the memory traffic.
* Contacts form a ragged list per batch member. The prototype pads to (B, K_max). On GPU, use
  compacted contact arrays with per-member offsets (prefix sum after detection), and a
  per-member dense W block in shared memory for K ≤ ~64 (larger K: global memory).
* The path quantities `(UL, A, Vp, ρ)` are (contact, depth) arrays: one thread per contact walks
  its path (depth ≤ 27 unrefined, 216 at ×8). The k² block assembly is one thread per (c, c′) pair,
  using the ancestor table `anc[nb, nlev]` (int32) for the LCA.
* PGS is one thread (or one warp, with the column update v += W[:, c]Δλ parallel over rows) per
  batch member. The kernel in `contact_wp.py` is already in that form.
* Detection: per feature × collider, embarrassingly parallel. Reduction per feature group should be
  a segmented argmax over (member, group, collider) keys, not a lexsort.
* Keep float64 for W and PGS. The leaf/stem W_nn ratio reaches 10⁴ on the tomato.

## 10. Limitations

1. No self-contact (leaf–leaf, leaf–stem) and no plant–plant contact. Both use the same machinery,
   but they need a broad phase and two-body Jacobians (rows on two paths).
2. Leaves are rigid point clouds (vertices only). A collider smaller than the vertex spacing
   (~1 cm) can pass between vertices. Edge/face contact is not implemented.
3. PGS can cycle in violent transients (§5). An exact Newton-type solver (semismooth / interior
   point on k) would be needed for a guaranteed residual.
4. Kinematic colliders translate only (no angular velocity).
5. Supported inverse: dense stability check (O(n³)), frictionless-surface-curvature approximation,
   prestress-minimal force choice (other criteria are possible), batch member 0 only.
6. Floating bases cannot be inverted (B2) and must have a single child edge.

## 11. GPU: B4's contact slot

The CPU solve is written as the five stages B4's slot expects (`contact.py`):
`prepare_contacts` (detect), `free_velocities` (free motion), the k×k operator, `solve_impulses`
(PGS with targets, warm start and safeguards), and `contact_body_forces` (apply).
`heliades/sim/gpu/contact.py::ContactSlotSolver` runs the same stages inside
`TreeSimGPU.step` (`sim.contact = solver`, between the factorisation and the integration sweep):

| stage | CPU (`TreeSim`) | GPU slot (`ContactSlotSolver`) |
|---|---|---|
| 1 detect | `prepare_contacts` at q_n | the same, on a float64 host mirror of the device kinematics |
| 2 free motion | contact-free `u̇` of the step | `slot_free_motion` |
| 3 operator | path formula, O(k·depth + k²) | 3·K_max `factor_response_device` columns (Baraff O(kn)), each batched over all envs |
| 4 PGS | `solve_impulses` (Warp CPU kernel) | the same |
| 5 apply | `TreeSim.response(−F)` | `slot_apply(F)` |

Agreement (test `test_gpu_contact_slot_matches_cpu`): finger push, 0.9 s, fp64 **7e-16 m** and fp32
**3e-7 m**; tomato lying on the ground, fp32 **3.5e-7 m**, with the same contacts and the reaction
equal to 4 digits.

**Current status: host-synchronous.** Every step reads device buffers (Q, u, response columns),
so the solver needs `TreeSimGPU(..., use_graph=False)`. It costs about 0.25 s per step for 63
contacts, because each of the 189 response columns is 2·nlev launches plus a host sync. It is a
correctness reference, not a fast path. A graph-resident version needs:

1. A **detection kernel**: per (feature, collider) gap and normal, compacted into fixed-capacity
   per-env slots (`k_max`) with atomic counters, and a stable key per slot for warm start.
2. A **device free-velocity kernel**: plain velocity propagation of `u + h u̇_free` to the contact
   points (one level sweep), not the modified accelerations `acc0/acc1`.
3. **The operator.** Either k_max response solves batched as extra right-hand sides (a
   `factor_response` variant over (n, k_max) buffers, giving one launch per level instead of k),
   or the path formula of §3 as kernels. The path formula is cheaper for deep, wide trees: one
   thread per contact walks its root path with the stored `UD, W, D⁻¹, X', S'` (the B4 buffers
   W0/W1, UD0/UD1, Dinv, Xa/Xc, Smt), and one thread per (c, c′) pair does the LCA gather.
4. **PGS** as one block per env on the dense k×k W in shared memory (k ≤ 64), including the disk
   QP and the safeguards. This is `contact_wp._pgs_kernel` with the column update parallelised
   over the block.
5. **Apply:** a kernel that scatters `[(p−o)×f; f]` into df0/df1 (atomics per body), then
   `slot_apply`. This step is already graph-safe.
