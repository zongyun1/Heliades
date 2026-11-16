# heliades.sim B2: exact discrete inverse statics, with a stability check

Status: prototype (CPU, numpy, batched). Code: `heliades/sim/inverse.py` (inverse, critical modulus,
fallback) and `heliades/sim/hessian.py` (subtree loads, gradient, tangent stiffness, O(n) tree
factorisation, Sturm counts). Tests: `tests/heliades/test_sim_inverse.py`. Results:
`migration/notes/B2.md`. Demo: `scripts/inverse_statics_demo.py` → `outputs/sim/tomato_inverse_hold.mp4`.

## 1. Problem

Helios, or any reconstruction from photos, gives the plant in its **loaded** shape, already in
equilibrium under gravity. If we use that shape as the rest shape, the forward simulator counts
gravity twice. B1 measured 1–45% extra sag on the Helios tomato, and the leaflets droop by
15–33 cm. We want the **rest shape** for which the given shape is the equilibrium *of the
discrete forward model B1 actually integrates*, and we want to know whether that equilibrium is
**stable**.

## 2. The forward model being inverted (B1 recap)

Bodies are rigid segments, and each has a spherical joint at its proximal node. Joint j carries
the deviation quaternion Q_j = (w, v) from its rest relative rotation:
R_rel,j = R0rel,j R(Q_j). The potential is

    V(Q) = Σ_j 2 vⱼᵀ Kⱼ vⱼ  −  Σ_loads F · p          (K = diag(k_b, k_b, k_t), dead loads F)

The loads are the gravity of segments, rigid leaf payloads and fruit point masses (plus optional
dead `PointForce`s). With right perturbations Q_j ← Q_j exp(δ_j) (the same chart as B1's
velocities u_j), the generalized gradient is

    g_j = 2 Γ_j K_j v_j − R_jᵀ N_j,        Γ = w I − [v]×,
    N_j = Σ_{loads in subtree(j)} (p − o_j) × F       (moment about joint j's origin)

B1's step has the fixed point u = 0, a = 0 iff τ_el + Sᵀ f_ext = −g = 0. The ABA bias terms
vanish at u = 0, and all of B1's implicit terms (armature, K_g⁺, damping) multiply velocities or
accelerations. So **{g(Q) = 0} is exactly the set of static equilibria of the time stepper,
independent of h.**

## 3. Inverse statics (exact, O(n))

A root-clamped tree without contact is **statically determinate**. N_j depends only on the
configuration (positions of the loads in subtree(j) and of o_j), never on the stiffness or the
rest shape. The algorithm:

1. **Target kinematics.** Take the target relative rotations R_rel,j: by default the asset's own
   parallel-transport frames, which is the Helios shape, or `target_from_state(sim)`. One
   root-to-leaf sweep gives R_j, o_j.
2. **Subtree loads.** One leaf-to-root segmented sum (the same `reduceat` pattern as B1's pass 2)
   of the resultant F and the world load matrix W = Σ p Fᵀ. Then L_j = W_j − o_j F_jᵀ, the moment
   N_j = vee(L_jᵀ − L_j), and A_j = L_j − tr(L_j) I (used in §4).
3. **Local joint solve.** Per joint, independently and vectorised over (B, nb): find the unit
   quaternion Q_j with

       2 Γ(Q_j) K_j v(Q_j) = m_j := R_jᵀ N_j

   by damped 3×3 Newton in v (w = √(1−|v|²)), starting from the linear guess v = K⁻¹m/2. We keep
   the branch connected to Q = 1, checked by det H_el > 0 (§4). If Newton lands on another branch,
   we re-solve by continuation m → t·m.
4. **Rest rotation.** R0rel_j ← R_rel,target,j · R(Q_j)ᵀ, per batch member (the rest depends on
   E). `apply_rest(sim, res)` calls the new `TreeSim.set_rest(R0rel)` and `reset(Q = Q_dev)`, so the
   simulator starts exactly at the target.

**Why it is exact for the discrete model.** Step 3 enforces g_j = 0 *joint by joint*, using the
same quaternion energy, the same Γ, the same series joint compliance and the same mass parts
(segments, leaves, fruit; `TreeTopology.*_part`) that the forward step uses. The equilibrium
equations decouple because m_j is independent of the unknowns. The only errors are floating
point and the 3×3 Newton tolerance (1e-15 relative). Measured: the residual of g at the
target is ≤ 2e-16 relative, and a forward step from the target gives |u̇| ~ 1e-10 rad/s². This
is "exact" with respect to the discrete model, not the continuum rod: the discrete model is
O(1/n²) from Euler–Bernoulli (B1 tests 1–3). The inverse does not add any error on top of that.

**Rest rotations carry twist.** An asymmetric load (a leaflet off to the side) needs torsional
pre-twist in the rest shape. Node positions plus parallel transport cannot express that, which is
why B2 returns rest *rotations* and why `TreeSim` gained a per-batch `R0rel`. Body frames, payload
geometry and stiffness are unchanged; only the zero of each joint spring moves.

**Existence: the spring's capacity.** The quaternion spring's torque saturates. For isotropic K
it is |m| = k sin θ, with maximum k at θ = π/2. A joint whose required moment exceeds the capacity
has *no* rest rotation, whatever the rest of the plant does. The solver flags such joints as
`joint_ok = False`, and the batch member is reported as `status = "infeasible"`. Part of this is a
discretisation effect (B1 §5.5: per-joint angles > 0.5 rad mean the rod should be refined), and
part of it is physical (a flaccid petiole cannot hold a leaf up).

## 4. Stability: exact tangent stiffness and an O(n) Sturm count

An inverse equilibrium may exist and still be unstable: an upright stem is a column under
compression. We need the joint-space Hessian *including the geometric (gravity) stiffness*.

### 4.1 The Hessian

Differentiate g. Variables are δ_j (child frame), or world e_j = R_j δ_j.

* Elastic, joint-local: H_el = Γ K Γᵀ + [K v]× Γᵀ − (K v) vᵀ (= K at v = 0).
* Loads: a perturbation of an ancestor-or-self k rotates subtree(j) rigidly about o_k, which gives
  d(R_jᵀN_j) = R_jᵀ([N_j]× + A_j) e_k = R_jᵀ A_jᵀ e_k. A perturbation of a strict descendant k
  rotates only subtree(k), which gives dN_j = A_k e_k. In world variables:

        diagonal       D_j     = R_j H_el,j R_jᵀ − A_jᵀ
        j below k      ∂g_j/∂e_k = −A_jᵀ,      ∂g_k/∂e_j = −A_j      (k a strict ancestor of j)

  A_j = L_j − tr(L_j) I. For pure gravity that is h_j gᵀ − (g·h_j) I, with h_j the subtree first
  moment about o_j.

The off-diagonal coupling is symmetric everywhere. The diagonal blocks are symmetric up to terms
proportional to the joint's own gradient, so **at an equilibrium H is exactly symmetric**. Verified
against central finite differences of g (error 1e-10, test 1). The negative part of −sym(A_j) is
the buckling (P-δ) stiffness of an upright subtree, and the positive part is the pendulum
stiffness of hanging leaves.

### 4.2 Ancestor structure → O(n) block LDLᵀ

Row j couples to all its ancestors only through **one 3-vector**, the accumulated rotation of
its parent Ω_p(j) = Σ_{k ∈ anc(j)} e_k. It couples to its descendants only through Σ A_k e_k. This
is the ancestor structure of an articulated-body mass matrix. Eliminating leaves first leaves a
3×3 "articulated stiffness" Φ per subtree:

    P_j = D_j − Φ̄_j,     C_j = A_jᵀ + Φ̄_j,     E_j = A_j + Φ̄_j,     Φ̄_j = Σ_children Φ_c
    Φ_j = Φ̄_j + E_j P_j⁻¹ C_j                            (leaf → root, segmented sum)
    solve:  r_j = b_j + φ̄_j,  φ_j = φ̄_j + E_j P_j⁻¹ r_j;   x_j = P_j⁻¹ (r_j + C_j Ω_p),  Ω_j = Ω_p + x_j

(`hessian.eliminate`, `tree_solve`, `tree_matvec`). Each joint costs a constant number of 3×3
products and one 3×3 inverse. There are 27 levels on the tomato, and everything is vectorised
over (B, level width), so this is the same Warp-portable pattern as B1.

* **Stability = Sylvester inertia.** Elimination is a congruence, so by Haynsworth's inertia
  additivity **#negative eigenvalues of H = Σ_j #negative eigenvalues of the 3×3 pivots P_j**.
  Exact, O(n), no eigen-solver. It agrees with dense `eigvalsh` (test 1).
* **Sturm bisection.** The pivots of H − σM count the eigenvalues below σ. M = I gives λ_min(H).
  M = K_el = blkdiag(R K Rᵀ) gives the smallest *generalized* eigenvalue μ_min of the pencil
  (H, K_el). A shift that hits an eigenvalue exactly is perturbed to +tiny, the usual Sturm
  convention. We bisect to 1e-10 relative. Dense `eigh` is kept as a cross-check for moderate n.
* **Localisation.** The pivots of subtree(j) are the inertia of the principal submatrix of
  subtree(j). So `joint_negative[j] > 0`, with no negative pivots below it, means "subtree(j),
  clamped at its base, is unstable while all its child subtrees are stable". This is an exact
  per-subtree criterion.
* **Newton statics for free.** The nonsymmetric version (`symmetric=False`) solves dg/dδ · δ = −g
  exactly in O(n). `newton_static` polishes B1's pseudo-time `solve_static` to round-off in 2–3
  iterations. The round-trip tests need this, because 1e-8 rest accuracy requires a forward
  equilibrium converged far beyond `solve_static`'s 1e-6.

### 4.3 Critical modulus

K_el ∝ E. Everything else in H at the target depends on E only through the (small) inverse
deviations v.

* **Linearised (generalized eigenproblem).** Treat H − K_el as E-independent. Then
  H(E') = (E'/E) K_el + (H − K_el), and **E_cr ≈ E (1 − μ_min)**. This is K_el φ = −K_geom φ in
  the brief's notation, and it is exact when the joints are unloaded (v = 0, e.g. a straight
  column).
* **Exact.** Bisect on E for the smallest E at which the inverse rest exists and its Hessian has
  no negative pivot (`critical_modulus`). Each probe is one O(n) inverse plus one O(n) Sturm count,
  with 16 moduli per round as one batch. Five rounds give 1e-4 relative. It takes about 1 s for
  the tomato.

### 4.4 When unstable: what is returned, and the fallback

`InverseResult.status` is `"stable"`, `"unstable"` (with `n_negative` and `joint_negative`) or
`"infeasible"` (`joint_ok` names the joints that exceed capacity). The rest rotations are
returned anyway. An unstable result is still an exact equilibrium, and it holds the target for
seconds without a disturbance (§6), so **the hold test alone cannot detect instability; the
eigen-check is required.**

Fallback (`stiffen_to_stable`, documented choice): **greedy stiffening of minimal unstable
subtrees.** Take the deepest joint j that has a negative pivot or is over capacity. Its clamped
subtree is unstable while its child subtrees are stable. Find, by Sturm-count multisection, the
smallest uniform factor f on subtree(j)'s joint stiffness that removes every negative pivot and
over-capacity joint inside it. Apply f·(1 + 2%) and repeat upward. The output is a per-joint
stiffness multiplier s_j ≥ 1, equivalently a required modulus E·s_j per joint. Stiffening a subtree
changes only its pivots and its ancestors' pivots, so earlier fixes stay fixed. This is
"minimal" per subtree, not a global optimum. A convex alternative, minimising Σ log s subject to
H ⪰ 0 (an SDP), is left for later. The other report is the global E_cr, which is the
uniform-stiffening answer.

## 5. Comparison with prior inverse statics (survey §4)

| | problem | global stage | local stage | stability |
|---|---|---|---|---|
| Twigg & Kačić-Alesić 2011 | sag-free rest parameters for mass-spring / rods | nonlinear least squares over rest lengths/angles | (inside the optimisation) | not addressed |
| Derouet-Jourdan 2010 / 2013 | super-helix hair, with frictional contact (2013) | underdetermined; regularised toward uniform parameters | per-element | not addressed |
| Bertails-Descoubes 2018 | one clamped hanging Kirchhoff rod | none: natural shape unique, 3 linear IVPs | integrate | hanging = tension |
| Hsu 2022 | general sag-free (rods, cloth, FEM) | global linear solve for equilibrium forces (indeterminate) | per-element rest recovery | not addressed |
| Hsu 2023 | sag-free hair (DER) | equilibrium forces | modified Darboux vectors (rest curvature/twist) per edge | hair hangs |
| Takahashi & Batty 2024/25 | DER rest shape making net force exactly zero, optionally with stiffness | active-set Cholesky optimisation | (joint) | single strands, no branching |
| **B2 (this)** | **branched plant tree with leaf/fruit payloads, reduced-coordinate quaternion joints** | **none needed: statically determinate, one O(n) leaf→root sum** | **3×3 Newton per joint on the simulator's own quaternion energy** | **exact O(n) Sturm count of the full Hessian incl. gravity stiffness; critical E; localisation; fallback** |

B2 is closest to Hsu 2023 / Takahashi & Batty in spirit: force balance, then local recovery of
rest curvature and twist, exact with respect to the discrete energy. It differs in three ways.

1. **Determinate trees remove the global solve.** Hsu's global stage exists because cloth and
   FEM (and rods with constraints in maximal coordinates) are statically indeterminate. In reduced
   coordinates a clamped tree's joint moments are explicit subtree sums. This is the discrete,
   branched, payload-carrying generalisation of Bertails-Descoubes' uniqueness result for one rod.
2. **The local stage inverts the forward model's own nonlinear spring.** It is not an
   approximation of it. That is why round trips close to 1e-12 rather than to a discretisation
   tolerance. It also exposes a failure mode the linear-curvature formulations do not have: the
   quaternion spring's capacity (§3).
3. **Compressive, buckling-prone stems: the plant-specific part.** Hair hangs. Every hair strand
   is in tension, so gravity's geometric stiffness is *positive* and any equilibrium the inverse
   produces is stable, which is why stability never comes up in those papers. A plant stands up.
   The main axis is a column under self-weight (Greenhill), side shoots are inclined cantilevers
   under end loads, and petioles that hold leaflets *above* the joint are spring-held inverted
   pendula. For those, the inverse gives an exact equilibrium that can be **unstable**, or, below
   the spring capacity, no equilibrium at all. On the Helios tomato this is not a corner case.
   It decides the result: at E = 5e8 the exact inverse has 11 unstable modes, and at 3e8 it is
   infeasible (§6). The O(n) Sturm count makes the check essentially free (≈1.5 ms, about half of
   the inverse's cost), so it can sit inside a Real2Sim fitting loop as a constraint
   (E ≥ E_cr(target)).

## 6. Results in brief (details: migration/notes/B2.md)

* Round trip rest → static → inverse: rest recovered to ≤ 4e-15 (cantilevers, 4 cases) and
  ≤ 1.5e-12 (tomato at 5e8 and 1e9).
* Hold, tomato `seed=1`:
  * E = 1e9: stable, drift 5e-16 m. Kicked with 1e-3 rad/s it drifts 1.4e-7 m.
  * E = 5e8: **unstable** (11 modes in petioles and side shoots). It holds to 1e-12 m unperturbed,
    but the same kick sends it 0.43 m away.
  * E = 3e8: **infeasible** (10 joints over capacity). It collapses (0.63 m).
  * Forward-only at 1e9 drifts 0.25 m.
* Greenhill: E_cr within +0.028% at n = 40 (O(1/n²)). Exact and linearised critical moduli
  coincide.
* Tomato E_cr = **7.32e8 Pa** (exact; linearised 7.32e8). With petioles made rigid it is 3.36e8.
  For the main axis alone it is **2.38e8**, consistent with B1's ≈2e8 sag-amplification estimate.
* Time: 3.3 ms per inverse with stability at B = 1, 120 ms at B = 64 (1.9 ms per member).

## 7. Limitations

1. Dead loads only (gravity, fixed point forces). No contact/ground in the inverse: a contact
   inverse is indeterminate (Derouet-Jourdan 2013) and needs B1 §7's Schur machinery.
2. Leaves and fruit are rigid payloads. A leaf shell is internally indeterminate (Hsu-style
   global solve inside the leaf), coupled back through the petiole moment.
3. The answer is exact for the *discrete* model. Coarse petioles (2 segments with ~1 rad per
   joint) make both feasibility and stability discretisation-sensitive. Refine asset edges before
   drawing biological conclusions from E_cr.
4. The fallback is greedy per subtree, not a global minimum-stiffening optimum.
5. Rest rotations are per batch member (they depend on E). There is no single rest *asset*
   geometry, because the rest carries twist that node positions cannot express.
6. The dense eigen-path (`min_eig_method="dense"`) is O(n³) and meant for checks. Sturm bisection
   is the production path.
