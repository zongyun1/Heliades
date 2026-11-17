# heliades.sim.gpu (B4): the tree solver and inverse statics on the GPU (NVIDIA Warp)

Status: working, tested. Code: `heliades/sim/gpu/` (`sim.py` TreeSimGPU, `kernels.py`, `layout.py`,
`inverse.py`, `contact.py`). Tests: `tests/sim/test_sim_gpu.py` (19 tests, ~12 s with a warm
kernel cache, skipped without CUDA). Benchmark: `scripts/bench/gpu_benchmark.py`. Results:
`migration/notes/archive/B4.md`. Hardware used: RTX 5060 Ti 16 GB (sm_120), Warp 1.19.

This is a 1:1 port of B1 (`TreeSim`, docs/sim/DESIGN.md) and B2 (`inverse_statics` + Sturm count,
docs/sim/INVERSE.md). The model, the linearly-implicit step, and the fixed point are unchanged.
In float64 the GPU reproduces the CPU solver to 1e-16 to 1e-15 over 3 s rollouts.

## 1. API

```python
from heliades.sim.gpu import TreeSimGPU, inverse_statics_gpu, critical_modulus_gpu
sim = TreeSimGPU(asset, [phys_1, ..., phys_B], precision="float32")   # or "float64"
sim.step(h, external=[PointForce(...), SphereCollider(...)] | (B,N,3) ndarray | wp.array, damping=None)
sim.settle(...); sim.solve_static(...); sim.run(...); sim.set_rest(R0rel); sim.reset(Q, u)
sim.state  # SimState (host copy), sim.leaf_vertices(), sim.point_speeds(), sim.energy()
sim.Q, sim.u, sim.K   # host views with setters (upload + forward kinematics)
res = inverse_statics_gpu(sim, target_Rrel=None, external=None, min_eig=False, full=True)
apply_rest(sim, res)  # heliades.sim.inverse.apply_rest works unchanged
TreeSimGPU([asset_a, asset_b, ...], [phys_a, phys_b, ...])            # heterogeneous batch
sim.env_state(e)
```

The constructor arguments, `step`, `settle`, `solve_static`, `run`, `set_rest`, `reset`, `state`,
`energy` and `point_speeds` have TreeSim's signatures and semantics. Additions:

* `precision`, `device`, `use_graph`.
* `settle(check_every=8)` and `solve_static(check_every=8)`: the convergence test runs on the device.
* `step(return_udot=False)`: returning udot would force a host sync.
* `factorize`, `factor_response`, and the `slot_*` primitives for the contact slot (§6).

## 2. Data layout: a flattened multi-tree

All environments share one **global body index** g. Bodies are sorted by
(depth level, rank inside the level, environment) (`layout.py`). This ordering gives four properties:

* Level d is one contiguous range `[L[d], L[d+1])` over the whole batch. One kernel launch per
  level has launch dim = (bodies at depth d summed over all envs), and g = L[d] + tid.
* For a **shared topology** the ordering reduces to the body-major layout `g = i*B + e`. Adjacent
  threads are the same body in adjacent environments, so every per-body load is coalesced.
* **Different topologies** use the identical code path, with no padding: a level just holds
  a different number of bodies from each env, and an env with fewer levels drops out of the
  deeper launches. This is implemented and tested: tomato, cantilever, tomato and an inclined
  cantilever in one batch match four single CPU runs to 1e-16.
* Children are listed in a CSR (`ch_start`, `ch_count`, `ch_list`). The child-to-parent
  accumulation of the up sweep is a per-parent loop, a segmented sum with **no atomics**. Results
  are therefore deterministic and batch-invariant: env i of a batch is bitwise equal to a single
  run, in both precisions.

Per-body device arrays are SoA arrays of `vec3`/`mat33`/`vec4`. Every per-env parameter (K, mass
moments, R0rel) is stored per global body. `R0rel` is per body because B2's rest rotations
depend on E. Contact sample points (distal node + leaf vertices, 10,416 on the tomato) are one
per-topology CSR shared by all envs, so the leaf meshes are not replicated B times.

Memory per env for the tomato, all buffers: **387 kB in float32, 758 kB in float64**. On a 16 GB
card that is about 40k envs in fp32 and about 20k in fp64. B = 16384 in fp32 uses 6.3 GB.

## 3. Kernels and the step

Each step is 2 × nlev launches (54 for the tomato), plus one force-scatter launch when
point forces are present:

| kernel | launch | work per thread |
|---|---|---|
| `aba_up` (level d, leaf→root) | (bodies in level d) | **bulk work fused**: spatial inertia, bias, gravity, mass damping, implicit gyroscopic terms, gravity K_g⁺, point force, sphere penalties (loop over the body's contact points), joint spring/armature, X′ and S′; pull the children's Xᵀ Ia X′ and Xᵀ pa; ABA projection; store W, D⁻¹, ũ, UD, c |
| `aba_down` (level d, root→leaf) | (bodies in level d) | a′ = X′a_p + c, u̇ = D⁻¹(ũ − W a′), a; **fused** u ← u + h u̇, Q ← Q ⊗ exp(h u), and the forward kinematics of the new state (the parent was updated by the previous launch) |

* All math is written in 3×3 blocks: S′ = [Smt; 0], X′ = [[Xa, 0], [Xc, I]], Xᵀ = [[I, [r]×], [0, I]].
  The 3×3 inverse uses the closed form.
* Fusing the bulk work into the up sweep and the kinematics into the down sweep removes the
  separate passes 0, 1 and 1b of DESIGN §2.4. The kinematics stay valid between steps, so
  `state` and `point_speeds` need no extra pass.
* **CUDA graphs.** Each launch sequence is captured once per key `(h, damping source, external
  layout, contact hook)` with `wp.ScopedCapture` and replayed with `wp.capture_launch`. At B = 1
  this gives 0.20 ms per step against 1.94 ms with plain launches (Python launch overhead).
* **No host syncs in a step.** External data (forces, spheres, damping overrides) is re-uploaded
  only when it changes. A device `wp.array` of node forces is a zero-copy path for RL.
* `settle` and `solve_static` also run on the device. Speed and increment reductions use
  `atomic_max` per env, the quiet-window/convergence bookkeeping is a kernel, and a device
  `done` flag turns later steps into no-ops. The host polls every `check_every` steps, and the
  result equals TreeSim's per-step loop exactly (same step count, same `t_converged`).
* The speed reductions treat NaN/inf as "moving". `atomic_max` would otherwise drop NaN, and a
  blown-up env would report as settled.

## 4. Inverse statics and stability on the GPU

`inverse_statics_gpu` is B2 in two sweeps for any B:

| kernel | direction | work |
|---|---|---|
| `inv_fk_level` | root→leaf | target kinematics R_j, o_j |
| `inv_up_level` | leaf→root, fused | own dead loads; pull the children's (F, Σ p Fᵀ, Φ); L_j, N_j, A_j; m_j = R_jᵀN_j; the local 3×3 damped Newton with continuation (branch check det H_el > 0); rest rotation; H_el, D_w, K_w; the symmetric pivot P_j = D_j − Φ̄_j with `eig3` (Sturm sign count, tiny-pivot convention as B2); Φ_j |

* The local Newton solve and the elimination fuse into the same sweep. Each joint's moment
  depends only on its subtree, and its pivot only on its children's Φ.
* `elim_level` repeats the shifted elimination (H − σI or H − σK_el) for Sturm bisection. All
  envs bisect in lock-step, and 8 iterations are captured as one graph, which gives λ_min and
  the buckling margin μ_min.
* `critical_modulus_gpu` runs B2's multisection with each round of 16 moduli as one batch.
* `full=False` reads back only the per-env status. This is the screening path for thousands of
  (asset, E) pairs: 1024 moduli in 8 ms, and 4096 in 22 ms (5.5 µs per pair, fp32).

## 5. Precision

Both builds come from one kernel source. `get_kernels(precision)` instantiates every kernel with
closure-captured types (F, vec3, mat33, vec4) and dtype-dependent constants (Newton tolerances,
guards). Measured against the CPU float64 reference:

| case | float64 GPU | float32 GPU |
|---|---|---|
| tomato, E = 1e9 and 5e8, 1 step | 8e-14 rel. to displacement (2e-16 m) | 2e-7 m |
| tomato 3 s rollout (180 steps, α = 5) | **7e-16 rel.** (2e-16 m) | 2.3e-7 m (7e-7 rel.), not growing |
| tomato 3 s, E = 1e8 / 1e10 / 1e11 | 1–3e-16 m | 1.4–2.7e-7 m |
| spheres + point forces + per-env params, 60 steps | 1.5e-15 m | (n/a) |
| settle t_converged, solve_static iterations | identical | identical (2.033 s) |
| cantilever tip load (B1 test 1) | | **+0.124%** = the model's 1/(2n²), h-independent |
| first frequency (B1 test 4) | | **−0.24%** |
| 1000:1 tip mass, h = 1/60 (B1 test 5) | | stable, +0.077% vs elastica, 1.1e-7 m from fp64 |
| energy, zero damping, 10 s | non-increasing | non-increasing (identical to 6 digits) |
| inverse: R0rel, Q vs CPU | 2e-15 | 1.2e-6 / 6e-7 |
| inverse: status, #negative modes, over-capacity joints | identical | identical (65/11/0, 10/0/0) |
| margin μ_min, λ_min | 1e-8 rel. | 2e-6 rel. |
| tomato E_cr | 7.3163e8 (= CPU) | 7.3163e8 |
| round trip rest → static → inverse | 1.5e-12 | 5e-7 |

**Verdict: float32 is sufficient for dynamics, settling, and stability classification.** The
1000:1 mass ratio does not break it: ABA factorises M exactly, the pivots D = SᵀIAS + h²K̃ stay
well conditioned, and fp32 matches fp64 to 1e-7. It also holds up to 10⁴:1 (fp32 − fp64 = 1.7e-7 m).
At 3·10⁴:1 and above, the CPU model itself leaves its valid range in either precision. The joints
exceed the quaternion spring's capacity (k sin θ, B2 §3), the chain whirls at 10⁴ rad/s, CPU
and GPU float64 diverge chaotically within about 1 s, and a run can produce NaN. NaN stays
confined to its own env.

**Where float32 is not enough, and what to do:**

1. **Statics below 1e-7 m.** fp32 position increments floor at one ulp (3e-8 m on a 0.5 m
   coordinate), so `solve_static(tol < ~5e-8)` never converges. Use the float64 build for
   tight statics.
2. **B2's round-trip tests at 1e-8.** These need float64. fp32 gives 5e-7, which is still far
   below any physical tolerance.
3. **Unstable equilibria.** In fp32, round-off (about 1e-7) seeds the unstable modes. The E = 5e8
   inverse rest, which has 11 negative modes, drifts 0.7 mm in 3 s in fp32 against 1e-12 m in fp64.
   This is physics amplifying noise, not error, and the Sturm count classifies both builds
   identically.

The mixed-precision policy is a choice of build per task, since every build is one kernel source:

* fp32 for RL and dataset rollouts;
* fp64 for inverse statics when the rest shape must reproduce the target to better than 1e-6
  (one inverse at B = 4096 costs 68 ms in fp64);
* fp64 for tight statics and parity tests.

Loading an fp64 inverse into an fp32 sim (`apply_rest` casts) adds only fp32 rounding. A
finer-grained mixed mode (fp64 state with fp32 factor) was not needed by any acceptance case,
so it is not implemented.

## 6. Throughput (tomato 75 d: 476 bodies, 497 leaflets, 27 levels, h = 1/60)

| B | fp32 ms/step | fp32 env-steps/s | fp32 ×CPU | fp64 ms/step | fp64 env-steps/s | fp64 ×CPU |
|---|---|---|---|---|---|---|
| 1 | 0.197 | 5,088 | 8× | 0.663 | 1,508 | 2.5× |
| 64 | 0.681 | 93,926 | 154× | 2.55 | 25,132 | 41× |
| 256 | 0.907 | 282,398 | 464× | 2.82 | 90,871 | 149× |
| 1024 | 2.32 | 440,908 | 724× | 6.79 | 150,805 | 248× |
| 4096 | 8.04 | **509,382** | **837×** | 23.3 | 175,604 | 288× |
| 16384 | 32.0 | 512,051 | 841× | (13 GB, skipped) | | |

The CPU numpy baseline on the same machine is 221 env-steps/s at B = 1 and 589–609 at B = 16–64.
Memory per env is 387 kB in fp32 and 758 kB in fp64. The fp64 rate is bound by the consumer
card's 1/64-rate fp64 units, and a data-centre GPU would close most of that gap. B = 1 is
latency-bound: 54 dependent launches of about 3.7 µs each. Throughput saturates near B ≈ 4096.

Inverse + Sturm count (summary readback), per (asset, E) pair:

| | fp32 | fp64 |
|---|---|---|
| B = 1 | 1.6 ms | 5.3 ms |
| B = 64 | 41 µs | 165 µs |
| B = 4096 | 5.5 µs | 16.7 µs |

The CPU takes 3.3 ms at B = 1 and 1.9 ms per member at B = 64. Batched fp32 is about 350× faster.

## 7. Heterogeneous topologies (plan and status)

**Chosen layout: the flattened multi-tree with per-env offsets (§2), not padded levels.**

* Padding every env to the widest level of the widest asset wastes work in proportion to the
  shape mismatch. A tomato (32-wide levels) next to a 1-wide cantilever would pad the cantilever
  by 32×, and every padded body would still need a dummy 6×6 factor.
* The flattened tree costs one extra index indirection: parent and children are global ids
  rather than i ± const. It costs nothing for a shared topology, where it degenerates to
  `g = i*B + e`.

Status:

* **Implemented and tested** for the forward step, settle, solve_static, externals (point forces
  and spheres), and `env_state(e)`.
* **Not yet implemented** for heterogeneous batches:
  * result packing of `inverse_statics_gpu` (the kernels are already generic; it raises
    NotImplementedError);
  * `state` and `leaf_vertices` (use `env_state`).

Remaining work for many-asset datasets: per-env result packing for the inverse, and sorting
envs by topology inside a level for better coalescing (the current order is rank-major).

## 8. Contact slot (interface for B3)

B3 had not merged when B4 finished, so this is the interface B4 expects. `sim.contact =
solver`, where `solver.solve(sim, h)` is called **inside the step graph, between the up sweep
(the complete factorisation of this step's implicit system, kinematics still at q_n) and the
down sweep**. It must only launch kernels on preallocated buffers. Three primitives on the stored
factor are provided and tested (`contact.py`):

| primitive | cost | meaning |
|---|---|---|
| `slot_free_motion(acc0, acc1, udot)` | one O(n) sweep | contact-free accelerations of this step, without integrating |
| `factor_response_device(df0, df1, …)` | two O(n) sweeps | linear response Δu̇, Δa to extra spatial body forces (no refactorisation; IA and D⁻¹ do not depend on the forces) |
| `slot_apply(df0, df1, …)` | one O(n) sweep | fold solved contact forces into ũ. The regular down sweep then integrates the contact-coupled motion exactly (linearity) |

Factor buffers available to B3, per global body: W0, W1, D⁻¹, UD0, UD1, Xa, Xc, Smt, ũ, c0, c1, r,
plus the kinematics R, o, tip, w, vO.

Tested properties:
* `factor_response` equals the difference of two steps with and without a point force: 7e-14
  in fp64 (1.5e-15 relative) and 1.7e-6 relative in fp32.
* A `ConstantBodyForces` hook through `slot_apply` reproduces the PointForce step to round-off.

Expected B3 pipeline, all envs in lock-step:

1. **Detect.** Contacts are compacted to k_max per env with device counters (fixed capacity, so
   the step stays graph-safe).
2. **Free motion.** `slot_free_motion` gives v_free = G(u_n + h a_free).
3. **Delassus operator.** W = G A⁻¹ Gᵀ from k response solves, batched as k right-hand sides
   (O(kn)). A path-only up pass is possible, because only the ancestors of a contact body change.
4. **Solve.** PGS / LCP on k unknowns per env, one thread block per env.
5. **Apply.** `slot_apply(Gᵀλ/h)`. Impulses p enter as forces p/h.

Once B3 merges, the ground plane and segment refinement only change the topology (refinement)
and add a detector kernel. The step structure is unchanged. B1's penalty spheres stay as the
fallback.

## 9. Limitations

* Rigid leaves, no self-contact, penalty spheres only: the same scope as B1.
* Penalty spheres overlapping hundreds of leaf vertices at high stiffness are unstable on the CPU
  too (a 12 cm sphere at k = 3e3 into the tomato top: singular D by step 7). This needs B3.
* `newton_static` (B2) is not ported. The O(n) nonsymmetric `tree_solve` would be two more
  sweeps on the same buffers.
* `stiffen_to_stable` is not ported. It can run on the CPU, or on top of `inverse_statics_gpu`
  with `sim.K` scaling.
* Differentiability (`wp.Tape`) is not wired up. The kernels are plain Warp and in-place, so
  gradients need either double-buffered state or a recorded tape per step.
* The heterogeneous gaps listed in §7.
