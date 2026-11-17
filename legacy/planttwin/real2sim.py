"""Real2Sim: fit physical parameters so simulated keypoint tracks match a video.

    theta* = argmin_theta  mean_t,k || Pi(Simulate(theta, u_1:t))[k] - y_t,k ||^2

theta lives in log10 space. The "video" is either synthetic (sim + pixel noise, used to
test identifiability) or real 2D tracks (e.g. CoTracker / TAPIR on a phone clip, with the
camera from COLMAP). Optimisation is gradient-free (differential evolution); every
generation is one *batched* rollout, split across processes.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution

from .asset import PhysicsParams, PlantAsset
from .camera import Camera
from .sim import PlantModel, Sphere

FIT_PARAMS = ("stem_youngs_modulus", "leaf_bend_compliance", "damping")


@dataclass
class PokeExperiment:
    """A finger (sphere) presses a leaf down, then lifts off; the plant rings down."""

    leaf: int = 2
    radius: float = 0.008
    approach: float = 0.02     # start this far above the leaf
    depth: float = 0.025       # press this far below the leaf's equilibrium position
    t_press: float = 0.35
    t_release: float = 0.45
    duration: float = 2.0
    fps: int = 60
    substeps: int = 64
    settle_time: float = 1.0

    def colliders(self, target: np.ndarray) -> tuple[Sphere, ...]:
        """target: (B,3) equilibrium position of the pressed point, per batch element."""
        start = target + np.array([0, 0, self.approach + self.radius])
        end = target + np.array([0, 0, -self.depth + self.radius])

        def center(t):
            s = min(t / self.t_press, 1.0)
            return (1 - s) * start + s * end

        return (Sphere(self.radius, center, lambda t: t < self.t_release),)


@dataclass
class Observation:
    camera: Camera
    keypoints: np.ndarray      # particle indices that are "tracked" in the video
    tracks: np.ndarray         # (frames, K, 2) pixels
    meta: dict = field(default_factory=dict)


def rollout(model: PlantModel, phys, exp: PokeExperiment, record, static_only=False):
    """Settle under gravity, then poke. Returns keypoint positions (F+1, B, K, 3)."""
    model.set_params(phys)
    x_eq, q_eq = model.settle(exp.settle_time, damping=15.0, substeps=exp.substeps)
    if static_only:
        return x_eq[None][:, :, record]
    press = model.leaf_idx[exp.leaf][model.asset.leaves[exp.leaf].keypoints["mid"]]
    X, _, _ = model.simulate(exp.duration, fps=exp.fps, substeps=exp.substeps,
                             colliders=exp.colliders(x_eq[:, press]), x0=x_eq, q_init=q_eq,
                             record=record)
    return X


def tracked_points(model: PlantModel) -> np.ndarray:
    return np.concatenate([model.leaf_keypoints("tip"), model.leaf_keypoints("mid")])


def make_synthetic_observation(asset: PlantAsset, phys_true: PhysicsParams, exp: PokeExperiment,
                               camera: Camera, pixel_noise=1.0, seed=0) -> Observation:
    model = PlantModel(asset)
    kps = tracked_points(model)
    X = rollout(model, phys_true, exp, kps)[:, 0]
    rng = np.random.default_rng(seed)
    tracks = camera.project(X) + rng.normal(0, pixel_noise, X.shape[:-1] + (2,))
    return Observation(camera, kps, tracks, meta={"true": phys_true, "pixel_noise": pixel_noise})


def theta_to_phys(theta, base: PhysicsParams) -> PhysicsParams:
    return base.copy(**{k: float(10.0 ** v) for k, v in zip(FIT_PARAMS, theta)})


def phys_to_theta(phys: PhysicsParams) -> np.ndarray:
    return np.log10([getattr(phys, k) for k in FIT_PARAMS])


# ---------------------------------------------------------------- batched, parallel loss
_W: dict = {}


def _init_worker(asset, obs, exp, base, static_only):
    _W.update(model=PlantModel(asset), obs=obs, exp=exp, base=base, static_only=static_only)


def _eval_chunk(thetas: np.ndarray) -> np.ndarray:
    """thetas: (b, D) -> losses (b,), px^2 per tracked point per frame."""
    if len(thetas) == 0:
        return np.zeros(0)
    obs, exp = _W["obs"], _W["exp"]
    phys = [theta_to_phys(t, _W["base"]) for t in thetas]
    with np.errstate(all="ignore"):
        X = rollout(_W["model"], phys, exp, obs.keypoints, _W["static_only"])
        pred = obs.camera.project(X)                             # (F, b, K, 2)
        target = obs.tracks[:1] if _W["static_only"] else obs.tracks
        err = np.mean(np.sum((pred - target[:, None]) ** 2, -1), axis=(0, 2))
    return np.where(np.isfinite(err), err, 1e12)


class BatchedLoss:
    def __init__(self, asset, obs, exp, base, static_only=False, processes=None):
        self.processes = processes or max(1, mp.cpu_count() - 1)
        self.pool = mp.get_context("fork").Pool(
            self.processes, _init_worker, (asset, obs, exp, base, static_only))
        self.n_evals = 0

    def __call__(self, thetas: np.ndarray) -> np.ndarray:
        """scipy vectorized convention: thetas (D, S) -> (S,)."""
        T = np.atleast_2d(thetas.T) if thetas.ndim == 2 else thetas[None]
        chunks = np.array_split(T, min(self.processes, len(T)))
        self.n_evals += len(T)
        return np.concatenate(self.pool.map(_eval_chunk, chunks))

    def close(self):
        self.pool.close(); self.pool.join()


def fit(asset, obs, exp, base: PhysicsParams, bounds, static_only=False, maxiter=12,
        popsize=10, seed=0, callback=None, processes=None):
    loss = BatchedLoss(asset, obs, exp, base, static_only, processes)
    try:
        res = differential_evolution(
            loss, bounds, maxiter=maxiter, popsize=popsize, seed=seed, vectorized=True,
            updating="deferred", polish=False, tol=1e-8, callback=callback, init="sobol",
        )
    finally:
        loss.close()
    res.n_evals = loss.n_evals
    return theta_to_phys(res.x, base), res
