"""Synthetic Real2Sim identifiability experiment.

1. procedural plant (stand-in for the reconstruction front end)
2. "real" video = simulate with hidden true params, project, add pixel noise
3. fit (E_stem, leaf bend compliance, damping) from
     (a) static photo only (equilibrium shape)    (b) static + poke video
4. report recovery error, save plots + JSON to outputs/

Run:  OMP_NUM_THREADS=1 python scripts/real2sim_demo.py [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planttwin.camera import Camera  # noqa: E402
from planttwin.procedural import make_plant  # noqa: E402
from planttwin.real2sim import (FIT_PARAMS, PokeExperiment, fit, make_synthetic_observation,  # noqa: E402
                                phys_to_theta, rollout)
from planttwin.sim import PlantModel  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny optimiser budget (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    asset = make_plant(seed=args.seed)
    asset.save(OUT / "plant_asset.json")
    true = asset.physics.copy(stem_youngs_modulus=1.0e8, leaf_bend_compliance=2e-2, damping=1.2)
    base = asset.physics.copy(stem_youngs_modulus=3e8, leaf_bend_compliance=5e-3, damping=3.0)  # wrong guess
    exp = PokeExperiment()
    cam = Camera.look_at([0.05, -0.6, 0.30], [0, 0, 0.18])
    obs = make_synthetic_observation(asset, true, exp, cam, pixel_noise=1.0, seed=args.seed)

    bounds = [(np.log10(3e7), 9.0), (-3.0, -1.0), (np.log10(0.3), 1.0)]
    budget = dict(maxiter=2, popsize=4) if args.quick else dict(maxiter=10, popsize=8)
    results = {"true": dict(zip(FIT_PARAMS, map(float, 10 ** phys_to_theta(true))))}

    for mode in ("static_only", "static+video"):
        t0 = time.time()
        log = []

        def cb(xk, convergence=None):
            log.append(xk.tolist())
            print(f"  [{mode}] gen {len(log):2d}  theta={np.round(10 ** xk, 4)}", flush=True)

        est, res = fit(asset, obs, exp, base, bounds, static_only=(mode == "static_only"),
                       seed=args.seed, callback=cb, **budget)
        th_t, th_e = phys_to_theta(true), phys_to_theta(est)
        results[mode] = {
            "estimate": dict(zip(FIT_PARAMS, map(float, 10 ** th_e))),
            "rel_error": dict(zip(FIT_PARAMS, map(float, 10 ** (th_e - th_t) - 1))),
            "loss_px2": float(res.fun), "evals": int(res.n_evals), "seconds": time.time() - t0,
            "history_log10": log,
        }
        print(f"{mode}: loss={res.fun:.2f}px^2  evals={res.n_evals}  {time.time() - t0:.0f}s")
        for k in FIT_PARAMS:
            print(f"   {k:22s} true={getattr(true, k):.4g}  est={getattr(est, k):.4g}  "
                  f"err={results[mode]['rel_error'][k] * 100:+.1f}%")
        results[mode]["_phys"] = est

    # ---- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = PlantModel(asset)
    kps = obs.keypoints
    preds = {m: cam.project(rollout(model, results[m]["_phys"], exp, kps)[:, 0]) for m in ("static_only", "static+video")}
    t = np.arange(len(obs.tracks)) / exp.fps
    poked = list(kps).index(model.leaf_keypoints("tip")[exp.leaf])
    other = (poked + 1) % 5
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for a, k, name in ((ax[0], poked, f"poked leaf {exp.leaf} tip"), (ax[1], other, f"leaf {other} tip")):
        a.plot(t, obs.tracks[:, k, 1], ".", ms=3, color="0.4", label="observed (synthetic video)")
        a.plot(t, preds["static+video"][:, k, 1], lw=1.8, label="fit: static+video")
        a.plot(t, preds["static_only"][:, k, 1], lw=1.2, ls="--", label="fit: static only")
        a.axvline(exp.t_release, color="0.7", lw=0.8)
        a.set_ylabel("image v [px]"); a.set_title(name, fontsize=10); a.invert_yaxis()
    ax[0].legend(fontsize=8); ax[1].set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig(OUT / "real2sim_tracks.png", dpi=130)

    for m in ("static_only", "static+video"):
        results[m].pop("_phys")
    (OUT / "real2sim_results.json").write_text(json.dumps(results, indent=1))
    print("wrote", OUT / "real2sim_results.json", OUT / "real2sim_tracks.png")


if __name__ == "__main__":
    main()
