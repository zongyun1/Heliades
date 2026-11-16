"""heliades.sim.gpu: the B1 tree solver and B2 inverse statics on the GPU (NVIDIA Warp, B4).

    from heliades.sim.gpu import TreeSimGPU, inverse_statics_gpu
    sim = TreeSimGPU(asset, [phys_1, ..., phys_B], precision="float32")   # or "float64"
    sim.step(1 / 60); sim.settle(); sim.state.x
    res = inverse_statics_gpu(sim)          # B2 inverse + Sturm stability count, batched

See docs/sim/GPU.md.
"""
from .sim import TreeSimGPU
from .inverse import critical_modulus_gpu, inverse_statics_gpu

__all__ = ["TreeSimGPU", "inverse_statics_gpu", "critical_modulus_gpu", "cuda_available"]


def cuda_available() -> bool:
    try:
        import warp as wp
        wp.init()
        return len(wp.get_cuda_devices()) > 0
    except Exception:
        return False
