import pytest

pytest.importorskip("pyhelios")

from heliades.parity import run_case  # noqa: E402
from heliades.parity.cases import CASES  # noqa: E402


@pytest.mark.parametrize("name", sorted(CASES))
def test_parity_with_helios(name):
    for seed in range(CASES[name]["fuzz"]):
        r = run_case(name, seed)
        assert r.ok, f"seed {seed}\n{r}"


def test_compare_detects_divergence():
    """The harness must catch a one-ulp-scale-changing bug, not just pass everything."""
    import numpy as np

    from heliades.parity.backends import ForgeBackend
    from heliades.parity.compare import compare_snapshots
    a, b = ForgeBackend(), ForgeBackend()
    for B in (a, b):
        B.add_patch((0.1, 0.2, 0.3), (1.0, 2.0), (0.4, 0.5))
    sb = b.snapshot()
    sb.vertices[0, 2, 1] = np.nextafter(sb.vertices[0, 2, 1], np.float32(10)) + np.float32(1e-5)
    r = compare_snapshots(a.snapshot(), sb)
    assert not r.ok and "vertices" in r.first_divergence
