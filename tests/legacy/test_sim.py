import numpy as np
import pytest

from planttwin.asset import PlantAsset
from planttwin.camera import Camera
from planttwin.procedural import make_plant
from planttwin.sim import PlantModel, Sphere, q_apply_rotvec, qconj, qd3, qmul, qrot


@pytest.fixture(scope="module")
def plant():
    return make_plant(seed=0)


def test_quaternion_helpers():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(5, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    v = rng.normal(size=(5, 3))
    # qrot matches q * v * q^-1 and qd3 matches R e3
    qv = np.concatenate([np.zeros((5, 1)), v], 1)
    np.testing.assert_allclose(qrot(q, v), qmul(qmul(q, qv), qconj(q))[:, 1:], atol=1e-12)
    np.testing.assert_allclose(qd3(q), qrot(q, np.broadcast_to([0, 0, 1.0], (5, 3))), atol=1e-12)
    # small rotation vector rotates d3 by ~ dtheta x d3
    dth = np.array([[1e-4, 0, 0]] * 5)
    d = qd3(q_apply_rotvec(q, dth)) - qd3(q)
    np.testing.assert_allclose(d, np.cross(dth, qd3(q)), atol=1e-7)


def test_asset_roundtrip(tmp_path, plant):
    plant.save(tmp_path / "p.json")
    p2 = PlantAsset.load(tmp_path / "p.json")
    np.testing.assert_allclose(p2.stem.nodes, plant.stem.nodes)
    assert len(p2.leaves) == len(plant.leaves)
    assert p2.physics == plant.physics


def test_equilibrium_is_static(plant):
    """After settling, the undisturbed plant must not drift (catches twist/flip modes)."""
    m = PlantModel(plant)
    xe, qe = m.settle()
    X, _, _ = m.simulate(0.5, x0=xe, q_init=qe)
    assert np.isfinite(X).all()
    assert np.abs(X - xe[None]).max() < 2e-3


def test_stiffer_stem_sags_less(plant):
    m = PlantModel(plant)
    m.set_params([plant.physics.copy(stem_youngs_modulus=E) for E in (5e7, 2e8, 1e9)])
    xe, _ = m.settle()
    tips = m.leaf_keypoints("tip")
    sag = (m.x_rest[tips, 2] - xe[:, tips, 2]).mean(1)
    assert sag[0] > sag[1] > sag[2] > 0


def test_batch_matches_single(plant):
    m = PlantModel(plant)
    phys = [plant.physics, plant.physics.copy(stem_youngs_modulus=3e8)]
    m.set_params(phys)
    xb, _ = m.settle(0.3)
    m.set_params(phys[1])
    xs, _ = m.settle(0.3)
    np.testing.assert_allclose(xb[1], xs[0], atol=1e-10)


def test_sphere_pushes_leaf(plant):
    m = PlantModel(plant)
    xe, qe = m.settle()
    k = m.leaf_idx[2][plant.leaves[2].keypoints["mid"]]
    target = xe[0, k]
    sph = Sphere(0.008, lambda t: target + np.array([0, 0, 0.03 - 0.05 * min(t / 0.3, 1)]))
    X, _, _ = m.simulate(0.4, colliders=(sph,), x0=xe, q_init=qe, record=np.array([k]))
    assert X[-1, 0, 0, 2] < target[2] - 0.01  # pushed down by > 1 cm


def test_camera_projects_target_to_center():
    cam = Camera.look_at([0, -1, 0.2], [0, 0, 0.2])
    np.testing.assert_allclose(cam.project(np.array([0, 0, 0.2])), [640, 360], atol=1e-9)
    assert cam.project(np.array([0, 0, 0.3]))[1] < 360  # up in the world is up in the image
