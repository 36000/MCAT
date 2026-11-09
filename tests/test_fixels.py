"""Fixels on synthetic voxels: lobes of single-fibre and crossing fODFs, a bundle's visits
partitioned among them, and the fixel directory written and read back."""
import numpy as np
import pytest

from mcat.fixels import (
    fixel_streamlines,
    read_mif,
    segment_lobes,
    write_fixel_directory,
    write_mif,
)
from mcat.geometry import OFFSETS, build_geometry


def _sh(directions, kappa=20.0):
    """SH (descoteaux07 legacy, lmax 8) of a sum of Watson fODFs along ``directions``."""
    from dipy.core.sphere import Sphere
    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    sph = get_sphere(name="repulsion724")
    _, inv = sh_to_sf_matrix(Sphere(xyz=sph.vertices), sh_order_max=8, return_inv=True)
    sf = sum(np.exp(kappa * (sph.vertices @ np.asarray(d, float)) ** 2) for d in directions)
    return (sf @ inv).astype(np.float32)


@pytest.fixture(scope="module")
def setup():
    geo = build_geometry((2.0, 2.0, 2.0))
    rows = np.stack([_sh([(1, 0, 0)]), _sh([(1, 0, 0), (0, 1, 0)]), _sh([(1, 0, 0), (0, 1, 0)], kappa=8.0)])
    return geo, rows          # a single fibre, a sharp crossing and a broader one


def _class(d):
    return int(np.flatnonzero((OFFSETS == np.asarray(d)).all(1))[0])


def test_lobes_of_single_and_crossing(setup):
    geo, rows = setup
    lobes = segment_lobes(rows, geo)
    assert lobes.count.tolist() == [1, 2, 2] and lobes.offset.tolist() == [0, 1, 3]
    assert abs(lobes.direction[0] @ [1, 0, 0]) > 0.98
    d = np.abs(lobes.direction[1:3] @ np.eye(3)[:2].T)       # (2 fixels, x / y)
    assert d.max(1).min() > 0.98 and set(d.argmax(1)) == {0, 1}
    assert lobes.peak[1] == pytest.approx(lobes.peak[2], rel=0.05)
    assert lobes.integral[1] == pytest.approx(lobes.integral[2], rel=0.05)


def test_visits_partition_among_fixels(setup):
    geo, rows = setup
    lobes = segment_lobes(rows, geo)
    x_fix = 1 + int(np.argmax(np.abs(lobes.direction[1:3] @ [1, 0, 0])))
    y_fix = 3 - x_fix
    dc = np.zeros((3, 26))
    dc[0, [_class((1, 0, 0)), _class((-1, 0, 0))]] = 3.0        # single fibre, walked along x
    dc[1, [_class((1, 0, 0)), _class((-1, 0, 0))]] = 5.0        # crossing, walked along x only
    values, unassigned = fixel_streamlines(dc, rows, geo, lobes)
    assert values[0] == pytest.approx(6.0) and unassigned.max() < 1e-12
    assert values[x_fix] == pytest.approx(10.0, rel=0.02) and values[y_fix] < 0.2
    # the broader crossing walked diagonally: both fixels get some, the total is kept
    dc[:] = 0.0
    dc[2, _class((1, 1, 0))] = 4.0
    values, unassigned = fixel_streamlines(dc, rows, geo, lobes)
    assert values[3:].sum() + unassigned[2] == pytest.approx(4.0)
    assert values[3] > 0.5 and values[4] > 0.5


def test_mif_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    affine = np.array([[0, -2.0, 0, 5], [2.0, 0, 0, -7], [0, 0, 2.5, 3], [0, 0, 0, 1]])
    for arr in (rng.random((4, 3, 2, 2)).astype(np.float32), rng.integers(0, 9, (3, 2, 2)).astype(np.uint32)):
        write_mif(tmp_path / "a.mif", arr, affine)
        back, A = read_mif(tmp_path / "a.mif")
        assert back.shape == arr.shape and np.array_equal(back, arr) and np.allclose(A, affine)


def test_fixel_directory_round_trip(setup, tmp_path):
    geo, rows = setup
    lobes = segment_lobes(rows, geo)
    ijk = np.array([[1, 1, 1], [2, 1, 1], [3, 1, 1]])
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    out = write_fixel_directory(tmp_path / "fixels", lobes, ijk, affine, (4, 3, 3),
                                {"demo": np.arange(5, dtype=float)})
    index, A = read_mif(out / "index.mif")
    assert index.shape == (4, 3, 3, 2) and np.allclose(A, affine)
    assert index[1, 1, 1].tolist() == [1, 0] and index[2, 1, 1].tolist() == [2, 1] and index[3, 1, 1].tolist() == [2, 3]
    dirs, _ = read_mif(out / "directions.mif")
    assert dirs.shape == (5, 3, 1) and np.allclose(np.linalg.norm(dirs[:, :, 0], axis=1), 1)
    assert read_mif(out / "demo.mif")[0].ravel().tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        write_fixel_directory(tmp_path / "bad", lobes, ijk, affine, (4, 3, 3), {"x": np.zeros(2)})
