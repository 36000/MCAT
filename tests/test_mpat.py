"""Minimal end-to-end checks of MPAT on a small synthetic volume.

The volume is a slab of white matter, one voxel thick, between two strips of grey matter
(left and right) and sandwiched between slices of CSF, which walkers never enter: every walk
stays in the plane.  Its fibres run left-right, so the bundle "from the left grey matter to
the right grey matter" holds most streamlines.
"""
import json

import nibabel as nib
import numpy as np
import pytest

NX, NY = 16, 9
Z = 1


@pytest.fixture(scope="module")
def geo():
    from mpat.geometry import build_geometry

    return build_geometry((2.0, 2.0, 2.0), 35.0)


def _fodf_sh(direction, vertices):
    """SH coefficients (DIPY descoteaux07, legacy, lmax 8) of a sharp lobe along direction."""
    from dipy.core.sphere import Sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    _, inv = sh_to_sf_matrix(Sphere(xyz=vertices), sh_order_max=8, return_inv=True)
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    return np.exp(30.0 * ((vertices @ d) ** 2 - 1.0)) @ inv


@pytest.fixture(scope="module")
def slab(geo):
    """sh, transient, absorb_class, and the left / right grey-matter strips."""
    coef = _fodf_sh([1, 0, 0], geo.vertices)
    iso = _iso(geo.vertices)                                # grey matter: isotropic
    sh = np.zeros((NX, NY, 3, coef.size), np.float32)
    transient = np.zeros((NX, NY, 3), bool)
    absorb = np.zeros((NX, NY, 3), np.int8)
    absorb[..., 0] = absorb[..., 2] = 2                     # CSF above and below
    transient[2:-2, 1:-1, Z] = True                         # white matter
    sh[1:-1, 1:-1, Z] = coef
    left = np.zeros((NX, NY, 3), bool)
    right = np.zeros((NX, NY, 3), bool)
    left[1, 1:-1, Z] = right[-2, 1:-1, Z] = True            # grey matter at both ends
    absorb[left | right] = 1
    sh[left | right] = iso
    edge = np.zeros((NX, NY), bool)                         # the brain edge around it all
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    absorb[..., Z][edge] = 3
    return sh, transient, absorb, left, right


def _iso(vertices):
    from dipy.core.sphere import Sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    _, inv = sh_to_sf_matrix(Sphere(xyz=vertices), sh_order_max=8, return_inv=True)
    return np.ones(len(vertices)) @ inv


@pytest.fixture(scope="module")
def chain(slab):
    from mpat import Chain, ChainConfig

    sh, transient, absorb, _, _ = slab
    return Chain.build(sh, (2.0, 2.0, 2.0), transient, absorb, ChainConfig(angle=35.0))


@pytest.fixture(scope="module")
def result(chain, slab):
    from mpat import prepare_launches, segment_bundles

    _, transient, _, left, right = slab
    seeds = np.argwhere(transient)
    launches = prepare_launches(chain, seeds)
    res = segment_bundles(chain, [dict(name="LR", start=left, end=right)], launches,
                          tol=1e-10, max_mem=2e9)
    return res["LR"], seeds


def test_geometry(geo):
    assert geo.n_dirs == 642
    assert geo.p_vol.size > 0 and np.all(geo.p_vol > 0)
    assert set(np.unique(geo.vbin)) == set(range(26))       # every direction class is used


def test_transition_rows_sum_to_one(chain):
    """Every state's transitions (to white matter, to absorption, or death) sum to 1."""
    def row_sums(ptr, w):
        return np.bincount(np.repeat(np.arange(chain.n_states), np.diff(ptr)), weights=w,
                           minlength=chain.n_states)

    total = row_sums(chain.Q_ptr, chain.Q_w) + row_sums(chain.R_ptr, chain.R_val) + chain.dead
    np.testing.assert_allclose(total, 1.0, atol=1e-9)


def test_bundle(result, chain):
    res, seeds = result
    info = res.info
    assert info["endpoint_mass_check"] == pytest.approx(1.0, abs=1e-6)
    assert np.all(res.density_t >= 0) and np.all(res.endpoints_a >= 0)
    assert np.all((res.seed_prob >= 0) & (res.seed_prob <= 1 + 1e-9))
    # fibres run straight from one end to the other: most streamlines are in the bundle
    assert info["expected_bundle_streamlines"] > 0.5 * len(seeds)
    # ... and every end of the bundle is in the left or right grey matter
    ends = chain.to_volume(res.endpoints_a, "absorbing")
    assert ends[:, :, [0, 2]].sum() == 0
    density = chain.to_volume(res.density_t)
    assert density.sum() == pytest.approx(res.density_t.sum(), rel=1e-6)


def test_core(result, chain, slab):
    from mpat import bundle_core

    res, _ = result
    _, _, _, left, right = slab
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    core = bundle_core(chain.to_volume(res.density_t), aff, chain.to_volume(res.endpoints_a, "absorbing"),
                       left, right, n_nodes=20)
    assert core.xyz.shape == (20, 3)
    # it runs left to right, along the fibres
    assert core.xyz[-1, 0] - core.xyz[0, 0] > 0.5 * (NX - 6) * 2.0


def test_tissue_labels():
    from mpat.afq import tissue_labels

    shape = (8, 8, 8)
    sh = np.zeros(shape + (45,), np.float32)
    sh[1:-1, 1:-1, 1:-1, 0] = 1.0                            # the fODF field
    pve = np.zeros(shape + (3,), np.float32)                 # (CSF, GM, WM)
    pve[..., 2] = 1.0                                        # white matter...
    pve[3, 3, 3] = (0, 1, 0)                                 # ...one grey-matter voxel
    pve[4, 4, 4] = (1, 0, 0)                                 # ...and one CSF voxel
    aff = np.eye(4)
    transient, absorb, dseg = tissue_labels(nib.Nifti1Image(sh, aff), nib.Nifti1Image(pve, aff))
    assert dseg[3, 3, 3] == 2 and absorb[3, 3, 3] == 1        # grey matter: absorbs
    assert dseg[4, 4, 4] == 3 and absorb[4, 4, 4] == 2        # CSF: never entered
    assert dseg[1, 3, 3] == 4 and absorb[1, 3, 3] == 3        # the field's rim: brain edge
    assert transient[2, 2, 2] and dseg[2, 2, 2] == 1
    assert not transient[0, 0, 0] and dseg[0, 0, 0] == 0      # outside


def test_cli_afq_kwargs():
    from mpat.cli import afq_kwargs

    assert afq_kwargs(json.dumps({"pve": "synthseg"})) == {"pve": "synthseg"}
    assert afq_kwargs("dict(a=1)") == {"a": 1}
    with pytest.raises(SystemExit):
        afq_kwargs("[1, 2]")                                 # not a dict
    with pytest.raises(SystemExit):
        afq_kwargs("this is not python")
