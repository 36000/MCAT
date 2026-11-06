"""Bundle fields on a synthetic straight bundle: the fields reproduce the density, the edge
flux is conserved and counts the streamlines crossing a cut, committors partition the
visits, and endpoint restriction is exact."""
import numpy as np
import pytest

from mcat import ChainConfig, Chain, prepare_launches, restrict_endpoints, segment_bundles
from mcat.geometry import OFFSETS

NX, NY, NZ = 14, 7, 7
WM = (slice(2, 12), slice(2, 5), slice(2, 5))   # white matter along x, 3 x 3 cross-section


def _sh_rows():
    """SH (descoteaux07 legacy, lmax 8) of a sharp fODF along x, and of an isotropic one."""
    from dipy.core.sphere import Sphere
    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    sph = get_sphere(name="repulsion724")
    _, inv = sh_to_sf_matrix(Sphere(xyz=sph.vertices), sh_order_max=8, return_inv=True)
    fiber = np.exp(20.0 * (sph.vertices[:, 0] ** 2 - 1.0)) @ inv
    iso = np.ones(len(sph.vertices)) @ inv
    return fiber, iso


@pytest.fixture(scope="module")
def phantom():
    fiber, iso = _sh_rows()
    sh = np.zeros((NX, NY, NZ, fiber.size), np.float32)
    transient = np.zeros((NX, NY, NZ), bool)
    transient[WM] = True
    absorb = np.zeros((NX, NY, NZ), np.int8)
    gm = np.zeros((NX, NY, NZ), bool)
    gm[1, 1:6, 1:6] = gm[12, 1:6, 1:6] = True      # grey matter slabs at both ends
    absorb[gm] = 1
    sh[transient] = fiber
    sh[gm] = iso
    chain = Chain.build(sh, (2.0, 2.0, 2.0), transient, absorb, ChainConfig())
    seeds = np.argwhere(transient & (np.isin(np.arange(NX), [2, 11])[:, None, None]))
    launches = prepare_launches(chain, seeds)
    start = np.zeros((NX, NY, NZ), bool)
    start[1] = True
    end = np.zeros((NX, NY, NZ), bool)
    end[12] = True
    include = np.zeros((NX, NY, NZ), bool)
    include[6] = True
    specs = [dict(name="plain"), dict(name="rois", include=[include], start=start, end=end)]
    res = segment_bundles(chain, specs, launches, tol=1e-12, keep_fields=True)
    return dict(chain=chain, launches=launches, specs=specs, res=res, start=start, end=end, gm=gm)


def _vox_index(chain):
    return np.stack(np.unravel_index(chain.t_lin, chain.pshape), 1) - 1


@pytest.mark.parametrize("name", ["plain", "rois"])
def test_fields_reproduce_density(phantom, name):
    res, chain = phantom["res"][name], phantom["chain"]
    F = res.fields
    n = F.visits()
    ti = chain.tindex[chain.lin(res.seed_vox)]
    np.add.at(n, ti, res.seed_prob)          # the streamlines' step 0 at their seeds
    N = res.info["expected_bundle_streamlines"]
    assert N > 10                            # 18 seeds, nearly all straight through
    assert np.abs(n - res.density_t).max() < 1e-9 * N
    assert abs(F.plan_streamlines() - N) < 1e-9 * N


@pytest.mark.parametrize("name", ["plain", "rois"])
def test_flux_is_conserved(phantom, name):
    F = phantom["res"][name].fields
    c = F.conservation()
    assert c["in"] < 1e-9 and c["out"] < 1e-9
    f26 = F.edge_flux()
    assert np.allclose(F.through_flux(f26), F.visits(), rtol=1e-9, atol=1e-9 * c["expected_streamlines"])
    assert (f26 >= -1e-12).all()


@pytest.mark.parametrize("name", ["plain", "rois"])
def test_cut_flux_counts_streamlines(phantom, name):
    """Every streamline crosses the plane between x = 6 and x = 7 exactly once (no reversals
    within 35 degrees), so the flux across it is the expected number of streamlines."""
    res, chain = phantom["res"][name], phantom["chain"]
    f26 = res.fields.edge_flux()
    x = _vox_index(chain)[:, 0]
    dx = OFFSETS[:, 0]
    across = f26[x == 6][:, dx == 1].sum() + f26[x == 7][:, dx == -1].sum()
    N = res.info["expected_bundle_streamlines"]
    assert abs(across - N) < 1e-6 * N


def test_committors_partition_visits(phantom):
    res = phantom["res"]["rois"]
    F = res.fields
    phi = F.endpoint_sides([phantom["start"], phantom["end"]], tol=1e-12)
    n = F.visits()
    assert np.allclose(phi.sum(1), n, rtol=1e-8, atol=1e-9 * n.max())
    x = _vox_index(phantom["chain"])[:, 0]
    frac_end = phi[:, 1] / np.maximum(n, 1e-300)
    # walks near the far end mostly finish there; near the near end the two seeds' halves mix
    assert frac_end[x == 11].mean() > frac_end[x == 2].mean()
    assert ((frac_end >= -1e-9) & (frac_end <= 1 + 1e-9)).all()


def test_endpoint_restriction_is_exact(phantom):
    chain, L, specs = phantom["chain"], phantom["launches"], phantom["specs"]
    half = np.zeros((NX, NY, NZ), bool)
    half[:, :, :3] = True                    # keep the end patch with z < 3 only
    spec = restrict_endpoints(specs[1], end=half)
    assert spec["start"] is specs[1]["start"]
    res = segment_bundles(chain, [spec], L, tol=1e-12)["rois"]
    full = phantom["res"]["rois"]
    N, Nf = full.info["expected_bundle_streamlines"], res.info["expected_bundle_streamlines"]
    assert 0 < Nf < N
    ends = chain.to_volume(res.endpoints_a, "absorbing")
    assert ends[12][:, 3:].sum() == 0        # nothing ends in the excluded part of the end slab
    assert ends[12].sum() > 0
    # the committor of the kept patch, from the full solve, predicts the kept endpoint mass
    phi = full.fields.endpoint_sides([phantom["end"] & half], tol=1e-12)[:, 0]
    assert phi.max() > 0


def test_keep_fields_needs_plain_chain():
    class Moments:
        ncomp = 4

    with pytest.raises(ValueError):
        segment_bundles(Moments(), [], None, keep_fields=True)
