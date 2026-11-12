"""MAT: geometry, chain, bundle solves (against walkers sampled from the same chain), the
Krylov tail, grouping, cores, the viewer, and an offline end-to-end run through pyAFQ."""
import nibabel as nib
import numpy as np
import pytest
from numba import njit
from phantoms import crossing_phantom
from scipy import ndimage

from mpat import kernel as K
from mpat.chain import Chain, ChainConfig
from mpat.geometry import build_geometry
from mpat.solve import prepare_launches, segment_bundles, solve_series


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def crossing():
    sh, wm, cls, reg = crossing_phantom(n=25, width=5)
    caps = reg["capx"] | reg["capy"]
    seeds = np.argwhere(ndimage.binary_dilation(caps, np.ones((3, 3, 3))) & wm)
    c = reg["center"][0]
    A = np.zeros_like(wm)
    A[c - 4] = True
    A &= wm
    B = np.zeros_like(wm)
    B[c + 4] = True
    B &= wm
    end = np.zeros_like(wm)
    end[-2:] = True
    end &= reg["capx"]
    ch = Chain.build(sh, (1.0, 1.0, 1.0), wm, cls, ChainConfig())
    return sh, wm, cls, reg, seeds, A, B, end, ch


def _csr(ch):
    """The stencil as row-form CSR (Q) plus R."""
    import scipy.sparse as sps

    nb = ch.nb
    rows = np.repeat(np.arange(ch.n_states), np.diff(ch.Q_ptr))
    vox, c = rows // nb, rows % nb
    sl = ch.Q_slot.astype(np.int64)
    cols = ch.nbr_t[vox, ch.tmpl_n[c, sl]].astype(np.int64) * nb + ch.tmpl_b[c, sl]
    Q = sps.csr_matrix((ch.Q_w, (rows, cols)), shape=(ch.n_states,) * 2)
    R = sps.csr_matrix((ch.R_val, ch.R_ind, ch.R_ptr), shape=(ch.n_states, ch.a_lin.size))
    return Q, R


# ------------------------------------------------------------------ geometry and chain
def test_visitation_odf():
    from dipy.core.sphere import Sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    from mpat.geometry import OFFSETS, visitation_odf

    g = build_geometry((2.0, 2.0, 2.0), 35.0)
    V = g.vertices
    B, inv = sh_to_sf_matrix(Sphere(xyz=V), sh_order_max=8, return_inv=True)
    x, y = np.eye(3)[0], np.eye(3)[1]
    sh = np.stack([(np.abs(V @ x) ** 20 + np.abs(V @ y) ** 20) @ inv] * 2)    # crossing fODF
    cls = {tuple(o): i for i, o in enumerate(OFFSETS)}
    D = np.zeros((2, 26))
    D[0, cls[1, 0, 0]], D[0, cls[-1, 0, 0]] = 3.0, 1.0       # walkers along x only
    D[1, cls[0, 1, 0]] = 2.0                                  # along y
    c = visitation_odf(D, sh, g)
    assert np.allclose(c[:, 0] * 2 * np.sqrt(np.pi), D.sum(1), rtol=1e-6)   # integral = density
    sf = c @ B
    for i, (own, other) in enumerate(((x, y), (y, x))):
        assert abs(V[np.argmax(sf[i])] @ own) > 0.99
        assert sf[i, np.argmax(V @ other)] < 0.05 * sf[i].max()             # the other fibre: not visited
    assert np.allclose(sf[0], sf[0, g.antipode()], atol=1e-6 * sf[0].max())  # symmetric


def test_geometry_boxes_and_anisotropy():
    g = build_geometry((2.0, 2.0, 2.0), 35.0)   # raises if any start box disagrees with MITTENS
    assert g.n_dirs == 642 and g.p_dirs.shape[1] == 2
    assert all(i in g.cone_idx[g.cone_ptr[i]:g.cone_ptr[i + 1]] for i in range(0, 642, 37))
    ga = build_geometry((1.75, 1.75, 2.35), 25.0)
    assert ga.p_vol.size > 0 and (ga.hi >= ga.lo).all()


def test_chain_rows_and_adjoint(crossing):
    *_, ch = crossing
    Q, R = _csr(ch)
    rows = np.asarray(Q.sum(1)).ravel() + np.asarray(R.sum(1)).ravel() + ch.dead
    assert np.abs(rows - 1).max() < 1e-13
    rng = np.random.default_rng(0)
    x, y = rng.random((ch.n_states, 3)), rng.random((ch.n_states, 3))
    fx, by = np.empty_like(x), np.empty_like(y)
    ch.mul_forward(x, fx)
    ch.mul_backward(y, by)
    assert np.abs(fx - Q.T @ x).max() < 1e-12 and np.abs(by - Q @ y).max() < 1e-12


def test_power_series_matches_direct_solve(crossing):
    import scipy.sparse as sps
    import scipy.sparse.linalg as spla

    *_, ch = crossing
    Q, _ = _csr(ch)
    rng = np.random.default_rng(1)
    x0 = rng.random((ch.n_states, 2))
    mask = (rng.random(x0.shape) > 0.1).astype(np.uint8)
    x0 *= mask
    tot, info = solve_series(ch.mul_forward, x0.copy(), mask, 1e-13, 5000, "test")
    for c in range(2):
        M = sps.diags(mask[:, c].astype(float))
        exact = spla.spsolve((sps.identity(ch.n_states) - M @ Q.T).tocsc(), x0[:, c])
        assert np.abs(tot[:, c] - exact).max() < 1e-9 * exact.max()


# ------------------------------------------------------------------ bundles vs sampled walkers
@njit(cache=True)
def _walks(n, seed, launch, v, noff, label, tindex, aindex, nb, Qp, Qi, Qv, Rp, Ri, Rv, inA, inB,
           inA_a, inB_a, a0, b0):
    """n half-walks from one launch: (hit A, hit B, absorbing voxel or -1)."""
    np.random.seed(seed)
    res = np.empty((n, 3), np.int64)
    for w in range(n):
        hitA, hitB, state, end = a0, b0, -1, -1
        u, acc, done = np.random.random(), 0.0, False
        for nn in range(26):
            for b in range(nb):
                acc += launch[nn, b]
                if acc >= u and not done:
                    t = v + noff[nn]
                    if label[t] == K.TRANSIENT:
                        state = tindex[t] * nb + b
                    else:
                        end = aindex[t]
                    done = True
        while state >= 0:
            hitA |= inA[state]
            hitB |= inB[state]
            u, acc, nxt = np.random.random(), 0.0, -1
            for p in range(Qp[state], Qp[state + 1]):
                acc += Qv[p]
                if acc >= u:
                    nxt = Qi[p]
                    break
            if nxt < 0:
                for p in range(Rp[state], Rp[state + 1]):
                    acc += Rv[p]
                    if acc >= u:
                        end = Ri[p]
                        break
            state = nxt
        if end >= 0:
            hitA |= inA_a[end]
            hitB |= inB_a[end]
        res[w, 0], res[w, 1], res[w, 2] = hitA, hitB, end
    return res


def test_bundle_matches_walkers_sampled_from_the_chain(crossing):
    sh, wm, cls, reg, seeds, A, B, end, ch = crossing
    L = prepare_launches(ch, seeds)
    res = segment_bundles(ch, [dict(name="x", include=[A, B], end=end)], L)["x"]
    assert abs(res.info["endpoint_mass_check"] - 1) < 1e-9
    assert np.abs(res.density_class.sum(1) - res.density_t).max() < 1e-6 * res.density_t.max()
    Q, R = _csr(ch)

    def flags(m):
        p = np.zeros(ch.pshape, bool)
        p[1:-1, 1:-1, 1:-1] = m
        return p.ravel()

    sv = np.repeat(ch.t_lin, ch.nb)
    fA, fB, fE = flags(A), flags(B), flags(end)[ch.a_lin]
    valid = np.isin(ch.a_class, [1, 3])
    n = 4000
    for si in np.argsort(res.seed_prob)[-3:]:
        p_mc = 0.0
        for pi in np.flatnonzero(L.pair_seed == si):
            v = L.seed_lin[si]
            hs = []
            for h in (2 * pi, 2 * pi + 1):
                launch = np.zeros((26, ch.nb))
                for k in range(L.lptr[h], L.lptr[h + 1]):
                    launch[L.lkey[k] // ch.nb, L.lkey[k] % ch.nb] = L.lval[k, 0]
                hs.append(_walks(n, 7 + h, launch, v, ch.noff, ch.label, ch.tindex, ch.aindex, ch.nb,
                                 Q.indptr, Q.indices, Q.data, R.indptr, R.indices, R.data, fA[sv], fB[sv],
                                 fA[ch.a_lin], fB[ch.a_lin], fA[v], fB[v]))
            e1, e2 = np.maximum(hs[0][:, 2], 0), np.maximum(hs[1][:, 2], 0)
            ok = (hs[0][:, 2] >= 0) & (hs[1][:, 2] >= 0) & valid[e1] & valid[e2] & (fE[e1] | fE[e2])
            ok &= ((hs[0][:, 0] | hs[1][:, 0]) & (hs[0][:, 1] | hs[1][:, 1])) > 0
            p_mc += L.pair_w[pi] * ok.mean()
        se = np.sqrt(max(p_mc * (1 - p_mc), 1e-4) / n)
        assert abs(res.seed_prob[si] - p_mc) < 4 * se + 2e-3, (res.seed_prob[si], p_mc)


def test_grouped_equals_separate_and_moments(crossing):
    sh, wm, cls, reg, seeds, A, B, end, ch = crossing
    L = prepare_launches(ch, seeds)
    specs = [dict(name="a", include=[A], end=end), dict(name="b", include=[A, B])]
    together = segment_bundles(ch, specs, L, max_mem=8e9)
    alone = {s["name"]: segment_bundles(ch, [s], L, max_mem=1)[s["name"]] for s in specs}  # 1 column
    for k in together:
        assert together[k].info["solved_with"] and not alone[k].info["solved_with"]
        assert np.abs(together[k].density_t - alone[k].density_t).max() < 1e-8 * alone[k].density_t.max()
    m = Chain.build(sh, (1.0, 1.0, 1.0), wm, cls, ChainConfig(moments=True))
    assert type(m).__name__ == "MomentChain"
    r = segment_bundles(m, specs[:1], prepare_launches(m, seeds))["a"]
    assert abs(r.info["endpoint_mass_check"] - 1) < 1e-6
    e = r.info["expected_bundle_streamlines"] / together["a"].info["expected_bundle_streamlines"]
    assert 0.5 < e < 2


# ------------------------------------------------------------------ Krylov tail
def _sticky_chain(n=60, loops=((20, 0.9995), (40, 0.999))):
    """Transport down a line with near-closed 2-cycles hanging off it (A[to, from])."""
    N = n + 2 * len(loops)
    A = np.zeros((N, N))
    for j in range(n - 1):
        A[j + 1, j] = 0.97
    for k, (at, p) in enumerate(loops):
        a, b = n + 2 * k, n + 2 * k + 1
        A[at + 1, at], A[a, at], A[b, a], A[a, b], A[at + 1, b] = 0.5, 0.47, p, p, 1 - p - 1e-5
    return A


def test_krylov_tail_where_power_crawls():
    A = _sticky_chain()
    x0 = np.zeros((A.shape[0], 2))
    x0[0, 0], x0[10, 1] = 1.0, 2.0
    exact = np.linalg.solve(np.eye(len(A)) - A, x0)
    tot, info = solve_series(lambda x, o: np.copyto(o, A @ x), x0.copy(), np.ones_like(x0), 1e-10,
                             100000, "test")
    assert info["tail_started_at"] < 300 and info["tail_products"] < 1500 and not info["stalled"]
    assert np.abs(tot - exact).max() < 1e-8 * exact.max()


def test_closed_loop_stalls_cleanly():
    A = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 1.0], [0.0, 1.0, 0.0]])
    tot, info = solve_series(lambda x, o: np.copyto(o, A @ x), np.array([[1.0], [0.0], [0.0]]),
                             np.ones((3, 1)), 1e-6, 5000, "test")
    assert info["stalled"] and np.isfinite(tot).all() and tot.max() < 1e3


def test_fast_decay_never_switches():
    rng = np.random.default_rng(3)
    A = rng.random((40, 40))
    A *= 0.9 / A.sum(0)
    x0 = rng.random((40, 2))
    tot, info = solve_series(lambda x, o: np.copyto(o, A @ x), x0.copy(), np.ones_like(x0), 1e-12,
                             5000, "test")
    assert "tail_started_at" not in info
    assert np.abs(tot - np.linalg.solve(np.eye(40) - A, x0)).max() < 1e-9


# ------------------------------------------------------------------ pyAFQ end to end
def test_pyafq_end_to_end(tmp_path):
    pytest.importorskip("AFQ")
    from synthetic_subject import make_afq, make_subject

    import mpat

    f = make_subject(tmp_path / "raw")
    res = mpat.run(make_afq(f, tmp_path / "afq"), n_proc=2, max_mem="2GB")
    out = tmp_path / "afq" / "mpat"
    for b, own in (("A", f["A"]), ("B", f["B"])):
        assert res[b].info["expected_bundle_streamlines"] > 10
        d = nib.load(out / f"sub-01_bundle-{b}_desc-density_map.nii.gz").get_fdata()
        assert d[own].sum() > 0.99 * d.sum()
        assert (out / f"sub-01_bundle-{b}_desc-core.trx").exists()
        # visitation ODF: integrates to the density, points along the bundle (A: x, B: y),
        # including where the two bundles cross
        from mpat.geometry import _sh_basis

        odf = nib.load(out / f"sub-01_bundle-{b}_desc-visitodf_dwimap.nii.gz").get_fdata()
        m = d > 1e-2 * d.max()
        assert np.allclose(odf[m][:, 0] * 2 * np.sqrt(np.pi), d[m], rtol=1e-4)
        u = np.eye(3)
        sf = odf[m & f["A"] & f["B"]] @ _sh_basis(np.r_[u, -u], odf.shape[-1])
        axis = 0 if b == "A" else 1
        assert (np.argmax(sf[:, :3], 1) == axis).all() and (sf[:, 1 - axis] < 0.05 * sf[:, axis]).all()
    assert (out / "sub-01_desc-mpat_table.tsv").exists()
    pytest.importorskip("ipyniivue")
    nv = mpat.view(out, glyphs=["A"])
    assert [m.name for m in nv.meshes] == ["A_core.obj", "B_core.obj", "A_glyphs.mz3"]
    assert nv.scene.crosshair_pos != [0.5, 0.5, 0.5]
    # glyphs follow the slices: every voxel >= 1e-2 of the peak is available, the three
    # slices through the crosshair are meshed
    from mpat.viewer import glyph_mesh, save_html, scene

    S = scene(out, glyphs=True)
    g = S["glyphs"][0]
    dA = nib.load(out / "sub-01_bundle-A_desc-density_map.nii.gz").get_fdata()
    assert len(g["ijk"]) == (dA >= 1e-2 * dA.max()).sum()
    c = tuple(g["ijk"][0])
    assert glyph_mesh(g, S["affine"], c) is not None and glyph_mesh(g, S["affine"], (-1, -1, -1)) is None
    page = save_html(tmp_path / "view.html", out, glyphs=True)
    html = open(page).read()
    assert '"role": "glyphs"' in html and "followGlyphs" in html and "cdn.jsdelivr.net/npm/@niivue/niivue@" in html
