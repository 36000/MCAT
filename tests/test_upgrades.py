"""Tests for MCAT performance optimizations and algorithmic upgrades:
1. Prefix-tree transition evaluation vs flat path evaluation
2. Cached class-to-fixel weights in segment_lobes / fixel_streamlines
3. Periodic Minimal Residual ("mr") and Aitken ("aitken") series acceleration
4. Non-negative endpoint partition and Strong-Markov hitting cascade (solve_bundle_cascade)
5. MITTENS stopping mass (stop_mass) and physical chord-length arc-length weighting
6. Positivity-limited and 2D transverse MomentChain closures
7. DirectedFixelChain (both unmasked and lobe-conditioned) on crossing fibers
"""
import numpy as np
import pytest

from mcat import (
    BundleQuantification,
    Chain,
    ChainConfig,
    ConnectomeResult,
    DirectedFixelChain,
    build_connectome,
    class_chord_lengths,
    fixel_streamlines,
    prepare_launches,
    quantify_bundle,
    segment_bundles,
    segment_lobes,
    solve_bundle_cascade,
)
from mcat.geometry import build_geometry
import mcat.kernel as K
from mcat.moments import MomentChain, _prep_m, _prep_m_tree
from mcat.solve import solve_series


def _sh(directions, kappa=20.0):
    from dipy.core.sphere import Sphere
    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    sph = get_sphere(name="repulsion724")
    _, inv = sh_to_sf_matrix(Sphere(xyz=sph.vertices), sh_order_max=8, return_inv=True)
    sf = sum(np.exp(kappa * (sph.vertices @ np.asarray(d, float)) ** 2) for d in directions)
    return (sf @ inv).astype(np.float32)


@pytest.fixture(scope="module")
def crossing_phantom():
    nx, ny, nz = 12, 12, 5
    zooms = (2.0, 2.0, 2.0)
    sh = np.zeros((nx, ny, nz, 45), np.float32)
    transient = np.zeros((nx, ny, nz), bool)
    absorb = np.zeros((nx, ny, nz), np.int8)

    u1 = (1.0, 0.0, 0.0)
    ang = np.deg2rad(42.0)
    u2 = (float(np.cos(ang)), float(np.sin(ang)), 0.0)
    sh1 = _sh([u1], kappa=18.0)
    sh2 = _sh([u2], kappa=18.0)
    sh12 = _sh([u1, u2], kappa=18.0)

    transient[2:-2, 2:-2, 1:-1] = True
    absorb[1, 2:-2, 1:-1] = 1
    absorb[-2, 2:-2, 1:-1] = 1
    absorb[2:-2, 1, 1:-1] = 1
    absorb[2:-2, -2, 1:-1] = 1

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                if 4 <= i <= 7 and 4 <= j <= 7:
                    sh[i, j, k] = sh12
                elif abs(j - 5.5) <= 1.5:
                    sh[i, j, k] = sh1
                else:
                    sh[i, j, k] = sh2

    seeds = np.argwhere(transient)
    return sh, zooms, transient, absorb, seeds


def test_prefix_tree_exact_equivalence(crossing_phantom):
    sh, zooms, transient, absorb, _ = crossing_phantom
    chain = Chain.build(sh, zooms, transient, absorb, ChainConfig())
    g = chain.geo
    vl = chain.t_lin[:16]
    out_flat = np.empty((vl.size, chain.nb, 26, chain.nb))
    dead_flat = np.empty((vl.size, chain.nb))
    K.chunk_blocks(
        vl, *chain._kargs(), g.bin_ptr, g.bin_idx, out_flat, dead_flat
    )
    out_tree = np.empty_like(out_flat)
    dead_tree = np.empty_like(dead_flat)
    K.chunk_blocks_tree(
        vl, *chain._kargs_tree(), g.bin_ptr, g.bin_idx, out_tree, dead_tree, 0.0
    )
    assert np.max(np.abs(out_tree - out_flat)) < 1e-14
    assert np.max(np.abs(dead_tree - dead_flat)) < 1e-14


def test_fixel_cached_vs_uncached_equivalence():
    geo = build_geometry((2.0, 2.0, 2.0))
    rows = np.stack([
        _sh([(1, 0, 0)]),
        _sh([(1, 0, 0), (0, 1, 0)]),
        _sh([(1, 0, 0), (0, 1, 0), (0, 0, 1)], kappa=14.0),
    ])
    lobes = segment_lobes(rows, geo)
    rng = np.random.default_rng(42)
    dc = rng.random((3, 26))
    val_fast, unass_fast = fixel_streamlines(dc, rows, geo, lobes)
    # Force fallback path by clearing _w_fix
    w_saved = lobes._w_fix
    lobes._w_fix = None
    val_ref, unass_ref = fixel_streamlines(dc, rows, geo, lobes)
    lobes._w_fix = w_saved
    assert np.max(np.abs(val_fast - val_ref)) < 1e-13
    assert np.max(np.abs(unass_fast - unass_ref)) < 1e-13


def test_series_acceleration_and_nonneg_cascade(crossing_phantom):
    sh, zooms, transient, absorb, seeds = crossing_phantom
    chain = Chain.build(sh, zooms, transient, absorb, ChainConfig())
    launches = prepare_launches(chain, seeds)

    inc = np.zeros(transient.shape, bool)
    inc[5:7, 4:8, 1:-1] = True
    start = np.zeros(transient.shape, bool)
    start[1, :, :] = True
    end = np.zeros(transient.shape, bool)
    end[-2, :, :] = True
    spec = dict(name="Horiz", include=[inc], start=start, end=end)

    res_std = segment_bundles(chain, [spec], launches, tol=1e-9, accel=False)["Horiz"]
    res_mr = segment_bundles(chain, [spec], launches, tol=1e-9, accel="mr")["Horiz"]
    res_ait = segment_bundles(chain, [spec], launches, tol=1e-9, accel="aitken")["Horiz"]
    res_cas = solve_bundle_cascade(chain, spec, launches, tol=1e-9, accel="mr")

    assert np.max(np.abs(res_mr.density_t - res_std.density_t)) / max(res_std.density_t.max(), 1e-12) < 1e-7
    assert np.max(np.abs(res_ait.density_t - res_std.density_t)) / max(res_std.density_t.max(), 1e-12) < 1e-6
    assert np.max(np.abs(res_cas.density_t - res_std.density_t)) / max(res_std.density_t.max(), 1e-12) < 1e-7
    assert np.max(np.abs(res_cas.seed_prob - res_std.seed_prob)) < 1e-7
    assert res_cas.info["negative_mass_before_clip"]["density"] == 0.0
    assert res_cas.info["negative_mass_before_clip"]["seed_prob"] == 0.0


def test_stop_mass_and_chord_lengths(crossing_phantom):
    sh, zooms, transient, absorb, seeds = crossing_phantom
    chords_vox = class_chord_lengths(build_geometry(zooms))
    chords_mm = class_chord_lengths(build_geometry(zooms), zooms=zooms)
    assert chords_vox.shape == (26,) and np.all((chords_vox >= 1.0) & (chords_vox <= 1.8))
    assert np.allclose(chords_mm, chords_vox * 2.0)

    chain_tau = Chain.build(sh, zooms, transient, absorb, ChainConfig(stop_mass=1e-3))
    assert np.all(chain_tau.dead > 0.0)
    launches = prepare_launches(chain_tau, seeds[:20])
    spec = dict(name="All")
    res_vox = segment_bundles(chain_tau, [spec], launches, tol=1e-6, arc_length=False)["All"]
    res_mm = segment_bundles(chain_tau, [spec], launches, tol=1e-6, arc_length=True, zooms=zooms)["All"]
    ratio = res_mm.density_t.sum() / max(res_vox.density_t.sum(), 1e-12)
    assert 2.0 <= ratio <= 3.5


def test_moment_closures_and_tree_prep(crossing_phantom):
    sh, zooms, transient, absorb, _ = crossing_phantom
    for closure in ("positivity", "transverse"):
        mc = Chain.build(sh, zooms, transient, absorb, ChainConfig(moments=True, moment_closure=closure))
        assert isinstance(mc, MomentChain)
        assert mc.stab_info["fallback_voxels"] == 0
        # Verify prefix-tree moment prep matches flat _prep_m
        v = mc.t_lin[0]
        af = mc._margs()
        at = mc._margs_tree()
        _, _, cK_f, zK_f, Zv_f = _prep_m(v, *af[:-5], af[-2], af[-1], mc.geo.t_nbr.shape[0])
        _, _, cK_t, zK_t, Zv_t = _prep_m_tree(v, *at[:-5], at[-2], at[-1], mc.geo.t_nbr.shape[0])
        assert np.max(np.abs(cK_t - cK_f)) < 1e-13
        assert np.max(np.abs(zK_t - zK_f)) < 1e-13
        assert np.max(np.abs(Zv_t - Zv_f)) < 1e-13


def test_directed_fixel_chain(crossing_phantom):
    sh, zooms, transient, absorb, _ = crossing_phantom
    chain26 = Chain.build(sh, zooms, transient, absorb, ChainConfig())
    fc_unmasked = DirectedFixelChain(chain26, lobe_conditioned=False)
    fc_cond = DirectedFixelChain(chain26, lobe_conditioned=True)
    assert fc_unmasked.n_states < chain26.n_states // 4
    assert fc_cond.n_transitions <= fc_unmasked.n_transitions
    # Verify row stochasticity: Q @ 1 + R @ 1 + dead == 1
    for fc in (fc_unmasked, fc_cond):
        row_sum = np.asarray(fc.Q.sum(1)).ravel() + np.asarray(fc.R.sum(1)).ravel() + fc.dead
        assert np.allclose(row_sum, 1.0, atol=1e-12)


def test_quantify_bundle_all_weightings(crossing_phantom):
    sh, zooms, transient, absorb, seeds = crossing_phantom
    chain = Chain.build(sh, zooms, transient, absorb, ChainConfig())
    launches = prepare_launches(chain, seeds)
    ijk_t = np.stack(np.unravel_index(chain.t_lin, chain.pshape), 1) - 1
    sh_rows = sh[tuple(ijk_t.T)]
    lobes = segment_lobes(sh_rows, chain.geo)

    inc = np.zeros(transient.shape, bool)
    inc[5:7, 4:8, 1:-1] = True
    start = np.zeros(transient.shape, bool)
    start[1, :, :] = True
    end = np.zeros(transient.shape, bool)
    end[-2, :, :] = True
    spec = dict(name="Horiz", include=[inc], start=start, end=end)

    res = segment_bundles(chain, [{"name": "_whole"}, spec], launches, tol=1e-8, keep_fields=True, accel="mr")
    whole_dc = res["_whole"].density_class
    horiz = res["Horiz"]
    affine = np.diag(list(zooms) + [1.0])

    q = quantify_bundle(
        chain, horiz, whole_dc, sh_rows, affine, lobes=lobes,
        start_mask=start, end_mask=end, n_bins=10,
        scalar_maps={"c00": sh_rows[:, 0]}, tol=1e-8
    )
    assert isinstance(q, BundleQuantification)
    assert q.mean_streamline_length_mm > 10.0
    assert q.tract_afd_volume_mm3 > 0.0 and q.effective_area_mm2 > 0.0
    assert np.all((q.fixel_share >= 0.0) & (q.fixel_share <= 1.0))
    assert np.all((q.q_harmonic_t >= 0.0) & (q.q_harmonic_t <= 1.0))
    assert 0.0 < q.boundary_hit_prob_fwd <= 1.0 and q.boundary_conductance > 0.0
    assert q.along_tract["afd_area_mm2"].shape == (10,)
    assert q.along_tract["c00"].shape == (10,)
    assert np.all(q.branching_k_eff_t[horiz.density_t > 0] >= 1.0)

    # Also verify default fast path without keep_fields=True (using BundleResult.flux26)
    res_nf = segment_bundles(chain, [spec], launches, tol=1e-8, keep_fields=False, accel="mr")["Horiz"]
    assert res_nf.flux26 is not None and res_nf.fields is None
    q_nf = quantify_bundle(chain, res_nf, whole_dc, sh_rows, affine, lobes=lobes, n_bins=10,
                           compute_boundary_conductance=False)
    assert np.allclose(q_nf.voxel_afd_t, q.voxel_afd_t, rtol=1e-5, atol=1e-6)
    assert np.allclose(q_nf.tdi_mm_t, q.tdi_mm_t, rtol=1e-5, atol=1e-6)
    assert np.allclose(q_nf.flux_coherence_t, q.flux_coherence_t, rtol=1e-5, atol=1e-5)
    assert np.allclose(q_nf.branching_k_eff_t, q.branching_k_eff_t, rtol=1e-5, atol=1e-5)
    assert np.all((q_nf.q_harmonic_t >= 0.0) & (q_nf.q_harmonic_t <= 1.0))


def test_build_connectome_phantom(crossing_phantom):
    sh, zooms, transient, absorb, seeds = crossing_phantom
    chain = Chain.build(sh, zooms, transient, absorb, ChainConfig())
    launches = prepare_launches(chain, seeds)
    ijk_t = np.stack(np.unravel_index(chain.t_lin, chain.pshape), 1) - 1
    sh_rows = sh[tuple(ijk_t.T)]
    lobes = segment_lobes(sh_rows, chain.geo)
    affine = np.diag(list(zooms) + [1.0])

    # Create 4 parcels at the 4 ends of the crossing phantom: Left(1), Right(2), Bottom(3), Top(4)
    atlas = np.zeros(transient.shape, dtype=np.int32)
    atlas[1, 4:8, 1:-1] = 1
    atlas[-2, 4:8, 1:-1] = 2
    atlas[4:8, 1, 1:-1] = 3
    atlas[4:8, -2, 1:-1] = 4

    conn = build_connectome(
        chain, sh_rows, atlas, affine, launches=launches, lobes=lobes,
        parcel_ids=[1, 2, 3, 4], parcel_names=["Left", "Right", "Bottom", "Top"],
        max_dist_mm=2.5, tol=1e-8, zero_diagonal=False
    )
    assert isinstance(conn, ConnectomeResult)
    assert conn.afd_volume_mm3.shape == (4, 4)
    assert np.allclose(conn.afd_volume_mm3, conn.afd_volume_mm3.T)
    assert np.allclose(conn.afd_area_mm2, conn.afd_area_mm2.T)
    assert np.allclose(conn.mean_length_mm, conn.mean_length_mm.T)
    assert np.allclose(conn.streamline_count, conn.streamline_count.T)
    # Total SIFT2-conserved AFD volume across all (i, j) pairs equals total connected fixel AFD volume
    assert np.isclose(conn.afd_volume_mm3.sum(), conn.info["total_wm_afd_volume_mm3"], rtol=1e-4)
    # Horizontal (0, 1) and Vertical (2, 3) pairs should have strong connectivity and realistic length (>15 mm)
    assert conn.afd_area_mm2[0, 1] > 0.0 and conn.afd_area_mm2[2, 3] > 0.0
    assert conn.mean_length_mm[0, 1] > 14.0 and conn.mean_length_mm[2, 3] > 14.0


def test_ptt_chain_config(crossing_phantom):
    sh, zooms, transient, absorb, seeds = crossing_phantom
    cfg_ptt = ChainConfig.ptt(fod_thresh=0.08, fod_power=1.6, curv_sigma=16.0, probe_radius=0.5, stop_mass=1e-4)
    chain_ptt = Chain.build(sh, zooms, transient, absorb, cfg_ptt)
    chain_base = Chain.build(sh, zooms, transient, absorb, ChainConfig())
    # PTT prunes weak/high-curvature transitions while keeping positive mass
    assert 0 < chain_ptt.n_transitions < chain_base.n_transitions
    launches_ptt = prepare_launches(chain_ptt, seeds)
    end_right = np.zeros(transient.shape, bool)
    end_right[-2, 4:8, 1:-1] = True
    start_left = np.zeros(transient.shape, bool)
    start_left[1, 4:8, 1:-1] = True
    res = segment_bundles(chain_ptt, [dict(name="H", start=start_left, end=end_right)], launches_ptt)["H"]
    assert res.density_t.sum() > 0.0
    assert res.flux26 is not None



