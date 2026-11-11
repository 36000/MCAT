"""Analytical structural connectivity matrices ("Analytical SIFT2") on the MCAT Markov chain.

Instead of solving O(R^2) pairwise bundle problems or running Monte Carlo tractography + SIFT2
optimization, whole-brain parcel-to-parcel connectivity across R grey-matter regions is solved
in TWO batched R-column backward passes sharing the unmasked white-matter transition operator Q:

1. Parcel-hitting probabilities B in R^(Ns x R):
       (I - Q) B = R_abs E_GM
   where B[s, r] is the exact probability that a half-walk from state s=(v, c) is absorbed
   into grey-matter parcel r.

2. Expected half-walk physical path lengths M_len in R^(Ns x R) (in mm):
       (I - Q) M_len = d_chord * B
   where d_chord(s) is the physical step advance (mm) of state s, so M_len[s, r] / B[s, r] is
   the exact conditional expected path length (mm) from state s to parcel r.

From B and M_len, all connectome matrices are assembled in closed form via matrix products
in < 50 ms:
- ``afd_volume_mm3`` (R, R): SIFT2-conserved intra-axonal white-matter volume (mm^3) per pair.
- ``afd_area_mm2`` (R, R): Length-corrected effective axonal cross-sectional area (mm^2),
  computed as afd_volume_mm3 / mean_length_mm ("Analytical SIFT2" connectome).
- ``mean_length_mm`` (R, R): Expected physical streamline length (mm) connecting each pair.
- ``streamline_count`` (R, R): Seed-launched expected streamline count per parcel pair.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import kernel as K
from .fixels import Lobes, segment_lobes
from .geometry import class_chord_lengths
from .solve import max_columns, solve_series

log = logging.getLogger("mcat")


@dataclass
class ConnectomeResult:
    """Whole-brain structural connectivity matrices across R parcels."""

    parcel_ids: np.ndarray              # (R,) int parcel labels
    parcel_names: list[str]             # length R
    afd_volume_mm3: np.ndarray          # (R, R) SIFT2-conserved intra-axonal volume (mm^3)
    afd_volume_raw_mm3: np.ndarray      # (R, R) unnormalized probabilistic AFD volume (mm^3)
    afd_area_mm2: np.ndarray            # (R, R) effective axonal cross-sectional area (mm^2)
    mean_length_mm: np.ndarray          # (R, R) AFD-weighted expected streamline length (mm)
    streamline_count: np.ndarray        # (R, R) seed-launched expected streamlines (if launches given)
    seed_mean_length_mm: np.ndarray     # (R, R) seed-launched expected streamline length (mm)
    parcel_gm_voxels: np.ndarray        # (R,) number of absorbing GM boundary voxels per parcel
    info: dict = field(default_factory=dict)
    _seed_Vp: np.ndarray = field(default=None, repr=False)
    _seed_pair_w: np.ndarray = field(default=None, repr=False)
    _seed_pair_seed: np.ndarray = field(default=None, repr=False)

    def seed_connectome_for_subset(self, seed_indices, zero_diagonal=True):
        """Evaluate ``streamline_count`` for a subset of seed indices without re-solving."""
        if self._seed_Vp is None:
            raise ValueError("ConnectomeResult was built without launches")
        keep = np.isin(self._seed_pair_seed, np.asarray(seed_indices))
        pw = self._seed_pair_w[keep, None]
        V0, V1 = self._seed_Vp[keep, 0], self._seed_Vp[keep, 1]
        mat = (V0 * pw).T @ V1 + (V1 * pw).T @ V0
        if zero_diagonal:
            np.fill_diagonal(mat, 0.0)
        return mat


def assign_absorbing_parcels(chain, atlas_vol, parcel_ids=None, parcel_names=None, max_dist_mm=4.0,
                             zooms=None, valid_classes=(1,)):
    """Map grey-matter absorbing states ``chain.a_lin`` to integer parcel labels in ``atlas_vol``.

    Absorbing states in ``valid_classes`` (default: 1 = GM) that lie within ``max_dist_mm`` of a
    nonzero label in ``atlas_vol`` are assigned to the nearest parcel.

    Returns (E_gm, parcel_ids, parcel_names, parcel_counts) where ``E_gm`` has shape (Na, R).
    """
    atlas = np.rint(np.asarray(atlas_vol)).astype(np.int32)
    if atlas.shape != chain.shape:
        raise ValueError(f"atlas_vol shape {atlas.shape} does not match chain.shape {chain.shape}")
    if parcel_ids is None:
        parcel_ids = np.unique(atlas)
        parcel_ids = parcel_ids[parcel_ids > 0]
    else:
        parcel_ids = np.asarray(parcel_ids, dtype=np.int32)
    R = len(parcel_ids)
    if R == 0:
        raise ValueError("atlas_vol contains no positive parcel labels")
    if parcel_names is None:
        parcel_names = [f"Parcel_{int(pid)}" for pid in parcel_ids]
    else:
        parcel_names = list(parcel_names)

    # Nearest-parcel assignment within max_dist_mm
    has_label = np.isin(atlas, parcel_ids)
    if max_dist_mm > 0 and np.any(has_label):
        sampling = tuple(float(z) for z in zooms[:3]) if zooms is not None else None
        dist, nearest_idx = distance_transform_edt(~has_label, sampling=sampling, return_indices=True)
        assigned_vol = np.where(dist <= max_dist_mm, atlas[tuple(nearest_idx)], 0)
    else:
        assigned_vol = np.where(has_label, atlas, 0)

    pad = np.zeros(chain.pshape, dtype=np.int32)
    pad[1:-1, 1:-1, 1:-1] = assigned_vol
    a_labels = pad.ravel()[chain.a_lin]
    a_valid = np.isin(chain.a_class, np.asarray(valid_classes))
    a_labels = np.where(a_valid, a_labels, 0)

    # Map parcel_ids -> 0..R-1
    E_gm = np.zeros((chain.a_lin.size, R), dtype=np.float64)
    for idx, pid in enumerate(parcel_ids):
        E_gm[a_labels == pid, idx] = 1.0
    counts = E_gm.sum(0).astype(np.int64)
    return E_gm, parcel_ids, parcel_names, counts


def _antipodal_fixel_weights(geo, lobes: Lobes):
    """Build antipodal class weights ``w_anti`` (n_fixels, nb) for walking along -u_f.

    For each fixel f in voxel v, ``lobes._w_fix[f, b]`` is the directional probability distribution
    along +u_f; ``w_anti[f, b] = lobes._w_fix[f, anti_b[b]]`` is the distribution along -u_f.
    """
    anti_v = geo.antipode()
    # Map each class b to its antipodal class anti_b
    # Every class b has exit face c_exit = b // 27 and entry face c_entry = b % 27 (for b < 312)
    # Use the representative vertex or exact antipodal class mapping:
    nb = lobes._w_fix.shape[1]
    anti_b = np.empty(nb, dtype=np.int64)
    for b in range(nb):
        i0, i1 = geo.bin_ptr[b], geo.bin_ptr[b + 1]
        if i1 > i0:
            v_mid = geo.bin_idx[(i0 + i1) // 2]
            anti_b[b] = int(geo.vbin[anti_v[v_mid]])
        else:
            anti_b[b] = b
    return np.ascontiguousarray(lobes._w_fix[:, anti_b])


def build_connectome(chain, sh_rows, atlas_vol, affine, launches=None, lobes: Lobes = None,
                     parcel_ids=None, parcel_names=None, max_dist_mm=4.0, tol=1e-4, len_tol=None,
                     max_iter=5000, max_mem=8e9, accel="mr", zero_diagonal=True):
    """Compute the whole-brain structural connectivity matrices across parcels in ``atlas_vol``.

    Parameters
    ----------
    chain : mcat.Chain
        Built MCAT Markov chain (plain chain, ``ncomp == 1``).
    sh_rows : (Nt, n_sh) ndarray
        fODF SH coefficients at the chain's transient voxels ``chain.t_lin``.
    atlas_vol : 3D ndarray
        Integer parcellation volume on ``chain.shape``.
    affine : (4, 4) ndarray
        Voxel-to-world affine matrix (used for voxel sizes and volume).
    launches : mcat.solve._Launches, optional
        Prepared seed launches; when provided, also computes ``streamline_count`` and
        ``seed_mean_length_mm``.
    lobes : mcat.fixels.Lobes, optional
        Pre-segmented fODF lobes at ``chain.t_lin`` (computed automatically if None).
    """
    if chain.ncomp != 1:
        raise ValueError("build_connectome requires the plain chain (ncomp == 1)")
    t0 = time.perf_counter()
    zooms = np.linalg.norm(np.asarray(affine, float)[:3, :3], axis=0)
    vox_vol_mm3 = float(np.abs(np.linalg.det(np.asarray(affine, float)[:3, :3])))
    if len_tol is None:
        len_tol = max(tol * 10.0, 1e-4) if tol >= 1e-5 else tol

    if lobes is None:
        lobes = segment_lobes(sh_rows, chain.geo)

    E_gm, parcel_ids, parcel_names, parcel_counts = assign_absorbing_parcels(
        chain, atlas_vol, parcel_ids=parcel_ids, parcel_names=parcel_names,
        max_dist_mm=max_dist_mm, zooms=zooms
    )
    R = len(parcel_ids)
    Ns, Nt, nb = chain.n_states, chain.t_lin.size, chain.nb
    n_fix = lobes.n_fixels

    # Base mask (respecting soft max length if set)
    base_mask = np.ones(Ns, dtype=np.uint8)
    if getattr(chain, "kill", None) is not None:
        base_mask[chain.kill] = 0

    chords_mm = class_chord_lengths(chain.geo, zooms=zooms)
    state_chords = np.tile(chords_mm, Nt)  # (Ns,)

    # Precompute normalized directional class weights per fixel (+u_f and -u_f)
    frow = lobes._frow          # (n_fixels,) voxel index in 0..Nt-1
    w_plus = lobes._w_fix       # (n_fixels, nb)
    w_minus = _antipodal_fixel_weights(chain.geo, lobes)  # (n_fixels, nb)
    wp_norm = w_plus / np.maximum(w_plus.sum(1, keepdims=True), 1e-30)
    wm_norm = w_minus / np.maximum(w_minus.sum(1, keepdims=True), 1e-30)

    # Compact fixel-space and seed-space accumulators (230x smaller than (Ns, R)!)
    U_plus = np.empty((n_fix, R), dtype=np.float64)
    U_minus = np.empty((n_fix, R), dtype=np.float64)
    L_plus = np.empty((n_fix, R), dtype=np.float64)
    L_minus = np.empty((n_fix, R), dtype=np.float64)

    if launches is not None:
        P = launches.pair_seed.size
        H = launches.hseed.size
        vals_prob = np.empty((H, R), dtype=np.float64)
        vals_len = np.empty((H, R), dtype=np.float64)

    # Solve backward hitting probabilities B_chunk (Ns, w) and path-length expectations L_chunk (Ns, w)
    # in memory-bounded column chunks, projecting immediately to fixel and seed space.
    cap = max(1, min(R, max_columns(chain, launches, max_mem) if launches is not None else 32))
    t_solve_b = 0.0
    t_solve_l = 0.0
    iters_b, iters_l = [], []
    for c0 in range(0, R, cap):
        c1 = min(R, c0 + cap)
        w = c1 - c0
        mask_chunk = np.repeat(base_mask[:, None], w, axis=1)
        r_chunk = chain.absorb_rhs(E_gm[:, c0:c1]) * mask_chunk

        tb0 = time.perf_counter()
        B_chunk, info_b = solve_series(
            chain.mul_backward, r_chunk, mask_chunk, tol, max_iter,
            "connectome_prob", 1, accel=accel
        )
        B_chunk = np.clip(B_chunk, 0.0, 1.0)
        t_solve_b += time.perf_counter() - tb0
        iters_b.append(info_b["iterations"])

        # Project B_chunk to fixels and seeds
        Bv_f = B_chunk.reshape(Nt, nb, w)[frow]
        U_plus[:, c0:c1] = np.einsum("fb,fbw->fw", wp_norm, Bv_f, optimize=True)
        U_minus[:, c0:c1] = np.einsum("fb,fbw->fw", wm_norm, Bv_f, optimize=True)
        del Bv_f

        if launches is not None:
            seed_ok_w = np.ones((H, w), dtype=np.float64)
            vp_w = np.empty((H, w), dtype=np.float64)
            K.half_values(launches.hseed, launches.lptr, launches.lkey, launches.lval,
                          chain.label, chain.noff, chain.tindex, chain.aindex, nb,
                          np.ascontiguousarray(B_chunk), E_gm[:, c0:c1], seed_ok_w, vp_w)
            vals_prob[:, c0:c1] = vp_w

        # Expected path length (mm): (I - Q) M_len = (d_chord * B) * mask
        tl0 = time.perf_counter()
        rhs_len = (B_chunk * state_chords[:, None]) * mask_chunk
        del B_chunk
        L_chunk, info_l = solve_series(
            chain.mul_backward, rhs_len, mask_chunk, len_tol, max_iter,
            "connectome_length", 1, accel=accel
        )
        L_chunk = np.maximum(L_chunk, 0.0)
        t_solve_l += time.perf_counter() - tl0
        iters_l.append(info_l["iterations"])

        Lv_f = L_chunk.reshape(Nt, nb, w)[frow]
        L_plus[:, c0:c1] = np.einsum("fb,fbw->fw", wp_norm, Lv_f, optimize=True)
        L_minus[:, c0:c1] = np.einsum("fb,fbw->fw", wm_norm, Lv_f, optimize=True)
        del Lv_f

        if launches is not None:
            zero_absw_w = np.zeros((chain.a_lin.size, w), dtype=np.float64)
            vl_w = np.empty((H, w), dtype=np.float64)
            K.half_values(launches.hseed, launches.lptr, launches.lkey, launches.lval,
                          chain.label, chain.noff, chain.tindex, chain.aindex, nb,
                          np.ascontiguousarray(L_chunk), zero_absw_w, seed_ok_w, vl_w)
            vals_len[:, c0:c1] = vl_w
        del L_chunk, mask_chunk

    # ------------------------------------------------------------------
    # 1. Fixel-Native Analytical SIFT2 Connectomes (AFD volume, area, length)
    # ------------------------------------------------------------------
    t_mat0 = time.perf_counter()

    # Physical intra-axonal volume per fixel (mm^3)
    voxel_total_afd = np.bincount(frow, weights=lobes.integral, minlength=Nt)
    mean_wm_afd = max(float(voxel_total_afd.mean()), 1e-30)
    afd_vol_f = (lobes.integral.astype(np.float64) / mean_wm_afd) * vox_vol_mm3  # (n_fixels,)

    # Unnormalized probabilistic AFD volume: 0.5 * sum_f V_AFD(f) (U+_fi U-_fj + U-_fi U+_fj)
    w_raw = 0.5 * afd_vol_f[:, None]
    afd_vol_raw = (U_plus * w_raw).T @ U_minus + (U_minus * w_raw).T @ U_plus

    # SIFT2-conserved AFD volume: partition each connected fixel's V_AFD(f) across parcel pairs
    Z_f = U_plus.sum(1) * U_minus.sum(1)
    conn_f = Z_f > 1e-12
    scale_f = np.zeros_like(afd_vol_f)
    scale_f[conn_f] = 0.5 * afd_vol_f[conn_f] / Z_f[conn_f]

    afd_vol = (U_plus * scale_f[:, None]).T @ U_minus + (U_minus * scale_f[:, None]).T @ U_plus

    # Length-weighted AFD volume (mm^3 * mm):
    # Each pair of half-walks (+ -> i, - -> j) has total expected length L+_{f,i}/U+_{f,i} + L-_{f,j}/U-_{f,j},
    # so length * probability = L+_{f,i} U-_{f,j} + U+_{f,i} L-_{f,j}.
    afd_vol_len = (
        (L_plus * scale_f[:, None]).T @ U_minus
        + (U_plus * scale_f[:, None]).T @ L_minus
        + (L_minus * scale_f[:, None]).T @ U_plus
        + (U_minus * scale_f[:, None]).T @ L_plus
    )
    mean_length_mm = np.divide(afd_vol_len, afd_vol, out=np.zeros_like(afd_vol), where=afd_vol > 1e-12)
    afd_area_mm2 = np.divide(afd_vol, mean_length_mm, out=np.zeros_like(afd_vol), where=mean_length_mm > 1e-6)

    # ------------------------------------------------------------------
    # 2. Seed-Launched Streamline Count & Length Connectomes (optional)
    # ------------------------------------------------------------------
    if launches is not None:
        Vp = vals_prob.reshape(P, 2, R)
        Vl = vals_len.reshape(P, 2, R)
        pw = launches.pair_w[:, None]

        V0, V1 = Vp[:, 0], Vp[:, 1]
        L0, L1 = Vl[:, 0], Vl[:, 1]
        streamline_count = (V0 * pw).T @ V1 + (V1 * pw).T @ V0
        seed_len_num = (
            (L0 * pw).T @ V1
            + (V0 * pw).T @ L1
            + (L1 * pw).T @ V0
            + (V1 * pw).T @ L0
        )
        seed_mean_length_mm = np.divide(
            seed_len_num, streamline_count, out=np.zeros_like(streamline_count), where=streamline_count > 1e-12
        )
    else:
        streamline_count = np.zeros((R, R), dtype=np.float64)
        seed_mean_length_mm = np.zeros((R, R), dtype=np.float64)

    if zero_diagonal:
        for mat in (afd_vol, afd_vol_raw, afd_area_mm2, mean_length_mm, streamline_count, seed_mean_length_mm):
            np.fill_diagonal(mat, 0.0)

    t_mat = time.perf_counter() - t_mat0
    total_time = time.perf_counter() - t0
    info = dict(
        n_parcels=int(R),
        n_active_parcels=int((parcel_counts > 0).sum()),
        n_fixels=int(lobes.n_fixels),
        n_connected_fixels=int(conn_f.sum()),
        total_wm_afd_volume_mm3=round(float(afd_vol_f.sum()), 2),
        connectome_afd_volume_sum_mm3=round(float(np.triu(afd_vol, 1 if zero_diagonal else 0).sum()), 2),
        connectome_afd_area_sum_mm2=round(float(np.triu(afd_area_mm2, 1 if zero_diagonal else 0).sum()), 2),
        total_streamlines=round(float(np.triu(streamline_count, 1 if zero_diagonal else 0).sum()), 2),
        backward_prob_iters=iters_b,
        backward_len_iters=iters_l,
        solve_prob_sec=round(t_solve_b, 2),
        solve_len_sec=round(t_solve_l, 2),
        matrix_assembly_sec=round(t_mat, 3),
        total_sec=round(total_time, 2),
    )
    return ConnectomeResult(
        parcel_ids=parcel_ids,
        parcel_names=parcel_names,
        afd_volume_mm3=afd_vol,
        afd_volume_raw_mm3=afd_vol_raw,
        afd_area_mm2=afd_area_mm2,
        mean_length_mm=mean_length_mm,
        streamline_count=streamline_count,
        seed_mean_length_mm=seed_mean_length_mm,
        parcel_gm_voxels=parcel_counts,
        info=info,
        _seed_Vp=Vp if launches is not None else None,
        _seed_pair_w=launches.pair_w if launches is not None else None,
        _seed_pair_seed=launches.pair_seed if launches is not None else None,
    )
