"""Biologically interpretable quantifications and weightings of MCAT bundle outputs.

Beyond raw voxel step-count visitation density (``BundleResult.density_t``), this module
provides five complementary quantifications:

1. **Physical Arc-Length Density (TDI, mm)** and **Expected Streamline Length (mm)**:
   weights each direction class by its geometric mean intra-voxel chord length (mm).
2. **AFD-Calibrated Fixel & Voxel Fiber Density ("Analytical SIFT2")**:
   partitions each fixel's physical Apparent Fiber Density (``lobes.integral``) by the
   bundle's Markov routing share ``n_bundle(f) / n_whole(f)``, yielding seed-rate-invariant
   bundle AFD maps, total intra-axonal tract volume (mm^3) and effective cross-sectional
   area (mm^2).
3. **Intrinsic Markov Conductance**:
   - Cross-sectional reactive cut-flux across harmonic level sets of the bundle.
   - Boundary-launched Markov hitting probability and conductance between the two endpoint
     ROIs (eliminating interior white-matter seed-volume bias).
4. **Harmonic Committor Coordinates & Along-Tract Flux-Weighted Tractometry**:
   computes both the Laplace-Beltrami harmonic potential ``q_harmonic in [0, 1]`` on the
   bundle's reactive flux graph and the forward endpoint committor ``q_committor in [0, 1]``,
   and extracts flux-weighted along-tract profiles of cross-sectional flux, AFD area,
   coherence, and any user-supplied scalar microstructure map (e.g. FA, MD).
5. **Markov Routing Entropy, Effective Branching Number (K_eff), and Flux Coherence**:
   quantifies where a bundle funnels through tight white-matter bottlenecks vs. fans/disperses
   into cortex.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from .core import split_ends
from .fields import flux_orientation
from .fixels import Lobes, fixel_streamlines, segment_lobes
from .geometry import OFFSETS, class_chord_lengths


@dataclass
class BundleQuantification:
    name: str
    # 1. Transition-weighted physical length (TDI)
    tdi_mm_t: np.ndarray              # (Nt,) expected streamline path length (mm) per WM voxel
    mean_streamline_length_mm: float  # exact expected length (mm) of a bundle streamline
    # 2. Local fixel AFD attribution (bundle-wise fixel AFD share)
    fixel_share: np.ndarray           # (N_fixels,) n_bundle(f) / n_whole(f) in [0, 1]
    fixel_afd: np.ndarray             # (N_fixels,) share(f) * lobes.integral(f)
    voxel_afd_t: np.ndarray           # (Nt,) sum of fixel_afd in each WM voxel
    tract_afd_volume_mm3: float       # sum(voxel_afd_t) * vox_vol_mm3 / mean_wm_afd
    effective_area_mm2: float         # tract_afd_volume_mm3 / mean_streamline_length_mm
    # 3. Intrinsic Markov conductance & boundary-launched metrics
    cut_flux_profile: np.ndarray      # (n_bins,) reactive streamline flux across harmonic cuts
    boundary_hit_prob_fwd: float      # mean P(reach End | launched from Start GM-WM interface)
    boundary_hit_prob_rev: float      # mean P(reach Start | launched from End GM-WM interface)
    boundary_conductance: float       # symmetric boundary-launched Markov conductance
    # 4. Harmonic committor coordinates & along-tract profiles
    q_harmonic_t: np.ndarray          # (Nt,) Dirichlet harmonic coordinate in [0, 1] on flux graph
    q_committor_t: np.ndarray         # (Nt,) endpoint_sides committor phi_E / (phi_S + phi_E) in [0, 1]
    along_tract: dict = field(default_factory=dict)  # bin centers & profiles along q_harmonic
    # 5. Routing entropy & coherence maps
    routing_entropy_t: np.ndarray = None   # (Nt,) Shannon entropy (nats) of outgoing 26-flux
    branching_k_eff_t: np.ndarray = None   # (Nt,) exp(routing_entropy_t) in [1, 26]
    flux_coherence_t: np.ndarray = None    # (Nt,) directional coherence in [0, 1]
    summary: dict = field(default_factory=dict)


def afd_tract_metrics(bundle_res, whole_density_class, sh_rows, geo, lobes: Lobes = None,
                      vox_vol_mm3: float = 1.0):
    """Compute local fixel routing share ``n_bundle(f) / n_whole(f)``, AFD-attributed fixel & voxel
    densities, AFD-normalized tract volume (mm^3), and mean white-matter AFD integral."""
    if lobes is None:
        lobes = segment_lobes(sh_rows, geo)
    n_b, u_b = fixel_streamlines(bundle_res.density_class, sh_rows, geo, lobes)
    n_w, u_w = fixel_streamlines(whole_density_class, sh_rows, geo, lobes)
    share = np.divide(n_b, n_w, out=np.zeros_like(n_b), where=n_w > 1e-30)
    share = np.clip(share, 0.0, 1.0)
    fixel_afd = share * lobes.integral
    Nt = len(sh_rows)
    voxel_afd = np.bincount(lobes._frow, weights=fixel_afd, minlength=Nt) if lobes.n_fixels else np.zeros(Nt)
    # Normalize by mean total fODF integral across positive-AFD WM voxels so 1 mean-AFD voxel = vox_vol_mm3
    total_vox_afd = np.bincount(lobes._frow, weights=lobes.integral, minlength=Nt) if lobes.n_fixels else np.ones(Nt)
    mean_wm_afd = float(np.mean(total_vox_afd[total_vox_afd > 0])) if np.any(total_vox_afd > 0) else 1.0
    tract_vol_mm3 = float(voxel_afd.sum() * vox_vol_mm3 / max(mean_wm_afd, 1e-30))
    return share, fixel_afd, voxel_afd, tract_vol_mm3, mean_wm_afd


def sift2_optimize_launches(chain, launches, sh_rows, lobes: Lobes = None, n_iter: int = 8,
                            reg_tikhonov: float = 0.05, step_size: float = 1.0, tol: float = 1e-5,
                            max_iter: int = 5000, accel: str = "mr", vox_vol_mm3: float = 1.0,
                            valid_classes=(1, 3)):
    """Global path-consistent SIFT2 optimization on the Markov chain.

    Finds a single constant cross-sectional multiplier ``w_p = w_{0,p} * exp(f_p)`` for each
    launched seed peak-pair ``p in 0..P-1`` so that the whole-brain Markov fixel visitation
    ``n_whole(f; w)`` matches the physical fODF lobe integrals ``AFD(f) = lobes.integral[f]``
    across all white-matter fixels simultaneously:

        min_{f_p}  0.5 * sum_f (mu * n_whole(f; w_0 * exp(f)) - AFD_norm(f))^2
                 + 0.5 * reg_tikhonov * sum_p w_{0,p} * L_p * f_p^2

    Because only the seed-launch weights ``launches.pair_w`` are scaled (one constant weight per
    trajectory from start to end), every bundle segmented with the returned ``sift2_launches``
    preserves exact reactive-flux conservation (``L + inflow = n = outflow + A``) and path
    consistency along its entire length.

    Returns ``(sift2_launches, info)`` where ``sift2_launches`` is a ``Launches`` object whose
    ``pair_w`` carries the SIFT2-calibrated cross-sectional weights, and ``info`` contains
    convergence diagnostics and fixel-wise ``AFD(f)`` fit metrics before and after optimization.
    """
    from . import kernel as K
    from .solve import _Plan, solve_series

    if lobes is None:
        lobes = segment_lobes(sh_rows, chain.geo)

    Nt, nb, Ns, nc = chain.t_lin.size, chain.nb, chain.n_states, chain.ncomp
    if nc != 1:
        raise ValueError("sift2_optimize_launches requires a scalar chain (ncomp == 1)")

    # Target fixel AFD volume (normalized so mean positive WM voxel has total AFD = vox_vol_mm3)
    total_vox_afd = np.bincount(lobes._frow, weights=lobes.integral, minlength=Nt) if lobes.n_fixels else np.ones(Nt)
    mean_wm_afd = float(np.mean(total_vox_afd[total_vox_afd > 0])) if np.any(total_vox_afd > 0) else 1.0
    afd_target = (lobes.integral.astype(np.float64) / max(mean_wm_afd, 1e-30)) * float(vox_vol_mm3)

    # 1. Solve single-column whole-brain backward completion beta_whole once
    plan = _Plan(chain, {"name": "_whole"}, valid_classes)
    mask = plan.mask
    r_rhs = chain.absorb_rhs(plan.absw) * mask
    beta_whole, _ = solve_series(chain.mul_backward, r_rhs, mask, tol, max_iter, "sift2_bwd", 1, accel=accel)
    beta_whole = np.ascontiguousarray(beta_whole)

    P = launches.pair_seed.size
    seed_ok = (~plan.taboo[launches.hseed][:, plan.xcol]).astype(np.float64)
    vals = np.empty((launches.hseed.size, 1), dtype=np.float64)
    K.half_values(launches.hseed, launches.lptr, launches.lkey, launches.lval, chain.label,
                  chain.noff, chain.tindex, chain.aindex, nb, beta_whole, plan.absw, seed_ok, vals)
    V = vals.reshape(P, 2)
    pb = V[:, 0] * V[:, 1]
    coef_base = V[:, ::-1].reshape(2 * P, 1) * seed_ok

    ti = chain.tindex[launches.seed_lin]
    pt = ti[launches.pair_seed]
    on = pt >= 0
    b0 = chain.geo.vbin[launches.pair_d[on, 0]]
    b1 = chain.geo.vbin[launches.pair_d[on, 1]]
    pt_on = pt[on]
    pb_on = pb[on]
    zero_absw = np.zeros((chain.a_lin.size, 1), dtype=np.float64)
    kill = getattr(chain, "kill", None)

    frow = lobes._frow
    w_fix = lobes._w_fix

    def forward_fixels(w_pair):
        """Exact forward operator A(w_pair): streamline weights (P,) -> fixel visits (N_fixels,)."""
        coef = coef_base * np.repeat(w_pair, 2)[:, None]
        pi0 = np.zeros((Ns, 1), dtype=np.float64)
        a0 = np.zeros((chain.a_lin.size, 1), dtype=np.float64)
        K.scatter_init(launches.hseed, launches.lptr, launches.lkey, launches.lval, coef,
                       chain.label, chain.noff, chain.tindex, chain.aindex, nb, plan.taboo,
                       plan.xcol, pi0, a0)
        if kill is not None:
            pi0[kill] = 0.0
        S, _ = solve_series(chain.mul_forward, pi0, mask, tol, max_iter, "sift2_fwd", 1, accel=accel)
        dcls = (S[:, 0] * beta_whole[:, 0]).reshape(Nt, nb)
        w0 = 0.5 * (w_pair[on] * pb_on)
        np.add.at(dcls, (pt_on, b0), w0)
        np.add.at(dcls, (pt_on, b1), w0)
        dcls = np.maximum(dcls, 0.0)
        n_fix = np.einsum("ij,ij->i", dcls[frow], w_fix) if lobes.n_fixels else np.zeros(0)
        return n_fix, dcls

    def adjoint_pairs(v_fix):
        """Exact transpose operator A^T(v_fix): fixel vector (N_fixels,) -> pair sensitivities (P,)."""
        g_dcls = np.zeros((Nt, nb), dtype=np.float64)
        if lobes.n_fixels:
            np.add.at(g_dcls, frow, np.asarray(v_fix, np.float64)[:, None] * w_fix)
        g_w = np.zeros(P, dtype=np.float64)
        g_w[on] = 0.5 * pb_on * (g_dcls[pt_on, b0] + g_dcls[pt_on, b1])
        rhs_adj = (g_dcls.reshape(-1) * beta_whole[:, 0])[:, None] * mask
        if kill is not None:
            rhs_adj[kill] = 0.0
        U_adj, _ = solve_series(chain.mul_backward, rhs_adj, mask, tol, max_iter, "sift2_adj", 1, accel=accel)
        vals_adj = np.empty((2 * P, 1), dtype=np.float64)
        K.half_values(launches.hseed, launches.lptr, launches.lkey, launches.lval, chain.label,
                      chain.noff, chain.tindex, chain.aindex, nb, np.ascontiguousarray(U_adj),
                      zero_absw, seed_ok, vals_adj)
        g_w += (vals_adj * coef_base).reshape(P, 2).sum(1)
        return np.maximum(g_w, 0.0)

    # Precompute target back-projection A^T(AFD_target) once (1 backward solve)
    c_target = adjoint_pairs(afd_target)

    w0_pair = launches.pair_w.astype(np.float64).copy()
    f_log = np.zeros(P, dtype=np.float64)

    # Evaluate initial uncalibrated fixel visits (k = 0)
    n_fix_init, dcls_init = forward_fixels(w0_pair)
    mu_init = float(np.dot(n_fix_init, afd_target) / max(np.dot(n_fix_init, n_fix_init), 1e-30))
    n_fix_cur = n_fix_init.copy()
    dcls_cur = dcls_init
    mu_cur = mu_init

    def _fit_stats(n_f, mu_val):
        pred = mu_val * n_f
        cos = float(np.dot(pred, afd_target) / max(np.linalg.norm(pred) * np.linalg.norm(afd_target), 1e-30))
        r = float(np.corrcoef(pred, afd_target)[0, 1]) if len(pred) > 1 else 1.0
        ss_res = float(np.sum((pred - afd_target) ** 2))
        ss_tot = float(np.sum((afd_target - afd_target.mean()) ** 2))
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-30))
        rel_l2 = float(np.linalg.norm(pred - afd_target) / max(np.linalg.norm(afd_target), 1e-30))
        return dict(cosine=round(cos, 4), pearson_r=round(r, 4), r2=round(r2, 4), rel_l2_err=round(rel_l2, 4))

    stats_init = _fit_stats(n_fix_init, mu_init)
    history = [dict(iter=0, mu=round(mu_init, 6), **stats_init)]

    for it in range(1, n_iter + 1):
        # Back-project current predicted fixel density A^T(mu * n_fix_cur) (1 backward solve)
        d_pred = adjoint_pairs(mu_cur * n_fix_cur)
        active_p = (c_target > 1e-15) & (d_pred > 1e-15)
        log_ratio = np.zeros(P, dtype=np.float64)
        log_ratio[active_p] = np.log(c_target[active_p] / d_pred[active_p])
        # Regularized multiplicative update in log-space (exact ISRA + Tikhonov damping)
        f_log = (f_log + step_size * log_ratio) / (1.0 + step_size * reg_tikhonov)
        f_log = np.clip(f_log, -4.0, 4.0)
        w_cur = w0_pair * np.exp(f_log)

        n_fix_cur, dcls_cur = forward_fixels(w_cur)
        mu_cur = float(np.dot(n_fix_cur, afd_target) / max(np.dot(n_fix_cur, n_fix_cur), 1e-30))
        st = _fit_stats(n_fix_cur, mu_cur)
        history.append(dict(iter=it, mu=round(mu_cur, 6), **st))

    w_final = w0_pair * mu_cur * np.exp(f_log)
    sift2_launches = replace(launches, pair_w=w_final)
    info = dict(
        n_fixels=int(lobes.n_fixels),
        n_streamlines=int(P),
        n_iter=int(n_iter),
        reg_tikhonov=float(reg_tikhonov),
        mean_wm_afd=round(mean_wm_afd, 6),
        mu_init=round(mu_init, 6),
        mu_final=round(mu_cur, 6),
        initial_fit=stats_init,
        final_fit=history[-1],
        history=history,
        log_weights=f_log,
        afd_target_mm3=afd_target,
        n_fix_init_mm3=mu_init * n_fix_init,
        n_fix_sift2_mm3=mu_cur * n_fix_cur,
        whole_density_class_sift2=mu_cur * dcls_cur,
    )
    return sift2_launches, info


def routing_entropy_and_coherence(flux26, zooms=(1.0, 1.0, 1.0)):
    """Per-voxel Shannon entropy H(v) in nats, effective branching number K_eff(v) = exp(H(v))
    in [1, 26], and directional coherence in [0, 1] from the (Nt, 26) reactive edge flux."""
    f = np.maximum(np.asarray(flux26, np.float64), 0.0)
    tot = f.sum(1, keepdims=True)
    p = np.divide(f, tot, out=np.zeros_like(f), where=tot > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 1e-30, np.log(p), 0.0)
    H = -(p * log_p).sum(1)
    k_eff = np.where(tot[:, 0] > 0, np.exp(H), 0.0)
    _, coh = flux_orientation(f, zooms)
    return H, k_eff, coh


def harmonic_flux_coordinate(chain, flux26, abs_start_t, abs_end_t, visits_t=None):
    """Solve the discrete Dirichlet Laplace problem on the bundle's symmetrized reactive flux
    graph:
        (L_W + diag(g_S + g_E)) q = g_E
    where W_uv = 0.5 * (F(u->v) + F(v->u)), g_S(u) is absorption into the start end (q = 0),
    and g_E(u) is absorption into the target end (q = 1).  By the maximum principle, q(u) is
    strictly in [0, 1] on every connected component and provides a smooth intrinsic coordinate
    from start (0) to end (1)."""
    Nt = chain.t_lin.size
    f = np.maximum(np.asarray(flux26, np.float64), 0.0)
    v_tot = f.sum(1) + abs_start_t + abs_end_t if visits_t is None else np.asarray(visits_t, np.float64)
    active = v_tot > 1e-12 * max(float(v_tot.max()), 1e-30)
    if not np.any(active):
        return np.full(Nt, 0.5)

    act_idx = np.flatnonzero(active)
    n_act = act_idx.size
    remap = np.full(Nt, -1, np.int64)
    remap[act_idx] = np.arange(n_act)

    rows, cols, vals = [], [], []
    for n in range(26):
        dst = chain.nbr_t[act_idx, n]
        ok = dst >= 0
        r_dst = np.where(ok, remap[np.maximum(dst, 0)], -1)
        m = ok & (r_dst >= 0) & (f[act_idx, n] > 0)
        if np.any(m):
            rows.append(np.flatnonzero(m))
            cols.append(r_dst[m])
            vals.append(f[act_idx[m], n])

    if rows:
        G = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(n_act, n_act))
        W = 0.5 * (G + G.T)
    else:
        W = sps.csr_matrix((n_act, n_act))

    gS = np.maximum(abs_start_t[act_idx], 0.0)
    gE = np.maximum(abs_end_t[act_idx], 0.0)
    # Small regularizing conductivity along active graph so isolated leaves stay well-posed
    deg = np.asarray(W.sum(1)).ravel()
    L = sps.diags(deg + gS + gE + 1e-12 * max(float(deg.max()), 1.0)) - W
    rhs = gE + 0.5 * 1e-12 * max(float(deg.max()), 1.0)
    q_act = spla.spsolve(L.tocsc(), rhs)
    q_act = np.clip(np.nan_to_num(q_act, nan=0.5), 0.0, 1.0)

    q = np.full(Nt, 0.5)
    q[act_idx] = q_act
    return q


def cut_flux_along_coordinate(chain, flux26, q_t, cut_points=None):
    """Total reactive streamline flux crossing each isosurface q(u) < c <= q(v) for c in cut_points.
    Because MCAT launches two outward half-walks from each seed, a streamline seeded on the q < c
    side crosses the cut in the forward direction (u -> v) while a streamline seeded on the q >= c
    side crosses the cut in the reverse direction (v -> u); their sum is the total streamline
    flux crossing the cut (see test_fields.py::test_cut_flux_counts_streamlines)."""
    if cut_points is None:
        cut_points = np.linspace(0.05, 0.95, 19)
    cut_points = np.asarray(cut_points, np.float64)
    f = np.maximum(np.asarray(flux26, np.float64), 0.0)
    dst_all = chain.nbr_t
    valid = (dst_all >= 0) & (f > 0)
    u_idx, n_idx = np.nonzero(valid)
    v_idx = dst_all[u_idx, n_idx]
    w_uv = f[u_idx, n_idx]
    qu = q_t[u_idx]
    qv = q_t[v_idx]

    flux_cuts = np.zeros(len(cut_points))
    for i, c in enumerate(cut_points):
        cross = ((qu < c) & (qv >= c)) | ((qu >= c) & (qv < c))
        flux_cuts[i] = float(w_uv[cross].sum())
    return cut_points, flux_cuts


def boundary_conductance_metrics(chain, fields, S_roi_vol, E_roi_vol, tol=1e-6, max_iter=5000):
    """Boundary-launched Markov hitting probability and conductance between S_roi_vol and E_roi_vol,
    without interior white-matter seeding bias.
    Uses the bundle's avoid-set masks and signs from ``fields.plan``."""
    from .solve import solve_series

    p = fields.plan
    pad = np.zeros(chain.pshape, bool)

    def to_a(m):
        q = pad.copy()
        q[1:-1, 1:-1, 1:-1] = m
        return q.ravel()[chain.a_lin]

    a_S = to_a(S_roi_vol)
    a_E = to_a(E_roi_vol)
    not_taboo_a = ~p.taboo[chain.a_lin]  # (Na, nX)
    absw_S = (a_S[:, None] & not_taboo_a).astype(np.float64)
    absw_E = (a_E[:, None] & not_taboo_a).astype(np.float64)

    st = np.repeat(chain.t_lin, chain.nb)
    mask_x = (~p.taboo[st]).astype(np.uint8)
    if getattr(chain, "kill", None) is not None:
        mask_x[chain.kill] = 0

    r_S = chain.absorb_rhs(absw_S) * mask_x
    r_E = chain.absorb_rhs(absw_E) * mask_x

    # Solve backward completion to E (and to S) across the nX avoid sets
    r_both = np.concatenate([r_E, r_S], axis=1)
    m_both = np.concatenate([mask_x, mask_x], axis=1)
    beta_both, _ = solve_series(chain.mul_backward, r_both.copy(), m_both, tol, max_iter,
                                "boundary_cond", chain.ncomp, accel="mr")
    beta_E = beta_both[:, :p.nX]
    beta_S = beta_both[:, p.nX:]

    # Boundary launch states: if state s=(v, b) steps into S_roi with weight r_S(s, 0),
    # then a walker entering WM from S_roi launches into the antipodal class (v, anti_b)!
    anti_v = chain.geo.antipode()
    anti_b = np.array([int(chain.geo.vbin[anti_v[chain.geo.bin_idx[chain.geo.bin_ptr[b]]]])
                       for b in range(chain.nb)], dtype=np.int64)
    Nt, nb = chain.t_lin.size, chain.nb
    pi_S = r_S[:, 0].reshape(Nt, nb)[:, anti_b].reshape(-1) * mask_x[:, 0]
    pi_E = r_E[:, 0].reshape(Nt, nb)[:, anti_b].reshape(-1) * mask_x[:, 0]

    # Signed inclusion-exclusion over the nX avoid sets
    beta_E_net = np.clip(beta_E @ p.signs, 0.0, 1.0)
    beta_S_net = np.clip(beta_S @ p.signs, 0.0, 1.0)

    cond_fwd = float(pi_S @ beta_E_net)
    cond_rev = float(pi_E @ beta_S_net)
    hit_fwd = float(cond_fwd / max(pi_S.sum(), 1e-30))
    hit_rev = float(cond_rev / max(pi_E.sum(), 1e-30))
    cond_sym = 0.5 * (cond_fwd + cond_rev)
    return hit_fwd, hit_rev, cond_sym, beta_S, beta_E, mask_x


def _project_endpoints_to_transient(chain, ep_a):
    """Distribute absorbing-voxel endpoint mass ``ep_a`` (Na,) onto 26-adjacent transient voxels (Nt,)."""
    from .geometry import OFFSETS

    t_map = np.full(chain.pshape, -1, np.int64)
    t_map.ravel()[chain.t_lin] = np.arange(chain.t_lin.size)
    a_ijk = np.stack(np.unravel_index(chain.a_lin, chain.pshape), 1)
    nz = np.flatnonzero(ep_a > 0)
    out = np.zeros(chain.t_lin.size, np.float64)
    if nz.size == 0:
        return out
    ijk = a_ijk[nz]
    w = ep_a[nz]
    nbr_t = np.empty((nz.size, 26), np.int64)
    for n, o in enumerate(OFFSETS):
        nb = ijk + o
        ok = np.all((nb >= 0) & (nb < chain.pshape), 1)
        idx = np.full(nz.size, -1, np.int64)
        idx[ok] = t_map[tuple(nb[ok].T)]
        nbr_t[:, n] = idx
    cnt = np.maximum((nbr_t >= 0).sum(1), 1)
    share = w / cnt
    for n in range(26):
        t_idx = nbr_t[:, n]
        m = t_idx >= 0
        np.add.at(out, t_idx[m], share[m])
    return out


def quantify_bundle(chain, bundle_res, whole_density_class, sh_rows, affine, lobes: Lobes = None,
                    start_mask=None, end_mask=None, n_bins=20, scalar_maps=None, tol=1e-6,
                    compute_boundary_conductance=True):
    """Compute all 5 biologically interpretable quantifications for a segmented bundle.
    Works with either ``bundle_res.fields`` (from ``keep_fields=True``) or ``bundle_res.flux26``
    (computed automatically by ``segment_bundles`` when ``chain.ncomp == 1``)."""
    fields = bundle_res.fields
    if fields is None and getattr(bundle_res, "flux26", None) is None:
        raise ValueError("quantify_bundle requires bundle_res.fields or bundle_res.flux26")
    if start_mask is None:
        start_mask = getattr(bundle_res, "start_mask", None)
    if end_mask is None:
        end_mask = getattr(bundle_res, "end_mask", None)
    zooms = np.linalg.norm(np.asarray(affine, float)[:3, :3], axis=0)
    vox_vol_mm3 = float(np.abs(np.linalg.det(np.asarray(affine, float)[:3, :3])))
    n_streamlines = max(float(bundle_res.info.get("expected_bundle_streamlines", bundle_res.seed_prob.sum())), 1e-30)

    # 3 & 4. Endpoint split, committors, harmonic flux coordinate, and boundary conductance
    ep_vol = chain.to_volume(bundle_res.endpoints_a, which="absorbing")
    try:
        S_roi, E_roi, ends_how = split_ends(ep_vol, affine, start_mask, end_mask)
        has_ends = True
    except ValueError:
        S_roi = np.zeros(ep_vol.shape, bool)
        E_roi = np.zeros(ep_vol.shape, bool)
        ends_how = "none"
        has_ends = False

    pad = np.zeros(chain.pshape, bool)
    qS, qE = pad.copy(), pad.copy()
    qS[1:-1, 1:-1, 1:-1] = S_roi
    qE[1:-1, 1:-1, 1:-1] = E_roi
    a_S = qS.ravel()[chain.a_lin]
    a_E = qE.ravel()[chain.a_lin]

    if fields is not None and has_ends:
        phi_SE = fields.endpoint_sides([S_roi, E_roi], tol=tol)  # (Nt, 2)
        phi_tot = phi_SE.sum(1)
        q_committor_t = np.divide(phi_SE[:, 1], phi_tot, out=np.full(chain.t_lin.size, 0.5), where=phi_tot > 1e-12)
        q_committor_t = np.clip(q_committor_t, 0.0, 1.0)
        r_S_col = chain.absorb_rhs(fields.absw * a_S[:, None])
        r_E_col = chain.absorb_rhs(fields.absw * a_E[:, None])
        abs_S_t = np.maximum(fields.per_voxel((fields.S * r_S_col) @ fields.colsign), 0.0)
        abs_E_t = np.maximum(fields.per_voxel((fields.S * r_E_col) @ fields.colsign), 0.0)
        flux26 = fields.edge_flux()
        visits_t = fields.visits()
    else:
        abs_S_t = _project_endpoints_to_transient(chain, bundle_res.endpoints_a * a_S) if has_ends else np.zeros(chain.t_lin.size)
        abs_E_t = _project_endpoints_to_transient(chain, bundle_res.endpoints_a * a_E) if has_ends else np.zeros(chain.t_lin.size)
        flux26 = fields.edge_flux() if fields is not None else np.asarray(bundle_res.flux26, np.float64)
        visits_t = fields.visits() if fields is not None else np.asarray(bundle_res.density_t, np.float64)
        q_committor_t = None

    # 1. Exact transition-weighted physical length (TDI in mm) & expected streamline length (mm)
    # Computed directly from the conserved transition flux F(u, n) * ||offset_n * zooms||_2,
    # avoiding step-0 seed inflation and class-average approximations.
    step_lengths_mm = np.linalg.norm(OFFSETS.astype(np.float64) * zooms[None, :], axis=1)  # (26,)
    tdi_mm_t = (np.maximum(flux26, 0.0) * step_lengths_mm[None, :]).sum(1)
    abs_total_mm = float(np.maximum(bundle_res.endpoints_a, 0.0).sum()) * 0.5 * float(np.mean(zooms))
    mean_length_mm = float((tdi_mm_t.sum() + abs_total_mm) / n_streamlines)

    # 2. Local fixel AFD attribution (bundle-wise fixel AFD share)
    share, fixel_afd, voxel_afd_t, tract_vol_mm3, mean_wm_afd = afd_tract_metrics(
        bundle_res, whole_density_class, sh_rows, chain.geo, lobes=lobes, vox_vol_mm3=vox_vol_mm3
    )
    eff_area_mm2 = float(tract_vol_mm3 / max(mean_length_mm, 1e-30))

    q_harmonic_t = harmonic_flux_coordinate(chain, flux26, abs_S_t, abs_E_t, visits_t=visits_t) if has_ends else np.full(chain.t_lin.size, 0.5)
    if q_committor_t is None:
        q_committor_t = q_harmonic_t

    # Cross-sectional cut flux along q_harmonic_t
    cut_edges = np.linspace(0.0, 1.0, n_bins + 1)
    cut_centers = 0.5 * (cut_edges[:-1] + cut_edges[1:])
    _, cut_flux = cut_flux_along_coordinate(chain, flux26, q_harmonic_t, cut_centers) if has_ends else (cut_centers, np.zeros(n_bins))

    # Boundary-launched conductance
    if fields is not None and compute_boundary_conductance and has_ends:
        hit_fwd, hit_rev, cond_sym, _, _, _ = boundary_conductance_metrics(
            chain, fields, S_roi, E_roi, tol=tol
        )
    else:
        hit_fwd = hit_rev = cond_sym = 0.0

    # 5. Routing entropy, effective branching number K_eff, and flux coherence
    H_t, k_eff_t, coh_t = routing_entropy_and_coherence(flux26, zooms=zooms)

    # Along-tract profiles binned by q_harmonic_t
    bin_idx = np.clip(np.digitize(q_harmonic_t, cut_edges) - 1, 0, n_bins - 1)
    w_flux = np.maximum(visits_t, 0.0)
    bin_w = np.bincount(bin_idx, weights=w_flux, minlength=n_bins)

    def _profile(arr):
        num = np.bincount(bin_idx, weights=w_flux * np.asarray(arr, np.float64), minlength=n_bins)
        return np.divide(num, bin_w, out=np.zeros(n_bins), where=bin_w > 0)

    # Cross-sectional AFD area per harmonic bin using actual physical path length spent in each bin
    bin_afd_vol = np.bincount(bin_idx, weights=voxel_afd_t, minlength=n_bins) * vox_vol_mm3 / max(mean_wm_afd, 1e-30)
    bin_tdi_mm = np.bincount(bin_idx, weights=tdi_mm_t, minlength=n_bins)
    bin_flux_ref = np.where(cut_flux > 1e-6 * n_streamlines, cut_flux, n_streamlines)
    bin_len_mm = bin_tdi_mm / np.maximum(bin_flux_ref, 1e-30)
    tot_bin_len = float(bin_len_mm.sum())
    if tot_bin_len > 1e-12:
        bin_len_mm = bin_len_mm * (mean_length_mm / tot_bin_len)
    else:
        bin_len_mm = np.full(n_bins, max(mean_length_mm / n_bins, 1e-30))
    bin_afd_area_mm2 = np.divide(bin_afd_vol, bin_len_mm, out=np.zeros(n_bins), where=bin_len_mm > 1e-12)

    along_tract = dict(
        q_centers=cut_centers,
        cut_flux=cut_flux,
        bin_length_mm=bin_len_mm,
        afd_area_mm2=bin_afd_area_mm2,
        voxel_afd=_profile(voxel_afd_t),
        coherence=_profile(coh_t),
        branching_k_eff=_profile(k_eff_t),
        routing_entropy=_profile(H_t),
    )
    for k_name, s_map in (scalar_maps or {}).items():
        s_arr = np.asarray(s_map, np.float64)
        if s_arr.ndim == 3:
            pad_s = np.zeros(chain.pshape, np.float64)
            pad_s[1:-1, 1:-1, 1:-1] = s_arr
            s_arr = pad_s.ravel()[chain.t_lin]
        along_tract[k_name] = _profile(s_arr)

    active = w_flux > 1e-3 * max(float(w_flux.max()), 1e-30)
    # Compare trunk (middle 50%: q in [0.25, 0.75]) vs cortical terminations (q < 0.15 or q > 0.85)
    in_trunk = active & (q_harmonic_t >= 0.25) & (q_harmonic_t <= 0.75)
    in_ends = active & ((q_harmonic_t < 0.15) | (q_harmonic_t > 0.85))
    w_sum = lambda m: max(float(w_flux[m].sum()), 1e-30)
    w_mean = lambda arr, m: float((w_flux[m] * arr[m]).sum() / w_sum(m)) if np.any(m) else 0.0

    summary = dict(
        ends_split=ends_how,
        expected_streamlines=round(n_streamlines, 2),
        mean_streamline_length_mm=round(mean_length_mm, 2),
        tract_afd_volume_mm3=round(tract_vol_mm3, 2),
        effective_afd_area_mm2=round(eff_area_mm2, 2),
        median_cut_flux=round(float(np.median(cut_flux)), 2),
        boundary_hit_prob_fwd=round(hit_fwd, 5),
        boundary_hit_prob_rev=round(hit_rev, 5),
        boundary_conductance=round(cond_sym, 3),
        trunk_coherence=round(w_mean(coh_t, in_trunk), 4),
        cortical_end_coherence=round(w_mean(coh_t, in_ends), 4),
        trunk_branching_k_eff=round(w_mean(k_eff_t, in_trunk), 3),
        cortical_end_branching_k_eff=round(w_mean(k_eff_t, in_ends), 3),
    )
    return BundleQuantification(
        name=bundle_res.name,
        tdi_mm_t=tdi_mm_t,
        mean_streamline_length_mm=mean_length_mm,
        fixel_share=share,
        fixel_afd=fixel_afd,
        voxel_afd_t=voxel_afd_t,
        tract_afd_volume_mm3=tract_vol_mm3,
        effective_area_mm2=eff_area_mm2,
        cut_flux_profile=cut_flux,
        boundary_hit_prob_fwd=hit_fwd,
        boundary_hit_prob_rev=hit_rev,
        boundary_conductance=cond_sym,
        q_harmonic_t=q_harmonic_t,
        q_committor_t=q_committor_t,
        along_tract=along_tract,
        routing_entropy_t=H_t,
        branching_k_eff_t=k_eff_t,
        flux_coherence_t=coh_t,
        summary=summary,
    )
