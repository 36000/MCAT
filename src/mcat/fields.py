"""Bundle fields: the solution columns behind a bundle's maps, and what they give.

``segment_bundles(..., keep_fields=True)`` keeps, per bundle, the forward occupancy ``S``
and the backward completion probability ``beta`` of every solution column (one column per
avoid set x end kind, see solve.py), the launch mass ``launch`` the forward series started
from, and the mass absorbed on the launch step ``a0``.  With the column signs of the
inclusion-exclusion, every bundle-conditioned expectation is a signed sum over columns:

  visits    n(s)     = sum_c sign_c S[s, c] beta[s, c]        (the density, per state)
  edge flux F(s, s') = sum_c sign_c S[s, c] Q(s, s') beta[s', c]
  absorption from s  = sum_c sign_c S[s, c] r_c(s)            (r_c = R absw_c: valid end next)
  launch mass at s   = sum_c sign_c launch[s, c] beta[s, c]

F(s, s') is the expected number of bundle-streamline steps along the transition s -> s', so
it is conserved: at every state, launch + inflow = visits = outflow + absorption.  Summed
over any cut through the bundle it gives the expected streamlines crossing it.  Reduced to
voxel -> neighbour edges (``edge_flux``, 26 columns in ``geometry.OFFSETS`` order) it is the
bundle's own routing through each voxel, independent of the other bundles using the same
fODF lobes.  A committor (``committors``) replaces the end kind of each column with a
subset of the valid endpoints: the fraction of the visits at a state whose walk ends in that
subset.

See docs/fields.md for the method, its uses and what was tried with it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from . import kernel as K
from .geometry import OFFSETS

log = logging.getLogger("mcat")


@dataclass
class BundleFields:
    chain: object
    plan: object             # solve._Plan: taboo sets, signs, end kinds, absw
    S: np.ndarray            # (Ns, C) occupancy of the weighted half-walks under each column
    beta: np.ndarray         # (Ns, C) P(avoid the column's set and end in its kind | state)
    launch: np.ndarray       # (Ns, C) mass the forward series started from (first exits of the seeds)
    a0: np.ndarray           # (Na, C) mass absorbed on the launch step

    # ---------------------------------------------------------------- bookkeeping
    @property
    def colsign(self):
        return self.plan.signs[self.plan.xcol]

    @property
    def absw(self):
        return self.plan.absw

    @property
    def n_columns(self):
        return self.S.shape[1]

    def mask(self):
        """(Ns, C) uint8: state alive under the column's avoid set."""
        ch, p = self.chain, self.plan
        st = np.repeat(ch.t_lin, ch.nb)
        m = ~p.taboo[st][:, p.xcol]
        kill = getattr(ch, "kill", None)     # states past the soft maximum length
        if kill is not None:
            m[kill] = False
        return m.astype(np.uint8)

    def per_voxel(self, x):
        """Sum a per-state quantity over the voxel's direction classes: (Ns, ...) -> (Nt, ...)."""
        x = np.asarray(x)
        return x.reshape((self.chain.t_lin.size, self.chain.nb) + x.shape[1:]).sum(1)

    # ---------------------------------------------------------------- fields
    def state_visits(self):
        """(Ns,) expected visits of bundle streamlines per (voxel, direction class) state.
        Summed per voxel this is ``result.density_t`` without the streamlines' step 0 at
        their seed (``result.seed_prob`` at the seed voxels)."""
        return (self.S * self.beta) @ self.colsign

    def visits(self):
        return self.per_voxel(self.state_visits())

    def launch_mass(self):
        """(Ns,) bundle-streamline mass entering each state on the launch step."""
        return (self.launch * self.beta) @ self.colsign

    def absorption(self):
        """(Ns,) expected bundle-streamline absorptions from each state (next step into a
        valid endpoint of the column)."""
        r = self.chain.absorb_rhs(self.absw)
        return (self.S * r) @ self.colsign

    def edge_flux(self):
        """(Nt, 26) expected bundle-streamline steps from each white-matter voxel into each
        of its 26 neighbours (columns in ``geometry.OFFSETS`` order; 0 where the neighbour is
        not white matter: steps out of white matter are ``absorption``)."""
        ch = self.chain
        out = np.empty((ch.t_lin.size, 26))
        K.stencil_flux(ch.Q_ptr, ch.Q_slot, ch.Q_w, ch.tmpl_n, ch.tmpl_b, ch.nbr_t, ch.nb,
                       np.ascontiguousarray(self.S), np.ascontiguousarray(self.beta),
                       np.ascontiguousarray(self.colsign), out)
        return out

    def inflow(self, flux26=None):
        """(Nt,) bundle-streamline steps arriving in each voxel from its neighbours."""
        f = self.edge_flux() if flux26 is None else flux26
        ch = self.chain
        out = np.zeros(ch.t_lin.size)
        for n in range(26):
            dst = ch.nbr_t[:, n]
            ok = dst >= 0
            np.add.at(out, dst[ok], f[ok, n])
        return out

    def through_flux(self, flux26=None):
        """(Nt,) bundle-streamline steps leaving each voxel (to neighbours or absorbed):
        the voxel's visits, by conservation."""
        f = self.edge_flux() if flux26 is None else flux26
        return f.sum(1) + self.per_voxel(self.absorption())

    def conservation(self, flux26=None):
        """Flux balance per voxel, relative to the bundle's expected streamlines:
        ``in`` = max |launch + inflow - visits|, ``out`` = max |visits - outflow - absorption|.
        Both are at the solve tolerance when the series converged."""
        f = self.edge_flux() if flux26 is None else flux26
        n = self.visits()
        N = max(float(self.plan_streamlines()), 1e-300)
        r_in = self.per_voxel(self.launch_mass()) + self.inflow(f) - n
        r_out = n - f.sum(1) - self.per_voxel(self.absorption())
        return dict(**{"in": float(np.abs(r_in).max() / N), "out": float(np.abs(r_out).max() / N)},
                    expected_streamlines=N, residual_in=r_in, residual_out=r_out)

    def plan_streamlines(self):
        """Expected bundle streamlines: half the valid endpoint mass (two ends per streamline)."""
        ends = (self.a0 + self.chain.absorbed_from(np.ascontiguousarray(self.S))) * self.absw
        return 0.5 * float((ends @ self.colsign).sum())

    # ---------------------------------------------------------------- committors
    def committors(self, targets, tol=1e-8, max_iter=5000, per_state=False):
        """Where the bundle's streamlines end.  ``targets``: (Na, T) bool over the chain's
        absorbing voxels (``chain.a_lin``), one column per endpoint subset.  Returns
        (Nt, T) (or (Ns, T) with ``per_state``): the expected visits at each voxel by
        bundle streamlines whose walk from there ends in the subset, still completing the
        bundle's requirements.  Divided by ``visits()`` it is the committor; the columns of
        subsets that partition the valid endpoints sum to the visits."""
        from .solve import solve_series

        ch = self.chain
        T = np.asarray(targets, np.float64).reshape(len(ch.a_lin), -1)
        C, nT = self.n_columns, T.shape[1]
        absw = (self.absw[:, :, None] * T[:, None, :]).reshape(len(ch.a_lin), C * nT)
        mask = np.repeat(self.mask(), nT, axis=1)
        r = ch.absorb_rhs(absw) * mask
        beta, info = solve_series(ch.mul_backward, r, mask, tol, max_iter, "committor", ch.ncomp)
        log.info("committors: %d columns, %d steps", C * nT, info["iterations"])
        beta = beta.reshape(ch.n_states, C, nT)
        phi = np.einsum("sc,sct,c->st", self.S, beta, self.colsign)
        return phi if per_state else self.per_voxel(phi)

    def endpoint_sides(self, side_masks, **kw):
        """``committors`` for sides given as boolean volumes on the chain grid (e.g. a
        bundle's start and end ROIs): returns (Nt, len(side_masks))."""
        ch = self.chain
        pad = np.zeros(ch.pshape, bool)
        cols = []
        for m in side_masks:
            q = pad.copy()
            q[1:-1, 1:-1, 1:-1] = m
            cols.append(q.ravel()[ch.a_lin])
        return self.committors(np.stack(cols, 1), **kw)


# -------------------------------------------------------------------- flux geometry
def flux_orientation(flux26, zooms=(1.0, 1.0, 1.0)):
    """Principal direction of the flux through each voxel from its 26 edge fluxes: the
    leading eigenvector of sum_n F_n d_n d_n^T with d_n the unit neighbour offsets (mm).
    Returns (dirs (Nt, 3) unit vectors, coherence (Nt,) in [0, 1]: 1 when all flux is along
    one axis, 0 when isotropic)."""
    d = OFFSETS * np.asarray(zooms, float)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    T = np.einsum("nb,bi,bj->nij", np.asarray(flux26, float), d, d)
    w, v = np.linalg.eigh(T)
    tot = w.sum(1)
    coh = np.divide(w[:, -1] - 0.5 * (w[:, 0] + w[:, 1]), tot, out=np.zeros_like(tot), where=tot > 0)
    return v[:, :, -1], coh


# -------------------------------------------------------------------- endpoint patches
def endpoint_patches(endpoints, sigma_vox=2.5, min_share=0.02, prominence=0.2):
    """Patches of a bundle's endpoint map: a marker watershed of the blurred map, basins
    merged when the saddle between them is at least (1 - prominence) x the lower peak,
    basins carrying under ``min_share`` of the mass merged into the neighbour with the
    highest saddle (isolated small basins: label 0).  Returns (labels (same shape, int16,
    1 = largest share), shares (list, descending)).

    Use it per side: the map of one endpoint ROI, or ``core.split_ends`` to separate the
    two ends.  Patches pooled over many subjects in a template space are more stable than
    one subject's (docs/fields.md)."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    vol = np.clip(np.nan_to_num(np.asarray(endpoints, np.float64)), 0, None)
    blur = ndi.gaussian_filter(vol, sigma_vox)
    if blur.max() <= 0:
        return np.zeros(vol.shape, np.int16), []
    sup = blur > 0.01 * blur.max()
    peaks = peak_local_max(blur, min_distance=3, threshold_abs=0.01 * blur.max(),
                           labels=sup.astype(int), exclude_border=False)
    markers = np.zeros(vol.shape, int)
    for k, p in enumerate(peaks):
        markers[tuple(p)] = k + 1
    ws = watershed(-blur, markers, mask=sup)
    npk = len(peaks)
    pk = np.concatenate([[0.0], [blur[tuple(p)] for p in peaks]])
    mass = np.bincount(ws.ravel(), weights=vol.ravel(), minlength=npk + 1)
    lo_all, hi_all, v_all = [], [], []
    for ax in range(3):                      # saddles between 6-adjacent basins
        a = np.moveaxis(ws, ax, 0)
        b, a = a[1:], a[:-1]
        ia = np.moveaxis(blur, ax, 0)
        ib, ia = ia[1:], ia[:-1]
        m = (a > 0) & (b > 0) & (a != b)
        lo_all.append(np.minimum(a[m], b[m]))
        hi_all.append(np.maximum(a[m], b[m]))
        v_all.append(np.minimum(ia[m], ib[m]))
    lo, hi, sv = (np.concatenate(x) for x in (lo_all, hi_all, v_all))
    if lo.size:
        key = lo.astype(np.int64) * (npk + 1) + hi
        ukey, inv = np.unique(key, return_inverse=True)
        smax = np.zeros(ukey.size)
        np.maximum.at(smax, inv, sv)
        lo, hi, sv = ukey // (npk + 1), ukey % (npk + 1), smax
    parent = np.arange(npk + 1)

    def roots():
        r = parent.copy()
        while True:
            rr = r[r]
            if np.array_equal(rr, r):
                return r
            r = rr

    def merge(drop, keep):
        parent[drop] = keep
        mass[keep] += mass[drop]
        mass[drop] = 0.0
        pk[keep] = max(pk[keep], pk[drop])

    while lo.size:                           # prominence merges, strongest saddle first
        r = roots()
        ra, rb = r[lo], r[hi]
        ok = ra != rb
        if not ok.any():
            break
        lowpk = np.minimum(pk[ra], pk[rb])
        score = np.where(ok & (sv >= (1 - prominence) * lowpk), sv / np.maximum(lowpk, 1e-300), -1.0)
        j = int(np.argmax(score))
        if score[j] < 0:
            break
        a_, b_ = ra[j], rb[j]
        keep, drop = (a_, b_) if pk[a_] >= pk[b_] else (b_, a_)
        merge(drop, keep)
    tot = vol.sum()
    unassigned = set()
    while True:                              # small basins
        r = roots()
        live = [k for k in np.unique(r) if k > 0 and k not in unassigned]
        small = [k for k in live if mass[k] < min_share * tot]
        if not small:
            break
        k = min(small, key=lambda q: mass[q])
        ra, rb = r[lo], r[hi]
        m = ((ra == k) | (rb == k)) & (ra != rb)
        if not m.any():
            unassigned.add(k)
            continue
        j = np.flatnonzero(m)[np.argmax(sv[m])]
        other = rb[j] if ra[j] == k else ra[j]
        if other in unassigned:
            unassigned.add(k)
            continue
        merge(k, other)
    lab = roots()[ws]
    for k in unassigned:
        lab[lab == k] = 0
    order = sorted([k for k in np.unique(lab) if k > 0], key=lambda k: -mass[k])
    out = np.zeros(vol.shape, np.int16)
    for i, k in enumerate(order, 1):
        out[lab == k] = i
    return out, [float(mass[k] / tot) for k in order]


def restrict_endpoints(spec, start=None, end=None):
    """A bundle spec whose start / end ROIs are cut down to the given masks (intersected
    with the spec's own ROI when it has one): the endpoint filter of docs/fields.md.
    Re-segment with ``segment_bundles``; the exact result needs the re-solve (the stored
    fields bake the partner half-walk's completion under the old endpoint rule)."""
    out = dict(spec)
    for k, m in (("start", start), ("end", end)):
        if m is not None:
            out[k] = m if spec.get(k) is None else (spec[k] & m)
    return out
