"""Bundle segmentation with include / exclude / endpoint ROIs, analytically.

A streamline from a seed is a pair of half-walks launched along +p and -p for each fODF peak
p.  The bundle event is

    passes every include ROI (either half) AND avoids every exclude ROI (both halves)
    AND both ends are valid (grey matter or brain edge) AND >= 1 end in the endpoint ROI.

Include ROIs are handled by inclusion-exclusion over "avoid" sets, P(hit A and B) =
sum_T (-1)^|T| P(avoid union(T)), and avoiding a set is an exclusion ROI.  For each avoid
set X and end kind k, one backward solve beta = (I - M Q)^-1 M r gives every state's
probability of completing the requirements (shared by all seeds); one forward solve from
every half's launch, weighted by its partner's completion probability, gives the occupancy.
Occupancy x completion is the bundle density (transition path theory's reactive density).
Each solve is a column; bundles are packed into groups of columns that share every pass
over the transition matrix.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from . import kernel as K
from .chain import VALID_ENDS

log = logging.getLogger("mcat")

class Clock:
    """Wall time per stage, summed over calls: ``with clock("forward"): ...``."""

    def __init__(self):
        self.t = {}

    def add(self, key, seconds):
        self.t[key] = self.t.get(key, 0.0) + seconds

    @contextmanager
    def __call__(self, key):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(key, time.perf_counter() - t0)

    def report(self, title, total=None):
        """One log line per stage, largest first, with its share of ``total`` (default: the sum)."""
        total = total or sum(self.t.values()) or 1e-9
        log.info("%s (%.0fs):", title, total)
        for k, v in sorted(self.t.items(), key=lambda kv: -kv[1]):
            log.info("  %-24s %8.1fs  %5.1f%%", k, v, 100 * v / total)


STALL_WINDOW, STALL_RATIO = 200, 0.9   # stalled: remaining mass fell < 10% in 200 steps
FLOAT_BLOCKS_PER_COLUMN = 6    # float64 arrays per solution column at the peak
OVERLENGTH_SURVIVAL = 0.5      # states where more walkers than this outlast the soft max length


def solve_series(mul, x0, mask, tol, max_iter, what, ncomp=1, verbose=False):
    """total = sum_t (M A)^t x0 per column (M = diag(mask)), A x = mul(x, out), by power
    iteration until the remaining mass (mass rows, relative to the largest column's start) is
    below ``tol``.  Stops early when it stalls (mass circulating in states that almost never
    get absorbed) and reports where that mass sits as ``info["trapped"]``.  ``x0`` is consumed.
    ``verbose``: log the time per step (matrix product / masking and summing) every 100 steps.
    Returns (total, info)."""
    t_start = time.perf_counter()
    t_mul = t_acc = 0.0
    w_mul = w_acc = 0.0      # since the last verbose line
    total = x0.copy()
    term, nxt = x0, np.empty_like(x0)
    mask = np.ascontiguousarray(mask)
    rows = slice(None, None, ncomp)
    scale = max(np.abs(x0[rows]).sum(0).max(), 1e-300)
    info = dict(iterations=0, residual=0.0, stalled=False)
    ref = None
    for it in range(1, max_iter + 1):
        t0 = time.perf_counter()
        mul(term, nxt)
        t1 = time.perf_counter()
        res = float(K.mask_accumulate(nxt, mask, total, ncomp).max()) / scale
        t2 = time.perf_counter()
        t_mul += t1 - t0
        t_acc += t2 - t1
        w_mul += t1 - t0
        w_acc += t2 - t1
        term, nxt = nxt, term
        info.update(iterations=it, residual=res)
        if it % 100 == 0:
            log.info("  %s step %d: remaining mass %.2e", what, it, res)
            if verbose:
                log.info("  [timing] %s steps %d-%d: %.1f ms/step (matrix product %.1f, mask+sum %.1f); "
                         "%d rows x %d columns", what, it - 99, it, 10 * (w_mul + w_acc), 10 * w_mul,
                         10 * w_acc, x0.shape[0], x0.shape[1])
                w_mul = w_acc = 0.0
        if res < tol:
            break
        if not np.isfinite(res) or res > 1e6:
            raise RuntimeError(f"{what} solve diverges (remaining mass {res:.1e} after {it} steps)")
        if it % STALL_WINDOW == 0:
            if ref is not None and res > STALL_RATIO * ref:
                info["stalled"] = True
                log.warning("%s solve stalled at step %d: remaining mass %.2e circulates in states "
                            "that (almost) never get absorbed; stopping here", what, it, res)
                break
            ref = res
    if info["residual"] >= tol:     # stalled or max_iter reached
        info["trapped"] = np.abs(term[rows]).sum(1).astype(np.float32) / scale
    n = max(info["iterations"], 1)
    info.update(seconds=round(time.perf_counter() - t_start, 1), ms_per_step=round(1e3 * (t_mul + t_acc) / n, 2))
    if verbose:
        log.info("  [timing] %s: %d steps in %.1fs, %.1f ms/step (matrix product %.1f, mask+sum %.1f)",
                 what, n, info["seconds"], 1e3 * (t_mul + t_acc) / n, 1e3 * t_mul / n, 1e3 * t_acc / n)
    return total, info


# ------------------------------------------------------------------ soft maximum length
def overlength_states(chain, max_length_mm, zooms):
    """Direction states from which most walkers are still walking after ``max_length_mm``:
    the survival probability u_T = Q^T 1 (Q: one step among white-matter states) after
    T = max_length / smallest voxel side steps -- every step crosses into a neighbouring voxel,
    so a walk of T steps is at least that long.  Traps (white matter the turning limit makes
    hard to leave, e.g. a ring around a cavity) keep u_T near 1, ordinary states drain
    within a tract's length.  Bundle ROIs only add ways out, so this holds for every bundle.
    Returns (kill: bool per state, survival per state, T)."""
    T = int(np.ceil(max_length_mm / float(np.min(np.abs(zooms)))))
    nc = chain.ncomp
    u = np.zeros((chain.n_states * nc, 1))
    u[::nc] = 1.0                                        # the mass rows (moments: position-averaged)
    out = np.empty_like(u)
    for _ in range(T):
        chain.mul_backward(u, out)
        u, out = out, u
    surv = u[::nc, 0].copy()
    return surv >= OVERLENGTH_SURVIVAL, surv, T


# ------------------------------------------------------------------ launches
@dataclass
class Launches:
    """First exits of every half-walk of every seed (independent of the bundle)."""
    seeds: np.ndarray      # (S, 3) voxel indices
    seed_lin: np.ndarray   # (S,) padded linear indices
    pair_seed: np.ndarray  # (P,) seed of each streamline (peak pair)
    pair_w: np.ndarray     # (P,) weight; sums to 1 per seed
    pair_d: np.ndarray     # (P, 2) the two signed peak directions (sphere vertices)
    hseed: np.ndarray      # (2P,) padded linear index of each half's seed (partner = h ^ 1)
    lptr: np.ndarray       # (2P + 1,) sparse first exits: keys n * nb + b, values (ncomp,)
    lkey: np.ndarray
    lval: np.ndarray
    dead: np.ndarray       # (2P,)

    @property
    def nbytes(self):
        return self.lval.nbytes + self.lkey.nbytes + self.lptr.nbytes


def prepare_launches(chain, seeds, chunk=4096):
    """One streamline per seed voxel, split over its fODF peak pairs {p, -p}."""
    t0 = time.time()
    seeds = np.asarray(seeds, np.int64).reshape(-1, 3)
    seed_lin = chain.lin(seeds)
    anti = chain.geo.antipode()
    pair_seed, pair_w, pair_d = [], [], []
    for si, v in enumerate(seed_lin):
        peaks = chain.seed_peaks(v)
        pairs = {}
        for p in peaks:
            key = (min(p, anti[p]), max(p, anti[p]))
            pairs[key] = pairs.get(key, 0.0) + 1.0 / peaks.size
        for (p, q), w in pairs.items():
            pair_seed.append(si), pair_w.append(w), pair_d.append((p, q))
    pair_seed = np.asarray(pair_seed, np.int64)
    pair_w = np.asarray(pair_w, np.float64)
    pair_d = np.asarray(pair_d, np.int64).reshape(-1, 2)
    P, nc, nb = pair_seed.size, chain.ncomp, chain.nb
    counts = np.zeros(2 * P, np.int64)
    keys, vals = [], []
    dead = np.empty(2 * P)
    for s0 in range(0, P, chunk):
        n = min(P, s0 + chunk) - s0
        o = np.empty((2 * n, 26, nb, nc))
        d = np.empty(2 * n)
        chain.launch_halves(seed_lin[pair_seed[s0:s0 + n]], np.arange(0, 2 * n + 1, 2, dtype=np.int64),
                            pair_d[s0:s0 + n].reshape(-1), o, d)
        hh, nn, bb = np.nonzero(np.any(o != 0, axis=-1))
        counts[2 * s0:2 * (s0 + n)] = np.bincount(hh, minlength=2 * n)
        keys.append((nn * nb + bb).astype(np.int16))
        vals.append(o[hh, nn, bb])
        dead[2 * s0:2 * (s0 + n)] = d
    L = Launches(seeds, seed_lin, pair_seed, pair_w, pair_d, np.repeat(seed_lin[pair_seed], 2),
                 np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
                 np.concatenate(keys) if keys else np.zeros(0, np.int16),
                 np.concatenate(vals) if vals else np.zeros((0, nc)), dead)
    log.info("launches: %d streamlines (peak pairs) from %d seeds in %.0fs (%.0f MB)", P, len(seeds),
             time.time() - t0, L.nbytes / 1e6)
    return L


# ------------------------------------------------------------------ bundles
@dataclass
class BundleResult:
    name: str
    seed_vox: np.ndarray     # (S, 3)
    seed_prob: np.ndarray    # (S,) P(the streamline from this seed is in the bundle)
    density_t: np.ndarray    # (Nt,) expected visits by bundle streamlines, all seeds
    endpoints_a: np.ndarray  # (Na,) expected bundle endpoints per absorbing voxel
    info: dict = field(default_factory=dict)
    density_class: np.ndarray = None  # (Nt, 26) the visits split by the walker's direction class
    fields: object = None    # mcat.fields.BundleFields when solved with keep_fields=True


def _endpoint_kinds(a_valid, a_start, a_end):
    """End categories per half and the bilinear form M: P(both ends acceptable) =
    sum_kl M[k, l] V_k(h1) V_l(h2)."""
    if a_start is None and a_end is None:
        return [a_valid], np.array([[1.0]]), "both ends valid"
    if a_start is None or a_end is None:
        E = a_end if a_end is not None else a_start
        return [a_valid, a_valid & ~E], np.diag([1.0, -1.0]), "one end in the endpoint ROI"
    S, E = a_valid & a_start, a_valid & a_end
    if not (S & E).any():
        return [S, E], np.array([[0.0, 1.0], [1.0, 0.0]]), "one end in start, the other in end"
    return [S, E, S & E], np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]), \
        "one end in start, the other in end"


class _Plan:
    """A bundle's avoid sets, end kinds and solution columns."""

    def __init__(self, chain, spec, valid_classes):
        self.name, self.t0 = spec["name"], time.time()
        pad = np.zeros(chain.pshape, bool)

        def padded(m):
            p = pad.copy()
            p[1:-1, 1:-1, 1:-1] = m
            return p.ravel()

        inc = [padded(m) for m in spec.get("include", ())]
        exc = padded(spec["exclude"]) if spec.get("exclude") is not None else pad.ravel()
        self.subsets = [T for r in range(len(inc) + 1) for T in combinations(range(len(inc)), r)]
        self.signs = np.array([(-1.0) ** len(T) for T in self.subsets])
        self.nX = len(self.subsets)
        self.taboo = np.zeros((pad.size, self.nX), np.bool_)
        for x, T in enumerate(self.subsets):
            tb = exc.copy()
            for i in T:
                tb |= inc[i]
            self.taboo[:, x] = tb
        a_valid = np.isin(chain.a_class, np.asarray(valid_classes))
        ends = [padded(spec[k])[chain.a_lin] if spec.get(k) is not None else None for k in ("start", "end")]
        kinds, self.M, self.end_rule = _endpoint_kinds(a_valid, *ends)
        self.nK = len(kinds)
        self.C = self.nX * self.nK
        self.xcol = np.repeat(np.arange(self.nX), self.nK)
        self.kcol = np.tile(np.arange(self.nK), self.nX)
        self.absw = np.stack([kinds[self.kcol[c]] & ~self.taboo[chain.a_lin, self.xcol[c]]
                              for c in range(self.C)], 1).astype(np.float64)
        st = np.repeat(chain.t_lin, chain.nb * chain.ncomp)
        self.mask = (~self.taboo[st][:, self.xcol]).astype(np.uint8)
        kill = getattr(chain, "kill", None)      # states past the soft maximum length: walkers end there
        if kill is not None:
            self.mask[np.repeat(kill, chain.ncomp)] = 0


def max_columns(chain, launches, max_mem):
    """Solution columns that fit in ``max_mem`` bytes next to the chain and the launches."""
    per_col = chain.n_states * chain.ncomp * (8 * FLOAT_BLOCKS_PER_COLUMN + 1)
    free = max_mem - chain.nbytes - launches.nbytes - 1e9
    return int(max(1, min(256, free // per_col)))


def _trapped(chain, info, what, names, on_trapped):
    """Hand the mass a solve could not finish (per voxel) to ``on_trapped``."""
    m = info.pop("trapped", None)
    if m is None or on_trapped is None:
        return
    vol = chain.to_volume(m.reshape(-1, chain.nb).sum(1))
    on_trapped(what, names, vol)


def _solve(chain, x0, mask, tol, max_iter, what, verbose=False):
    mul = chain.mul_forward if what == "forward" else chain.mul_backward
    return solve_series(mul, x0, mask, tol, max_iter, what, chain.ncomp, verbose)


def _solve_group(chain, plans, L, tol, max_iter, on_result, on_trapped=None, verbose=False, clock=None,
                 keep_fields=False):
    t0 = time.time()
    gclock = Clock()
    nb, Ns, nc = chain.nb, chain.n_states, chain.ncomp
    offs = np.cumsum([0] + [p.C for p in plans])
    names = [p.name for p in plans]
    if len(plans) > 1:
        log.info("solving %d bundles together (%d columns): %s", len(plans), offs[-1], ", ".join(names))
    with gclock("masks and absorption"):
        mask = np.concatenate([p.mask for p in plans], axis=1)
        for p in plans:
            p.mask = None
        r = np.concatenate([chain.absorb_rhs(p.absw) for p in plans], axis=1)
        r *= mask
    with gclock("backward solve"):
        beta, info_b = _solve(chain, r, mask, tol, max_iter, "backward", verbose)
    with gclock("trapped maps"):
        _trapped(chain, info_b, "backward", names, on_trapped)
    del r
    t_init = time.perf_counter()
    P = L.pair_seed.size
    pi0 = np.zeros((Ns * nc, int(offs[-1])))
    starts = []
    for p, o0, o1 in zip(plans, offs[:-1], offs[1:]):
        seed_ok = (~p.taboo[L.hseed][:, p.xcol]).astype(np.float64)
        vals = np.empty((L.hseed.size, p.C))
        K.half_values(L.hseed, L.lptr, L.lkey, L.lval, chain.label, chain.noff, chain.tindex,
                      chain.aindex, nb, np.ascontiguousarray(beta[:, o0:o1]), p.absw, seed_ok, vals)
        V = vals.reshape(P, 2, p.nX, p.nK)
        pb = np.einsum("pxk,kl,pxl->px", V[:, 0], p.M, V[:, 1]) @ p.signs   # P(pair in bundle)
        seed_prob = np.bincount(L.pair_seed, weights=L.pair_w * pb, minlength=len(L.seeds))
        coef = np.einsum("phxl,kl->phxk", V[:, ::-1], p.M).reshape(2 * P, p.C)
        coef *= np.repeat(L.pair_w, 2)[:, None] * seed_ok
        pi0_p = np.zeros((Ns * nc, p.C))
        a0 = np.zeros((chain.a_lin.size, p.C))
        K.scatter_init(L.hseed, L.lptr, L.lkey, L.lval, coef, chain.label, chain.noff, chain.tindex,
                       chain.aindex, nb, p.taboo, p.xcol, pi0_p, a0)
        pi0[:, o0:o1] = pi0_p
        starts.append((a0, seed_prob, pb))
    if getattr(chain, "kill", None) is not None:
        pi0[np.repeat(chain.kill, nc)] = 0.0
    gclock.add("seed weights (forward start)", time.perf_counter() - t_init)
    launch = pi0.copy() if keep_fields else None   # solve_series consumes pi0
    with gclock("forward solve"):
        S, info_f = _solve(chain, pi0, mask, tol, max_iter, "forward", verbose)
    with gclock("trapped maps"):
        _trapped(chain, info_f, "forward", names, on_trapped)
    del pi0, mask
    t_dens = time.perf_counter()
    out = {}
    ti = chain.tindex[L.seed_lin]
    pt = ti[L.pair_seed]
    on = pt >= 0
    for p, o0, o1, (a0, seed_prob, pb) in zip(plans, offs[:-1], offs[1:], starts):
        colsign = p.signs[p.xcol]
        S_p = np.ascontiguousarray(S[:, o0:o1])
        dcls = K.paired_density(S_p, np.ascontiguousarray(beta[:, o0:o1]), colsign, Ns, nc)
        dcls = dcls.reshape(chain.t_lin.size, nb)
        # step 0 at the seed, half along each direction of the streamline's peak
        w0 = 0.5 * (L.pair_w * pb)[on]
        for k in (0, 1):
            np.add.at(dcls, (pt[on], chain.geo.vbin[L.pair_d[on, k]]), w0)
        density = dcls.sum(1)
        endpoints = ((a0 + chain.absorbed_from(S_p)) * p.absw) @ colsign
        info = dict(n_seeds=int(len(L.seeds)), n_streamlines=int(P),
                    expected_bundle_streamlines=float(seed_prob.sum()), endpoint_rule=p.end_rule,
                    endpoint_mass_check=float(endpoints.sum() / max(2 * seed_prob.sum(), 1e-300)),
                    n_columns=int(p.C), tolerance=tol, backward=info_b, forward=info_f,
                    solved_with=[n for n in names if n != p.name],
                    negative_mass_before_clip=dict(density=float(-density[density < 0].sum()),
                                                   seed_prob=float(-seed_prob[seed_prob < 0].sum())),
                    seconds=round(time.time() - p.t0, 1))
        res = BundleResult(p.name, L.seeds, np.clip(seed_prob, 0, None), np.clip(density, 0, None),
                           np.clip(endpoints, 0, None), info, np.clip(dcls, 0, None).astype(np.float32))
        if keep_fields:
            from .fields import BundleFields

            res.fields = BundleFields(chain, p, S_p, np.ascontiguousarray(beta[:, o0:o1]),
                                      np.ascontiguousarray(launch[:, o0:o1]), a0)
        if on_result is not None:
            tw = time.perf_counter()
            on_result(res)
            gclock.add("writing outputs", time.perf_counter() - tw)
            t_dens += time.perf_counter() - tw
        out[p.name] = res
    gclock.add("densities and endpoints", time.perf_counter() - t_dens)
    log.info("group done in %.0fs (backward %d steps, forward %d steps)", time.time() - t0,
             info_b["iterations"], info_f["iterations"])
    log.info("  group time: " + ", ".join(f"{k} {v:.3g}s" for k, v in
                                          sorted(gclock.t.items(), key=lambda kv: -kv[1]) if v >= 0.005))
    if clock is not None:
        for k, v in gclock.t.items():
            clock.add(k, v)
    return out


def segment_bundles(chain, specs, launches, tol=1e-6, max_mem=8e9, max_iter=5000,
                    valid_classes=VALID_ENDS, on_result=None, on_trapped=None, verbose=False,
                    clock=None, keep_fields=False):
    """Segment bundles, sharing solves.  ``specs``: dicts with ``name`` and optional
    ``include`` (list of masks), ``exclude``, ``start``, ``end`` (boolean volumes on the chain
    grid).  Bundles are packed in order into groups whose columns fit in ``max_mem`` bytes;
    each group costs one backward and one forward series.
    ``on_trapped(direction, names, volume)``: called when a solve stalls or runs out of steps,
    with the mass it could not finish per voxel (relative to the largest column's start).
    ``verbose``: timing inside the solves.  ``clock``: a Clock to add the time per stage to
    (default: a new one, reported at the end).
    ``keep_fields``: also keep each bundle's solution columns (occupancy, completion and
    launch mass per state) as ``result.fields`` (mcat.fields.BundleFields), for edge flux,
    committors and conservation checks.  Three more float64 arrays of n_states x columns per
    bundle; the plain chain only (not with spatial moments).
    Returns {name: BundleResult}."""
    if keep_fields and chain.ncomp != 1:
        raise ValueError("keep_fields needs the plain chain (ncomp == 1), not the moment chain")
    if keep_fields:
        max_mem = max_mem / 2    # the launch copy during the forward solve
    own = clock is None
    clock = clock or Clock()
    t_all = time.perf_counter()
    cap = max_columns(chain, launches, max_mem)
    log.info("column groups of up to %d columns (%.1f GB budget)", cap, max_mem / 1e9)
    out, group, used = {}, [], 0
    it = iter(specs)
    while True:
        with clock("bundle ROIs"):          # specs may warp each bundle's ROIs as it is asked for
            spec = next(it, None)
        if spec is None:
            break
        with clock("bundle plans"):
            p = _Plan(chain, spec, valid_classes)
        if p.C > cap:
            log.warning("%s needs %d columns, more than fit in max_mem (%d); solving it alone",
                        p.name, p.C, cap)
        if group and used + p.C > cap:
            out.update(_solve_group(chain, group, launches, tol, max_iter, on_result, on_trapped,
                                    verbose, clock, keep_fields))
            group, used = [], 0
        group.append(p)
        used += p.C
    if group:
        out.update(_solve_group(chain, group, launches, tol, max_iter, on_result, on_trapped,
                                verbose, clock, keep_fields))
    if own:
        clock.report("bundle segmentation time", time.perf_counter() - t_all)
    return out
