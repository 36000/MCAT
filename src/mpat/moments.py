"""Spatial first moments: the chain also carries each state's mean sub-voxel position.

The default chain restarts every voxel's paths from a uniform position, which makes
bundles several times too wide.  Here a state carries (M, Q) with Q = M x the mean position
(voxel units, from the centre), and the within-voxel density is reconstructed linearly,
rho(y) = 1 + 12 Q.y / M (Prather 1986).  Every MITTENS path has an axis-aligned box of
start points (centre c, widths w) and lands at l in the next voxel, so per path exactly

    mass        = vol (1 + 12 c.x)
    landing Q   = vol (l + (12 l c^T + diag(w^2)) x),    x = Q / M.

The closure is linear, so the chain stays linear (adjoints, inclusion-exclusion and the
forward-backward products are unchanged), but it is not positivity-preserving: masses can go
slightly negative, and a few loops of states can amplify.  ``stabilize`` finds such loops
and falls back to the uniform closure there.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from numba import njit, prange

from . import kernel as K
from .chain import Chain, _Grower
from .geometry import OFFSETS

log = logging.getLogger("mpat")
NCOMP = 4          # (M, Q1, Q2, Q3)
MAX_RATE = 0.95    # slowest allowed decay per step of the moment operator (see stabilize)


def spatial_channels(geo):
    """(P, 16) per unit volume: [1, 12 c (3), l (3), 12 l c^T + diag(w^2) (9)]."""
    c = 0.5 * (geo.lo + geo.hi) - 0.5
    w = geo.hi - geo.lo
    land = c + geo.p_disp - OFFSETS[geo.p_nbr].astype(np.float64)
    E = 12.0 * land[:, :, None] * c[:, None, :]
    E[:, [0, 1, 2], [0, 1, 2]] += w ** 2
    P = len(c)
    return np.ascontiguousarray(np.concatenate([np.ones((P, 1)), 12.0 * c, land, E.reshape(P, 9)], 1))


@njit(cache=True)
def _prep_m(vlin, Xall, Yall, xrow, label, noff, double_odf, p_vol, p_dirs, p_len, p_nbr, p_trip,
            chan, cone_ptr, cone_idx, nT):
    """Channel version of kernel._voxel_prep: cK[t, k], zK[d1, k] (channel 0 = plain sums)."""
    X = Xall[xrow[vlin]].astype(np.float64)
    Y = Yall[xrow[vlin]].astype(np.float64)
    V = X.shape[0]
    wrows = np.empty(26, np.int64)
    K._lookahead_rows(vlin, label, xrow, noff, double_odf, wrows)
    nk = chan.shape[1]
    cK = np.zeros((nT, nk))
    zK = np.zeros((V, nk))
    for p in range(p_vol.shape[0]):
        d1 = p_dirs[p, 0]
        if X[d1] <= 0.0:
            continue
        val = p_vol[p]
        L = p_len[p]
        for i in range(1, L):
            ya = Y[p_dirs[p, i - 1]]
            if ya <= 0.0:
                val = 0.0
                break
            val *= X[p_dirs[p, i]] / ya
        if val == 0.0:
            continue
        r = wrows[p_nbr[p]]
        if r >= 0:
            val *= Yall[r, p_dirs[p, L - 1]]
        elif r == -2:
            continue
        for j in range(nk):
            cK[p_trip[p], j] += val * chan[p, j]
            zK[d1, j] += val * chan[p, j]
    Zv = np.empty(V)
    K._incoming_norm(X, Y, np.ascontiguousarray(zK[:, 0]), cone_ptr, cone_idx, Zv)
    return X, Y, cK, zK, Zv


@njit(cache=True)
def _incoming_m(q_e, q_w, X, Y, Zv, zK, cK, cone_ptr, cone_idx, d1_ptr, t_nbr, t_bin, acc, accz,
                mark, touched, out):
    """out[n, b', 4, 4] for a walker whose incoming direction is distributed as q.
    Returns the dead mass."""
    dead = 0.0
    ntouch = 0
    zeta = np.empty(3)
    for i in range(q_e.shape[0]):
        e = q_e[i]
        qe = q_w[i]
        if qe <= 0.0:
            continue
        if Zv[e] <= 0.0:
            dead += qe
            continue
        zeta[:] = 0.0   # first-order change of the normaliser per unit offset
        for k in range(cone_ptr[e], cone_ptr[e + 1]):
            d = cone_idx[k]
            for j in range(3):
                zeta[j] += X[d] * zK[d, 1 + j]
        for j in range(3):
            zeta[j] /= Y[e] * Zv[e]
        f = qe / (Zv[e] * Y[e])
        for k in range(cone_ptr[e], cone_ptr[e + 1]):
            d = cone_idx[k]
            if not mark[d]:
                mark[d] = True
                touched[ntouch] = d
                ntouch += 1
            acc[d] += f
            for j in range(3):
                accz[d, j] += f * zeta[j]
    for i in range(ntouch):
        d = touched[i]
        mark[d] = False
        ud = X[d] * acc[d]
        uz0, uz1, uz2 = X[d] * accz[d, 0], X[d] * accz[d, 1], X[d] * accz[d, 2]
        acc[d] = 0.0
        accz[d, :] = 0.0
        if ud <= 0.0:
            continue
        for t in range(d1_ptr[d], d1_ptr[d + 1]):
            o = out[t_nbr[t], t_bin[t]]
            a = cK[t, 0]
            o[0, 0] += ud * a
            o[0, 1] += ud * cK[t, 1] - uz0 * a
            o[0, 2] += ud * cK[t, 2] - uz1 * a
            o[0, 3] += ud * cK[t, 3] - uz2 * a
            for r in range(3):
                lr = cK[t, 4 + r]
                o[1 + r, 0] += ud * lr
                o[1 + r, 1] += ud * cK[t, 7 + 3 * r] - lr * uz0
                o[1 + r, 2] += ud * cK[t, 8 + 3 * r] - lr * uz1
                o[1 + r, 3] += ud * cK[t, 9 + 3 * r] - lr * uz2
    return dead


@njit(parallel=True, cache=True)
def chunk_blocks_m(vlist, Xall, Yall, xrow, label, noff, double_odf, p_vol, p_dirs, p_len, p_nbr,
                   p_trip, chan, d1_ptr, t_nbr, t_bin, cone_ptr, cone_idx, bin_ptr, bin_idx, out, dead):
    """out[i, b, n, b', 4, 4]: block transitions of every class of voxel vlist[i]; dead[i, b, 4]."""
    nb = out.shape[1]
    V = Xall.shape[1]
    for i in prange(vlist.shape[0]):
        v = vlist[i]
        o = np.zeros((nb, 26, nb, 4, 4))
        dd = np.zeros((nb, 4))
        if xrow[v] < 0:
            dd[:, 0] = 1.0
        else:
            X, Y, cK, zK, Zv = _prep_m(v, Xall, Yall, xrow, label, noff, double_odf, p_vol, p_dirs,
                                       p_len, p_nbr, p_trip, chan, cone_ptr, cone_idx, t_nbr.shape[0])
            acc = np.zeros(V)
            accz = np.zeros((V, 3))
            mark = np.zeros(V, np.bool_)
            touched = np.empty(V, np.int64)
            for b in range(nb):
                q_e = bin_idx[bin_ptr[b]:bin_ptr[b + 1]]
                dd[b, 0] = _incoming_m(q_e, K.class_q(X, Y, q_e), X, Y, Zv, zK, cK, cone_ptr,
                                       cone_idx, d1_ptr, t_nbr, t_bin, acc, accz, mark, touched, o[b])
        out[i] = o.astype(np.float32)
        dead[i] = dd


@njit(parallel=True, cache=True)
def launch_batch_m(seed_vlin, dptr, dirs, Xall, Yall, xrow, label, noff, double_odf, p_vol, p_dirs,
                   p_len, p_nbr, p_trip, chan, d1_ptr, t_nbr, t_bin, cone_ptr, cone_idx, nb, out, dead):
    """out[h] (26, nb, 4): (mass, Q) after the first exit of half h, launched from a uniform
    position (Q = 0) along dirs[h]."""
    V = Xall.shape[1]
    for i in prange(seed_vlin.shape[0]):
        v = seed_vlin[i]
        if xrow[v] < 0:
            for h in range(dptr[i], dptr[i + 1]):
                out[h] = 0.0
                dead[h] = 1.0
            continue
        X, Y, cK, zK, Zv = _prep_m(v, Xall, Yall, xrow, label, noff, double_odf, p_vol, p_dirs,
                                   p_len, p_nbr, p_trip, chan, cone_ptr, cone_idx, t_nbr.shape[0])
        acc = np.zeros(V)
        accz = np.zeros((V, 3))
        mark = np.zeros(V, np.bool_)
        touched = np.empty(V, np.int64)
        for h in range(dptr[i], dptr[i + 1]):
            o = np.zeros((26, nb, 4, 4))
            dead[h] = _incoming_m(np.full(1, dirs[h], np.int64), np.ones(1), X, Y, Zv, zK, cK,
                                  cone_ptr, cone_idx, d1_ptr, t_nbr, t_bin, acc, accz, mark, touched, o)
            out[h] = o[:, :, :, 0]   # input (M = 1, Q = 0): the first column


@njit(cache=True)
def blocks_to_rows(vlist, out, dead, label, noff, tindex, aindex):
    """Dense chunk blocks -> block rows (row = source state) of Q (16 floats) and R (the 4
    mass-row entries) + dead (4)."""
    C, nb = out.shape[0], out.shape[1]
    nq = 0
    nr = 0
    for i in range(C):
        for b in range(nb):
            for n in range(26):
                if label[vlist[i] + noff[n]] == K.TRANSIENT:
                    for bb in range(nb):
                        if np.any(out[i, b, n, bb] != 0.0):
                            nq += 1
                else:
                    nr += 1
    q_ptr = np.zeros(C * nb + 1, np.int64)
    q_ind = np.empty(nq, np.int32)
    q_blk = np.empty((nq, 16), np.float32)
    r_ptr = np.zeros(C * nb + 1, np.int64)
    r_ind = np.empty(nr, np.int32)
    r_blk = np.empty((nr, 4), np.float32)
    kq = 0
    kr = 0
    for i in range(C):
        v = vlist[i]
        for b in range(nb):
            row = i * nb + b
            for n in range(26):
                w = v + noff[n]
                if label[w] == K.TRANSIENT:
                    for bb in range(nb):
                        blk = out[i, b, n, bb]
                        if np.any(blk != 0.0):
                            q_ind[kq] = tindex[w] * nb + bb
                            for r in range(4):
                                for c in range(4):
                                    q_blk[kq, 4 * r + c] = blk[r, c]
                            kq += 1
                else:
                    s = 0.0
                    for bb in range(nb):
                        for c in range(4):
                            s += abs(out[i, b, n, bb, 0, c])
                    if s > 0.0:
                        r_ind[kr] = aindex[w]
                        for c in range(4):
                            acc = 0.0
                            for bb in range(nb):
                                acc += out[i, b, n, bb, 0, c]
                            r_blk[kr, c] = acc
                        kr += 1
            q_ptr[row + 1] = kq
            r_ptr[row + 1] = kr
    return q_ptr, q_ind[:kq].copy(), q_blk[:kq].copy(), r_ptr, r_ind[:kr].copy(), r_blk[:kr].copy(), \
        dead.reshape(C * nb, 4).copy()


@njit(parallel=True, cache=True)
def bsr_backward(ptr, ind, blk, x, out):
    """out[s] = sum_k B_k^T x[dst_k]  (x, out: (Ns, 4, C)); gather over source rows."""
    Ns = ptr.shape[0] - 1
    C = x.shape[2]
    for s in prange(Ns):
        out[s] = 0.0
        for k in range(ptr[s], ptr[s + 1]):
            j = ind[k]
            for r in range(4):
                for c in range(C):
                    acc = 0.0
                    for i in range(4):
                        acc += blk[k, 4 * i + r] * x[j, i, c]
                    out[s, r, c] += acc


@njit(parallel=True, cache=True)
def bsr_forward(ptr, ind, blk, x, out, chunk_ptr):
    """out[dst_k] += B_k x[s] (scatter), red-black over two-slab chunks as kernel.stencil_forward."""
    Ns = ptr.shape[0] - 1
    C = x.shape[2]
    for s in prange(Ns):
        out[s] = 0.0
    nch = chunk_ptr.shape[0] - 1
    for phase in range(2):
        for q in prange((nch - phase + 1) // 2):
            k = 2 * q + phase
            for s in range(chunk_ptr[k], chunk_ptr[k + 1]):
                if not np.any(x[s] != 0.0):
                    continue
                for kk in range(ptr[s], ptr[s + 1]):
                    j = ind[kk]
                    for i in range(4):
                        for c in range(C):
                            acc = 0.0
                            for r in range(4):
                                acc += blk[kk, 4 * i + r] * x[s, r, c]
                            out[j, i, c] += acc


@njit(parallel=True, cache=True)
def r_backward(ptr, ind, blk, absw, out):
    """out[s, r, c] = sum_k R_k[r] absw[a_k, c]."""
    Ns = ptr.shape[0] - 1
    C = absw.shape[1]
    for s in prange(Ns):
        out[s] = 0.0
        for k in range(ptr[s], ptr[s + 1]):
            for r in range(4):
                for c in range(C):
                    out[s, r, c] += blk[k, r] * absw[ind[k], c]


@njit(cache=True)
def r_forward(ptr, ind, blk, x, out):
    """out[a, c] += sum_s sum_r R_k[r] x[s, r, c]."""
    for s in range(ptr.shape[0] - 1):
        for k in range(ptr[s], ptr[s + 1]):
            for c in range(x.shape[2]):
                acc = 0.0
                for r in range(4):
                    acc += blk[k, r] * x[s, r, c]
                out[ind[k], c] += acc


@njit(parallel=True, cache=True)
def _mass_forward(ptr, ind, blk, x, out, chunk_ptr):
    """The mass channel alone (the default chain inside the moment chain)."""
    for s in prange(x.shape[0]):
        out[s] = 0.0
    nch = chunk_ptr.shape[0] - 1
    for phase in range(2):
        for q in prange((nch - phase + 1) // 2):
            k = 2 * q + phase
            for s in range(chunk_ptr[k], chunk_ptr[k + 1]):
                for kk in range(ptr[s], ptr[s + 1]):
                    for c in range(x.shape[1]):
                        out[ind[kk], c] += blk[kk, 0] * x[s, c]


def _rate(mul, x, iters):
    """Power iteration on independent columns; (largest rate per step over the last third of
    the iterations, that column's vector)."""
    x = np.array(x, np.float64, order="C")
    y = np.empty_like(x)
    n = np.abs(x).sum(0)
    alive = n > 0
    x[:, alive] /= n[alive]
    logs = []
    for _ in range(iters):
        mul(x, y)
        x, y = y, x
        n = np.abs(x).sum(0)
        alive &= n > 0
        logs.append(np.log(np.where(alive, n, 1.0)))
        x[:, alive] /= n[alive]
    rates = np.where(alive, np.exp(np.mean(logs[-max(1, iters // 3):], axis=0)), 0.0)
    j = int(np.argmax(rates))
    return float(rates[j]), x[:, j:j + 1].copy()


def stabilize(chain, max_rate=MAX_RATE, iters=150, max_rounds=12, cover=0.99, seed=0):
    """Find loops of states where the moment closure amplifies (the operator decays slower than
    ``max_rate`` per step and slower than its mass channel alone) and drop the moment inputs of
    the voxels carrying ``cover`` of that mode (uniform closure there).  Repeats until fine."""
    Ns, nb = chain.n_states, chain.nb
    rng = np.random.default_rng(seed)
    t0 = time.time()
    k = 4
    rho0, _ = _rate(lambda a, b: _mass_forward(chain.Q_ptr, chain.Q_ind, chain.Q_blk, a, b,
                                               chain.chunk_ptr), rng.random((Ns, k)), iters)
    target = max(max_rate, rho0 + 0.02)
    bad = np.zeros(Ns, np.bool_)
    rho = None
    for rnd in range(max_rounds):
        x = rng.standard_normal((Ns * NCOMP, k))
        x[::NCOMP] = np.abs(x[::NCOMP])
        rho, x = _rate(chain.mul_forward, x, iters)
        log.info("moment stability, round %d: slowest rate %.4f per step (mass alone %.4f, "
                 "target %.4f)", rnd, rho, rho0, target)
        if rho <= target:
            break
        _, z = _rate(chain.mul_backward, rng.standard_normal((Ns * NCOMP, k)), iters)
        w = (np.abs(x.reshape(Ns, NCOMP)) * np.abs(z.reshape(Ns, NCOMP))).sum(1)
        vw = np.bincount(np.arange(Ns) // nb, weights=w, minlength=Ns // nb)
        order = np.argsort(vw)[::-1]
        vox = order[:np.searchsorted(np.cumsum(vw[order]) / vw.sum(), cover) + 1]
        new = np.zeros(Ns, np.bool_)
        new.reshape(-1, nb)[vox] = True
        new &= ~bad
        if not new.any():
            log.warning("moment stability: mode support already damped; stopping")
            break
        bad |= new
        rows = np.repeat(new, np.diff(chain.Q_ptr))
        blk = chain.Q_blk[rows]      # zero the responses to an incoming moment
        blk[:, [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]] = 0.0
        chain.Q_blk[rows] = blk
        chain.R_blk[np.repeat(new, np.diff(chain.R_ptr)), 1:] = 0.0
        chain.dead4[new, 1:] = 0.0
        log.info("  uniform closure in %d more voxels (%d total)", vox.size, bad.sum() // nb)
    chain.stab_info = dict(rate=rho, rate_mass_channel=rho0, target=target,
                           fallback_voxels=int(bad.sum() // nb), seconds=round(time.time() - t0, 1))
    return bad


class MomentChain(Chain):
    ncomp = NCOMP
    _CACHE_KEYS = ("Q_ptr", "Q_ind", "Q_blk", "R_ptr", "R_ind", "R_blk", "dead4")

    def _margs(self):
        X, Y, xrow, label, noff, dbl, p_vol, p_dirs, p_len, p_nbr, p_trip, d1_ptr, t_nbr, t_bin, \
            cptr, cidx = self._kargs()
        if getattr(self, "_chan", None) is None:
            self._chan = spatial_channels(self.geo)
        return (X, Y, xrow, label, noff, dbl, p_vol, p_dirs, p_len, p_nbr, p_trip, self._chan,
                d1_ptr, t_nbr, t_bin, cptr, cidx)

    def _build_matrices(self):
        nb, Nt = self.nb, self.t_lin.size
        args = self._margs()
        chunk = max(16, self.cfg.chunk // 4)
        Ns = Nt * nb
        q_ind, q_blk = _Grower(np.int32), _Grower(np.float32, 16)
        r_ind, r_blk = _Grower(np.int32), _Grower(np.float32, 4)
        Q_ptr = np.zeros(Ns + 1, np.int64)
        R_ptr = np.zeros(Ns + 1, np.int64)
        self.dead4 = np.empty((Ns, 4))
        tt = time.time()
        for s in range(0, Nt, chunk):
            vl = self.t_lin[s:s + chunk]
            out = np.empty((vl.size, nb, 26, nb, 4, 4), np.float32)
            dead = np.empty((vl.size, nb, 4))
            chunk_blocks_m(vl, *args, self.geo.bin_ptr, self.geo.bin_idx, out, dead)
            qp, qi, qb, rp, ri, rb, dd = blocks_to_rows(vl, out, dead, self.label, self.noff,
                                                       self.tindex, self.aindex)
            del out
            r0, r1 = s * nb, (s + vl.size) * nb
            Q_ptr[r0 + 1:r1 + 1] = qp[1:] + q_ind.n
            R_ptr[r0 + 1:r1 + 1] = rp[1:] + r_ind.n
            frac = (s + vl.size) / Nt
            q_ind.add(qi, frac), q_blk.add(qb, frac), r_ind.add(ri, frac), r_blk.add(rb, frac)
            self.dead4[r0:r1] = dd
            if s and (s // chunk) % 40 == 0:
                el = time.time() - tt
                log.info("  moment transitions %d/%d voxels (%.0fs, eta %.0fs)", s, Nt, el,
                         el / s * (Nt - s))
        self.Q_ptr, self.R_ptr = Q_ptr, R_ptr
        self.Q_ind, self.Q_blk = q_ind.array(), q_blk.array()
        self.R_ind, self.R_blk = r_ind.array(), r_blk.array()

    def _finalize(self):
        self.n_states = self.t_lin.size * self.nb
        self.nbr_t = None
        self.chunk_ptr = K.slab_chunks(self.t_lin, self.pshape, self.nb)
        self.n_transitions = int(self.Q_ind.size)
        self.nbytes = sum(getattr(self, k).nbytes for k in self._CACHE_KEYS)
        stabilize(self)

    def launch_halves(self, seed_vlin, dptr, dirs, out, dead):
        launch_batch_m(seed_vlin, dptr, dirs, *self._margs(), self.nb, out, dead)

    def mul_forward(self, x, out):
        Ns = self.n_states
        bsr_forward(self.Q_ptr, self.Q_ind, self.Q_blk, x.reshape(Ns, 4, -1), out.reshape(Ns, 4, -1),
                    self.chunk_ptr)
        return out

    def mul_backward(self, x, out):
        Ns = self.n_states
        bsr_backward(self.Q_ptr, self.Q_ind, self.Q_blk, x.reshape(Ns, 4, -1), out.reshape(Ns, 4, -1))
        return out

    def absorb_rhs(self, absw):
        out = np.empty((self.n_states, 4, absw.shape[1]))
        r_backward(self.R_ptr, self.R_ind, self.R_blk, np.ascontiguousarray(absw), out)
        return out.reshape(self.n_states * 4, -1)

    def absorbed_from(self, S):
        out = np.zeros((self.a_lin.size, S.shape[1]))
        r_forward(self.R_ptr, self.R_ind, self.R_blk, S.reshape(self.n_states, 4, -1), out)
        return out
