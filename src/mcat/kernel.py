"""Numba kernels: per-voxel transition blocks, stencil storage and products, launches.

For a walker that *enters* voxel v moving in direction e (Cieslak et al. 2018, with the
walker's direction remembered across voxels):

  P(exit to neighbour n, last hop d_L | e)
      = sum over paths p = (d_1..d_L) of G_v(e, d_1) vol_p prod_i G_v(d_i, d_i+1) L_n(d_L) / Z_v(e)

G_v(a, d) = X_v(d) / Y_v(a) for d in the cone of a (X the fODF on the sphere, Y its cone
sums), L_n the double-ODF look-ahead (the neighbour's Y; 1 for singleODF) and Z_v(e) the
normaliser over all outcomes.  A state is (voxel, direction class b); within its class the
incoming direction is taken as q_{v,b}(e) ~ X_v(e) Y_v(e).  If Z_v(e) = 0 the walker has no
admissible continuation and dies in white matter ("dead").

Voxels labelled BLOCKED (CSF) are never entered: every path that would exit into one gets
weight 0, in both doubleODF and singleODF, so the walker's probability goes to its other
hops through the normaliser Z_v.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

TRANSIENT, ABSORBING, BLOCKED = 1, 2, 3   # BLOCKED: CSF, zero transition probability


@njit(cache=True)
def _lookahead_rows(vlin, label, xrow, noff, double_odf, wrows):
    """Row of Y used as look-ahead for each neighbour: -1 none (singleODF: weight 1),
    -2 never entered (weight 0): CSF, or no fODF there."""
    for n in range(26):
        w = vlin + noff[n]
        if label[w] == BLOCKED:
            wrows[n] = -2
        elif not double_odf:
            wrows[n] = -1
        elif label[w] != 0 and xrow[w] >= 0:
            wrows[n] = xrow[w]
        else:
            wrows[n] = -2


@njit(cache=True)
def _path_sums(X, Yall, Y, wrows, p_vol, p_dirs, p_len, p_nbr, p_trip, c, z):
    """c[t]: sum over the paths of triple t of vol * prod G * look-ahead; z[d]: same by first
    hop (X_v(d_1) not included)."""
    c[:] = 0.0
    z[:] = 0.0
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
        c[p_trip[p]] += val
        z[d1] += val


@njit(cache=True)
def _incoming_norm(X, Y, z, cone_ptr, cone_idx, Zv):
    """Z_v(e) = sum_{d in cone(e)} G_v(e, d) z(d)."""
    for e in range(X.shape[0]):
        if Y[e] <= 0.0:
            Zv[e] = 0.0
            continue
        s = 0.0
        for k in range(cone_ptr[e], cone_ptr[e + 1]):
            s += X[cone_idx[k]] * z[cone_idx[k]]
        Zv[e] = s / Y[e]


@njit(cache=True)
def _apply_incoming(q_e, q_w, X, Y, Zv, c, cone_ptr, cone_idx, d1_ptr, t_nbr, t_bin, acc,
                    touched, out):
    """Outcomes out[n, b'] of a walker whose incoming direction is distributed as q.
    Returns the dead mass."""
    dead = 0.0
    ntouch = 0
    for i in range(q_e.shape[0]):
        e = q_e[i]
        qe = q_w[i]
        if qe <= 0.0:
            continue
        if Zv[e] <= 0.0:
            dead += qe
            continue
        f = qe / (Zv[e] * Y[e])
        for k in range(cone_ptr[e], cone_ptr[e + 1]):
            d = cone_idx[k]
            if acc[d] == 0.0:
                touched[ntouch] = d
                ntouch += 1
            acc[d] += f
    for i in range(ntouch):
        d = touched[i]
        ud = X[d] * acc[d]
        acc[d] = 0.0
        if ud <= 0.0:
            continue
        for t in range(d1_ptr[d], d1_ptr[d + 1]):
            out[t_nbr[t], t_bin[t]] += ud * c[t]
    return dead


@njit(cache=True)
def class_q(X, Y, q_e):
    """Within-class incoming distribution q ~ X Y (fallbacks: Y, then uniform)."""
    m = q_e.shape[0]
    q_w = np.empty(m)
    s = 0.0
    for i in range(m):
        q_w[i] = X[q_e[i]] * Y[q_e[i]]
        s += q_w[i]
    if s <= 0.0:
        for i in range(m):
            q_w[i] = Y[q_e[i]]
            s += q_w[i]
    if s <= 0.0:
        for i in range(m):
            q_w[i] = 1.0
        s = m
    for i in range(m):
        q_w[i] /= s
    return q_w


@njit(cache=True)
def _voxel_prep(vlin, Xall, Yall, xrow, label, noff, double_odf, p_vol, p_dirs, p_len, p_nbr,
                p_trip, cone_ptr, cone_idx, nT):
    X = Xall[xrow[vlin]].astype(np.float64)
    Y = Yall[xrow[vlin]].astype(np.float64)
    wrows = np.empty(26, np.int64)
    _lookahead_rows(vlin, label, xrow, noff, double_odf, wrows)
    c = np.empty(nT)
    z = np.empty(X.shape[0])
    _path_sums(X, Yall, Y, wrows, p_vol, p_dirs, p_len, p_nbr, p_trip, c, z)
    Zv = np.empty(X.shape[0])
    _incoming_norm(X, Y, z, cone_ptr, cone_idx, Zv)
    return X, Y, c, Zv


@njit(parallel=True, cache=True)
def chunk_blocks(vlist, Xall, Yall, xrow, label, noff, double_odf, p_vol, p_dirs, p_len, p_nbr,
                 p_trip, d1_ptr, t_nbr, t_bin, cone_ptr, cone_idx, bin_ptr, bin_idx, out, dead):
    """out[i, b, n, b'] = P((v_i, b) -> (neighbour n, class b')); dead[i, b]."""
    nb = out.shape[1]
    V = Xall.shape[1]
    for i in prange(vlist.shape[0]):
        v = vlist[i]
        out[i] = 0.0
        if xrow[v] < 0:
            dead[i] = 1.0
            continue
        X, Y, c, Zv = _voxel_prep(v, Xall, Yall, xrow, label, noff, double_odf, p_vol, p_dirs,
                                  p_len, p_nbr, p_trip, cone_ptr, cone_idx, t_nbr.shape[0])
        acc = np.zeros(V)
        touched = np.empty(V, np.int64)
        for b in range(nb):
            q_e = bin_idx[bin_ptr[b]:bin_ptr[b + 1]]
            dead[i, b] = _apply_incoming(q_e, class_q(X, Y, q_e), X, Y, Zv, c, cone_ptr, cone_idx,
                                         d1_ptr, t_nbr, t_bin, acc, touched, out[i, b])


@njit(parallel=True, cache=True)
def launch_batch(seed_vlin, dptr, dirs, Xall, Yall, xrow, label, noff, double_odf, p_vol, p_dirs,
                 p_len, p_nbr, p_trip, d1_ptr, t_nbr, t_bin, cone_ptr, cone_idx, nb, out, dead):
    """out[h] (26, nb): first exit of half-walk h, launched along dirs[h] from seed voxel i
    (halves dptr[i]:dptr[i+1]) -- the first step of probabilistic tracking from a peak."""
    V = Xall.shape[1]
    for i in prange(seed_vlin.shape[0]):
        v = seed_vlin[i]
        if xrow[v] < 0:
            for h in range(dptr[i], dptr[i + 1]):
                out[h] = 0.0
                dead[h] = 1.0
            continue
        X, Y, c, Zv = _voxel_prep(v, Xall, Yall, xrow, label, noff, double_odf, p_vol, p_dirs,
                                  p_len, p_nbr, p_trip, cone_ptr, cone_idx, t_nbr.shape[0])
        acc = np.zeros(V)
        touched = np.empty(V, np.int64)
        for h in range(dptr[i], dptr[i + 1]):
            out[h] = 0.0
            dead[h] = _apply_incoming(np.full(1, dirs[h], np.int64), np.ones(1), X, Y, Zv, c,
                                      cone_ptr, cone_idx, d1_ptr, t_nbr, t_bin, acc, touched, out[h])


# ------------------------------------------------------------------ stencil storage
@njit(cache=True)
def stencil_rows(vlist, out, dead, label, noff, aindex, slot_of):
    """Dense chunk blocks -> stencil rows: per transition a slot id (its (neighbour, class)
    within the source class's template) and a float64 weight; absorption as (absorbing
    voxel, weight).  Rows are renormalised to sum to 1 with their dead mass."""
    C, nb = out.shape[0], out.shape[1]
    nq = 0
    nr = 0
    for i in range(C):
        for b in range(nb):
            for n in range(26):
                if label[vlist[i] + noff[n]] == TRANSIENT:
                    for bb in range(nb):
                        if out[i, b, n, bb] > 0.0:
                            nq += 1
                else:
                    nr += 1
    q_ptr = np.zeros(C * nb + 1, np.int64)
    q_slot = np.empty(nq, np.int16)
    q_w = np.empty(nq)
    r_ptr = np.zeros(C * nb + 1, np.int64)
    r_ind = np.empty(nr, np.int32)
    r_w = np.empty(nr)
    dead_out = np.zeros(C * nb)
    kq = 0
    kr = 0
    for i in range(C):
        v = vlist[i]
        for b in range(nb):
            row = i * nb + b
            tot = dead[i, b]
            for n in range(26):
                for bb in range(nb):
                    tot += out[i, b, n, bb]
            inv = 1.0 / tot if tot > 0.0 else 0.0
            for n in range(26):
                w = v + noff[n]
                if label[w] == TRANSIENT:
                    for bb in range(nb):
                        val = out[i, b, n, bb]
                        if val > 0.0:
                            sl = slot_of[b, n * nb + bb]
                            if sl < 0:
                                raise ValueError("transition outside its class template")
                            q_slot[kq] = sl
                            q_w[kq] = val * inv
                            kq += 1
                else:
                    s = 0.0
                    for bb in range(nb):
                        s += out[i, b, n, bb]
                    if s > 0.0:
                        r_ind[kr] = aindex[w]
                        r_w[kr] = s * inv
                        kr += 1
            dead_out[row] = dead[i, b] * inv if tot > 0.0 else 1.0
            q_ptr[row + 1] = kq
            r_ptr[row + 1] = kr
    return q_ptr, q_slot[:kq].copy(), q_w[:kq].copy(), r_ptr, r_ind[:kr].copy(), r_w[:kr].copy(), dead_out


@njit(parallel=True, cache=True)
def stencil_forward(ptr, slot, w, tn, tb, nbr_t, nb, x, out, chunk_ptr):
    """out = Q^T x: every source state scatters to its targets.  States are ordered by voxel
    (x slowest) and ``chunk_ptr`` cuts them into two-slab chunks; a step moves at most one
    slab, so chunks of equal parity never write the same target (red-black, no races)."""
    Ns, C = x.shape
    for s in prange(Ns):
        for c in range(C):
            out[s, c] = 0.0
    nch = chunk_ptr.shape[0] - 1
    for phase in range(2):
        for q in prange((nch - phase + 1) // 2):
            k = 2 * q + phase
            for s in range(chunk_ptr[k], chunk_ptr[k + 1]):
                nz = False
                for c in range(C):
                    if x[s, c] != 0.0:
                        nz = True
                        break
                if not nz:
                    continue
                vox = s // nb
                cls = s - vox * nb
                for kk in range(ptr[s], ptr[s + 1]):
                    sl = slot[kk]
                    j = np.int64(nbr_t[vox, tn[cls, sl]]) * nb + tb[cls, sl]
                    a = w[kk]
                    for c in range(C):
                        out[j, c] += a * x[s, c]


@njit(parallel=True, cache=True)
def stencil_backward(ptr, slot, w, tn, tb, nbr_t, nb, x, out):
    """out = Q x (gather per state)."""
    Ns, C = out.shape
    for s in prange(Ns):
        vox = s // nb
        cls = s - vox * nb
        for c in range(C):
            out[s, c] = 0.0
        for kk in range(ptr[s], ptr[s + 1]):
            sl = slot[kk]
            j = np.int64(nbr_t[vox, tn[cls, sl]]) * nb + tb[cls, sl]
            a = w[kk]
            for c in range(C):
                out[s, c] += a * x[j, c]


def slab_chunks(t_lin, pshape, nb):
    """State-index boundaries of two-x-slab chunks (see stencil_forward)."""
    x_t = np.asarray(t_lin) // (pshape[1] * pshape[2])
    cuts = np.searchsorted(x_t, np.arange(0, pshape[0] + 2, 2)) * nb
    return np.unique(np.concatenate([[0], cuts, [x_t.size * nb]])).astype(np.int64)


@njit(parallel=True, cache=True)
def csr_tmatmat(ptr, ind, val, x, out):
    """out = A^T x for CSR A (rows = states): columns split across threads."""
    n_rows = ptr.shape[0] - 1
    C = x.shape[1]
    out[:] = 0.0
    for c in prange(C):
        for i in range(n_rows):
            xi = x[i, c]
            if xi == 0.0:
                continue
            for p in range(ptr[i], ptr[i + 1]):
                out[ind[p], c] += val[p] * xi


# ------------------------------------------------------------------ bundle solves
@njit(parallel=True, cache=True)
def mask_accumulate(nxt, mask, total, ncomp):
    """nxt *= mask; total += nxt; returns per-column sum |nxt| over the mass rows."""
    R, C = nxt.shape
    nblk = 512
    bs = (R + nblk - 1) // nblk
    part = np.zeros((nblk, C))
    for b in prange(nblk):
        for i in range(b * bs, min(R, (b + 1) * bs)):
            mass_row = i % ncomp == 0
            for c in range(C):
                v = nxt[i, c] * mask[i, c]
                nxt[i, c] = v
                total[i, c] += v
                if mass_row:
                    part[b, c] += abs(v)
    return part.sum(0)


@njit(parallel=True, cache=True)
def paired_density(S, beta, colsign, n_vox, rows_per_vox):
    """Per voxel: sum over its rows and the columns of S * beta * colsign."""
    out = np.zeros(n_vox)
    C = S.shape[1]
    for v in prange(n_vox):
        acc = 0.0
        for r in range(v * rows_per_vox, (v + 1) * rows_per_vox):
            for c in range(C):
                acc += S[r, c] * beta[r, c] * colsign[c]
        out[v] = acc
    return out


@njit(cache=True)
def half_values(hseed, lptr, lkey, lval, label, noff, tindex, aindex, nb, beta, absw, seed_ok, vals):
    """vals[h, c] = P(half h completes column c's requirement): launched into a transient
    state -> beta there (paired over the values a state carries); into an absorbing voxel ->
    absw.  Launches are sparse: keys n * nb + b sorted, values (ncomp,)."""
    H, C = vals.shape
    nc = lval.shape[1]
    for h in range(H):
        v = hseed[h]
        for c in range(C):
            vals[h, c] = 0.0
        k = lptr[h]
        end = lptr[h + 1]
        while k < end:
            n = lkey[k] // nb
            w = v + noff[n]
            if label[w] == TRANSIENT:
                base = tindex[w] * nb
                while k < end and lkey[k] // nb == n:
                    b = lkey[k] % nb
                    for r in range(nc):
                        o = lval[k, r]
                        if o != 0.0:
                            for c in range(C):
                                vals[h, c] += o * beta[(base + b) * nc + r, c]
                    k += 1
            else:
                s = 0.0
                while k < end and lkey[k] // nb == n:
                    s += lval[k, 0]
                    k += 1
                if s != 0.0:
                    for c in range(C):
                        vals[h, c] += s * absw[aindex[w], c]
        for c in range(C):
            vals[h, c] *= seed_ok[h, c]


@njit(cache=True)
def scatter_init(hseed, lptr, lkey, lval, coef, label, noff, tindex, aindex, nb, taboo, xcol,
                 pi0, a0):
    """pi0[:, c] += coef[h, c] * launch of h (taboo-masked); a0: absorbed on the launch step."""
    H, C = coef.shape
    nc = lval.shape[1]
    for h in range(H):
        v = hseed[h]
        k = lptr[h]
        end = lptr[h + 1]
        while k < end:
            n = lkey[k] // nb
            w = v + noff[n]
            k0 = k
            while k < end and lkey[k] // nb == n:
                k += 1
            if label[w] == TRANSIENT:
                base = tindex[w] * nb
                for c in range(C):
                    cf = coef[h, c]
                    if cf == 0.0 or taboo[w, xcol[c]]:
                        continue
                    for j in range(k0, k):
                        for r in range(nc):
                            pi0[(base + lkey[j] % nb) * nc + r, c] += cf * lval[j, r]
            else:
                s = 0.0
                for j in range(k0, k):
                    s += lval[j, 0]
                if s != 0.0:
                    for c in range(C):
                        a0[aindex[w], c] += coef[h, c] * s


# ------------------------------------------------------------------ bundle fields
@njit(parallel=True, cache=True)
def stencil_flux(ptr, slot, w, tn, tb, nbr_t, nb, S, beta, colsign, out26):
    """out26[vox, n] = sum over the voxel's states s, their transitions s -> j into neighbour n,
    and the columns c, of colsign[c] S[s, c] Q(s, j) beta[j, c]: the expected number of
    bundle-streamline steps from the voxel into its neighbour n (mcat.fields.edge_flux)."""
    n_vox = out26.shape[0]
    C = S.shape[1]
    for v in prange(n_vox):
        for n in range(26):
            out26[v, n] = 0.0
        for b in range(nb):
            s = v * nb + b
            nz = False
            for c in range(C):
                if S[s, c] != 0.0:
                    nz = True
                    break
            if not nz:
                continue
            for kk in range(ptr[s], ptr[s + 1]):
                sl = slot[kk]
                n = tn[b, sl]
                j = np.int64(nbr_t[v, n]) * nb + tb[b, sl]
                acc = 0.0
                for c in range(C):
                    acc += colsign[c] * S[s, c] * beta[j, c]
                out26[v, n] += w[kk] * acc
