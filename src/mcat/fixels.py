"""Fixels: the fODF's lobes in each voxel, and a bundle's expected streamlines through each.

A fixel is one fibre population of a voxel, a lobe of its fODF.  Fixel-based analysis
(Raffelt et al. 2017) measures each fixel (apparent fibre density, cross-section) and needs
to know which fixels a tract runs through; from tractography that is the number of
streamlines assigned to each fixel.  MCAT gives the same number without streamlines.  The
walkers' headings in a voxel are known per direction class, and within a class they are
spread as the chain's own transitions assume (geometry.vertex_visits); summing the expected
visits over a lobe's directions gives the expected number of bundle-streamline steps in that
fixel.  Every step crosses into a neighbouring voxel, so this is the expected number of
bundle streamlines passing through the fixel (re-entries counted).  The fixels of a voxel
partition its density.

    ijk = np.stack(np.unravel_index(chain.t_lin, chain.pshape), 1) - 1     # the chain's voxels
    lobes = segment_lobes(sh[tuple(ijk.T)], chain.geo)                     # the subject's fixels
    n_cst = fixel_streamlines(res.density_class, sh[tuple(ijk.T)], chain.geo, lobes)
    n_all = fixel_streamlines(whole.density_class, ...)   # the unconditioned chain (no ROIs)
    share = np.divide(n_cst, n_all, out=np.zeros_like(n_cst), where=n_all > 0)
    write_fixel_directory(out_dir, lobes, ijk, affine, shape,
                          {"CST_L_streamlines": n_cst, "CST_L_share": share})

``n_cst`` is the bundle's traffic through the fixel in streamlines; ``share`` is the fraction
of all traffic through the fixel that the bundle accounts for, a soft fixel segmentation
(times the fixel's fibre density: the bundle's share of it).

Lobes: each direction of the chain's sphere climbs the fODF to its local maximum over the
sphere's triangulation (as MRtrix's fod2fixel segments lobes); the two antipodal halves are
one fixel; lobes whose saddle is at least ``merge_ratio`` x the lower peak are one fixel (the
SH fit's ringing splits a lobe's shoulder off); lobes whose peak is under ``peak_min`` x the
voxel's maximum are not fixels, and their directions join the nearest fixel by angle, so
every direction with a positive fODF belongs to a fixel and the partition is complete.
The fODF directions are in voxel axes, as the chain's; the fixel directory stores them in
world (scanner) axes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange

from . import kernel as K
from .geometry import Geometry, PMF_THRESHOLD, _sh_basis, vertex_visits

log = logging.getLogger("mcat")
PEAK_MIN = 0.1        # a lobe is a fixel if its peak is at least this fraction of the voxel's maximum
MERGE_RATIO = 0.8     # lobes whose saddle is at least this fraction of the lower peak are one lobe


@dataclass
class Lobes:
    """The fixels of a set of voxels (rows), in MRtrix's fixel-directory layout: the fixels
    of row i are ``offset[i] : offset[i] + count[i]``, in descending peak order."""
    count: np.ndarray       # (n,) fixels per row
    offset: np.ndarray      # (n,) first fixel of each row
    direction: np.ndarray   # (N, 3) unit peak direction, voxel axes (one of the antipodal pair)
    peak: np.ndarray        # (N,) fODF amplitude at the peak
    integral: np.ndarray    # (N,) fODF integrated over the lobe, both halves (steradians)
    peak_min: float
    merge_ratio: float
    _frow: np.ndarray = None
    _w_fix: np.ndarray = None
    _w_unass: np.ndarray = None
    _sh_fp: tuple = None

    @property
    def n_fixels(self):
        return int(self.direction.shape[0])


def _sphere_edges(geo: Geometry):
    """(E, 2) edges of the sphere's triangulation."""
    from dipy.core.sphere import Sphere

    return np.asarray(Sphere(xyz=geo.vertices).edges, np.int64)


def _sphere_neighbours(geo: Geometry):
    """(V, k) neighbours of each sphere vertex in its triangulation, padded with the vertex itself."""
    e = _sphere_edges(geo)
    V = geo.n_dirs
    a = np.concatenate([e[:, 0], e[:, 1]])
    b = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.bincount(a, minlength=V)
    nbr = np.repeat(np.arange(V)[:, None], deg.max(), axis=1)
    order = np.argsort(a, kind="stable")
    pos = np.arange(a.size) - np.repeat(np.cumsum(deg) - deg, deg)
    nbr[a[order], pos] = b[order]
    return nbr


def _lobe_roots(sf, nbr):
    """Steepest ascent: the peak vertex each direction climbs to, (n, V)."""
    V = sf.shape[1]
    cand = sf[:, nbr]                                      # (n, V, k)
    j = cand.argmax(2)
    up = nbr[np.arange(V)[None, :], j]                     # (n, V)
    up = np.where(np.take_along_axis(sf, up, 1) > sf, up, np.arange(V)[None, :])
    return _follow(up)


def _follow(up):
    rows = np.arange(up.shape[0])[:, None]
    while True:
        nxt = up[rows, up]
        if np.array_equal(nxt, up):
            return up
        up = nxt


def _merge_lobes(sf, root, edges, anti, ratio, rounds=8):
    """Join lobes whose saddle (the highest fODF value on an edge between them) is at least
    ``ratio`` x the lower peak: the lower lobe joins the higher one it has the best saddle
    with.  Keeps the antipodal halves together."""
    V = sf.shape[1]
    rows = np.arange(sf.shape[0])[:, None]
    ea, eb = edges[:, 0], edges[:, 1]
    sad_e = np.minimum(sf[:, ea], sf[:, eb])               # (n, E)
    for _ in range(rounds):
        ra, rb = root[:, ea], root[:, eb]
        pa, pb = np.take_along_axis(sf, ra, 1), np.take_along_axis(sf, rb, 1)
        swap = pa > pb
        lo, hi = np.where(swap, rb, ra), np.where(swap, ra, rb)
        plo = np.minimum(pa, pb)
        ok = (ra != rb) & (sad_e >= ratio * plo) & (plo > 0)
        if not ok.any():
            break
        r_ok = np.broadcast_to(rows, lo.shape)[ok]
        key = r_ok * V + lo[ok]
        order = np.lexsort((-sad_e[ok], key))
        first = np.r_[True, key[order][1:] != key[order][:-1]]
        sel = order[first]
        parent = np.tile(np.arange(V), (sf.shape[0], 1))
        parent[r_ok[sel], lo[ok][sel]] = hi[ok][sel]
        parent = _follow(parent)
        root = parent[rows, root]
        root = np.minimum(root, root[rows, anti[root]])
    return root


@njit(parallel=True, cache=True)
def _segment_numba(sf, vertices, nbr, edges, anti, peak_min, merge_ratio, rounds=8):
    """Parallel per-voxel lobe segmentation in L1 cache, bit-for-bit identical to the
    steepest-ascent + saddle-merge partition without multi-GB (N, E) edge allocations."""
    n, V = sf.shape
    k_deg = nbr.shape[1]
    E = edges.shape[0]
    label = np.full((n, V), -1, np.int16)
    count = np.zeros(n, np.int64)
    max_k = 32
    froot_pad = np.zeros((n, max_k), np.int64)
    pk_pad = np.zeros((n, max_k), np.float64)
    mass_pad = np.zeros((n, max_k), np.float64)
    scale_sterad = 4.0 * np.pi / V

    for i in prange(n):
        s = sf[i]
        mx = 0.0
        for v in range(V):
            if s[v] > mx:
                mx = s[v]
        if mx <= 0.0:
            continue

        up = np.empty(V, np.int64)
        for v in range(V):
            best_val = s[nbr[v, 0]]
            best_u = nbr[v, 0]
            for j in range(1, k_deg):
                u = nbr[v, j]
                val = s[u]
                if val > best_val:
                    best_val = val
                    best_u = u
            up[v] = best_u if best_val > s[v] else v

        while True:
            changed = False
            for v in range(V):
                u = up[up[v]]
                if u != up[v]:
                    up[v] = u
                    changed = True
            if not changed:
                break

        root = np.empty(V, np.int64)
        for v in range(V):
            r = up[v]
            ar = up[anti[r]]
            root[v] = r if r <= ar else ar

        parent = np.empty(V, np.int64)
        best_sad = np.empty(V, np.float64)
        for _ in range(rounds):
            for v in range(V):
                parent[v] = v
                best_sad[v] = -1.0
            any_ok = False
            for e in range(E):
                ea = edges[e, 0]
                eb = edges[e, 1]
                ra = root[ea]
                rb = root[eb]
                if ra == rb:
                    continue
                sa = s[ea]
                sb = s[eb]
                sad = sa if sa <= sb else sb
                pa = s[ra]
                pb = s[rb]
                if pa > pb:
                    lo, hi, plo = rb, ra, pb
                else:
                    lo, hi, plo = ra, rb, pa
                if plo > 0.0 and sad >= merge_ratio * plo:
                    any_ok = True
                    if sad > best_sad[lo]:
                        best_sad[lo] = sad
                        parent[lo] = hi
            if not any_ok:
                break
            while True:
                changed = False
                for v in range(V):
                    u = parent[parent[v]]
                    if u != parent[v]:
                        parent[v] = u
                        changed = True
                if not changed:
                    break
            for v in range(V):
                root[v] = parent[root[v]]
            for v in range(V):
                r = root[v]
                ar = root[anti[r]]
                if ar < r:
                    root[v] = ar

        mass_by_r = np.zeros(V, np.float64)
        seen_r = np.zeros(V, np.bool_)
        for v in range(V):
            if s[v] > 0.0:
                r = root[v]
                seen_r[r] = True
                mass_by_r[r] += s[v]

        kept_r = np.empty(max_k, np.int64)
        kept_pk = np.empty(max_k, np.float64)
        kept_mass = np.empty(max_k, np.float64)
        k_cnt = 0
        thresh = peak_min * mx
        for r in range(V):
            if not seen_r[r]:
                continue
            pk = s[r]
            pk_anti = s[anti[r]]
            if pk_anti > pk:
                pk = pk_anti
            if pk >= thresh and k_cnt < max_k:
                pos = k_cnt
                while pos > 0 and pk > kept_pk[pos - 1]:
                    kept_r[pos] = kept_r[pos - 1]
                    kept_pk[pos] = kept_pk[pos - 1]
                    kept_mass[pos] = kept_mass[pos - 1]
                    pos -= 1
                kept_r[pos] = r
                kept_pk[pos] = pk
                kept_mass[pos] = mass_by_r[r] * scale_sterad
                k_cnt += 1

        count[i] = k_cnt
        rank_of_r = np.full(V, -1, np.int16)
        for k in range(k_cnt):
            r = kept_r[k]
            rank_of_r[r] = k
            froot_pad[i, k] = r
            pk_pad[i, k] = kept_pk[k]
            mass_pad[i, k] = kept_mass[k]

        if k_cnt > 0:
            for v in range(V):
                if s[v] > 0.0:
                    rk = rank_of_r[root[v]]
                    if rk >= 0:
                        label[i, v] = rk
                    else:
                        best_k = 0
                        best_sc = -1.0
                        vx, vy, vz = vertices[v, 0], vertices[v, 1], vertices[v, 2]
                        for k in range(k_cnt):
                            kr = kept_r[k]
                            sc = abs(vertices[kr, 0] * vx + vertices[kr, 1] * vy + vertices[kr, 2] * vz)
                            if sc > best_sc:
                                best_sc = sc
                                best_k = k
                        label[i, v] = best_k

    return label, count, froot_pad, pk_pad, mass_pad


def _segment(sf, geo: Geometry, nbr, edges, anti, peak_min, merge_ratio):
    """Lobes of a chunk of fODFs sampled on the sphere.  Returns (label (n, V): the fixel of
    each direction, local index in the voxel or -1; count (n,); per fixel: row, root vertex,
    peak, integral), fixels in row order then descending peak."""
    sf = np.ascontiguousarray(sf, np.float64)
    label, count, froot_pad, pk_pad, mass_pad = _segment_numba(
        sf, geo.vertices, nbr, edges, anti, float(peak_min), float(merge_ratio)
    )
    n = sf.shape[0]
    mask = np.arange(froot_pad.shape[1])[None, :] < count[:, None]
    frow = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], froot_pad.shape)[mask]
    return label, count, frow, froot_pad[mask], pk_pad[mask], mass_pad[mask]


def _sh_fingerprint(sh_rows):
    n = len(sh_rows)
    if n == 0:
        return (0, 0.0, 0.0)
    head = np.asarray(sh_rows[:min(n, 16)], np.float64)
    tail = np.asarray(sh_rows[max(0, n - 16):], np.float64)
    return (n, float(head.sum()), float(tail.sum()))


def segment_lobes(sh_rows, geo: Geometry, peak_min=PEAK_MIN, merge_ratio=MERGE_RATIO, chunk=20000) -> Lobes:
    """The fixels of the voxels whose fODFs (SH rows, the chain's basis) are given.
    ``peak_min``: lobes whose peak is under this fraction of the voxel's maximum are not
    fixels (their directions join the nearest fixel); ``merge_ratio``: lobes whose saddle is
    at least this fraction of the lower peak are one fixel."""
    import scipy.sparse as sps

    nbr, edges, anti = _sphere_neighbours(geo), _sphere_edges(geo), geo.antipode()
    B = _sh_basis(geo.vertices, sh_rows.shape[1])
    V, nb, vbin = geo.n_dirs, geo.nb, geo.vbin
    C = sps.csr_matrix((np.ones(geo.cone_idx.size), geo.cone_idx, geo.cone_ptr), shape=(V, V))
    cls = sps.csr_matrix((np.ones(V), (np.arange(V), vbin)), shape=(V, nb))

    counts, dirs, peaks, ints = [], [], [], []
    w_fix_list, w_unass_list, frow_list = [], [], []
    for s in range(0, len(sh_rows), chunk):
        rows = sh_rows[s:s + chunk]
        n = len(rows)
        sf = np.maximum(np.asarray(rows, np.float64) @ B, 0.0)
        label, count, frow_local, froot, pk, mass = _segment(sf, geo, nbr, edges, anti, peak_min, merge_ratio)
        counts.append(count)
        dirs.append(geo.vertices[froot])
        peaks.append(pk)
        ints.append(mass)

        # Precompute class-to-fixel weights directly from sf so fixel_streamlines is O(N_fixels * 26)
        sf_pmf = sf.copy()
        sf_pmf[sf_pmf < PMF_THRESHOLD * sf_pmf.max(1, keepdims=True)] = 0.0
        tot_pmf = sf_pmf.sum(1, keepdims=True)
        X = np.divide(sf_pmf, tot_pmf, out=np.zeros_like(sf_pmf), where=tot_pmf > 0).astype(np.float32).astype(np.float64)
        Y = (np.asarray(C @ X.T).T.astype(np.float32).astype(np.float64)) if len(X) else X.copy()
        q = np.zeros_like(X)
        todo = np.ones((n, nb), bool)
        for w in (X * Y, Y, np.ones_like(X)):
            tot_w = np.asarray(cls.T @ w.T).T
            use = todo & (tot_w > 0)
            q += np.where(use[:, vbin], w / np.where(tot_w > 0, tot_w, 1)[:, vbin], 0)
            todo &= ~use

        offset_local = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(np.int64) if count.size else count
        n_fix_chunk = int(count.sum())
        on = label >= 0
        f_idx = (offset_local[:, None] + label)[on]
        b_on = np.broadcast_to(vbin[None, :], (n, V))[on]
        w_fix = np.bincount(f_idx * nb + b_on, weights=q[on], minlength=n_fix_chunk * nb).reshape(n_fix_chunk, nb)
        off = ~on
        r_off = np.broadcast_to(np.arange(n)[:, None], (n, V))[off]
        b_off = np.broadcast_to(vbin[None, :], (n, V))[off]
        w_unass = np.bincount(r_off * nb + b_off, weights=q[off], minlength=n * nb).reshape(n, nb)
        w_fix_list.append(w_fix)
        w_unass_list.append(w_unass)
        frow_list.append(frow_local + s)

    count = np.concatenate(counts) if counts else np.zeros(0, np.int64)
    offset = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(np.int64) if count.size else count
    cat = lambda xs, w: np.concatenate(xs) if xs else np.zeros((0,) + w)
    lobes = Lobes(count.astype(np.int64), offset, cat(dirs, (3,)), cat(peaks, ()), cat(ints, ()), float(peak_min),
                  float(merge_ratio), cat(frow_list, ()).astype(np.int64), cat(w_fix_list, (nb,)),
                  cat(w_unass_list, (nb,)), _sh_fingerprint(sh_rows))
    log.info("fixels: %d in %d voxels (%.2f per voxel; %d voxels with none)", lobes.n_fixels, len(count),
             lobes.n_fixels / max(len(count), 1), int((count == 0).sum()))
    return lobes


def fixel_streamlines(density_class, sh_rows, geo: Geometry, lobes: Lobes = None, chunk=20000):
    """Expected bundle streamlines through each fixel: the voxel's visits by direction class
    (``density_class``, (n, 26), e.g. BundleResult.density_class) spread over the sphere as
    the chain assumes and summed over each lobe's directions.  ``lobes``: the voxels' fixels
    (segment_lobes of the same rows; default: segmented here).  Returns (values (N,),
    unassigned (n,): visits in directions that belong to no fixel)."""
    if lobes is None:
        lobes = segment_lobes(sh_rows, geo, chunk=chunk)
    if lobes._w_fix is not None and lobes._sh_fp == _sh_fingerprint(sh_rows):
        dc = np.asarray(density_class, np.float64)
        values = np.einsum("ij,ij->i", dc[lobes._frow], lobes._w_fix) if lobes.n_fixels else np.zeros(0)
        unassigned = np.einsum("ij,ij->i", dc, lobes._w_unass) if len(dc) else np.zeros(0)
        return values, unassigned
    nbr, edges, anti = _sphere_neighbours(geo), _sphere_edges(geo), geo.antipode()
    B = _sh_basis(geo.vertices, sh_rows.shape[1])
    values = np.zeros(lobes.n_fixels)
    unassigned = np.zeros(len(sh_rows))
    for s in range(0, len(sh_rows), chunk):
        rows = sh_rows[s:s + chunk]
        sf = np.maximum(np.asarray(rows, np.float64) @ B, 0.0)
        label, count, *_ = _segment(sf, geo, nbr, edges, anti, lobes.peak_min, lobes.merge_ratio)
        if not np.array_equal(count, lobes.count[s:s + len(rows)]):
            raise ValueError("lobes were not segmented from these fODF rows (fixel counts differ)")
        f = vertex_visits(density_class[s:s + len(rows)], rows, geo)
        on = label >= 0
        idx = (lobes.offset[s:s + len(rows)][:, None] + label)[on]
        values += np.bincount(idx, weights=f[on], minlength=lobes.n_fixels)
        unassigned[s:s + len(rows)] = np.where(on, 0.0, f).sum(1)
    return values, unassigned


_MIF_TYPES = {np.dtype(np.float32): "Float32LE", np.dtype(np.float64): "Float64LE",
              np.dtype(np.int32): "Int32LE", np.dtype(np.uint32): "UInt32LE"}


def write_mif(path, data, affine=None):
    """An MRtrix .mif image (little-endian, first axis fastest).  ``affine``: voxel -> scanner
    mm (RAS, as NIfTI); default identity."""
    data = np.asarray(data)
    if data.dtype not in _MIF_TYPES:
        raise ValueError(f"no .mif datatype for {data.dtype}")
    A = np.eye(4) if affine is None else np.asarray(affine, float)
    vox = np.linalg.norm(A[:3, :3], axis=0)
    R = A[:3, :3] / vox
    lines = ["mrtrix image", "dim: " + ",".join(str(int(d)) for d in data.shape),
             "vox: " + ",".join(f"{v:.17g}" for v in list(vox) + [1.0] * (data.ndim - 3)),
             "layout: " + ",".join(f"+{i}" for i in range(data.ndim)), f"datatype: {_MIF_TYPES[data.dtype]}"]
    lines += ["transform: " + ",".join(f"{v:.17g}" for v in list(R[i]) + [A[i, 3]]) for i in range(3)]
    head = "\n".join(lines) + "\nfile: . {offset}\nEND\n"
    offset = len(head.format(offset=0)) + 2
    hdr = head.format(offset=offset).encode()
    hdr += b" " * (offset - len(hdr))
    with open(path, "wb") as fh:
        fh.write(hdr)
        fh.write(np.ascontiguousarray(data.T).tobytes())     # +0,+1,+2 layout: Fortran order
    return path


def read_mif(path):
    """(data, affine) of a .mif written by write_mif (or MRtrix, for the layouts it writes here)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    end = raw.index(b"\nEND\n")
    kv = {}
    for line in raw[:end].decode().split("\n")[1:]:
        k, _, v = line.partition(": ")
        kv.setdefault(k, []).append(v)
    dims = [int(x) for x in kv["dim"][0].split(",")]
    layout = [int(x) for x in kv["layout"][0].replace("+", "").split(",")]
    if layout != list(range(len(dims))):
        raise ValueError(f"layout {kv['layout'][0]} not supported")
    dt = {v: k for k, v in _MIF_TYPES.items()}[kv["datatype"][0]]
    offset = int(kv["file"][0].split()[1])
    data = np.frombuffer(raw[offset:], dt, count=int(np.prod(dims))).reshape(dims[::-1]).T
    A = np.eye(4)
    vox = [float(x) for x in kv["vox"][0].split(",")]
    for i, t in enumerate(kv.get("transform", [])[:3]):
        A[i] = [float(x) for x in t.split(",")]
    A[:3, :3] *= np.asarray(vox[:3])
    return data, A


def write_fixel_directory(path, lobes: Lobes, ijk, affine, shape, data=None):
    """An MRtrix fixel directory: ``index.mif`` (two volumes: fixels per voxel, first fixel),
    ``directions.mif`` (N x 3 x 1, unit vectors in world axes) and one ``<name>.mif``
    (N x 1 x 1, float32) per entry of ``data`` ({name: (N,) values}).  ``ijk`` (n, 3): the
    voxel of each row of ``lobes``; ``affine``, ``shape``: the image grid.  The .mif files are
    written by hand (write_mif); not yet checked against MRtrix's reader."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    index = np.zeros(tuple(shape) + (2,), np.uint32)
    ijk = tuple(np.asarray(ijk).T)
    index[ijk + (0,)] = lobes.count
    index[ijk + (1,)] = lobes.offset
    write_mif(path / "index.mif", index, affine)
    R = np.asarray(affine, float)[:3, :3]
    d = lobes.direction @ R.T
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-300)
    write_mif(path / "directions.mif", d.astype(np.float32)[:, :, None])
    for name, values in (data or {}).items():
        v = np.asarray(values, np.float32).reshape(-1, 1, 1)
        if v.shape[0] != lobes.n_fixels:
            raise ValueError(f"{name}: {v.shape[0]} values for {lobes.n_fixels} fixels")
        write_mif(path / f"{name}.mif", v)
    return path


class DirectedFixelChain:
    """Absorbing Markov chain whose transient states are directed fixel half-lobes
    ``(v, lobe, sign in {+, -})`` instead of the 26 cube-offset Voronoi classes (~4 states/voxel
    in white matter instead of 26).  When ``lobe_conditioned=True``, a walker in state
    ``(v, lobe, sign)`` propagates through voxel ``v`` using only that lobe's fODF partition
    ``X_{v, lobe}``, preventing intra-voxel multi-hop switching across crossing fiber lobes."""

    ncomp = 1

    def __init__(self, chain_26, lobes: Lobes = None, lobe_conditioned: bool = False):
        from . import kernel as K

        self.lobe_conditioned = bool(lobe_conditioned)
        self.cfg = chain_26.cfg
        self.geo = geo = chain_26.geo
        self.shape = chain_26.shape
        self.pshape = chain_26.pshape
        self.noff = chain_26.noff
        self.label = chain_26.label
        self.t_lin = chain_26.t_lin
        self.tindex = chain_26.tindex
        self.a_lin = chain_26.a_lin
        self.aindex = chain_26.aindex
        self.a_class = chain_26.a_class
        Nt = self.t_lin.size
        V = geo.n_dirs

        nbr, edges, anti = _sphere_neighbours(geo), _sphere_edges(geo), geo.antipode()
        X_t = chain_26.X[chain_26.xrow[self.t_lin]].astype(np.float64)
        Y_t = chain_26.Y[chain_26.xrow[self.t_lin]].astype(np.float64)
        lobe_label, count, frow, froot, _, _ = _segment(X_t, geo, nbr, edges, anti, PEAK_MIN, MERGE_RATIO)
        count = np.maximum(count, 1)
        lobe_offset = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(np.int64) if count.size else count
        n_lobes = int(count.sum())

        self.state_ptr = 2 * np.concatenate([[0], np.cumsum(count)]).astype(np.int64)
        self.n_states = 2 * n_lobes

        lobe_dirs = np.zeros((n_lobes, 3))
        if frow.size:
            lobe_dirs[lobe_offset[frow] + lobe_label[frow, froot]] = geo.vertices[froot]
        v_lobe = np.maximum(lobe_label, 0)
        v_fidx = lobe_offset[:, None] + v_lobe
        dots = np.einsum("ivj,vj->iv", lobe_dirs[v_fidx], geo.vertices) if Nt else np.zeros((0, V))
        sign_bit = (dots < 0).astype(np.int16)
        self.dir_state_local = (2 * v_lobe + sign_bit).astype(np.int16)

        self._build_transitions(chain_26, X_t, Y_t, K)

    def _build_transitions(self, chain_26, X_t, Y_t, K):
        geo = self.geo
        Nt = self.t_lin.size
        V = geo.n_dirs
        Yall = chain_26.Y
        xrow, label, noff = chain_26.xrow, self.label, self.noff
        dbl = chain_26.cfg.double_odf
        tau = float(getattr(chain_26.cfg, "stop_mass", 0.0))
        cptr, cidx = geo.cone_ptr, geo.cone_idx
        u_dir, u_parent, u_d1, u_skip, u_eptr, e_vol, e_nbr = (
            geo.u_dir, geo.u_parent, geo.u_d1, geo.u_skip, geo.u_eptr, geo.e_vol, geo.e_nbr
        )
        n_nodes = u_dir.shape[0]

        Q_ptr = np.zeros(self.n_states + 1, np.int64)
        R_ptr = np.zeros(self.n_states + 1, np.int64)
        q_ind_l, q_w_l = [], []
        r_ind_l, r_w_l = [], []
        self.dead = np.zeros(self.n_states)

        pi = np.empty(n_nodes)
        z = np.empty(V)
        Zv = np.empty(V)
        wrows = np.empty(26, np.int64)

        def _traverse(X_vec, v):
            c_trans, c_abs = {}, {}
            z[:] = 0.0
            u = 0
            while u < n_nodes:
                d = u_dir[u]
                par = u_parent[u]
                if par < 0:
                    if X_vec[d] <= 0.0:
                        u = u_skip[u]
                        continue
                    val_u = 1.0
                else:
                    if X_vec[d] <= 0.0:
                        u = u_skip[u]
                        continue
                    val_u = pi[par] * (X_vec[d] / Y[u_dir[par]])
                    if val_u == 0.0:
                        u = u_skip[u]
                        continue
                pi[u] = val_u
                d1 = int(u_d1[u])
                e0, e1 = u_eptr[u], u_eptr[u + 1]
                for e in range(e0, e1):
                    n = int(e_nbr[e])
                    r = wrows[n]
                    if r == -2:
                        continue
                    w_val = e_vol[e] * val_u
                    if r >= 0:
                        w_val *= Yall[r, d]
                    z[d1] += w_val
                    w_lin = v + noff[n]
                    if label[w_lin] == K.TRANSIENT:
                        wt = self.tindex[w_lin]
                        dst_s = int(self.state_ptr[wt] + self.dir_state_local[wt, d])
                        d_map = c_trans.setdefault(d1, {})
                        d_map[dst_s] = d_map.get(dst_s, 0.0) + w_val
                    else:
                        ai = int(self.aindex[w_lin])
                        a_map = c_abs.setdefault(d1, {})
                        a_map[ai] = a_map.get(ai, 0.0) + w_val
                if Y[d] <= 0.0:
                    u = u_skip[u]
                else:
                    u += 1
            K._incoming_norm(X_vec, Y, z, cptr, cidx, Zv)
            return c_trans, c_abs

        for i in range(Nt):
            v = self.t_lin[i]
            X, Y = X_t[i], Y_t[i]
            K._lookahead_rows(v, label, xrow, noff, dbl, wrows)
            s0, s1 = self.state_ptr[i], self.state_ptr[i + 1]
            if not self.lobe_conditioned:
                c_trans, c_abs = _traverse(X, v)
            for loc_s in range(s1 - s0):
                g_s = s0 + loc_s
                q_e = np.flatnonzero(self.dir_state_local[i] == loc_s)
                if q_e.size == 0:
                    self.dead[g_s] = 1.0
                    Q_ptr[g_s + 1] = Q_ptr[g_s]
                    R_ptr[g_s + 1] = R_ptr[g_s]
                    continue
                if self.lobe_conditioned:
                    X_loc = np.where(self.dir_state_local[i] == loc_s, X, 0.0)
                    c_trans, c_abs = _traverse(X_loc, v)
                else:
                    X_loc = X
                q_w = K.class_q(X_loc, Y, q_e)
                dead_s = 0.0
                acc_d = {}
                for ii in range(q_e.size):
                    e = int(q_e[ii])
                    qe = q_w[ii]
                    if qe <= 0.0:
                        continue
                    denom = Zv[e] + tau
                    if denom <= 0.0 or Y[e] <= 0.0:
                        dead_s += qe
                        continue
                    if tau > 0.0:
                        dead_s += qe * (tau / denom)
                    f = qe / (denom * Y[e])
                    for k in range(cptr[e], cptr[e + 1]):
                        d1 = int(cidx[k])
                        if X_loc[d1] > 0.0 and z[d1] > 0.0:
                            acc_d[d1] = acc_d.get(d1, 0.0) + f
                t_out, a_out = {}, {}
                for d1, f_acc in acc_d.items():
                    ud = X_loc[d1] * f_acc
                    for dst_s, cw in c_trans.get(d1, {}).items():
                        t_out[dst_s] = t_out.get(dst_s, 0.0) + ud * cw
                    for ai, aw in c_abs.get(d1, {}).items():
                        a_out[ai] = a_out.get(ai, 0.0) + ud * aw
                tot = dead_s + sum(t_out.values()) + sum(a_out.values())
                inv = 1.0 / tot if tot > 0 else 0.0
                self.dead[g_s] = dead_s * inv if tot > 0 else 1.0
                for dst_s, val in sorted(t_out.items()):
                    if val > 0:
                        q_ind_l.append(dst_s)
                        q_w_l.append(val * inv)
                for ai, val in sorted(a_out.items()):
                    if val > 0:
                        r_ind_l.append(ai)
                        r_w_l.append(val * inv)
                Q_ptr[g_s + 1] = len(q_ind_l)
                R_ptr[g_s + 1] = len(r_ind_l)

        import scipy.sparse as sps

        self.Q = sps.csr_matrix(
            (np.asarray(q_w_l, np.float64), np.asarray(q_ind_l, np.int32), Q_ptr),
            shape=(self.n_states, self.n_states),
        )
        self.R = sps.csr_matrix(
            (np.asarray(r_w_l, np.float64), np.asarray(r_ind_l, np.int32), R_ptr),
            shape=(self.n_states, self.a_lin.size),
        )
        self.n_transitions = int(self.Q.nnz)
        self.nbytes = self.Q.data.nbytes + self.Q.indices.nbytes + self.Q.indptr.nbytes

    def mul_forward(self, x, out):
        np.copyto(out, self.Q.T @ x)
        return out

    def mul_backward(self, x, out):
        np.copyto(out, self.Q @ x)
        return out

    def absorb_rhs(self, absw):
        return np.asarray(self.R @ absw)

    def absorbed_from(self, S):
        return np.asarray(self.R.T @ S)

