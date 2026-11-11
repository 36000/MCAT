"""Directions, turning cone and within-voxel paths (MITTENS v2), and fODFs on them (DIPY).

MITTENS enumerates every hop sequence a fiber can take inside a voxel before it exits
into one of the 26 neighbours, with the exact volume of start points that realise it.
MCAT adds what MITTENS's first-order model does not need: the last hop's direction class
(the walker's memory, carried into the next voxel) and each path's box of start points
(the spatial-moment closure).
"""
from __future__ import annotations

from dataclasses import dataclass

import mittens
import numpy as np
from mittens.io import default_sphere

OFFSETS = np.asarray(mittens.neighbor_offsets(), np.int64)  # (26, 3), lexicographic
PMF_THRESHOLD = 0.05      # sphere values below this x max are zeroed (pyAFQ's tracker)
PEAK_THRESHOLD = 0.25     # seed peaks: relative threshold, min separation (pyAFQ's tracker)
PEAK_SEPARATION = 45.0


@dataclass
class Geometry:
    vertices: np.ndarray   # (V, 3) signed directions, voxel axes
    cone_ptr: np.ndarray   # CSR: directions within the turning angle of each direction
    cone_idx: np.ndarray
    p_vol: np.ndarray      # (P,) start volume of each path (fraction of the voxel)
    p_dirs: np.ndarray     # (P, Lmax) hop directions, -1 padded
    p_len: np.ndarray      # (P,)
    p_nbr: np.ndarray      # (P,) exit neighbour
    p_disp: np.ndarray     # (P, 3) total displacement, voxel units
    lo: np.ndarray         # (P, 3) box of start points in [0, 1]^3
    hi: np.ndarray
    vbin: np.ndarray       # (V,) direction class (nearest of the 26 neighbour directions)
    bin_ptr: np.ndarray    # CSR: vertices of each class
    bin_idx: np.ndarray
    p_trip: np.ndarray     # (P,) index of the path's (first hop, neighbour, last class) triple
    d1_ptr: np.ndarray     # triples sorted by first hop: CSR over directions
    t_nbr: np.ndarray
    t_bin: np.ndarray
    nb: int = 26
    # Preorder prefix tree over hop sequences (for O(1) zero-ODF subtree skipping)
    u_dir: np.ndarray = None
    u_parent: np.ndarray = None
    u_d1: np.ndarray = None
    u_skip: np.ndarray = None
    u_eptr: np.ndarray = None
    e_vol: np.ndarray = None
    e_nbr: np.ndarray = None
    e_trip: np.ndarray = None
    e_orig: np.ndarray = None

    @property
    def n_dirs(self):
        return self.vertices.shape[0]

    def antipode(self):
        return np.argmin(self.vertices @ self.vertices.T, axis=1)


def _build_prefix_tree(p_dirs, p_len, p_vol, p_nbr, p_trip):
    """Compile flat path emissions into a preorder prefix tree over hop sequences."""
    P = p_dirs.shape[0]
    children = [{}]
    node_dir = [-1]
    node_parent = [-1]
    node_d1 = [-1]
    node_emiss = [[]]

    for p in range(P):
        L = int(p_len[p])
        cur = 0
        d1 = int(p_dirs[p, 0])
        for i in range(L):
            d = int(p_dirs[p, i])
            ch = children[cur].get(d)
            if ch is None:
                ch = len(children)
                children[cur][d] = ch
                children.append({})
                node_dir.append(d)
                node_parent.append(cur)
                node_d1.append(d1)
                node_emiss.append([])
            cur = ch
        node_emiss[cur].append(p)

    n_nodes = len(children) - 1
    u_dir = np.empty(n_nodes, np.int64)
    u_parent = np.empty(n_nodes, np.int64)
    u_d1 = np.empty(n_nodes, np.int64)
    u_skip = np.empty(n_nodes, np.int64)
    u_eptr = np.zeros(n_nodes + 1, np.int64)
    e_orig = np.empty(P, np.int64)

    old_to_pre = np.full(len(children), -1, np.int64)
    stack = [(children[0][d], False) for d in sorted(children[0].keys(), reverse=True)]
    pre_idx = 0
    emiss_idx = 0
    while stack:
        nid, exiting = stack.pop()
        if exiting:
            u_skip[old_to_pre[nid]] = pre_idx
            continue
        u = pre_idx
        pre_idx += 1
        old_to_pre[nid] = u
        u_dir[u] = node_dir[nid]
        par = node_parent[nid]
        u_parent[u] = -1 if par == 0 else old_to_pre[par]
        u_d1[u] = node_d1[nid]
        for p in node_emiss[nid]:
            e_orig[emiss_idx] = p
            emiss_idx += 1
        u_eptr[u + 1] = emiss_idx
        stack.append((nid, True))
        for d in sorted(children[nid].keys(), reverse=True):
            stack.append((children[nid][d], False))

    return (u_dir, u_parent, u_d1, u_skip, u_eptr,
            np.ascontiguousarray(p_vol[e_orig]),
            np.ascontiguousarray(p_nbr[e_orig]),
            np.ascontiguousarray(p_trip[e_orig]),
            e_orig)


def class_chord_lengths(geo: Geometry, zooms=None):
    """Expected inter-voxel step length (in voxel units if ``zooms is None``, else in mm)
    for each of the 26 direction classes, weighted by path volume over each class's cone."""
    nbr_disp = OFFSETS.astype(np.float64)
    if zooms is not None:
        nbr_disp = nbr_disp * np.abs(np.asarray(zooms, float))[None, :]
    nbr_len = np.linalg.norm(nbr_disp, axis=1)
    p_len = nbr_len[geo.p_nbr]
    V = geo.n_dirs
    d1 = geo.p_dirs[:, 0]
    vol_d1 = np.bincount(d1, weights=geo.p_vol, minlength=V)
    len_d1 = np.bincount(d1, weights=geo.p_vol * p_len, minlength=V)
    out = np.zeros(geo.nb)
    for b in range(geo.nb):
        es = geo.bin_idx[geo.bin_ptr[b]:geo.bin_ptr[b + 1]]
        w_tot = l_tot = 0.0
        for e in es:
            ds = geo.cone_idx[geo.cone_ptr[e]:geo.cone_ptr[e + 1]]
            w_tot += vol_d1[ds].sum()
            l_tot += len_d1[ds].sum()
        out[b] = l_tot / max(w_tot, 1e-30)
    return out


def build_geometry(zooms, angle_max=35.0, step_vox=1.0) -> Geometry:
    """Paths for hops of ``step_vox`` voxels (of the smallest voxel side) turning less than
    ``angle_max`` degrees, on DSI Studio's 642-direction sphere; anisotropic voxels are exact."""
    V = np.ascontiguousarray(default_sphere(), np.float64)
    zooms = np.abs(np.asarray(zooms, float))
    table = mittens.PathTable(V, float(step_vox * zooms.min()), float(angle_max), np.diag(zooms))
    e = table.emissions()
    dirs = np.asarray(e["chain"], np.int64)
    length = (dirs >= 0).sum(1)
    nbr = np.asarray(e["neighbor"], np.int64)
    vol = np.asarray(e["volume"], np.float64)
    disp = np.asarray(table.displacements(), np.float64)[np.maximum(dirs, 0)]  # (P, L, 3)
    disp[dirs < 0] = 0.0
    cum = np.cumsum(disp, axis=1)
    total = cum[np.arange(len(dirs)), length - 1]
    # start points x in [0,1]^3 stay inside for every hop but the last, which lands in o
    inner = np.where((np.arange(dirs.shape[1])[None, :] < (length - 1)[:, None])[..., None], cum, 0.0)
    o = OFFSETS[nbr].astype(np.float64)
    lo = np.maximum(np.maximum(0.0, (-inner).max(1)), o - total)
    hi = np.minimum(np.minimum(1.0, (1.0 - inner).min(1)), o + 1.0 - total)
    if not np.allclose(np.clip(hi - lo, 0, None).prod(1), vol, rtol=1e-9, atol=1e-14):
        raise RuntimeError("path start boxes disagree with MITTENS's path volumes")
    ip, ix, _ = table.turning_matrix()
    bins = OFFSETS / np.linalg.norm(OFFSETS, axis=1, keepdims=True)
    vbin = np.argmax(V @ bins.T, axis=1).astype(np.int64)
    bin_idx = np.argsort(vbin, kind="stable").astype(np.int64)
    bin_ptr = np.concatenate([[0], np.cumsum(np.bincount(vbin, minlength=26))]).astype(np.int64)
    last = dirs[np.arange(len(dirs)), length - 1]
    key = dirs[:, 0] * (26 * 26) + nbr * 26 + vbin[last]
    ukey, trip = np.unique(key, return_inverse=True)
    d1 = ukey // (26 * 26)
    trip = trip.astype(np.int64)
    tree = _build_prefix_tree(dirs, length, vol, nbr, trip)
    return Geometry(V, np.asarray(ip, np.int64), np.asarray(ix, np.int64), vol, dirs, length, nbr,
                    total, lo, hi, vbin, bin_ptr, bin_idx, trip,
                    np.searchsorted(d1, np.arange(len(V) + 1)).astype(np.int64),
                    ((ukey // 26) % 26).astype(np.int64), (ukey % 26).astype(np.int64), 26, *tree)


def _sh_basis(vertices, n_coef):
    """(n_coef, V) so that sf = sh @ B: DIPY's descoteaux07 basis (legacy), symmetric or full,
    as pyAFQ's tracker evaluates its (a)ODFs."""
    from dipy.core.sphere import Sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    sym = (-3.0 + np.sqrt(1.0 + 8.0 * n_coef)) / 2.0
    full = not float(sym).is_integer()
    order = int(round(np.sqrt(n_coef) - 1)) if full else int(round(sym))
    return np.asarray(sh_to_sf_matrix(Sphere(xyz=vertices), sh_order_max=order, full_basis=full,
                                      return_inv=False), np.float64)


def pmf_and_cone_mass(sh_rows, geo: Geometry, chunk=20000):
    """X: fODF on the sphere, clipped at 0, values below PMF_THRESHOLD x max zeroed,
    normalised to sum 1 per voxel (rows of zeros stay zero); Y: X summed over each cone."""
    import scipy.sparse as sps

    B = _sh_basis(geo.vertices, sh_rows.shape[1])
    X = np.empty((sh_rows.shape[0], geo.n_dirs), np.float32)
    for s in range(0, sh_rows.shape[0], chunk):
        sf = np.maximum(np.asarray(sh_rows[s:s + chunk], np.float64) @ B, 0.0)
        sf[sf < PMF_THRESHOLD * sf.max(1, keepdims=True)] = 0.0
        tot = sf.sum(1, keepdims=True)
        X[s:s + chunk] = np.divide(sf, tot, out=np.zeros_like(sf), where=tot > 0)
    C = sps.csr_matrix((np.ones(geo.cone_idx.size), geo.cone_idx, geo.cone_ptr),
                       shape=(geo.n_dirs,) * 2)
    Y = np.asarray(C @ X.T.astype(np.float64)).T.astype(np.float32) if len(X) else X.copy()
    return X, Y


def vertex_visits(density_class, sh_rows, geo: Geometry):
    """Expected visits per sphere direction, (n, V): a voxel's visits by direction class
    (density_class, (n, 26)) spread over the class's directions as the chain assumes,
    q ~ X Y (kernel.class_q, with its fallbacks), from the fODF there (sh_rows, (n, ncoef)).
    Rows sum to the voxel's visits.  Not symmetrised: a direction is the walker's heading."""
    import scipy.sparse as sps

    V = geo.n_dirs
    cls = sps.csr_matrix((np.ones(V), (np.arange(V), geo.vbin)), shape=(V, geo.nb))
    X, Y = pmf_and_cone_mass(sh_rows, geo)
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    q = np.zeros_like(X)
    todo = np.ones((len(X), geo.nb), bool)
    for w in (X * Y, Y, np.ones_like(X)):           # class_q's fallbacks
        tot = np.asarray(cls.T @ w.T).T              # (n, 26)
        use = todo & (tot > 0)
        q += np.where(use[:, geo.vbin], w / np.where(tot > 0, tot, 1)[:, geo.vbin], 0)
        todo &= ~use
    return np.asarray(density_class, np.float64)[:, geo.vbin] * q


def visitation_odf(density_class, sh_rows, geo: Geometry, sh_order=8, chunk=20000):
    """A bundle's visitation ODF: the directions its walkers travel in each voxel, as SH
    coefficients in the fODF's basis (DIPY descoteaux07, legacy, symmetric).

    density_class (n, 26): expected visits by direction class, per voxel; sh_rows (n, ncoef):
    the fODF there.  Within a class the direction is spread as the chain assumes,
    q ~ X Y (kernel.class_q; vertex_visits).  The ODF is antipodally symmetrised and in
    expected visits per steradian, so it integrates to the voxel's density
    (c_00 = density / (2 sqrt(pi)))."""
    from dipy.core.sphere import Sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    _, inv = sh_to_sf_matrix(Sphere(xyz=geo.vertices), sh_order_max=sh_order, return_inv=True)
    anti = geo.antipode()
    out = np.zeros((len(sh_rows), inv.shape[1]), np.float32)
    for s in range(0, len(sh_rows), chunk):
        f = vertex_visits(density_class[s:s + chunk], sh_rows[s:s + chunk], geo)
        c = (0.5 * (f + f[:, anti])) @ inv
        # exact integral (the sphere's vertices do not cover equal solid angles)
        c00 = f.sum(1) / (2 * np.sqrt(np.pi))
        out[s:s + chunk] = c * np.divide(c00, c[:, 0], out=np.zeros_like(c00), where=c[:, 0] > 0)[:, None]
    return out


def seed_peaks(pmf, geo: Geometry):
    """Vertex indices of the fODF peaks of one voxel (DIPY's peak finder; pyAFQ's thresholds)."""
    from dipy.core.sphere import Sphere
    from dipy.direction.peaks import peak_directions

    if not hasattr(geo, "_dipy_sphere"):
        geo._dipy_sphere = Sphere(xyz=geo.vertices)
    _, _, idx = peak_directions(np.asarray(pmf, np.float64), geo._dipy_sphere,
                                relative_peak_threshold=PEAK_THRESHOLD,
                                min_separation_angle=PEAK_SEPARATION, is_symmetric=False)
    return np.asarray(idx, np.int64)
