"""Tract profiles: scalar values along a bundle's core, from its MPAT density map (issue 6).

For one bundle and its core (``mpat.core``, n nodes in world mm):

1. Support.  The voxels ``bundle_core`` builds its graph on: density above ``floor`` x peak, in
   the connected component that holds the peak.
2. Assignment.  Each support voxel gets the core node nearest to it along the bundle, in MPAT's
   cross-section metric (``bundle_core`` step 3: -log(density / peak) per mm on the support's
   26-neighbour graph, same step lengths and tie-break), by one multi-source Dijkstra.  A node
   enters the graph through the support voxels at the corners of the voxel cell it lies in.
   Nodes are never moved or renumbered; one with no support corner, or one that coincides with
   another, gets no voxels here.
3. Spreading.  A voxel is a cube that spans a stretch of the core, so it is shared between the
   nodes along that stretch.  Its centre is projected onto the core segment next to its node from
   step 2 (arc length s, direction u); projected onto u the cube is a sum of three uniform
   lengths |a_k . u| (a_k the affine's columns), and its share of node n is the fraction of that
   between the midpoints to n's neighbours.  Shares sum to one.  Without this, nodes closer
   together than a voxel get no voxels (about half the nodes of a 100 mm bundle on 2 mm voxels).
4. Weights.  d_i = density / peak x the voxel's share of the node.  ``weighting``:
   "mahalanobis"  w_i = d_i / D_i, D_i the Mahalanobis distance of the voxel centre from the
                  node's d-weighted mean in its d-weighted covariance (pinvh, so thin cross-sections
                  are kept): pyAFQ's ``gaussian_weights`` with every sum weighted by d.  As there, a
                  voxel exactly at the mean gets zero weight; a node whose weights are all zero
                  (one voxel) is weighted by d alone.
   "density"      w_i = d_i: the expected value over MPAT's walkers, no cross-section geometry.
   Which should be the default is open; on Stanford HARDI and one HBN subject, density-only
   correlated better with pyAFQ's profiles and was smoother than d / D.
5. Profile.  P_n = sum w_i s_i / sum w_i over node n's voxels with a finite scalar s_i; NaN if
   there are none (info lists nodes with no voxels in ``empty_nodes``).  Scalars are put on the
   density grid with ``afq._on_grid`` (linear); voxels outside a scalar's grid read 0 there.
"""
from __future__ import annotations

import csv
import json
import os.path as op

import nibabel as nib
import numpy as np

from .afq import _on_grid
from .core import _graph, _support, bundle_files

CELL = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def _assign(dn, keep, affine, xyz):
    """(ijk (m, 3) support voxels, owner (m,) node index or -1) -- see step 2."""
    import scipy.sparse as sps
    from scipy.sparse.csgraph import dijkstra

    vox_mm = float(np.abs(np.linalg.det(affine[:3, :3])) ** (1 / 3))  # as bundle_core
    cost = np.where(keep, -np.log(np.where(keep, dn, 1.0)), 0.0)   # as bundle_core's cost1
    G, idx, lin = _graph(cost, keep, vox_mm)
    m, n = len(idx), len(xyz)
    ijk = nib.affines.apply_affine(np.linalg.inv(affine), xyz)
    corner = np.floor(ijk).astype(np.int64)[:, None] + CELL            # (n, 8, 3)
    inside = np.all((corner >= 0) & (corner < keep.shape), -1)
    vox = np.full(inside.shape, -1, np.int64)
    vox[inside] = lin[tuple(corner[inside].T)]
    on = vox >= 0
    mm = np.linalg.norm(corner - ijk[:, None], axis=-1) * vox_mm
    c = np.zeros(vox.shape)
    c[on] = cost[tuple(idx[vox[on]].T)]
    E = sps.csr_matrix((mm[on] * c[on] + 1e-9, (m + np.nonzero(on)[0], vox[on])), shape=(m + n, m + n))
    G = sps.block_diag((G, sps.csr_matrix((n, n)))).tocsr() + E
    dist, _, src = dijkstra(G, indices=m + np.arange(n), min_only=True, return_predecessors=True)
    return idx, np.where(np.isfinite(dist[:m]), src[:m] - m, -1)


def _box_cdf(z, w):
    """P(U_1 + U_2 + U_3 <= z) for independent U_k uniform on [-w_k/2, w_k/2]: z (m, K), w (m, 3).
    The box-spline formula sum_e (-1)^|e| (y - e.w)_+^q / (q! prod w), y = z + sum(w) / 2, over the
    q widths that are not ~0 (an edge across u adds no length)."""
    w = -np.sort(-w, axis=1)
    q = (w > 1e-6 * w[:, :1]).sum(1)
    out = np.empty(z.shape)
    for k in (1, 2, 3):
        i = q == k
        wk = w[i, :k]
        tot = wk.sum(1)[:, None]
        y = np.clip(z[i] + tot / 2, 0, tot)
        f = sum((-1) ** sum(e) * np.clip(y - (wk @ np.array(e))[:, None], 0, None) ** k
                for e in np.ndindex(*[2] * k))
        out[i] = np.clip(f / (np.prod(wk, 1)[:, None] * [1, 1, 2, 6][k]), 0, 1)
    return out


def _spread(x, owner, core, affine):
    """(voxel index, node, share) of every voxel's shares -- see step 3."""
    seg = np.diff(core, axis=0)
    L = np.linalg.norm(seg, axis=1)
    s_node = np.r_[0, np.cumsum(L)]
    edges = np.r_[-np.inf, (s_node[:-1] + s_node[1:]) / 2, np.inf]      # node n spans edges n..n+1
    v = np.flatnonzero(owner >= 0)
    best = np.full(len(v), np.inf)
    s = np.empty(len(v))
    j_of = np.zeros(len(v), int)
    for j in (owner[v] - 1, owner[v]):                     # the two segments at the voxel's node
        ok = (j >= 0) & (j < len(seg)) & (L[np.clip(j, 0, len(seg) - 1)] > 0)
        jj = np.clip(j, 0, len(seg) - 1)
        t = np.einsum("ij,ij->i", x[v] - core[jj], seg[jj]) / np.where(ok, L[jj] ** 2, 1)
        t = np.clip(t, np.where(jj == 0, -np.inf, 0), np.where(jj == len(seg) - 1, np.inf, 1))
        dist = np.where(ok, np.linalg.norm(x[v] - core[jj] - t[:, None] * seg[jj], axis=1), np.inf)
        better = dist < best
        best[better], s[better], j_of[better] = dist[better], (s_node[jj] + t * L[jj])[better], jj[better]
    whole = ~np.isfinite(best)                               # no segment of length > 0 at its node
    w = np.where(whole[:, None], 1.0, np.abs((seg[j_of] / np.where(whole, 1, L[j_of])[:, None]) @ affine[:3, :3]))
    half = w.sum(1) / 2
    lo = np.searchsorted(edges, s - half) - 1
    hi = np.searchsorted(edges, s + half) - 1
    nodes = lo[:, None] + np.arange((hi - lo).max() + 1 if len(v) else 1)
    past = nodes > hi[:, None]                               # padding beyond the voxel's last node
    nodes[past] = 0
    F = _box_cdf(edges[nodes + 1] - s[:, None], w) - _box_cdf(edges[nodes] - s[:, None], w)
    F[past] = 0
    F[whole] = 0
    F[whole, 0], nodes[whole, 0] = 1.0, owner[v[whole]]
    keep = F > 1e-12
    return np.broadcast_to(v[:, None], F.shape)[keep], nodes[keep], F[keep]


def _mahalanobis(x, d):
    """Mahalanobis distance of each x_i from the d-weighted mean, in the d-weighted population
    covariance (pseudo-inverted).  pyAFQ's ``gaussian_weights`` (population covariance, pinvh),
    with every sum weighted by d:
    https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/AFQ/_fixes.py#L335-L350"""
    from scipy.linalg import pinvh

    r = x - d @ x / d.sum()
    S = (d[:, None] * r).T @ r / d.sum()
    D2 = np.einsum("ij,jk,ik->i", r, pinvh(0.5 * (S + S.T)), r)
    return np.sqrt(np.clip(D2, 0, None))


def _weights(x, d, owner, n, weighting="mahalanobis"):
    """Unnormalised profile weight of every voxel (step 4)."""
    if weighting == "density":
        return d
    if weighting != "mahalanobis":
        raise ValueError(f"weighting must be 'mahalanobis' or 'density', not {weighting!r}")
    w = np.zeros(len(x))
    for k in range(n):
        i = np.flatnonzero(owner == k)
        if not len(i):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            wk = d[i] / _mahalanobis(x[i], d[i])
        # inverse distances that are not finite get no weight, as in pyAFQ:
        # https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/AFQ/_fixes.py#L355-L357
        wk[~np.isfinite(wk)] = 0
        w[i] = wk if wk.sum() > 0 else d[i]
    return w


def tract_profiles(density, affine, core, scalars, n_nodes=100, floor=1e-6,
                   weighting="mahalanobis"):
    """Profiles of one bundle: ({scalar: (n,) values}, info).

    density   3-D bundle density map (NaN / negative read as 0, as bundle_core does)
    core      (n, 3) core nodes in world mm (``Core.xyz``), or None when there is none
    scalars   {name: nibabel image}
    n_nodes   the node count when there is no core; otherwise the core's own count is kept
    weighting "mahalanobis" or "density" (step 4)
    """
    d = np.clip(np.nan_to_num(np.asarray(density, float)), 0, None)   # as bundle_core
    if d.max() <= 0 or core is None:
        status = "empty" if d.max() <= 0 else "no core"
        return {k: np.full(n_nodes, np.nan) for k in scalars}, dict(status=status)
    n = len(core)
    dn = d / d.max()
    ijk, owner = _assign(dn, _support(dn, floor), affine, core)
    x = nib.affines.apply_affine(affine, ijk)
    vox, node, share = _spread(x, owner, core, affine)
    w = _weights(x[vox], dn[tuple(ijk.T)][vox] * share, node, n, weighting)
    out = {}
    ref = nib.Nifti1Image(d, affine)
    for name, img in scalars.items():
        s = _on_grid(img, ref)[tuple(ijk.T)].astype(float)[vox]
        ok = np.isfinite(s) & (w > 0)
        num = np.bincount(node[ok], w[ok] * s[ok], n)
        den = np.bincount(node[ok], w[ok], n)
        out[name] = np.divide(num, den, out=np.full(n, np.nan), where=den > 0)
    count = np.bincount(node, minlength=n)
    return out, dict(status="ok", n_support=len(owner), empty_nodes=np.flatnonzero(count == 0).tolist())


def write_profiles(folder, scalars, weighting="mahalanobis"):
    """Profiles of every MPAT bundle under ``folder`` from its saved core (``write_cores``), as
    pyAFQ-style ``<stem>_desc-profiles_tractography.csv`` (tractID, nodeID, one column per scalar;
    missing values empty) and a ``.json`` with each bundle's status.

    scalars   {column name: path of a scalar image}, e.g. pyAFQ's exported ``dti_fa``
    weighting passed to ``tract_profiles``
    """
    files = bundle_files(str(folder))
    out = next(iter(files.values()))["prefix"].split("_bundle-")[0] + "_desc-profiles_tractography.csv"
    images = {k: nib.load(p) for k, p in scalars.items()}
    rows, meta = [], {}
    for key, f in files.items():
        summary = f["prefix"] + "_desc-summary.json"
        name = json.load(open(summary)).get("bundle", key) if op.exists(summary) else key
        img = nib.load(f["density"])
        core_tsv = f["prefix"] + "_desc-core_path.tsv"
        core = np.loadtxt(core_tsv, skiprows=1, usecols=(1, 2, 3), ndmin=2) if op.exists(core_tsv) else None
        prof, meta[name] = tract_profiles(img.dataobj, img.affine, core, images, weighting=weighting)
        for i, values in enumerate(zip(*prof.values())):
            rows.append([name, i] + ["" if np.isnan(v) else repr(float(v)) for v in values])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tractID", "nodeID", *images])
        w.writerows(rows)
    with open(out[: -len(".csv")] + ".json", "w") as fh:
        json.dump(dict(method="mpat.profiles (see its docstring); nodeID 0 is the core's start end",
                       weighting=weighting,
                       scalars={k: str(p) for k, p in scalars.items()}, bundles=meta), fh, indent=1)
    return out
