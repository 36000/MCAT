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

from .geometry import Geometry, _sh_basis, vertex_visits

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


def _segment(sf, geo: Geometry, nbr, edges, anti, peak_min, merge_ratio):
    """Lobes of a chunk of fODFs sampled on the sphere.  Returns (label (n, V): the fixel of
    each direction, local index in the voxel or -1; count (n,); per fixel: row, root vertex,
    peak, integral), fixels in row order then descending peak."""
    n, V = sf.shape
    mx = sf.max(1)
    rows = np.arange(n)[:, None]
    root = _lobe_roots(sf, nbr)
    root = np.minimum(root, root[rows, anti[root]])        # join the antipodal halves
    root = _merge_lobes(sf, root, edges, anti, merge_ratio)
    alive = (sf > 0) & (mx[:, None] > 0)
    key = np.where(alive, rows * V + root, -1).ravel()
    ukey, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(n, V)
    keep = ukey >= 0
    frow, froot = ukey[keep] // V, ukey[keep] % V
    pk = np.maximum(sf[frow, froot], sf[frow, anti[froot]])
    ok = pk >= peak_min * mx[frow]
    mass = np.bincount(inv.ravel(), weights=sf.ravel(), minlength=ukey.size)[keep] * (4 * np.pi / V)
    # order: row, then descending peak; dropped lobes go last and get no label
    order = np.lexsort((-pk, ~ok, frow))
    frow, froot, pk, mass, ok = (x[order] for x in (frow, froot, pk, mass, ok))
    first = np.searchsorted(frow, np.arange(n))
    local = np.arange(frow.size) - first[frow]
    local[~ok] = -1
    count = np.bincount(frow[ok], minlength=n)
    rank = np.full(ukey.size, -1, np.int64)
    rank[np.flatnonzero(keep)[order]] = local
    label = np.where(alive, rank[inv], -1)
    # directions of dropped lobes join the nearest kept fixel of their voxel (by angle)
    r_d, v_d = np.nonzero(alive & (label < 0) & (count[:, None] > 0))
    if r_d.size:
        kmax = int(count.max())
        dirs = np.zeros((n, kmax, 3))
        kept_rows, kept_local = frow[ok], local[ok]
        dirs[kept_rows, kept_local] = geo.vertices[froot[ok]]
        score = np.abs(np.einsum("mkj,mj->mk", dirs[r_d], geo.vertices[v_d]))
        label[r_d, v_d] = score.argmax(1)
    return label.astype(np.int16), count, frow[ok], froot[ok], pk[ok], mass[ok]


def segment_lobes(sh_rows, geo: Geometry, peak_min=PEAK_MIN, merge_ratio=MERGE_RATIO, chunk=20000) -> Lobes:
    """The fixels of the voxels whose fODFs (SH rows, the chain's basis) are given.
    ``peak_min``: lobes whose peak is under this fraction of the voxel's maximum are not
    fixels (their directions join the nearest fixel); ``merge_ratio``: lobes whose saddle is
    at least this fraction of the lower peak are one fixel."""
    nbr, edges, anti = _sphere_neighbours(geo), _sphere_edges(geo), geo.antipode()
    B = _sh_basis(geo.vertices, sh_rows.shape[1])
    counts, dirs, peaks, ints = [], [], [], []
    for s in range(0, len(sh_rows), chunk):
        sf = np.maximum(np.asarray(sh_rows[s:s + chunk], np.float64) @ B, 0.0)
        _, count, _, froot, pk, mass = _segment(sf, geo, nbr, edges, anti, peak_min, merge_ratio)
        counts.append(count)
        dirs.append(geo.vertices[froot])
        peaks.append(pk)
        ints.append(mass)
    count = np.concatenate(counts) if counts else np.zeros(0, np.int64)
    offset = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(np.int64) if count.size else count
    cat = lambda xs, w: np.concatenate(xs) if xs else np.zeros((0,) + w)
    lobes = Lobes(count.astype(np.int64), offset, cat(dirs, (3,)), cat(peaks, ()), cat(ints, ()), float(peak_min),
                  float(merge_ratio))
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
