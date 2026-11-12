"""Synthetic fODF phantoms for testing (SH in DIPY descoteaux07 legacy basis)."""
from __future__ import annotations

import numpy as np


def _sh_fit_matrix(sh_order=8, sphere_name="repulsion724"):
    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    sp = get_sphere(name=sphere_name)
    B, invB = sh_to_sf_matrix(sp, sh_order_max=sh_order, basis_type="descoteaux07",
                              legacy=True, return_inv=True)
    return np.asarray(sp.vertices), invB


def lobes_to_sh(dirs_list, kappa=30.0, sh_order=8):
    """fODF = sum of axially symmetric Watson lobes exp(kappa (d.mu)^2)."""
    verts, invB = _sh_fit_matrix(sh_order)
    out = []
    for dirs in dirs_list:
        sf = np.zeros(verts.shape[0])
        for mu in dirs:
            mu = np.asarray(mu, float)
            mu /= np.linalg.norm(mu)
            sf += np.exp(kappa * ((verts @ mu) ** 2 - 1.0))
        out.append(sf @ invB)
    return np.array(out)


def tube_phantom(shape=(40, 15, 15), radius=3.0, gm_caps=True, kappa=30.0):
    """Straight fibre bundle along i, GM at both ends, CSF around the tube."""
    sh_one = lobes_to_sh([[(1, 0, 0)]], kappa)[0]
    sh = np.zeros(shape + (sh_one.size,), np.float32)
    ii, jj, kk = np.indices(shape)
    cj, ck = (shape[1] - 1) / 2, (shape[2] - 1) / 2
    tube = (jj - cj) ** 2 + (kk - ck) ** 2 <= radius**2
    wm = tube & (ii >= 3) & (ii < shape[0] - 3)
    gm = tube & ~wm
    sh[wm] = sh_one
    sh[gm] = lobes_to_sh([[(1, 0, 0), (0, 1, 0), (0, 0, 1)]], 2.0)[0]  # near-isotropic GM
    cls = np.zeros(shape, np.int8)
    cls[gm] = 1
    csf = ~tube
    cls[csf] = 2
    sh[csf] = lobes_to_sh([[(1, 0, 0), (0, 1, 0), (0, 0, 1)]], 0.5)[0]
    return sh, wm, cls


def crossing_phantom(n=41, width=7, kappa=30.0):
    """Two orthogonal bundles (along i and along j) crossing at the centre.
    Each bundle ends in its own GM caps, so absorption reveals which bundle a
    walker finished in."""
    shape = (n, n, width + 4)
    sh_x, sh_y, sh_xy, sh_gm = lobes_to_sh(
        [[(1, 0, 0)], [(0, 1, 0)], [(1, 0, 0), (0, 1, 0)], [(1, 0, 0), (0, 1, 0), (0, 0, 1)]],
        kappa)
    sh = np.zeros(shape + (sh_x.size,), np.float32)
    ii, jj, kk = np.indices(shape)
    c = (n - 1) // 2
    h = width // 2
    zin = (kk >= 2) & (kk < 2 + width)
    bx = (np.abs(jj - c) <= h) & zin  # bundle along i
    by = (np.abs(ii - c) <= h) & zin  # bundle along j
    capx = bx & ((ii < 2) | (ii >= n - 2))
    capy = by & ((jj < 2) | (jj >= n - 2))
    wm = (bx | by) & ~capx & ~capy
    sh[bx & ~by] = sh_x
    sh[by & ~bx] = sh_y
    sh[bx & by] = sh_xy
    cls = np.full(shape, 2, np.int8)  # CSF everywhere else
    cls[capx | capy] = 1
    cls[wm] = 0
    sh[cls == 2] = lobes_to_sh([[(1, 0, 0), (0, 1, 0), (0, 0, 1)]], 0.5)[0]
    sh[capx | capy] = sh_gm
    regions = dict(capx=capx, capy=capy, center=(c, c, 2 + h))
    return sh, wm, cls, regions


def arc_phantom(R=14.0, width=5, n=40, kappa=30.0):
    """Quarter-circle bundle in the i-j plane (tests turning across voxels)."""
    shape = (n, n, width + 4)
    ii, jj, kk = np.indices(shape).astype(float)
    r = np.hypot(ii - 2, jj - 2)
    zin = (kk >= 2) & (kk < 2 + width)
    band = (np.abs(r - R) <= width / 2) & zin
    theta = np.arctan2(jj - 2, ii - 2)
    wm = band & (theta > 0.08) & (theta < np.pi / 2 - 0.08)
    caps = band & ~wm & (ii >= 2) & (jj >= 2)
    dirs = [[(-np.sin(t), np.cos(t), 0.0)] for t in theta[wm]]
    sh_n = lobes_to_sh([[(1, 0, 0)]], kappa).shape[1]
    sh = np.zeros(shape + (sh_n,), np.float32)
    sh[wm] = lobes_to_sh(dirs, kappa)
    cls = np.full(shape, 2, np.int8)
    cls[caps] = 1
    sh[~wm] = lobes_to_sh([[(1, 0, 0), (0, 1, 0), (0, 0, 1)]], 1.0)[0]
    return sh, wm, cls, dict(cap_start=caps & (theta <= 0.08), cap_end=caps & (theta >= np.pi / 2 - 0.08))
