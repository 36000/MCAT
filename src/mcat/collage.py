"""MIP collage of a run's bundles: one tile per bundle, with sagittal, coronal and axial
maximum-intensity projections of its density (log scale, each bundle divided by its own
peak) on a grey silhouette of the run's tissue labels.  Written by ``mcat.run``."""
from __future__ import annotations

import glob
import os.path as op
import re

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from .core import bundle_files

VIEWS = (("sagittal", 0), ("coronal", 1), ("axial", 2))


def _sort_key(name):
    """Left / right partners next to each other."""
    base = re.sub(r"^(Left|Right)", "", name)
    side = 0 if name.startswith("Left") else 1 if name.startswith("Right") else 2
    return base.lower(), side


def write_collage(folder, out=None, bundles=None, floor=1e-3, cols=3, pad=3, tile=2.2, dpi=150):
    """PNG collage of the bundles under ``folder`` (default: all); returns its path.

    out      default: <folder>/<subject>_desc-mcat_collage.png
    floor    faintest density shown, as a fraction of each bundle's peak
    cols     bundles per row; pad: voxels of margin; tile: projection height (inches)"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    files = bundle_files(folder)
    names = sorted(bundles or files, key=_sort_key)
    if not names:
        raise ValueError(f"no *_bundle-*_desc-density_map.nii.gz under {folder}")
    ref = nib.as_closest_canonical(nib.load(files[names[0]]["density"]))
    shape, aff = ref.shape[:3], ref.affine

    def on_ref(path, order=0):
        img = nib.as_closest_canonical(nib.load(path))
        if img.shape[:3] != tuple(shape) or not np.allclose(img.affine, aff, atol=1e-3):
            img = resample_from_to(img, (shape, aff), order=order)
        return np.asarray(img.dataobj, float)

    dseg = sorted(glob.glob(op.join(folder, "**", "*_dseg.nii.gz"), recursive=True),
                  key=lambda x: "_desc-mcat_dseg" not in x)
    brain = on_ref(dseg[0]) > 0 if dseg else None
    dens = []
    for name in names:
        d = np.clip(on_ref(files[name]["density"], order=1), 0, None)
        if d.max() > 0:
            dens.append((name, d / d.max()))
    if not dens:
        raise ValueError("every density map is empty")

    # one crop for every tile: all that any bundle shows, plus a margin
    keep = np.zeros(shape, bool)
    for _, d in dens:
        keep |= d >= floor
    ijk = np.argwhere(keep)
    lo, hi = np.maximum(ijk.min(0) - pad, 0), np.minimum(ijk.max(0) + pad + 1, shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    vs = np.abs(nib.affines.voxel_sizes(aff))
    ext_mm = (hi - lo) * vs
    widths, heights = [ext_mm[1], ext_mm[0], ext_mm[0]], [ext_mm[2], ext_mm[2], ext_mm[1]]
    cols = min(cols, len(dens))
    rows = int(np.ceil(len(dens) / cols))
    tile_w = sum(w / max(heights) * tile for w in widths)
    fig = plt.figure(figsize=(cols * tile_w + 0.6, rows * (tile + 0.45) + 0.7), facecolor="black")
    gs = fig.add_gridspec(rows, cols * 3, width_ratios=widths * cols, wspace=0.02, hspace=0.18)
    norm = LogNorm(floor, 1.0)
    edge_lo, edge_hi = (lo - 0.5) * vs, (hi - 0.5) * vs
    im = None
    for k, (name, d) in enumerate(dens):
        r, c = divmod(k, cols)
        for v, (_, axis) in enumerate(VIEWS):
            ax = fig.add_subplot(gs[r, c * 3 + v])
            ax.set_facecolor("black")
            o = [i for i in range(3) if i != axis]
            ext = [edge_lo[o[0]], edge_hi[o[0]], edge_lo[o[1]], edge_hi[o[1]]]
            if brain is not None:
                ax.imshow(brain[sl].max(axis).T, cmap="gray", vmin=0, vmax=6, origin="lower", extent=ext,
                          interpolation="nearest")
            im = ax.imshow(np.ma.masked_less(d[sl].max(axis).T, floor), cmap="magma", norm=norm,
                           origin="lower", extent=ext, interpolation="nearest")
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_aspect("equal")
            ax.set_axis_off()
            if v == 1:
                ax.set_title(name, color="white", fontsize=10)
    cb = fig.colorbar(im, cax=fig.add_axes([0.35, 0.02, 0.3, 0.012]), orientation="horizontal")
    cb.set_label("density / own peak (log) · sagittal | coronal | axial MIP", color="white", fontsize=9)
    cb.ax.tick_params(colors="white", labelsize=8)
    if out is None:
        subject = op.basename(files[names[0]]["prefix"]).split("_bundle-")[0]
        out = op.join(folder, f"{subject}_desc-mcat_collage.png")
    fig.savefig(out, dpi=dpi, facecolor="black", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out
