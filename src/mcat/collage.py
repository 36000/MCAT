"""MIP collage of a run's bundles: one panel per bundle, in the one view that shows it best
(coronal for the corticospinal tract, sagittal for the arcuate, axial for the callosum, ...),
with left / right partners sharing a panel where they do not overlap (axial and coronal
views).  Each panel is the maximum-intensity projection of the density (log scale, each bundle
divided by its own peak) over a T1 slice through the bundle; every panel has the same mm scale.
Written by ``mcat.run``."""
from __future__ import annotations

import glob
import os.path as op
import re

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from .core import bundle_files

AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}
ORDER = ("axial", "coronal", "sagittal")
# the view each known bundle is shown in (matched against its name without Left / Right,
# lower case, letters only); anything else gets the view it is thinnest across
VIEW_RULES = (
    (r"corticospinal|corticopontine|corticobulbar|^cst$|verticaloccipital|^vof$", "coronal"),
    (r"forceps|callosum|commissur|^cc|^fmajor$|^fminor$|anteriorthalamic|^atr$|thalamicradiation"
     r"|opticradiation|^or$|retinogeniculate|rgvp|eye", "axial"),
    (r"arcuate|^arc$|^parc$|longitudinal|^slf|^ilf$|^mdlf$|uncinate|^unc$|cingulum|^cgc$|^cgh$|^cing"
     r"|frontooccipital|^ifo|parahippocampal|aslant|^fat$|fornix", "sagittal"),
)

# bundles that share a panel (each pattern one kind of bundle, both sides; matched like
# VIEW_RULES), when they are all shown in the same view; a last element naming a view puts
# them in it.  In a sagittal view the two sides would overlap, so each side gets its own panel.
GROUPS = (
    (r"anteriorthalamic|^atr$", r"opticradiation|^or$"),
    (r"forcepsmajor|^fmajor$|^fma$", r"forcepsminor|^fminor$|^fmi$"),
    (r"callosumoccipital", r"callosumorbital"),
    (r"callosumanteriorfrontal", r"callosumposteriorparietal"),
    (r"callosumsuperiorparietal", r"callosumsuperiorfrontal"),
    (r"callosummotor", r"callosumtemporal", "coronal"),
    (r"uncinate|^unc$", r"posteriorarcuate|^parc$", "sagittal"),
    (r"fornix", r"cingulumhippocampus|^cgh$|^hcc$", "sagittal"),
)


def _key(name):
    return re.sub(r"[^a-z]", "", split_side(name)[0].lower())


def _title(name):
    return re.sub(r"(?<=[a-z])(?=[A-Z][a-z])", " ", name.strip("_ "))


def split_side(name):
    """(base, side, partner name) for "LeftX" / "RightX" and "X_L" / "XL" style names
    (side "L" / "R"); (name, None, None) for the rest."""
    m = re.fullmatch(r"(Left|Right)([A-Z].*)", name)
    if m:
        other = "Right" if m.group(1) == "Left" else "Left"
        return m.group(2), m.group(1)[0], other + m.group(2)
    m = re.fullmatch(r"(.*?[A-Z0-9])(_?)(L|R)", name)
    if m:
        return m.group(1), m.group(3), m.group(1) + m.group(2) + ("R" if m.group(3) == "L" else "L")
    return name, None, None


def _sort_key(name):
    """Left / right partners next to each other."""
    base, side, _ = split_side(name)
    return base.lower(), {"L": 0, "R": 1}.get(side, 2)


def view_of(name, density=None, aff=None):
    """The view a bundle is shown in: by name (VIEW_RULES), else the canonical axis the
    density is thinnest across (density-weighted spread in mm)."""
    key = re.sub(r"[^a-z]", "", split_side(name)[0].lower())
    for pattern, view in VIEW_RULES:
        if re.search(pattern, key):
            return view
    if density is None:
        return "sagittal"
    ijk = np.argwhere(density > 0)
    w = density[tuple(ijk.T)]
    xyz = nib.affines.apply_affine(aff, ijk)
    sd = np.sqrt(np.average((xyz - np.average(xyz, axis=0, weights=w)) ** 2, axis=0, weights=w))
    return {0: "sagittal", 1: "coronal", 2: "axial"}[int(np.argmin(sd))]


def _find_t1(folder):
    """pyAFQ's masked T1 next to the run (the run folder sits in pyAFQ's output folder)."""
    for d in (folder, op.dirname(op.abspath(folder).rstrip("/"))):
        cand = sorted(glob.glob(op.join(d, "*_desc-masked_T1w.nii.gz")) + glob.glob(op.join(d, "*_T1w.nii.gz")),
                      key=lambda x: "_desc-masked_T1w" not in x)
        if cand:
            return cand[0]
    return None


def write_collage(folder, out=None, bundles=None, floor=1e-3, cols=4, pad=0, tile=2.4, dpi=200, t1=None,
                  views=None, pair=True, groups=GROUPS, mask=True, fontsize=13, brain_mask=None, titles=None):
    """PNG collage of the bundles under ``folder`` (default: all); returns its path.

    out      default: <folder>/<subject>_desc-mcat_collage.png
    floor    faintest density shown, as a fraction of each bundle's peak
    t1       background: a T1 slice through each panel's bundle(s), at the density-weighted
             centre along the projection (default: pyAFQ's masked T1 next to the run; without
             one, the grey silhouette of the run's tissue labels)
    mask     black out the background outside the brain, so an unmasked T1 / T2 shows no skull
    brain_mask  the brain (default: pyAFQ's ``*_desc-brain_mask`` next to the run, else the
             run's tissue labels); each view is cropped to what its panels draw of it
    fontsize titles and colour bar text (points)
    views    {bundle: "axial" | "coronal" | "sagittal"} overriding the automatic choice
    pair     left / right partners ("LeftX" / "RightX", "X_L" / "X_R") share one panel when
             their view is axial or coronal (in a sagittal view they would overlap)
    groups   kinds of bundle that share a panel, both sides (default GROUPS: ATR with the optic
             radiation, the forceps, callosal segments in pairs, uncinate with posterior
             arcuate): tuples of name patterns (regular expressions on the lower-case name
             without its side), optionally ending in the view they share; in a sagittal view
             each side gets its own panel; () for none
    titles   {lower-case name without its side, letters only: title}, e.g.
             {"superiorlongitudinaliii": "SLF"}
    cols     panels per row: each view fills rows of its own, and rows with room left are
             merged while they hold no more than ``cols`` panels together; every column is
             as wide as the widest panel, each panel centred in its cell, titles written on
             them (white, outlined in black, wrapped to the panel); pad: voxels of margin
             around the brain; tile: inches for the brain's largest extent"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    files = bundle_files(folder)
    names = sorted(bundles or files, key=_sort_key)
    names = [n for n in names if n in files]
    if not names:
        raise ValueError(f"no *_bundle-*_desc-density_map.nii.gz under {folder}")
    ref = nib.as_closest_canonical(nib.load(files[names[0]]["density"]))
    shape, aff = ref.shape[:3], ref.affine

    def on_ref(path, order=0):
        img = nib.as_closest_canonical(nib.load(path))
        if img.shape[:3] != tuple(shape) or not np.allclose(img.affine, aff, atol=1e-3):
            img = resample_from_to(img, (shape, aff), order=order)
        d = np.asarray(img.dataobj, float)
        return d[..., 0] if d.ndim == 4 else d

    # the brain: ``brain_mask``, else pyAFQ's brain mask next to the run, else the run's tissue
    # labels (non-zero inside that mask); without any, the T1 above 5% of its bright end (the
    # largest piece, holes filled)
    from scipy import ndimage as ndi

    up = op.dirname(op.abspath(folder).rstrip("/"))
    dseg = sorted(glob.glob(op.join(folder, "**", "*_desc-mcat_dseg.nii.gz"), recursive=True))
    bmask = sorted(glob.glob(op.join(folder, "*_desc-brain_mask.nii.gz")) + glob.glob(op.join(up, "*_desc-brain_mask.nii.gz")))
    src = brain_mask or (bmask[0] if bmask else dseg[0] if dseg else None)
    brain = on_ref(src) > 0 if src is not None else None
    t1 = t1 or _find_t1(folder)
    back, is_t1 = None, False
    if t1 is not None:
        back, is_t1 = on_ref(t1, order=1), True
        if brain is None and (back > 0).any():
            fg = back > 0.05 * np.percentile(back[back > 0], 99)
            lab, n = ndi.label(fg)
            if n:
                brain = ndi.binary_fill_holes(lab == 1 + int(np.argmax(np.bincount(lab.ravel())[1:])))
        if mask and brain is not None:
            back = np.where(brain, back, 0.0)
    elif brain is not None:
        back = brain.astype(float)
    dens = {}
    for name in names:
        d = np.clip(np.nan_to_num(on_ref(files[name]["density"], order=1)), 0, None)
        if d.max() > 0:
            dens[name] = d / d.max()
    if not dens:
        raise ValueError("every density map is empty")

    # panels: one per bundle in its view; Left / Right partners together in axial / coronal
    views = dict(views or {})
    titles = {re.sub(r"[^a-z]", "", k.lower()): v for k, v in (titles or {}).items()}

    def title_of(base):
        return titles.get(re.sub(r"[^a-z]", "", base.lower()), _title(base))

    panels, used = [], set()
    for group in groups or ():
        pats, forced = (group[:-1], group[-1]) if group[-1] in ORDER else (group, None)
        kinds = [[n for n in dens if n not in used and re.search(pt, _key(n))] for pt in pats]
        if sum(bool(k) for k in kinds) < 2:
            continue                                     # fewer than two of the kinds found
        members = [n for k in kinds for n in k]
        vs_ = {forced} if forced else {views.get(m) or view_of(m, dens[m], aff) for m in members}
        if len(vs_) > 1:                                 # shown in different views
            continue
        view = vs_.pop()
        by_side = ([[m for m in members if split_side(m)[1] == sd] for sd in ("L", "R", None)]
                   if view == "sagittal" else [members])  # sagittal: the sides would overlap
        for mem in by_side:
            if len({next(i for i, k in enumerate(kinds) if m in k) for m in mem}) < 2:
                continue                                 # this side lacks one of the kinds
            bases = list(dict.fromkeys(title_of(split_side(m)[0]) for m in mem))
            first = {b.split(" ", 1)[0] for b in bases}
            if len(first) == 1 and all(" " in b for b in bases):   # "Callosum A + Callosum B" -> "Callosum A + B"
                bases = [bases[0]] + [b.split(" ", 1)[1] for b in bases[1:]]
            sides = {split_side(m)[1] for m in mem} - {None}
            suffix = " (L + R)" if sides == {"L", "R"} else f" ({sides.pop()})" if sides else ""
            panels.append(dict(view=view, members=mem, title=" + ".join(bases) + suffix))
            used.update(mem)
    for n in dens:
        if n in used:
            continue
        base, side, partner = split_side(n)
        members = [n]
        if pair and side and partner in dens and partner not in used:
            members = [n, partner] if side == "L" else [partner, n]
        both = np.max([dens[m] for m in members], axis=0)
        v = next((views[m] for m in members if m in views), None) or view_of(n, both, aff)
        if len(members) == 2 and v == "sagittal":       # left and right overlap in a sagittal view
            members = [n]
            v = views.get(n) or view_of(n, dens[n], aff)
        used.update(members)
        if len(members) == 2:
            title = title_of(base) + " (L + R)"
        elif side and re.sub(r"[^a-z]", "", base.lower()) in titles:
            title = {"L": "Left ", "R": "Right "}[side] + title_of(base)
        else:
            title = title_of(n)
        panels.append(dict(view=v, members=members, title=title))
    panels.sort(key=lambda p: ORDER.index(p["view"]))

    # crop: first to the brain (not the whole T1: its field of view, skull and noise go) and
    # every bundle shown; then each view to what its panels actually draw (the T1 slices,
    # masked to the brain, and the projections), plus ``pad`` voxels
    keep = brain.copy() if brain is not None else np.zeros(shape, bool)
    for d in dens.values():
        keep |= d >= floor
    ijk = np.argwhere(keep)
    lo, hi = ijk.min(0), ijk.max(0) + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    vs = np.abs(nib.affines.voxel_sizes(aff))
    if back is not None and is_t1:
        b = back[sl]
        vlo, vhi = np.percentile(b[b > 0], [1, 99.5]) if (b > 0).any() else (0, 1)
    o = {v: [i for i in range(3) if i != AXIS[v]] for v in AXIS}
    for p in panels:
        axis = AXIS[p["view"]]
        d = np.max([dens[m] for m in p["members"]], axis=0)[sl]
        p["mip"] = d.max(axis).T
        p["grey"] = np.zeros(p["mip"].shape)
        if back is not None:
            if is_t1:
                # the T1 slice at the density-weighted centre of the bundle(s) along the projection
                prof = d.sum(tuple(i for i in range(3) if i != axis))
                k = int(round(np.average(np.arange(len(prof)), weights=prof))) if prof.sum() > 0 \
                    else d.shape[axis] // 2
                p["grey"] = np.clip((np.take(back[sl], k, axis=axis) - vlo) / max(vhi - vlo, 1e-9), 0, 1).T ** 0.9
            else:
                p["grey"] = back[sl].max(axis).T
    box = {}
    for v in AXIS:
        ps = [p for p in panels if p["view"] == v]
        if not ps:
            continue
        drawn = np.any([(p["grey"] > 0) | (p["mip"] >= floor) for p in ps], axis=0)
        rc = np.argwhere(drawn) if drawn.any() else np.argwhere(np.ones_like(drawn))
        r0, c0 = np.maximum(rc.min(0) - pad, 0)
        r1, c1 = np.minimum(rc.max(0) + pad + 1, drawn.shape)
        # the cropped image's extent in mm (voxel edges): columns run along o[v][0], rows along o[v][1]
        a0, a1 = o[v]
        box[v] = dict(rows=slice(r0, r1), cols=slice(c0, c1),
                      ext=[(lo[a0] + c0 - 0.5) * vs[a0], (lo[a0] + c1 - 0.5) * vs[a0],
                           (lo[a1] + r0 - 0.5) * vs[a1], (lo[a1] + r1 - 0.5) * vs[a1]])

    # layout: rows of up to ``cols`` panels of one view, every panel at the same mm scale, no
    # space between panels; rows with room left are merged (first fit) while they hold no more
    # than ``cols`` panels together; titles are written on the panels
    import textwrap

    from matplotlib import patheffects as pe

    s = tile / max(max(e[1] - e[0], e[3] - e[2]) for e in (bx["ext"] for bx in box.values()))   # inches per mm
    bottom = 0.25 + 0.035 * fontsize                     # the colour bar
    rows = []
    for v in ORDER:
        ps = [p for p in panels if p["view"] == v]
        rows += [ps[i:i + cols] for i in range(0, len(ps), cols)]
    merged = []
    for r in rows:
        room = next((m for m in merged if len(m) + len(r) <= cols), None) if len(r) < cols else None
        if room is None:
            merged.append(list(r))
        else:
            room.extend(r)
    rows = merged
    size = {v: ((bx["ext"][1] - bx["ext"][0]) * s, (bx["ext"][3] - bx["ext"][2]) * s) for v, bx in box.items()}
    cw = max(size[p["view"]][0] for p in panels)      # every column as wide as the widest panel
    W = cols * cw if len(rows) > 1 else sum(cw for _ in rows[0])
    H = sum(max(size[p["view"]][1] for p in r) for r in rows) + bottom
    fig = plt.figure(figsize=(W, H), facecolor="black")
    norm = LogNorm(floor, 1.0)
    cmap = plt.get_cmap("magma")
    fs = fontsize
    y_top = H
    im = None
    for r in rows:
        rh = max(size[p["view"]][1] for p in r)
        for c, p in enumerate(r):
            v = p["view"]
            w, h = size[v]
            # centred in its column and row
            x, y = c * cw + (cw - w) / 2, y_top - rh + (rh - h) / 2
            ax = fig.add_axes([x / W, y / H, w / W, h / H])
            ax.set_facecolor("black")
            bx = box[v]
            e = bx["ext"]
            mip, grey = p["mip"][bx["rows"], bx["cols"]], p["grey"][bx["rows"], bx["cols"]]
            if back is not None:
                if is_t1:
                    ax.imshow(grey, cmap="gray", vmin=0, vmax=1.05, origin="lower", extent=e,
                              interpolation="bilinear")
                else:
                    ax.imshow(grey, cmap="gray", vmin=0, vmax=6, origin="lower", extent=e,
                              interpolation="nearest")
            im = ax.imshow(np.ma.masked_less(mip, floor), cmap=cmap, norm=norm, origin="lower",
                           extent=e, interpolation="bilinear" if is_t1 else "nearest")
            ax.set_xlim(e[0], e[1])
            ax.set_ylim(e[2], e[3])
            ax.set_aspect("equal")
            ax.set_axis_off()
            # the title on the image, wrapped to the panel: white with a black outline
            per_line = max(8, int(w * 72 / (fs * 0.58)) - 1)
            text = textwrap.fill(f"{p['title']} \u00b7 {v}", per_line)
            ax.text(0.03, 0.97, text, transform=ax.transAxes, color="white", fontsize=fs, ha="left", va="top",
                    linespacing=1.15, path_effects=[pe.withStroke(linewidth=2.2, foreground="black")], zorder=5)
        y_top -= rh
    cax = fig.add_axes([0.04, (bottom - 0.2) / H, 0.3, 0.12 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(colors="white", labelsize=0.85 * fs)
    bg = "a T1 slice through the bundle" if is_t1 else "the brain's silhouette"
    label = f"density / own peak (log): maximum-intensity projection over {bg}"
    room = W * (1 - 0.04 - 0.3) - 0.1 * W * 0.3 - 0.1          # inches right of the bar
    label = textwrap.fill(label, max(10, int(room * 72 / (fs * 0.55))))
    cax.text(1.08, 0.5, label, transform=cax.transAxes, color="white", fontsize=fs, ha="left", va="center",
             linespacing=1.15)
    if out is None:
        subject = op.basename(files[names[0]]["prefix"]).split("_bundle-")[0]
        out = op.join(folder, f"{subject}_desc-mcat_collage.png")
    fig.savefig(out, dpi=dpi, facecolor="black")
    plt.close(fig)
    return out
