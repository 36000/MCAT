"""Bundle densities and cores in NiiVue: a Jupyter widget (ipyniivue) or a standalone page.

In a notebook (JupyterLab, Notebook 7, VS Code, Voila)::

    from mpat import view
    view("derivatives/afq/sub-01/mpat", t1="sub-01_desc-masked_T1w.nii.gz",
         bundles=["LeftArcuate", "LeftCorticospinal"])

or as a web page to open in any browser or share: ``mpat.viewer.save_html("bundles.html",
folder, ...)`` with the same options.

Each bundle's density is shown on a log scale, from ``floor`` x its peak (faint) to its peak
(bright), with its own colour map; its core (written by ``mpat.run``, computed here if
missing) is drawn as a tube in the matching colour.  ``glyphs=True`` adds the bundles'
visitation ODFs -- the directions their walkers travel in each voxel, drawn like fODF
glyphs -- on three planes through each core's midpoint.  Slices and 3-D render are both
live: click to move the slices, drag in the render to rotate, scroll to zoom.
"""
from __future__ import annotations

import glob
import gzip
import logging
import os.path as op
import struct

import nibabel as nib
import numpy as np

from .core import bundle_core, bundle_files, write_core

log = logging.getLogger("mpat")

# (NiiVue colour map for the density, matching RGBA for the core tube)
PALETTE = [("warm", (255, 196, 40)), ("cool", (60, 230, 255)), ("green", (80, 255, 100)),
           ("violet", (210, 110, 255)), ("redyell", (255, 90, 40)), ("blue2cyan", (60, 140, 255)),
           ("gold", (255, 230, 120)), ("bluegrn", (60, 255, 200)), ("hot", (255, 60, 60)),
           ("electric_blue", (120, 180, 255))]


def _nifti_bytes(arr, affine, dtype=np.uint8):
    img = nib.Nifti1Image(np.asarray(arr, dtype), affine)
    img.header.set_xyzt_units("mm")
    return gzip.compress(img.to_bytes(), compresslevel=6)


DENSITY_RANGE = 1e-6   # density maps are encoded down to this x the peak (the slider's range)
GLYPH_RANGE = 1e-3     # glyph voxels are kept down to this x the peak (mpat.afq.ODF_FLOOR)


def _u8_level(frac, enc_floor):
    """The uint8 level of log_density_u8(..., enc_floor) at frac x the peak."""
    return float(1 + 254 * (np.log10(frac) - np.log10(enc_floor)) / -np.log10(enc_floor))


def _bbox_mm(mask, affine):
    """The 8 corners (mm) of a mask's voxel bounding box, or None if it is empty."""
    ijk = np.argwhere(mask)
    if not len(ijk):
        return None
    lo, hi = ijk.min(0), ijk.max(0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return nib.affines.apply_affine(affine, corners)


def _crop_index(shape, affine, box):
    """(lo, hi) voxel indices of a grid covering the world box (mm); the whole grid if None."""
    shape = np.array(shape[:3])
    if box is None:
        return np.zeros(3, int), shape
    corners = np.array([[x, y, z] for x in box[:, 0] for y in box[:, 1] for z in box[:, 2]])
    ijk = nib.affines.apply_affine(np.linalg.inv(affine), corners)
    lo = np.clip(np.floor(ijk.min(0)), 0, shape - 1).astype(int)
    hi = np.clip(np.ceil(ijk.max(0)) + 1, lo + 1, shape).astype(int)
    return lo, hi


def _shifted(affine, lo):
    """The affine of a grid cropped to start at voxel lo."""
    A = np.array(affine, float)
    A[:3, 3] = nib.affines.apply_affine(affine, lo)
    return A


def log_density_u8(density, floor=1e-3):
    """density -> uint8: 0 below floor x peak, 1..255 for log10(density / peak) in [log10 floor, 0]."""
    d = np.clip(np.nan_to_num(np.asarray(density, float)), 0, None)
    out = np.zeros(d.shape, np.uint8)
    if d.max() > 0:
        r = d / d.max()
        m = r >= floor
        out[m] = np.clip(np.rint(1 + 254 * (np.log10(r[m]) - np.log10(floor)) / -np.log10(floor)), 1, 255)
    return out


def scene(folder, bundles=None, t1=None, floor=1e-3, show_density=True, density_opacity=1.0, crop=True,
          core_radius=1.2, glyphs=False, glyph_planes=None, glyph_floor=1e-2, glyph_range=GLYPH_RANGE,
          glyph_color=None, isosurface=True,
          glyph_norm=True, max_glyphs=4000, fodf=None, write_missing_cores=True, rois=False,
          roi_opacity=1.0):
    """What ``view`` and ``save_html`` draw, as data: volumes and meshes (file bytes plus
    display settings, each with its role), glyph sets (voxels and SH coefficients, meshed
    per slice by ``glyph_mesh``), the bundles, the point the slices start at (``focus``, mm)
    and the 2-D mesh slab.

    bundles          names to show (default: all)
    t1               background image (default: the run's ``*_desc-mpat_dseg``)
    floor            faintest density shown, as a fraction of each bundle's peak
    show_density     overlay the density maps (False: cores only, on the background)
    density_opacity  of the density over the background (1: opaque; below ``floor`` the
                     background shows)
    crop             crop every volume to the content box (the background's non-empty voxels
                     and every bundle above ``floor``, plus 2 voxels), so views fill their space
    core_radius      tube radius in mm
    glyphs           visitation-ODF glyphs: True for every bundle shown, or bundle names
    glyph_planes     glyphs follow the slices: every voxel of the three slices through the
                     crosshair that reaches glyph_floor, redrawn as the slices move.  None:
                     start at the first core's midpoint; (x, y, z) in mm: start there;
                     "all": every voxel at once, fixed (heavy; 2-D views still show one slice)
    glyph_floor      voxels whose density is below this x the bundle's peak get no glyph
    glyph_range      voxels down to this x the peak are kept, so a viewer can lower glyph_floor
                     live (default GLYPH_RANGE, 1e-3; raise it for smaller pages)
    glyph_color      "direction" (red left-right, green front-back, blue up-down) or "bundle"
                     (its core's colour); default: direction for one bundle, else bundle
    glyph_norm       each glyph scaled to its own maximum, as fODF glyphs usually are (False:
                     size proportional to the visits, the bundle's largest filling its voxel).
                     Glyphs read best zoomed in on a slice, with show_density=False.
    max_glyphs       per bundle with glyph_planes="all"; the densest voxels are kept
    isosurface       draw each bundle's density as an opaque surface at ``floor`` x its peak, in
                     its colour; the 3-D view's only picture of the density (the standalone page
                     redraws it at its slider's level)
    fodf             an SH image on the same grid (e.g. pyAFQ's ``*_model-csd_param-wm_dwimap``):
                     grey glyphs at the same voxels, a little smaller, for comparison
    write_missing_cores  compute cores that are not on disk yet and save them next to the maps
    rois             show each bundle's include ROIs (green) and exclude ROIs (red) as surfaces,
                     from the ``_desc-include_mask`` / ``_desc-exclude_mask`` files mpat.run writes
    roi_opacity      of those surfaces (1: opaque)
    """
    import json

    files = bundle_files(folder)
    if bundles:
        missing = [b for b in bundles if b not in files]
        if missing:
            raise ValueError(f"not found under {folder}: {missing}; available: {sorted(files)}")
        files = {b: files[b] for b in bundles}
    if not files:
        raise ValueError(f"no *_bundle-*_desc-density_map.nii.gz under {folder}")
    if t1 is None:  # the run's tissue labels (0 outside, 1 WM, 2-4 GM / CSF / edge)
        cand = sorted(glob.glob(op.join(folder, "**", "*_dseg.nii.gz"), recursive=True),
                      key=lambda x: "_desc-mpat_dseg" not in x)
        t1 = cand[0] if cand else None
    S = dict(volumes=[], meshes=[], glyphs=[], bundles=[], focus=None, slab=None, isosurface=bool(isosurface),
             subject=op.basename(next(iter(files.values()))["prefix"]).split("_bundle-")[0])
    dens = {name: nib.load(f["density"]) for name, f in files.items()}
    dens = {name: (img, np.asarray(img.dataobj, np.float32)) for name, img in dens.items()}
    ref, _ = next(iter(dens.values()))
    back = None
    if t1 is not None:
        bimg = nib.load(t1)
        bdat = np.asarray(bimg.dataobj)
        back = (bimg, bdat[..., 0] if bdat.ndim == 4 else bdat)
    # the content box (mm), and each grid's crop to it
    pts = [_bbox_mm(d >= floor * d.max(), img.affine) for img, d in dens.values() if d.max() > 0]
    if back is not None:
        pts.append(_bbox_mm(back[1] > 0, back[0].affine))
    pts = [p for p in pts if p is not None]
    box = None
    if crop and pts:
        pts = np.concatenate(pts)
        margin = 2.0 * float(max(ref.header.get_zooms()[:3]))
        box = np.array([pts.min(0) - margin, pts.max(0) + margin])
    lo_d, hi_d = _crop_index(ref.shape[:3], ref.affine, box)
    cut = tuple(slice(a, b) for a, b in zip(lo_d, hi_d))
    A = _shifted(ref.affine, lo_d)            # the bundles' grid, cropped
    if back is not None:
        dseg = "dseg" in op.basename(str(t1))
        lo_b, hi_b = _crop_index(back[1].shape, back[0].affine, box)
        bc = back[1][tuple(slice(a, b) for a, b in zip(lo_b, hi_b))]
        bc_img = nib.Nifti1Image(np.asarray(bc, np.uint8 if dseg else np.float32), _shifted(back[0].affine, lo_b))
        bc_img.header.set_xyzt_units("mm")
        S["volumes"].append(dict(role="background", name=op.basename(str(t1)).split(".nii")[0] + ".nii.gz",
                                 data=gzip.compress(bc_img.to_bytes(), compresslevel=6),
                                 colormap="gray", opacity=1.0, cal_min=0 if dseg else None,
                                 cal_max=4 if dseg else None))
    names = list(files) if glyphs is True else ([glyphs] if isinstance(glyphs, str) else list(glyphs or []))
    if glyph_color is None:
        glyph_color = "direction" if len(names) == 1 else "bundle"
    S["glyph_color"] = glyph_color
    every = []
    for i, (name, f) in enumerate(files.items()):
        img, dfull = dens[name]
        d = dfull[cut]                        # cropped; on the grid A
        cmap, rgb = PALETTE[i % len(PALETTE)]
        meta = dict(name=name, label=name, rgb=list(rgb), colormap=cmap)
        if op.exists(f["prefix"] + "_desc-summary.json"):
            with open(f["prefix"] + "_desc-summary.json") as fh:
                info = json.load(fh)
            meta.update(label=info.get("bundle", name),
                        expected_streamlines=info.get("result", {}).get("expected_bundle_streamlines"))
        S["bundles"].append(meta)
        if show_density or not S["volumes"]:
            # encoded down to DENSITY_RANGE; cal_min shows it from `floor`, so a viewer can move it
            enc = min(floor, DENSITY_RANGE)
            S["volumes"].append(dict(role="density", bundle=name, name=f"{name}_logdensity.nii.gz",
                                     data=_nifti_bytes(log_density_u8(d, enc), A), colormap=cmap,
                                     cal_min=_u8_level(floor, enc), cal_max=255, floor=floor, enc_floor=enc,
                                     opacity=density_opacity if (S["volumes"] or not show_density) else 1.0))
        if isosurface and d.max() > 0:
            S["meshes"].append(dict(role="surface", bundle=name, name=f"{name}_surface.mz3", rgba255=[*rgb, 255],
                                    opacity=1.0, data=isosurface_mz3(d / d.max(), floor, A, rgb)))
        if rois:
            for role, rgb_roi in (("include", ROI_COLORS["include"]), ("exclude", ROI_COLORS["exclude"])):
                if f.get(role) is None:
                    continue
                m = np.asarray(nib.load(f[role]).dataobj)[cut] > 0
                if m.any():
                    S["meshes"].append(dict(role=role, bundle=name, name=f"{name}_{role}.mz3",
                                            rgba255=[*rgb_roi, 255], opacity=roi_opacity,
                                            data=isosurface_mz3(m.astype(np.float32), 0.5, A, rgb_roi)))
            if not (f.get("include") or f.get("exclude")):
                log.warning("%s: no include / exclude masks on disk (written by mpat.run since 2026-11-06)", name)
        # core: written by mpat.run; computed here if missing
        tck, xyz = f["prefix"] + "_desc-core.tck", None
        if op.exists(tck):
            xyz = nib.streamlines.load(tck).streamlines[0]
        elif f["endpoints"] is not None and dfull.max() > 0:
            load = lambda k: (np.asarray(nib.load(f[k]).dataobj) > 0) if f[k] else None  # noqa: E731
            try:
                c = bundle_core(dfull, img.affine, np.asarray(nib.load(f["endpoints"]).dataobj), load("start"),
                                load("end"))
            except ValueError as e:
                log.warning("%s: no core (%s)", name, e)
            else:
                if write_missing_cores:
                    write_core(c, f["prefix"])
                xyz = c.xyz
        if xyz is not None:
            # a triangulated tube: NiiVue's streamline (.tck) renderer draws nothing in some
            # WebGL builds, while surfaces always render
            S["meshes"].append(dict(role="core", bundle=name, name=f"{name}_core.obj",
                                    data=tube_obj(xyz, core_radius), rgba255=[*rgb, 255], opacity=1.0))
        if name not in names or d.max() <= 0:      # empty bundles: no glyphs
            continue
        if f["odf"] is None:
            log.warning("%s: no visitation ODF on disk (written by mpat.run since 2026-11-04)", name)
            continue
        static = isinstance(glyph_planes, str) and glyph_planes == "all"
        keep = glyph_floor if static else min(glyph_floor, glyph_range)
        ijk = np.argwhere(d >= keep * d.max())
        if static and len(ijk) > max_glyphs:
            log.warning("%s: %d glyph voxels, showing the densest %d", name, len(ijk), max_glyphs)
            ijk = ijk[np.argsort(d[tuple(ijk.T)])[::-1][:max_glyphs]]
        every.append(ijk)
        if S["focus"] is None:   # slices start here; meshes in 2-D: a one-voxel slab
            point = (np.asarray(glyph_planes, float) if glyph_planes is not None and not static else
                     xyz[len(xyz) // 2] if xyz is not None else
                     nib.affines.apply_affine(A, np.unravel_index(np.argmax(d), d.shape)))
            S["focus"] = [float(x) for x in point]
            S["slab"] = float(min(img.header.get_zooms()[:3]))
        coef = np.asarray(nib.load(f["odf"]).dataobj, np.float32)[tuple((ijk + lo_d).T)]
        S["glyphs"].append(dict(role="glyphs", bundle=name, name=f"{name}_glyphs.mz3", opacity=1.0, scale=1.0,
                                ijk=ijk.astype(np.int32), coef=coef, norm=glyph_norm, floor=glyph_floor,
                                rel=(d[tuple(ijk.T)] / d.max()).astype(np.float32),
                                rgb=None if glyph_color == "direction" else list(rgb)))
        S["glyph_floor"], S["glyph_range"] = glyph_floor, keep
    if fodf is not None and every:
        fimg = nib.load(fodf)
        if fimg.shape[:3] != ref.shape[:3] or not np.allclose(fimg.affine, ref.affine, atol=1e-3):
            raise ValueError(f"{fodf} is not on the bundles' grid {ref.shape[:3]}")
        ijk, inv = np.unique(np.concatenate(every), axis=0, return_inverse=True)
        rel = np.zeros(len(ijk), np.float32)     # shown where any bundle's glyph is
        np.maximum.at(rel, inv.ravel(), np.concatenate([g["rel"] for g in S["glyphs"]]))
        ijk = ijk.astype(np.int32)
        S["glyphs"].append(dict(role="fodf", name="fodf_glyphs.mz3", opacity=1.0, scale=0.85, ijk=ijk, norm=True,
                                floor=glyph_floor, rel=rel, rgb=[150, 150, 150],
                                coef=np.asarray(fimg.dataobj, np.float32)[tuple((ijk + lo_d).T)]))
    for g in S["glyphs"]:      # largest radius of the set, for glyph_norm=False
        g["vmax"] = float(max(_odf_max(g["coef"]), 1e-30))
    if S["glyphs"]:
        S["affine"] = np.asarray(A, float).tolist()
        S["shape"] = list(d.shape)
        if static:             # every glyph at once; the slab still shows one slice in 2-D
            S["meshes"] += [dict(role=g["role"], bundle=g.get("bundle"), name=g["name"], opacity=g["opacity"],
                                 rgba255=[255, 255, 255, 255], data=glyph_mesh(g, S["affine"]))
                            for g in S.pop("glyphs")]
            S["glyphs"] = []
    if S["focus"] is not None:   # as NiiVue's crosshair: fraction of the RAS-ordered background grid
        bg = nib.as_closest_canonical(bc_img if back is not None else nib.Nifti1Image(d, A))
        vox = nib.affines.apply_affine(np.linalg.inv(bg.affine), S["focus"])
        S["focus_frac"] = [float(x) for x in np.clip((vox + 0.5) / np.array(bg.shape[:3]), 0, 1)]
    return S


def view(folder, height=640, **kwargs):
    """ipyniivue ``NiiVue`` widget with the MPAT bundles under ``folder``: the three slices
    and a 3-D render in an equal 2 x 2 grid, cropped to the content and opaque throughout
    (the 3-D view: each bundle as a surface in a lit head cut open towards the camera).
    Double-click a view to enlarge it to the whole widget; double-click again for the grid.  ``kwargs``: ``scene``'s options (bundles, t1, floor, glyphs, fodf, ...),
    then any ``ipyniivue.NiiVue`` option (e.g. back_color=(1, 1, 1, 1))."""
    import asyncio
    import inspect

    from ipyniivue import Mesh, MultiplanarType, NiiVue, ShowRender, SliceType, Volume

    own = set(inspect.signature(scene).parameters) - {"folder"}
    S = scene(folder, **{k: v for k, v in kwargs.items() if k in own})
    nv_options = {k: v for k, v in kwargs.items() if k not in own}
    # the three slices and the 3-D render as an equal 2 x 2 grid, using all their space.
    # Nothing transparent: meshes are not seen through the volume (x-ray 0), and the 3-D view
    # leaves the density out (the bundles are opaque surfaces there) and cuts the head open,
    # lit, on the side facing the camera
    opts = dict(back_color=(0, 0, 0, 1), slice_type=SliceType.MULTIPLANAR,
                multiplanar_show_render=ShowRender.ALWAYS, multiplanar_layout=MultiplanarType.GRID,
                multiplanar_equal_size=True, hero_image_fraction=0.0, is_colorbar=False, mesh_xray=0.0,
                render_overlay_blend=0.0, clip_plane_color=(0.0, 0.0, 0.0, 0.0), tile_margin=0.0,
                multiplanar_pad_pixels=0, is_orient_cube=False, mesh_thickness_on_2d=S["slab"] or 4.0)
    opts.update(nv_options)
    nv = NiiVue(height=height, **opts)
    nv.load_volumes([Volume(data=v["data"], name=v["name"], colormap=v["colormap"], opacity=v["opacity"],
                            **{k: v[k] for k in ("cal_min", "cal_max") if v[k] is not None})
                     for v in S["volumes"]])
    fixed = [Mesh(data=m["data"], name=m["name"], rgba255=m["rgba255"], opacity=m["opacity"]) for m in S["meshes"]]
    shown = {"c": None}

    def follow(mm):   # glyphs on the three slices through the crosshair
        c = tuple(int(x) for x in np.clip(np.rint(nib.affines.apply_affine(np.linalg.inv(S["affine"]), mm)),
                                          0, np.array(S["shape"]) - 1))
        if c == shown["c"]:
            return
        shown["c"] = c
        meshes = [Mesh(data=data, name=g["name"], opacity=g["opacity"]) for g in S["glyphs"]
                  if (data := glyph_mesh(g, S["affine"], c)) is not None]
        nv.meshes = [*fixed, *meshes]

    def moved(d):     # debounced: while scrolling, only the slice you stop on is meshed
        shown["mm"] = d["mm"][:3]
        if shown.get("timer") is not None:
            shown["timer"].cancel()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            shown["timer"] = loop.call_later(0.3, lambda: follow(shown["mm"]))
        else:
            follow(shown["mm"])

    if S["glyphs"]:
        follow(S["focus"])
        nv.on_location_change(moved)
    elif fixed:
        nv.load_meshes(fixed)

    # double-click a view to have it fill the widget; double-click again for the grid.
    # ipyniivue reports mouse-ups and crosshair moves only: two quick mouse-ups in one place
    # are a double-click; a crosshair move just before says which slice, none means 3-D
    import time

    click = {"t": 0.0, "pos": None, "view": None, "moved": 0.0}
    views = {0: SliceType.AXIAL, 1: SliceType.CORONAL, 2: SliceType.SAGITTAL}

    def located(d):
        click["moved"], click["view"] = time.monotonic(), d.get("ax_cor_sag")

    def released(d):
        if d.get("is_dragging"):
            return
        now, pos = time.monotonic(), tuple(d.get("mouse_pos") or (0, 0))
        double = now - click["t"] < 0.45 and click["pos"] is not None and max(
            abs(a - b) for a, b in zip(pos, click["pos"])) < 8
        click["t"], click["pos"] = now, pos
        if not double:
            return
        click["t"] = 0.0
        if nv.opts.slice_type != SliceType.MULTIPLANAR:
            nv.set_slice_type(SliceType.MULTIPLANAR)
        elif now - click["moved"] < 0.6 and click["view"] in views:
            nv.set_slice_type(views[click["view"]])
        else:
            nv.set_slice_type(SliceType.RENDER)

    nv.on_location_change(located)
    nv.on_mouse_up(released)
    if S["focus"] is not None:
        nv.scene.crosshair_pos = S["focus_frac"]
    nv.set_clip_planes([[0.0, nv.scene.render_azimuth + 180, -nv.scene.render_elevation]])
    nv.on_canvas_attached(lambda *_: nv.set_volume_render_illumination(0.6))   # lit, solid-looking head
    nv.bundle_names = [b["name"] for b in S["bundles"]]   # overlay order after the background
    return nv


ROI_COLORS = {"include": (70, 200, 110), "exclude": (225, 60, 60)}
NIIVUE_CDN = "https://cdn.jsdelivr.net/npm/@niivue/niivue@0.65.0/dist/niivue.umd.js"


def save_html(path, folder, title=None, niivue_js=None, **kwargs):
    """A standalone web page with the same scene as ``view`` (``kwargs``: ``scene``'s
    options), with controls for the layout, each bundle's density / core / glyphs, and the
    opacities.  The data are embedded.  NiiVue loads from a CDN, or from ``niivue_js``, a
    local ``niivue.umd.js`` (npm package @niivue/niivue, dist/) embedded too, so the page
    works offline.  Returns the path."""
    import base64
    import json
    from importlib.resources import files as resources

    from dipy.data import get_sphere

    from .geometry import _sh_basis

    S = scene(folder, **kwargs)
    S["meshes"] = [m for m in S["meshes"] if m["role"] != "surface"]   # the page draws them live
    b64 = lambda a, t: base64.b64encode(np.ascontiguousarray(a, t).tobytes()).decode()  # noqa: E731
    for item in S["volumes"] + S["meshes"]:
        item["data"] = base64.b64encode(item["data"]).decode()
    if S["glyphs"]:   # the page meshes the slices through the crosshair itself, as view does
        sph = get_sphere(name="symmetric362")
        u = np.asarray(sph.vertices, float)
        A = np.asarray(S["affine"])[:3, :3]
        zooms = np.linalg.norm(A, axis=0)
        widths = sorted({g["coef"].shape[1] for g in S["glyphs"]})
        S["sphere"] = dict(n=len(u), w=b64(u @ (A / zooms).T, "<f4"), faces=b64(sph.faces, "<u4"),
                           radius=float(0.5 * zooms.min()), inv_affine=np.linalg.inv(S["affine"]).tolist(),
                           basis={str(k): b64(_sh_basis(u, k), "<f4") for k in widths})
        for g in S["glyphs"]:
            g["K"] = int(g["coef"].shape[1])
            g["ijk"], g["coef"], g["rel"] = b64(g["ijk"], "<i4"), b64(g["coef"], "<f4"), b64(g["rel"], "<f4")
    title = title or f"MPAT \u00b7 {S['subject']}"
    if niivue_js is None:
        lib = f'<script src="{NIIVUE_CDN}"></script>'
    else:
        with open(niivue_js) as fh:
            lib = "<script>" + fh.read().replace("</script", "<\\/script") + "</script>"
    page = (resources("mpat") / "viewer.html").read_text()
    page = (page.replace("__TITLE__", title.replace("<", "&lt;")).replace("__NIIVUE__", lib)
            .replace("__DATA__", json.dumps(S).replace("</", "<\\/")))
    with open(path, "w") as fh:
        fh.write(page)
    return path


def _odf_max(coef, sphere="symmetric362"):
    from dipy.data import get_sphere

    from .geometry import _sh_basis

    if not len(coef):
        return 0.0
    B = _sh_basis(np.asarray(get_sphere(name=sphere).vertices, float), coef.shape[1])
    return max(float(np.max(np.asarray(coef[s:s + 4096], float) @ B)) for s in range(0, len(coef), 4096))


def glyph_mesh(g, affine, c=None, floor=None):
    """MZ3 bytes of a glyph set (from ``scene``) on the three slices through voxel c (all
    voxels if None), where the density reaches ``floor`` x the peak (default: the set's
    glyph_floor), or None if there are none there."""
    on = np.ones(len(g["ijk"]), bool) if c is None else (g["ijk"] == np.asarray(c)).any(1)
    on &= g["rel"] >= (g["floor"] if floor is None else floor)
    if not on.any():
        return None
    return glyph_mz3(g["coef"][on], g["ijk"][on], affine, rgb=g["rgb"], norm=g["norm"], vmax=g["vmax"],
                     scale=g.get("scale", 1.0))


def glyph_mz3(coef, ijk, affine, rgb=None, norm=True, scale=1.0, sphere="symmetric362", vmax=None):
    """Gzipped MZ3 surface (NiiVue's native mesh format) of ODF glyphs, one per voxel.

    coef (n, ncoef) SH coefficients (DIPY descoteaux07, as pyAFQ's fODFs and MPAT's visitation
    ODFs); ijk (n, 3) voxel indices; affine voxel -> world (mm).  Radii are the ODF clipped
    at 0, scaled to each glyph's maximum (norm) or to the largest of all (or vmax), which reaches
    ``scale`` x half the smallest voxel side.  Colour: rgb, or by direction (|x|, |y|, |z|)."""
    from dipy.data import get_sphere

    from .geometry import _sh_basis

    s = get_sphere(name=sphere)
    u, faces = np.asarray(s.vertices, float), np.asarray(s.faces, np.int64)
    coef, ijk = np.asarray(coef, float).reshape(len(ijk), -1), np.asarray(ijk).reshape(-1, 3)
    r = np.maximum(coef @ _sh_basis(u, coef.shape[1]), 0) if len(ijk) else np.zeros((0, len(u)))
    m = r.max(1)
    keep = m > 0
    r, m, ijk = r[keep], m[keep], ijk[keep]
    r /= m[:, None] if norm else (vmax or max(m.max(initial=0), 1e-300))
    A = np.asarray(affine, float)[:3, :3]
    zooms = np.linalg.norm(A, axis=0)
    w = u @ (A / zooms).T                                  # glyph directions in world axes
    centre = nib.affines.apply_affine(affine, ijk)
    verts = (centre[:, None] + (scale * 0.5 * zooms.min()) * r[..., None] * w[None]).reshape(-1, 3)
    col = np.abs(w) / np.abs(w).max(1, keepdims=True) * 255 if rgb is None else np.tile(rgb, (len(u), 1))
    rgba = np.tile(np.c_[col, np.full(len(u), 255)].astype(np.uint8), (len(ijk), 1))
    # drop the triangles collapsed at the centre (where the ODF is ~0), then unused vertices
    g, fi = np.nonzero(r[:, faces].max(-1) > 0.02)
    tri = faces[fi] + len(u) * g[:, None]
    used = np.zeros(len(verts), bool)
    used[tri.ravel()] = True
    tri = (np.cumsum(used) - 1)[tri]
    verts, rgba = verts[used], rgba[used]
    return _mz3(tri, verts, rgba)


def _mz3(tri, verts, rgba):
    """Gzipped MZ3 (NiiVue's native mesh format): triangles, vertices, per-vertex RGBA."""
    head = struct.pack("<HHIII", 23117, 7, len(tri), len(verts), 0)   # faces | vertices | RGBA
    body = (np.asarray(tri).astype("<u4").tobytes() + np.asarray(verts).astype("<f4").tobytes()
            + np.asarray(rgba, np.uint8).tobytes())
    return gzip.compress(head + body, compresslevel=6)


def isosurface_mz3(values, level, affine, rgb):
    """Gzipped MZ3 of the surface where ``values`` crosses ``level`` (marching cubes, closed
    at the volume's edges), in world mm, one colour."""
    from skimage.measure import marching_cubes

    v = np.pad(np.asarray(values, np.float32), 1)
    verts, faces, _, _ = marching_cubes(v, level=level)
    verts = nib.affines.apply_affine(affine, verts - 1.0)
    return _mz3(faces, verts, np.tile([*rgb, 255], (len(verts), 1)))


def tube_obj(xyz, radius=1.0, sides=12):
    """Wavefront OBJ text of a closed tube (triangles) around a polyline -- renders as an
    ordinary surface in any NiiVue build, independent of its streamline renderer."""
    xyz = np.asarray(xyz, float)
    n = len(xyz)
    t = np.gradient(xyz, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    # rotation-minimising frame (double reflection), so the tube does not twist
    a = np.cross(t[0], [1.0, 0, 0] if abs(t[0][0]) < 0.9 else [0, 1.0, 0])
    U = np.empty((n, 3))
    U[0] = a / np.linalg.norm(a)
    for i in range(n - 1):
        v1 = xyz[i + 1] - xyz[i]
        c1 = v1 @ v1
        if c1 < 1e-20:
            U[i + 1] = U[i]
            continue
        rL = U[i] - (2 / c1) * (v1 @ U[i]) * v1
        tL = t[i] - (2 / c1) * (v1 @ t[i]) * v1
        v2 = t[i + 1] - tL
        c2 = v2 @ v2
        U[i + 1] = rL - (2 / c2) * (v2 @ rL) * v2 if c2 > 1e-20 else rL
    V = np.cross(t, U)
    ang = 2 * np.pi * np.arange(sides) / sides
    ring = xyz[:, None] + radius * (np.cos(ang)[None, :, None] * U[:, None] + np.sin(ang)[None, :, None] * V[:, None])
    pts = np.vstack([ring.reshape(-1, 3), xyz[0], xyz[-1]])
    faces = []
    for i in range(n - 1):
        for k in range(sides):
            a0, a1 = i * sides + k, i * sides + (k + 1) % sides
            b0, b1 = a0 + sides, a1 + sides
            faces += [(a0, b0, a1), (a1, b0, b1)]
    c0, c1 = n * sides, n * sides + 1
    for k in range(sides):
        faces.append((c0, (k + 1) % sides, k))
        faces.append((c1, (n - 1) * sides + k, (n - 1) * sides + (k + 1) % sides))
    lines = [f"v {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}" for p in pts]
    lines += [f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}" for f in faces]
    return ("\n".join(lines) + "\n").encode()
