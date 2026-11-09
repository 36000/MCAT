"""MCAT on top of pyAFQ.  Everything except the Markov chain comes from a pyAFQ
``ParticipantAFQ`` (or each participant of a ``GroupAFQ``): the fODF pyAFQ tracks on, the
tissue PVE and brain mask, the white/grey-matter interface (seeds), the bundle
definitions, and their ROIs warped to the subject.  Outputs go to ``<output_dir>/mcat``
(``mcat-moments`` with spatial moments), next to pyAFQ's own, or to ``out_dir`` (pyAFQ's
derivatives are then only read).
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from .chain import ABSORB_CLASSES, Chain, ChainConfig
from .geometry import visitation_odf
from .solve import Clock, overlength_states, prepare_launches, segment_bundles

log = logging.getLogger("mcat")
WM_THRESHOLD = 0.4   # pyAFQ's tracker stops where WM PVE <= gm_threshold (0.4)
ODF_FLOOR = 1e-3     # visitation ODFs are written where the density reaches this x its peak


def _on_grid(img, ref, order=1):
    """img on ref's grid: linear as pyAFQ's tracker does (dipy.align.resample), or nearest
    neighbour (order=0) for masks."""
    from dipy.align import resample
    from nibabel.processing import resample_from_to

    data = np.asarray(img.dataobj, np.float32)
    if img.shape[:3] == ref.shape[:3] and np.allclose(img.affine, ref.affine, atol=1e-3):
        return data
    if order == 0:
        return np.asarray(resample_from_to(nib.Nifti1Image(data, img.affine), (ref.shape[:3], ref.affine),
                                           order=0).dataobj, np.float32)
    vols = [data] if data.ndim == 3 else [data[..., i] for i in range(data.shape[-1])]
    out = [np.asarray(resample(v, np.zeros(ref.shape[:3], np.float32), moving_affine=img.affine,
                               static_affine=ref.affine).get_fdata(), np.float32) for v in vols]
    return out[0] if data.ndim == 3 else np.stack(out, -1)


HOLE_WARN = 0.01   # warn when this fraction of the fODF region is all-zero holes inside it


def tissue_labels(sh_img, pve_img, brain_mask_img=None):
    """As pyAFQ's tracker: walkers move where WM PVE > 0.4 inside the fODF mask; the mask's
    inner boundary counts as grey matter ("edge", so fibres leaving the field end validly);
    other in-mask voxels are GM (GM PVE >= CSF PVE), which absorbs walkers, or CSF, which
    walkers never enter (zero transition probability).  White matter that cannot reach any
    ending -- a pocket walled in by CSF and the edge of the fODF field (in a tumour, say) --
    would trap walkers seeded in it forever and stall the solves: it is taken out of the walk
    (treated as CSF, never entered), labelled 5 in dseg, and logged.
    Returns transient, absorbing class (0 outside, 1 GM, 2 CSF, 3 edge), dseg (1 WM, 2 GM,
    3 CSF, 4 edge, 5 isolated WM)."""
    from skimage.segmentation import find_boundaries

    mask = np.any(np.asarray(sh_img.dataobj) != 0, axis=-1)
    if brain_mask_img is not None:
        mask &= _on_grid(brain_mask_img, sh_img) > 0.5
    csf, gm, wm = np.moveaxis(_on_grid(pve_img, sh_img), -1, 0)
    # holes inside the fODF region (voxels whose fit is all zero) each get a rim of "edge",
    # where walkers end validly: a few are harmless, many cut every bundle short
    from scipy import ndimage as ndi

    holes = int((ndi.binary_fill_holes(mask) & ~mask).sum())
    if holes > HOLE_WARN * mask.sum():
        log.warning("%d voxels (%.1f%% of the fODF region) inside it have an all-zero fODF; each hole is "
                    "ringed by edge voxels where walkers end, so bundles may come out near zero.  Check "
                    "the fODF fit (a stale or partial model file, a failed fit, NaN / zero signal)",
                    holes, 100 * holes / mask.sum())
    elif holes:
        log.info("%d voxels inside the fODF region have an all-zero fODF", holes)
    edge = find_boundaries(mask, mode="inner") & mask
    transient = mask & ~edge & (wm > WM_THRESHOLD)
    absorb = np.zeros(mask.shape, np.int8)
    absorb[mask & ~transient & (gm >= csf)] = 1
    absorb[mask & ~transient & (gm < csf)] = 2
    absorb[edge] = 3
    # white matter with no way to an ending (walkers step to any of the 26 neighbours)
    cube = np.ones((3, 3, 3), bool)
    lab, n = ndi.label(transient, cube)
    trapped = np.zeros(mask.shape, bool)
    if n:
        ends = ndi.binary_dilation((absorb == 1) | (absorb == 3), cube)
        reach = np.zeros(n + 1, bool)
        reach[np.unique(lab[ends & transient])] = True
        trapped = transient & ~reach[lab]
        if trapped.any():
            sizes = np.bincount(lab[trapped])
            sizes = np.sort(sizes[sizes > 0])[::-1]
            log.info("isolated white matter: %d voxels in %d pockets with no way to grey matter or the "
                     "edge (largest %d voxels) are left out of the walk (dseg 5)", trapped.sum(),
                     len(sizes), sizes[0])
            transient = transient & ~trapped
            absorb[trapped] = 2
    dseg = np.where(transient, 1, np.where(absorb > 0, absorb + 1, 0)).astype(np.uint8)
    dseg[trapped] = 5
    return transient, absorb, dseg


def seal_26(mask):
    """Make a thin ROI impassable for 26-connected walks.  Streamlines cannot slip through a
    one-voxel sheet; walkers stepping diagonally can.  Adding the sheet's 6-neighbours closes
    those gaps."""
    from scipy import ndimage

    return ndimage.binary_dilation(mask, ndimage.generate_binary_structure(3, 1))


def _subject_thalamus(name, k, roi, thalamus, x, tol):
    """``roi``, or the subject's thalamus on its side if ``roi`` is the template's thalamus."""
    from scipy.ndimage import distance_transform_edt

    either = thalamus["left"] | thalamus["right"]
    if not either.any():
        log.warning("%s %s: SynthSeg found no thalamus on this grid; keeping the template's ROI", name, k)
        return roi
    near = distance_transform_edt(~either) <= max(tol, 1)
    centre = tuple(np.round(np.argwhere(roi).mean(0)).astype(int))
    frac = float(near[roi].mean())
    if frac < 0.5 and not either[centre]:
        if frac >= 0.1:   # thalamus-ish but not taken: say so, so it can be checked
            log.info("%s %s: %.0f%% on the subject's thalamus, centre outside it: kept the template's ROI",
                     name, k, 100 * frac)
        return roi
    side = "left" if x[roi].mean() < 0 else "right"
    log.info("%s %s: the template thalamus (%.0f%% on the subject's) -> SynthSeg's %s thalamus "
             "(%d voxels, was %d)", name, k, 100 * frac, side, thalamus[side].sum(), roi.sum())
    return thalamus[side].copy()


def bundle_specs(afq, ref_img, transient, names=None, thalamus=None, no_midline=(), midline_tol=10.0):
    """bundle_specs_iter as a list (every bundle's ROIs at once)."""
    return list(bundle_specs_iter(afq, ref_img, transient, names, thalamus, no_midline, midline_tol))


def bundle_specs_iter(afq, ref_img, transient, names=None, thalamus=None, no_midline=(), midline_tol=10.0):
    """MCAT bundle specs from pyAFQ's bundle dictionary: include / exclude ROIs (sealed), start /
    end ROIs grown by pyAFQ's ``dist_to_atlas``, and the other hemisphere excluded for bundles
    that may not cross the midline.  Other pyAFQ criteria (curvature, length, cleaning) are
    streamline-based and do not apply.  Yields one bundle at a time, each warped only when
    it is asked for, so a run holds the ROIs of the bundles it is solving and no others.

    no_midline  bundle names whose cross_midline=False is ignored (no other-hemisphere exclusion)
    midline_tol mm past the midline (world x = 0) a cross_midline=False bundle may still go before
                the other hemisphere is excluded.  pyAFQ checks only 20 points per streamline, so it
                forgives short dips across; a voxel-exact cut at 0 can sever bundles that run
                next to the midline (the CST in the brainstem)
    thalamus    {"left": mask, "right": mask} on ``ref_img``'s grid (SynthSeg's): a start / end
                ROI that is the template's thalamus -- at least half of it within ``dist_to_atlas``
                of a thalamus, or its centre inside one -- is replaced by the subject's thalamus
                on its side.  The thalamus is grey matter, where walkers end, so a template
                thalamus warped a few mm off would make most real endings invalid."""
    from scipy.ndimage import distance_transform_edt

    bd, mapping = afq.export("bundle_dict"), afq.export("mapping")
    dist = afq.export("segmentation_params").get("dist_to_atlas", 4)
    R = ref_img.affine[:3, :3]
    tol = int(dist / np.mean(np.diag(np.linalg.cholesky(R.T @ R))))   # pyAFQ's tolerance_mm_to_vox
    x = nib.affines.apply_affine(ref_img.affine, np.indices(transient.shape).reshape(3, -1).T)[:, 0]
    x = x.reshape(transient.shape)
    for name in (names or list(bd)):
        rois = bd.transform_rois(name, mapping, ref_img)
        arr = lambda img: _on_grid(img, ref_img, order=0) > 0  # noqa: E731
        spec = dict(name=name, include=[seal_26(arr(m)) for m in rois.get("include", [])])
        exc = [seal_26(arr(m)) for m in rois.get("exclude", [])]
        spec["exclude_rois"] = np.any(exc, axis=0) if exc else None      # without the hemisphere (for display)
        for k in ("start", "end"):
            if k in rois:
                roi = arr(rois[k])
                if thalamus is not None and roi.any():
                    roi = _subject_thalamus(name, k, roi, thalamus, x, tol)
                spec[k] = distance_transform_edt(~roi) - 0.5 <= tol   # pyAFQ's endpoint test
        if bd[name].get("cross_midline") is False and name in no_midline:
            log.info("%s: no midline exclusion (no_midline)", name)
        elif bd[name].get("cross_midline") is False:
            own = np.concatenate([np.atleast_1d(x[m]) for m in spec["include"] +
                                  [spec[k] for k in ("start", "end") if k in spec]])
            exc.append(x > midline_tol if own.mean() < 0 else x < -midline_tol)
        spec["exclude"] = np.any(exc, axis=0) if exc else None
        yield spec


def _subject_thalami(afq, sh_img):
    """{"left", "right"} SynthSeg thalami on the fODF grid."""
    from AFQ.nn.synthseg import SynthSegLabels

    seg = afq.export("synthseg_model")
    seg = nib.load(seg) if isinstance(seg, str) else seg
    lab = np.rint(_on_grid(seg, sh_img, order=0)).astype(np.int32)
    lab = lab[..., 0] if lab.ndim == 4 else lab
    return {"left": lab == int(SynthSegLabels.LEFT_THALAMUS), "right": lab == int(SynthSegLabels.RIGHT_THALAMUS)}


def _thalami_if_synthseg(afq, sh_img, synthseg_thalamus):
    """SynthSeg's thalami when pyAFQ's tissue comes from SynthSeg (``pve`` unset, the default
    "synthseg", or a built-in method that includes it), else None; logged."""
    pve = afq.kwargs.get("pve")
    from_synthseg = pve is None or (isinstance(pve, str) and "synthseg" in pve.lower())
    use = synthseg_thalamus and from_synthseg
    why = "--mni-thalamus" if not synthseg_thalamus else f"pve={pve if isinstance(pve, str) else type(pve).__name__}"
    log.info("thalamic start / end ROIs: %s", "SynthSeg's thalami wherever the template's thalamus is used"
             if use else f"the template's ({why})")
    return _subject_thalami(afq, sh_img) if use else None


def roi_specs(afq, sh_img, transient, stem, bundles=None, thalamus=None, no_midline=(), midline_tol=10.0):
    """bundle_specs_iter, writing each bundle's ROI masks as it goes: start / end, the include
    ROIs (labelled 1, 2, ... in pyAFQ's order) and the exclude ROIs (without the other
    hemisphere, which cross_midline=False bundles exclude too)."""
    for spec in bundle_specs_iter(afq, sh_img, transient, bundles, thalamus=thalamus, no_midline=no_midline,
                                  midline_tol=midline_tol):
        b = f"{stem}_bundle-{re.sub(r'[^A-Za-z0-9]', '', spec['name'])}"
        save = lambda m, desc: nib.save(nib.Nifti1Image(np.asarray(m, np.uint8), sh_img.affine),  # noqa: E731
                                        f"{b}_desc-{desc}_mask.nii.gz")
        for k in ("start", "end"):
            if spec.get(k) is not None:
                save(spec[k], k)
        if spec["include"]:
            lab = np.zeros(transient.shape, np.uint8)
            for i, m in enumerate(spec["include"], 1):
                lab[m & (lab == 0)] = i
            save(lab, "include")
        exc = spec.pop("exclude_rois", None)
        if exc is not None:
            save(exc, "exclude")
        yield spec


def write_rois(afq, bundles=None, moments=False, synthseg_thalamus=True, no_midline=(), page=True,
               midline_tol=10.0, out_dir=None):
    """Write the ROI masks of a finished run's bundles (start / end / include / exclude) without
    solving anything, and a viewer page showing them (bundles with no streamlines included).
    Returns the page's path (or None)."""
    from .core import bundle_files
    from .viewer import save_html

    sh_img = nib.load(afq.export(f"{afq.export('tracking_params')['odf_model'].lower()}_params"))
    transient, _, _ = tissue_labels(sh_img, nib.load(afq.export("pve_internal")), nib.load(afq.export("brain_mask")))
    out = mcat_dir(afq, moments, out_dir)
    stem = str(out / Path(afq.kwargs["base_fname"]).name)
    done = bundle_files(out)
    names = list(bundles or afq.export("bundle_dict"))
    run_names = [n for n in names if re.sub(r"[^A-Za-z0-9]", "", n) in done]
    if len(run_names) < len(names):
        log.warning("no run outputs for %s: their ROIs are not written",
                    [n for n in names if n not in run_names])
    thal = _thalami_if_synthseg(afq, sh_img, synthseg_thalamus)
    for spec in roi_specs(afq, sh_img, transient, stem, run_names, thal, no_midline, midline_tol):
        log.info("%s: ROI masks written", spec["name"])
    if not page or not run_names:
        return None
    return save_html(f"{stem}_desc-mcat_rois.html", str(out),
                     bundles=[re.sub(r"[^A-Za-z0-9]", "", n) for n in run_names], rois=True)


def visualize(out, stem, bundles, fodf=None, t1=None, rois=False):
    """The run's figures: a MIP collage of the bundles, a viewer page with all of them
    (densities and cores), and a page per bundle with its glyphs (and the fODF, if given);
    ``rois``: the pages also show each bundle's include / exclude ROIs.
    Failures are logged, not raised: the bundles are already on disk."""
    from .collage import write_collage
    from .core import bundle_files
    from .viewer import save_html

    jobs = [("collage", lambda: write_collage(out, f"{stem}_desc-mcat_collage.png", bundles, t1=t1)),
            ("viewer", lambda: save_html(f"{stem}_desc-mcat_view.html", out, bundles=bundles, rois=rois))]
    files = bundle_files(out)
    for b in bundles:
        if b in files:   # glyphs from 1e-2 of the peak (the slider can raise it); keeps pages small
            jobs.append((f"{b} viewer", lambda b=b: save_html(
                f"{files[b]['prefix']}_desc-view.html", out, bundles=[b], glyphs=True, fodf=fodf,
                glyph_range=1e-2, rois=rois)))
    for what, job in jobs:
        try:
            log.info("%s: %s", what, job())
        except Exception as err:  # noqa: BLE001
            log.warning("%s not written: %s", what, err)


def mcat_dir(afq, moments=False, out_dir=None):
    """Where MCAT's outputs go: ``out_dir`` if given, else <pyAFQ output_dir>/mcat (mcat-moments)."""
    if out_dir:
        return Path(out_dir)
    return Path(afq.kwargs["output_dir"]) / ("mcat-moments" if moments else "mcat")


GiB = 1024 ** 3
AUTO_FRACTION, AUTO_HEADROOM = 0.8, 10 * GiB    # max_mem="auto": min(80% of the limit, limit - 10 GB)


def memory_limit():
    """(bytes this process may use, where that came from): the smallest of its cgroup's
    memory limits (what SLURM and containers enforce), SLURM's request, and physical memory."""
    found = []
    try:
        for line in open("/proc/self/cgroup"):
            hid, ctrl, path = line.rstrip("\n").split(":", 2)
            if hid == "0" and ctrl == "":                       # cgroup v2
                files = ["memory.max"]
                root = "/sys/fs/cgroup"
            elif "memory" in ctrl.split(","):                   # cgroup v1
                files = ["memory.limit_in_bytes"]
                root = "/sys/fs/cgroup/memory"
            else:
                continue
            p = Path(path)
            for d in [p, *p.parents]:                          # the job's cgroup and its parents
                for f in files:
                    try:
                        v = (Path(root) / d.relative_to("/") / f).read_text().strip()
                    except OSError:
                        continue
                    if v.isdigit() and int(v) < 1 << 60:
                        found.append((int(v), f"cgroup {d}"))
    except OSError:
        pass
    env = os.environ
    if env.get("SLURM_MEM_PER_NODE", "").isdigit():
        found.append((int(env["SLURM_MEM_PER_NODE"]) * 1024 ** 2, "SLURM_MEM_PER_NODE"))
    elif env.get("SLURM_MEM_PER_CPU", "").isdigit():
        cpus = env.get("SLURM_CPUS_ON_NODE") or env.get("SLURM_CPUS_PER_TASK") or "1"
        if cpus.isdigit():
            found.append((int(env["SLURM_MEM_PER_CPU"]) * 1024 ** 2 * int(cpus), "SLURM_MEM_PER_CPU"))
    try:
        found.append((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), "physical memory"))
    except (ValueError, OSError, AttributeError):
        pass
    return min(found) if found else (None, "unknown")


def memory_in_use():
    """Resident memory of this process (bytes; 0 if unknown)."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _budget(max_mem, chain, launches):
    """max_mem as segment_bundles takes it (bytes for the chain, the launches and the solve
    columns).  "auto": min(80% of the memory limit, limit - 10 GB), at least half the limit,
    less what the process already holds besides the chain and the launches (pyAFQ's data,
    the fODF, ...).  An explicit size is used as given, with a warning above the limit."""
    limit, src = memory_limit()
    held = memory_in_use()
    other = max(held - chain.nbytes - launches.nbytes, 0)
    if isinstance(max_mem, str) and max_mem.strip().lower() == "auto":
        if limit is None:
            raise ValueError("max_mem='auto': cannot tell this machine's memory limit; give a size")
        total = max(min(AUTO_FRACTION * limit, limit - AUTO_HEADROOM), 0.5 * limit)
        budget = total - other
        log.info("memory: limit %.1f GB (%s), MCAT may use %.1f GB; %.1f GB already held besides the chain "
                 "(%.1f GB) and launches (%.1f GB) -> %.1f GB budget", limit / GiB, src, total / GiB,
                 other / GiB, chain.nbytes / GiB, launches.nbytes / GiB, budget / GiB)
        return budget
    b = _bytes(max_mem)
    if limit is not None and b + other > limit:
        log.warning("max_mem %.1f GB plus the %.1f GB already held exceeds this job's memory limit "
                    "(%.1f GB, %s): it may be killed; try max_mem='auto'", b / GiB, other / GiB, limit / GiB, src)
    return b


def _bytes(x):
    if isinstance(x, str):
        m = re.fullmatch(r"\s*([\d.]+)\s*([kmgt]?)b?\s*", x.lower())
        if not m:
            raise ValueError(f"cannot read memory size {x!r} (e.g. '16GB')")
        return float(m.group(1)) * 1024 ** "kmgt".find(m.group(2)) * (1024 if m.group(2) else 1)
    return float(x)


def run(afq, n_proc, max_mem="auto", bundles=None, moments=False, tol=1e-6, angle=35.0, double_odf=True,
        cores=True, viz=True, synthseg_thalamus=True, max_length=250.0, viz_rois=False,
        no_midline=(), midline_tol=10.0, out_dir=None, verbose=False, flux=False):
    """Segment pyAFQ's bundles with MCAT.

    afq         ``ParticipantAFQ`` or ``GroupAFQ`` (every participant in turn)
    n_proc      threads
    max_mem     memory budget, bytes, e.g. "16GB", or "auto" (default): sets how many bundles
                share each solve.  "auto" reads the job's limit (cgroup / SLURM / physical),
                keeps min(80%, limit - 10 GB) of it and subtracts what is already held
    bundles     pyAFQ bundle names (default: all in the bundle dictionary)
    moments     spatial first moments (narrower, more accurate bundles; ~5x slower)
    tol         remaining walker mass at which the solves stop (relative)
    angle       maximum turn per one-voxel hop (degrees)
    double_odf  weight each hop by the fODF where it lands (False: singleODF, like pyAFQ's
                first-order probabilistic tracker)
    cores       also write each bundle's ridge-path core (mcat.core)
    viz         also write a MIP collage of all bundles (PNG), a viewer page of all bundles
                (densities and cores), and one page per bundle with its visitation-ODF glyphs
                and the fODF (mcat.viewer.save_html).  Pages load NiiVue from a CDN.
    viz_rois    the viewer pages also show each bundle's include / exclude ROIs
    no_midline  bundle names whose cross_midline=False is ignored (no other-hemisphere exclusion)
    midline_tol mm past the midline a one-sided bundle may go before the other hemisphere is
                excluded (default 10; 0: a strict cut at x = 0)
    max_length  soft maximum streamline length (mm; 0: none).  Before any bundle is solved,
                direction states from which most walkers would still be walking after this
                long -- traps where they circle, unable to turn out under the angle limit --
                become invalid ends for every bundle (mcat.solve.overlength_states); the
                survival per voxel is written to <stem>_desc-overlength_map.nii.gz
    out_dir     write MCAT's outputs here instead of <pyAFQ output_dir>/mcat (mcat-moments);
                pyAFQ's derivatives are only read.  For a GroupAFQ: <out_dir>/sub-<s>/ses-<t>
    verbose     log timing inside the solves (ms per power step: matrix product / mask and
                sum); the time per stage is logged at the end anyway
    synthseg_thalamus  with pyAFQ's SynthSeg tissue (pve unset or "synthseg"), a template start / end
                ROI that is the thalamus is replaced by SynthSeg's thalamus (see bundle_specs)
    flux        also write each bundle's edge flux (mcat.fields): the expected streamline steps
                from every voxel into each of its 26 neighbours, and the flux through each
                voxel.  Not with moments.  The fields themselves are not kept here; use
                ``segment_bundles(keep_fields=True)`` to work with them.
    Returns {bundle: BundleResult} (a dict per participant for a GroupAFQ)."""
    import numba

    if hasattr(afq, "pAFQ_list"):
        kw = dict(cores=cores, viz=viz, synthseg_thalamus=synthseg_thalamus, max_length=max_length,
                  viz_rois=viz_rois, no_midline=no_midline, midline_tol=midline_tol,
                  verbose=verbose, flux=flux)
        return {f"sub-{s}_ses-{t}": run(p, n_proc, max_mem, bundles, moments, tol, angle, double_odf, **kw,
                                        out_dir=out_dir and str(Path(out_dir) / f"sub-{s}" / f"ses-{t}"))
                for s, t, p in zip(afq.valid_sub_list, afq.valid_ses_list, afq.pAFQ_list)}
    numba.set_num_threads(max(1, min(int(n_proc), numba.config.NUMBA_NUM_THREADS)))
    t0 = time.time()
    clock = Clock()
    with clock("pyAFQ inputs"):
        sh_path = afq.export(f"{afq.export('tracking_params')['odf_model'].lower()}_params")
        sh_img = nib.load(sh_path)
        pve_img, bm_img = nib.load(afq.export("pve_internal")), nib.load(afq.export("brain_mask"))
    with clock("tissue labels"):
        transient, absorb, dseg = tissue_labels(sh_img, pve_img, bm_img)
    out = mcat_dir(afq, moments, out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = str(out / Path(afq.kwargs["base_fname"]).name)
    nib.save(nib.Nifti1Image(dseg, sh_img.affine), f"{stem}_desc-mcat_dseg.nii.gz")
    # pyAFQ's tissue from SynthSeg: its thalami stand in for the template's thalamic ROIs
    with clock("pyAFQ inputs"):
        thal = _thalami_if_synthseg(afq, sh_img, synthseg_thalamus)

    def specs():
        # each bundle's ROIs warped just before its group is solved, its ROI masks written then
        # (the solve keeps what it needs, so the ROIs can go)
        yield from roi_specs(afq, sh_img, transient, stem, bundles, thal, no_midline, midline_tol)

    cfg = ChainConfig(angle=angle, double_odf=double_odf, moments=moments)
    with clock("fODF load"):
        sh = np.asarray(sh_img.dataobj, np.float32)
    with clock("chain build / cache"):
        chain = Chain.build(sh, sh_img.header.get_zooms()[:3], transient, absorb, cfg,
                            cache=f"{stem}_desc-chain_cache.npz")
    with clock("pyAFQ inputs"):
        seeds = np.argwhere((_on_grid(nib.load(afq.export("wm_gm_interface")), sh_img, 0) > 0) & transient)
    t_ol = time.perf_counter()
    if max_length and max_length > 0:
        t1_ = time.time()
        kill, surv, T = overlength_states(chain, max_length, sh_img.header.get_zooms()[:3])
        vox = surv.reshape(-1, chain.nb)
        nib.save(nib.Nifti1Image(chain.to_volume(vox.max(1)), sh_img.affine), f"{stem}_desc-overlength_map.nii.gz")
        chain.kill = kill if kill.any() else None
        log.info("soft maximum streamline length %.0f mm (%d steps): %d of %d direction states, in %d voxels, "
                 "keep most walkers longer than that and end them invalidly (%.0fs)", max_length, T,
                 kill.sum(), kill.size, int(kill.reshape(-1, chain.nb).any(1).sum()), time.time() - t1_)
    clock.add("soft max length", time.perf_counter() - t_ol)
    with clock("seed launches (prepare)"):
        launches = prepare_launches(chain, seeds)
    chain.release_pmfs()
    rows = []

    def write(res):
        name = re.sub(r"[^A-Za-z0-9]", "", res.name)
        b = f"{stem}_bundle-{name}"
        sp = np.zeros(chain.shape, np.float32)
        np.add.at(sp, tuple(res.seed_vox.T), res.seed_prob)
        maps = dict(seedprob=(sp, "probmap"), density=(chain.to_volume(res.density_t), "map"),
                    endpoints=(chain.to_volume(res.endpoints_a, "absorbing"), "map"))
        d = res.density_t
        keep = np.flatnonzero(d >= ODF_FLOOR * d.max()) if d.max() > 0 else np.zeros(0, np.int64)
        ijk = tuple((np.stack(np.unravel_index(chain.t_lin[keep], chain.pshape), 1) - 1).T)
        coef = visitation_odf(res.density_class[keep], sh[ijk], chain.geo)
        odf = np.zeros(chain.shape + coef.shape[1:], np.float32)
        odf[ijk] = coef
        maps["visitodf"] = (odf, "dwimap")
        if res.fields is not None:
            f26 = res.fields.edge_flux()
            maps["throughflux"] = (chain.to_volume(res.fields.through_flux(f26)), "map")
            flux = np.zeros(chain.shape + (26,), np.float32)
            flux[tuple((np.stack(np.unravel_index(chain.t_lin, chain.pshape), 1) - 1).T)] = f26
            maps["edgeflux"] = (flux, "map")
            res.info["flux_conservation"] = {k: v for k, v in res.fields.conservation(f26).items()
                                             if not isinstance(v, np.ndarray)}
            res.fields = None
        for desc, (vol, suffix) in maps.items():
            nib.save(nib.Nifti1Image(vol, sh_img.affine), f"{b}_desc-{desc}_{suffix}.nii.gz")
        with open(f"{b}_desc-summary.json", "w") as fh:
            json.dump(dict(bundle=res.name, config=cfg.key(), result=res.info,
                           valid_ends=[ABSORB_CLASSES[c] for c in (1, 3)],
                           density="expected steps of bundle streamlines in each voxel, all seeds",
                           endpoints="expected bundle endpoints per voxel (two per streamline)",
                           visitodf="the walkers' directions per voxel: SH (descoteaux07, legacy, "
                                    "lmax 8), expected visits per steradian",
                           seedprob="P(the streamline seeded here belongs to the bundle)",
                           **({"throughflux": "expected bundle-streamline steps leaving each voxel",
                               "edgeflux": "the same per neighbour: 26 volumes, mcat.geometry.OFFSETS order"}
                              if flux else {})), fh, indent=1)
        i = res.info
        rows.append(dict(bundle=res.name, expected_streamlines=round(i["expected_bundle_streamlines"], 4),
                         columns=i["n_columns"], backward_steps=i["backward"]["iterations"],
                         forward_steps=i["forward"]["iterations"],
                         seconds=i["seconds"]))
        log.info("%s: %.2f expected streamlines", res.name, i["expected_bundle_streamlines"])

    n_trapped = [0]

    def trapped(what, names, vol):
        # where the mass a stalled solve could not finish sits: open it over the T1
        n_trapped[0] += 1
        f = f"{stem}_desc-trapped{what}{n_trapped[0]}_map.nii.gz"
        nib.save(nib.Nifti1Image(vol, sh_img.affine), f)
        peak = nib.affines.apply_affine(sh_img.affine, np.unravel_index(np.argmax(vol), vol.shape))
        log.warning("%s solve of %s did not finish: the mass left (%.1e of the start) is mapped in %s; "
                    "most of it at (%.0f, %.0f, %.0f) mm", what, ", ".join(names), vol.sum(), f, *peak)

    results = segment_bundles(chain, specs(), launches, tol=tol, max_mem=_budget(max_mem, chain, launches),
                              on_result=write,
                               on_trapped=trapped, verbose=verbose, clock=clock, keep_fields=flux)
    with open(f"{stem}_desc-mcat_table.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    if cores:
        from .core import write_cores
        with clock("cores"):
            write_cores(out)
    t_viz = time.perf_counter()
    if viz:
        try:
            t1 = afq.export("t1_masked")
        except Exception:  # noqa: BLE001 -- the collage falls back to the tissue labels
            t1 = None
        visualize(out, stem, [re.sub(r"[^A-Za-z0-9]", "", n) for n in results], sh_path, t1=t1, rois=viz_rois)
        clock.add("viz", time.perf_counter() - t_viz)
    clock.report("MCAT time per stage", time.time() - t0)
    log.info("MCAT: %d bundles in %.0fs -> %s", len(results), time.time() - t0, out)
    return results
