"""MCAT on one participant of the Healthy Brain Network (HBN-POD2).

This follows pyAFQ's "Getting started with pyAFQ - ParticipantAFQ" tutorial
(https://tractometry.org/pyAFQ/examples/tutorial_examples/plot_002_participant_afq_api.html)
up to the ``ParticipantAFQ`` object, then segments the bundles with MCAT instead of
tractography + recognition, and draws the results.

    python examples/hbn_participant.py --n-proc 8 --max-mem 32GB
    python examples/hbn_participant.py --n-proc 8 --max-mem 32GB --data-dir /scratch/AFQ_data \\
        --bundles "Left Arcuate" "Left Corticospinal"

The data (about 400 MB) go to ``~/AFQ_data/HBN`` by default, as in pyAFQ's tutorial, and MCAT
writes to ``.../derivatives/afq/sub-NDARAA948VFH/ses-HBNsiteRU/dwi/mcat/``.  ``--figures DIR``
saves the figures used in MCAT's documentation.
"""
import argparse
import os
import os.path as op
import time

import nibabel as nib
import numpy as np

SUBJECT = "NDARAA948VFH"
SESSION = "HBNsiteRU"
SHOWCASE = ["Left Arcuate", "Left Corticospinal", "Left Inferior Fronto-occipital",
            "Left Inferior Longitudinal", "Left Uncinate", "Callosum Motor"]


def participant(data_dir):
    """The tutorial's ParticipantAFQ for one HBN subject (downloaded if needed)."""
    from AFQ.api.participant import ParticipantAFQ
    import AFQ.data.fetch as afd
    import AFQ.definitions.image as afm

    afd.fetch_hbn_preproc([SUBJECT], path=data_dir)
    sub_dir = op.join(data_dir, "HBN", "derivatives", "qsiprep", f"sub-{SUBJECT}")
    stem = op.join(sub_dir, f"ses-{SESSION}", "dwi",
                   f"sub-{SUBJECT}_ses-{SESSION}_acq-64dir_space-T1w_desc-preproc_dwi")
    anat = op.join(sub_dir, "anat", f"sub-{SUBJECT}")
    output_dir = op.join(data_dir, "HBN", "derivatives", "afq", f"sub-{SUBJECT}",
                         f"ses-{SESSION}", "dwi")
    os.makedirs(output_dir, exist_ok=True)

    # partial volume estimates and brain mask from qsiprep, as in the tutorial
    pve = afm.PVEImages(afm.ImageFile(path=f"{anat}_label-CSF_probseg.nii.gz"),
                        afm.ImageFile(path=f"{anat}_label-GM_probseg.nii.gz"),
                        afm.ImageFile(path=f"{anat}_label-WM_probseg.nii.gz"))
    brain_mask_definition = afm.ImageFile(path=f"{anat}_desc-brain_mask.nii.gz")

    return ParticipantAFQ(
        dwi_data_file=f"{stem}.nii.gz",
        bval_file=f"{stem}.bval",
        bvec_file=f"{stem}.bvec",
        t1_file=f"{anat}_desc-preproc_T1w.nii.gz",
        output_dir=output_dir,
        # MCAT walks on the fODF pyAFQ would track on; CSD, as the mcat command line uses.
        # (pyAFQ's seeding options -- n_seeds, random_seeds, ... -- do not apply: MCAT
        # seeds every white/grey-matter interface voxel, with no sampling.)
        tracking_params=dict(odf_model="CSD"),
        pve=pve,
        brain_mask_definition=brain_mask_definition,
    )


# ------------------------------------------------------------------ figures
def _style():
    import matplotlib as mpl

    mpl.rcParams.update({"figure.dpi": 100, "savefig.dpi": 110, "font.size": 11,
                         "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                         "text.color": "#0b0b0b", "axes.titlesize": 11, "font.family": "DejaVu Sans"})


def _mass_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    cm = LinearSegmentedColormap.from_list(
        "mcat_mass", ["#fdf1ea", "#f9c4a8", "#f28d5e", "#eb6834", "#b8461a", "#6e2508"])
    cm.set_bad(alpha=0.0)
    return cm


def _load(f):
    """Data and affine, reoriented to RAS (qsiprep's images are LPS), so that every view has
    the subject's left on the left and anterior up / to the right."""
    img = nib.as_closest_canonical(nib.load(f))
    return np.asarray(img.dataobj, np.float32), img.affine


def figure_bundles(mcat_dir, fa_file, path, bundles=SHOWCASE):
    """Small multiples: each bundle's density as a maximum-intensity projection (log scale)
    over the FA's projection, with its core."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from mcat.core import bundle_files

    _style()
    files = bundle_files(mcat_dir)
    fa, aff = _load(fa_file)
    names = [b for b in bundles if b.replace(" ", "").replace("-", "") in files]
    views = [("sagittal", 0), ("axial", 2)]
    fig, axs = plt.subplots(2, len(names), figsize=(2.3 * len(names), 5.0),
                            gridspec_kw=dict(wspace=0.02, hspace=0.02))
    axs = np.atleast_2d(axs).reshape(2, -1)
    cm = _mass_cmap()
    for j, name in enumerate(names):
        f = files[name.replace(" ", "").replace("-", "")]
        d, _ = _load(f["density"])
        core = np.loadtxt(f["prefix"] + "_desc-core_path.tsv", skiprows=1, usecols=(1, 2, 3)) \
            if op.exists(f["prefix"] + "_desc-core_path.tsv") else None
        ijk = nib.affines.apply_affine(np.linalg.inv(aff), core) if core is not None else None
        for i, (view, ax_) in enumerate(views):
            ax = axs[i, j]
            bg = fa.max(ax_)
            mip = d.max(ax_)
            # RAS: sagittal shows (y, z), anterior right; axial shows (x, y), anterior up
            ax.imshow(bg.T, origin="lower", cmap="gray", vmin=0, vmax=np.percentile(bg, 99.5))
            ax.imshow(np.ma.masked_less(mip, mip.max() * 1e-3).T, origin="lower", cmap=cm,
                      norm=LogNorm(mip.max() * 1e-3, mip.max()), interpolation="nearest")
            if ijk is not None:
                keep = [k for k in range(3) if k != ax_]
                ax.plot(ijk[:, keep[0]], ijk[:, keep[1]], color="#2a78d6", lw=1.6)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(name.replace("Inferior ", "Inf. ").replace("Longitudinal", "Long."), fontsize=10)
    fig.text(0.01, 0.5, "sagittal  ·  axial", rotation=90, va="center", fontsize=10, color="#52514e")
    fig.suptitle("MCAT bundle densities (maximum-intensity projections, log scale, 1e-3 of peak "
                 "to peak) and cores (blue)", fontsize=11, x=0.01, ha="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_slices(mcat_dir, fa_file, path, name="Left Arcuate"):
    """One bundle on three slices through its core's midpoint: density over FA, and its core."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from mcat.core import bundle_files

    _style()
    f = bundle_files(mcat_dir)[name.replace(" ", "")]
    fa, aff = _load(fa_file)
    d, _ = _load(f["density"])
    core = np.loadtxt(f["prefix"] + "_desc-core_path.tsv", skiprows=1, usecols=(1, 2, 3))
    ijk = nib.affines.apply_affine(np.linalg.inv(aff), core)
    c = np.round(ijk[len(ijk) // 2]).astype(int)
    cm = _mass_cmap()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.4), gridspec_kw=dict(wspace=0.02))
    norm = LogNorm(d.max() * 1e-3, d.max())
    for ax, (view, k) in zip(axs, [("sagittal", 0), ("coronal", 1), ("axial", 2)]):
        sl = [slice(None)] * 3
        sl[k] = c[k]
        bg, dd = fa[tuple(sl)], d[tuple(sl)]
        ax.imshow(bg.T, origin="lower", cmap="gray", vmin=0, vmax=np.percentile(fa, 99.5))
        im = ax.imshow(np.ma.masked_less(dd, norm.vmin).T, origin="lower", cmap=cm, norm=norm,
                       interpolation="nearest")
        keep = [q for q in range(3) if q != k]
        near = np.abs(ijk[:, k] - c[k]) < 3
        ax.plot(np.where(near, ijk[:, keep[0]], np.nan), np.where(near, ijk[:, keep[1]], np.nan),
                color="#2a78d6", lw=2)
        ax.set_axis_off()
        ax.set_title(f"{view} slice through the core's midpoint", fontsize=10)
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01)
    cb.set_label("expected visits per voxel")
    cb.outline.set_visible(False)
    fig.suptitle(f"{name}: density over FA, core in blue (shown within 3 voxels of each slice)",
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    import AFQ.data.fetch as afd

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-proc", type=int, default=os.cpu_count())
    p.add_argument("--max-mem", default="16GB")
    p.add_argument("--data-dir", default=afd.afq_home, help="where HBN/ goes (default: ~/AFQ_data)")
    p.add_argument("--bundles", nargs="+", help="pyAFQ bundle names (default: all)")
    p.add_argument("--moments", action="store_true")
    p.add_argument("--figures", help="also save the documentation's figures in this folder")
    p.add_argument("--figures-only", action="store_true",
                   help="redraw the figures from an earlier run's outputs (needs --figures)")
    a = p.parse_args(argv)

    import logging

    import mcat

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    myafq = participant(a.data_dir)
    fa_file = myafq.export("dti_fa")             # for the figures' background
    out = op.join(myafq.kwargs["output_dir"], "mcat-moments" if a.moments else "mcat")
    if not a.figures_only:
        results = mcat.run(myafq, n_proc=a.n_proc, max_mem=a.max_mem, bundles=a.bundles,
                           moments=a.moments)
        print(f"\n{len(results)} bundles in {(time.time() - t0) / 60:.1f} min -> {out}")
        for name, r in results.items():
            print(f"  {name:32s} {r.info['expected_bundle_streamlines']:9.1f} expected streamlines")
    if a.figures:
        os.makedirs(a.figures, exist_ok=True)
        figure_bundles(out, fa_file, op.join(a.figures, "hbn_bundles.png"))
        figure_slices(out, fa_file, op.join(a.figures, "hbn_left_arcuate.png"))
        mcat.save_html(op.join(a.figures, "hbn_bundles.html"), out,
                              bundles=["LeftArcuate", "LeftCorticospinal"], glyphs=True)
        print("figures ->", a.figures)


if __name__ == "__main__":
    main()
