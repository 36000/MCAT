"""``mcat``: run pyAFQ (for its derivatives) and MCAT on one participant."""
import argparse
import json
import logging


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dwi"), p.add_argument("bval"), p.add_argument("bvec")
    p.add_argument("t1", help="T1w image, registered to the DWI (pyAFQ's t1_file)")
    p.add_argument("output_dir")
    p.add_argument("--n-proc", type=int, help="threads (required unless --rois-only)")
    p.add_argument("--max-mem", help="memory budget, e.g. 16GB (required unless --rois-only)")
    p.add_argument("--moments", action="store_true", help="spatial first moments (narrower bundles)")
    p.add_argument("--tol", type=float, default=1e-10, help="remaining walker mass at which solves stop")
    p.add_argument("--angle", type=float, default=35.0, help="maximum turn per one-voxel hop (deg)")
    p.add_argument("--single-odf", action="store_true", help="first-order hops (pyAFQ-like), not doubleODF")
    p.add_argument("--bundles", nargs="+", help="pyAFQ bundle names (default: all)")
    p.add_argument("--max-length", type=float, default=250.0,
                   help="soft maximum streamline length, mm: states where walkers would mostly outlast "
                        "it (traps) end them, for every bundle (0: off)")
    p.add_argument("--no-krylov", action="store_true", help=argparse.SUPPRESS)   # no-op (kept for old scripts)
    p.add_argument("--mni-thalamus", action="store_true",
                   help="keep the template's thalamic start / end ROIs (default: SynthSeg's thalami)")
    p.add_argument("--no-viz", action="store_true",
                   help="skip the collage and viewer pages (written by default)")
    p.add_argument("--midline-tol", type=float, default=10.0,
                   help="mm past the midline a one-sided (cross_midline=False) bundle may go before the "
                        "other hemisphere is excluded (0: strict cut at x = 0)")
    p.add_argument("--no-midline", nargs="+", default=(), metavar="BUNDLE",
                   help="ignore cross_midline=False for these bundles (no other-hemisphere exclusion)")
    p.add_argument("--rois-only", action="store_true",
                   help="solve nothing: write a finished run's ROI masks (start / end / include / exclude) "
                        "and a viewer page showing them, <stem>_desc-mcat_rois.html")
    p.add_argument("--viz-rois", action="store_true",
                   help="viewer pages also show each bundle's include (green) / exclude (red) ROIs")
    p.add_argument("--flux", action="store_true", help="also write edge-flux maps per bundle (mcat.fields)")
    p.add_argument("--afq-kwargs", default="{}",
                   help="extra ParticipantAFQ arguments: JSON, or a Python expression giving a dict, "
                        "with afd (AFQ.data.fetch), abd (AFQ.api.bundle_dict) and the "
                        "AFQ.definitions image / mapping classes in scope, e.g. "
                        "'dict(bundle_info=abd.baby_bd(), brain_mask_definition=BabyBrainMask())'")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    from AFQ.api.participant import ParticipantAFQ

    from .afq import run, write_rois

    kwargs = afq_kwargs(a.afq_kwargs)
    if "tracking_params" in kwargs:
        if "odf_model" not in kwargs["tracking_params"]:
            kwargs["tracking_params"]["odf_model"] = "CSD"
    else:
        kwargs["tracking_params"] = {"odf_model": "CSD"}

    afq = ParticipantAFQ(a.dwi, a.bval, a.bvec, a.t1, a.output_dir, **kwargs)
    if a.rois_only:
        page = write_rois(afq, a.bundles, a.moments, synthseg_thalamus=not a.mni_thalamus, no_midline=a.no_midline,
                          midline_tol=a.midline_tol, flux=a.flux)
        logging.getLogger("mcat").info("ROI page: %s", page)
        return
    if a.n_proc is None or a.max_mem is None:
        p.error("--n-proc and --max-mem are required (except with --rois-only)")
    run(afq, a.n_proc, a.max_mem, a.bundles, a.moments, a.tol, a.angle, not a.single_odf, viz=not a.no_viz,
        synthseg_thalamus=not a.mni_thalamus,
        max_length=a.max_length, viz_rois=a.viz_rois, no_midline=a.no_midline,
        midline_tol=a.midline_tol, flux=a.flux)


def afq_kwargs(text):
    """ParticipantAFQ keyword arguments from the command line: JSON, or else a Python
    expression evaluated with pyAFQ's usual helpers in scope (as in pyAFQ's examples):
    ``afd`` (AFQ.data.fetch), ``abd`` (AFQ.api.bundle_dict), every class in
    AFQ.definitions.image and AFQ.definitions.mapping, ``np`` and ``nib``."""
    try:
        kwargs = json.loads(text)
    except json.JSONDecodeError:
        import AFQ.api.bundle_dict as abd
        import AFQ.data.fetch as afd
        import AFQ.definitions.image as afm
        import AFQ.definitions.mapping as amp
        import nibabel as nib
        import numpy as np

        scope = dict(afd=afd, abd=abd, np=np, nib=nib)
        scope.update({k: getattr(afm, k) for k in afm.__all__})
        scope.update({k: getattr(amp, k) for k in amp.__all__})
        try:
            kwargs = eval(text, scope)  # noqa: S307 -- the user's own command line
        except Exception as err:
            raise SystemExit(f"--afq-kwargs is neither JSON nor a valid Python expression: {err}") from err
    if not isinstance(kwargs, dict):
        raise SystemExit(f"--afq-kwargs must give a dict, got {type(kwargs).__name__}")
    return kwargs


if __name__ == "__main__":
    main()
