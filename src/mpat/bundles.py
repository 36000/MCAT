"""MPAT's bundles: pyAFQ bundle dictionaries plus MPAT's own rules.

A bundle is defined by a pyAFQ ``BundleDict`` entry (include / exclude / start / end ROIs,
``cross_midline``, ``space``), which pyAFQ warps into the subject for MPAT.  What MPAT does
beyond that is a rule, kept as data in one place:

* ``MPAT_RULES`` -- rules for pyAFQ's own bundles, by name, applied to whichever bundle
  dictionary a run uses (so a plain pyAFQ dictionary gets them too);
* an entry's ``"mpat"`` sub-dictionary -- rules carried by the bundle itself, which override
  ``MPAT_RULES``.  Newer pyAFQ rejects unknown criteria, so pyafq_bd(bd) strips this key
  before pyAFQ's own recognition.

Rules (``RULE_DEFAULTS`` hold for every bundle that does not set them):

    midline_tol        mm past the midline (world x = 0) a cross_midline=False bundle may go
                       before the other hemisphere is excluded (0: cut at x = 0)
    exclude_mni_boxes  extra exclusion ROIs, MNI boxes {x: (lo, hi), y: ..., z: ...} in mm,
                       warped with pyAFQ's mapping
    include_mni_boxes  the same, as inclusion ROIs
    start_synthseg     start / end ROIs as SynthSeg labels of the subject (AFQ.nn.synthseg
    end_synthseg       SynthSegLabels names or values); they replace any template start / end
    synthseg_tol_mm    how far start / end ROIs given as SynthSeg labels are grown, mm (None: pyAFQ's
                       dist_to_atlas, 4 mm by default, as for template ROIs; 0: the labels as they are)
    thalamus           "synthseg": a start / end ROI that is the template's thalamus becomes the
                       subject's SynthSeg thalamus (when pyAFQ's tissue comes from SynthSeg);
                       "template": keep it

``mpat_bd()`` is MPAT's default dictionary: pyAFQ's defaults (default + SLF + callosal) and
the hippocampal cingulum and the fornix (MPAT only: its ends are SynthSeg labels, which
pyAFQ's segmentation does not read; ``fornix=False`` leaves it out; pyafq_bd(bd) gives the
dictionary pyAFQ's own recognition accepts).  ``mpat_bd(synthseg_cst=True)`` (``--synthseg-cst``)
swaps pyAFQ's corticospinal tracts for ones defined by SynthSeg endpoints only
(synthseg_cst_bd).
"""
from __future__ import annotations

# The fornix and septum pellucidum on the midline, under the corpus callosum and over the
# thalami, from the fornix's columns (in front of the thalami) back along its body (MNI mm).
FORNIX_BOX_MNI = dict(x=(-3.0, 3.0), y=(-28.0, 15.0), z=(-5.0, 22.0))
# The fornix's body alone: on the midline over the thalami, behind the columns.  The fornix's
# waypoint.  FORNIX_BOX_MNI reaches down to the roof of the hypothalamus, so a walker going
# straight across from the hippocampal head to the hypothalamus (the short way, under the
# thalamus) touched it; the body is only reached the long way, up the crus behind the
# thalamus and forward over it, as the fornix runs.
FORNIX_BODY_MNI = dict(x=(-4.0, 4.0), y=(-25.0, -5.0), z=(8.0, 22.0))

RULE_DEFAULTS = dict(midline_tol=0.0, exclude_mni_boxes=(), include_mni_boxes=(), start_synthseg=None,
                     end_synthseg=None, synthseg_tol_mm=None, thalamus="synthseg")

MPAT_RULES = {
    # next to the midline in the brainstem: a voxel-exact cut severs it (pyAFQ checks only 20
    # points per streamline, so it forgives short dips across)
    "Left Corticospinal": dict(midline_tol=10.0),
    "Right Corticospinal": dict(midline_tol=10.0),
    # walkers otherwise reach the thalamus along the fornix on the midline; the real ATR runs
    # lateral to the ventricles
    "Left Anterior Thalamic": dict(exclude_mni_boxes=[FORNIX_BOX_MNI]),
    "Right Anterior Thalamic": dict(exclude_mni_boxes=[FORNIX_BOX_MNI]),
}


def rules(bd, name):
    """The MPAT rules of bundle ``name`` in ``bd``: RULE_DEFAULTS, then MPAT_RULES, then the
    entry's own "mpat" sub-dictionary."""
    out = dict(RULE_DEFAULTS)
    out.update(MPAT_RULES.get(name, {}))
    try:
        own = bd[name].get("mpat") or bd[name].get("mcat") or {}   # "mcat": the old name, still read
    except (KeyError, AttributeError):
        own = {}
    out.update(own)
    return out


def synthseg_values(labels):
    """SynthSeg label values from names ("LEFT_HIPPOCAMPUS") or numbers."""
    from AFQ.nn.synthseg import SynthSegLabels

    return [int(SynthSegLabels[v]) if isinstance(v, str) else int(v) for v in labels]


def hcc_bd():
    """The hippocampal cingulum (Cingulum Hippocampus): MATLAB AFQ's two waypoints from pyAFQ's
    templates (which have no endpoint ROIs), and for MPAT one end in SynthSeg's hippocampus
    (the other anywhere valid).  Not among pyAFQ's defaults (hard to delineate at low SNR /
    resolution)."""
    import AFQ.api.bundle_dict as abd
    import AFQ.data.fetch as afd

    t = afd.read_templates()
    return abd.BundleDict({
        f"{side} Cingulum Hippocampus": dict(
            include=[t[f"HCC_roi1_{s}"], t[f"HCC_roi2_{s}"]], prob_map=t[f"HCC_{s}_prob_map"],
            cross_midline=False, mpat=dict(end_synthseg=[f"{side.upper()}_HIPPOCAMPUS"]))
        for side, s in (("Left", "L"), ("Right", "R"))})


def fornix_bd(resample_to=None):
    """The fornix, from the hippocampus (fimbria) along the body on the midline to the
    hypothalamus (columns, mammillary bodies): SynthSeg's hippocampus and hypothalamus as its
    ends, its body (FORNIX_BODY_MNI) as a waypoint, each side kept to its own hemisphere but for 3 mm.
    MPAT only.  pyAFQ's SynthSeg run has no hypothalamus, so MPAT reruns SynthSeg with its
    hypothalamus model for these ROIs when it is missing (<stem>_model-synthseg2hypo_dseg.nii.gz,
    reused afterwards; mpat.afq._synthseg_on_grid).  pyAFQ's tissue is not changed: there the
    hypothalamus is ventral diencephalon (mixed), whose rim is gray matter, and the end ROI,
    grown by dist_to_atlas as every end ROI is, reaches it."""
    import AFQ.api.bundle_dict as abd

    out = {}
    for side, s in (("Left", "LEFT"), ("Right", "RIGHT")):
        out[f"{side} Fornix"] = dict(
            cross_midline=False,
            mpat=dict(start_synthseg=[f"{s}_HIPPOCAMPUS"], end_synthseg=[f"{s}_HYPOTHALAMUS"],
                      include_mni_boxes=[FORNIX_BODY_MNI], midline_tol=3.0))
    kw = {} if resample_to is None else dict(resample_to=resample_to)
    return abd.BundleDict(out, **kw)


PYAFQ_ROI_KEYS = ("include", "start", "end", "prob_map", "recobundles")


def pyafq_bd(bd):
    """``bd`` as pyAFQ's recognition accepts it: each bundle's "mpat" rules removed (newer
    pyAFQ rejects unknown criteria), and bundles left with no ROI pyAFQ uses (the fornix,
    defined only by MPAT's SynthSeg ends) dropped."""
    out = bd.copy()
    for name in list(out._dict):
        entry = {k: v for k, v in out._dict[name].items() if k not in ("mpat", "mcat")}
        if not any(k in entry for k in PYAFQ_ROI_KEYS):
            del out[name]
            continue
        out._dict[name] = entry
    return out


SYNTHSEG_CST_CORTEX = ("PRECENTRAL", "PARACENTRAL", "POSTCENTRAL")   # the central (sensorimotor) cortex


def synthseg_cst_labels(side):
    """(start, end) SynthSeg labels of the SynthSeg corticospinal tract on ``side``: the
    precentral, paracentral and postcentral cortex (SYNTHSEG_CST_CORTEX, SynthSeg's cortical
    parcellation), and the brainstem."""
    h = {"Left": "LH", "Right": "RH"}[side]
    return [f"CTX_{h}_{p}" for p in SYNTHSEG_CST_CORTEX], ["BRAIN_STEM"]


def synthseg_cst_bd(resample_to=None):
    """The corticospinal tracts from SynthSeg endpoints only (no waypoints): each side's
    precentral, paracentral and postcentral cortex to the brainstem (synthseg_cst_labels), kept to its own
    hemisphere as pyAFQ's CST is (MPAT_RULES' midline tolerance).  The labels are used as they
    are (synthseg_tol_mm 0), not grown by pyAFQ's 4 mm dist_to_atlas: they are the subject's
    own, with no registration error to allow for.  Named as pyAFQ's CSTs, so
    they replace them in a dictionary.  The cortical parcels come from rerunning SynthSeg with
    its parcellation (mpat.afq.synthseg_parcellated); pyAFQ's own recognition reads them as
    subject-space masks (mpat.afq.synthseg_cst_for_pyafq)."""
    import AFQ.api.bundle_dict as abd

    out = {}
    for side in ("Left", "Right"):
        start, end = synthseg_cst_labels(side)
        out[f"{side} Corticospinal"] = dict(cross_midline=False, mpat=dict(start_synthseg=start, end_synthseg=end,
                                                                            synthseg_tol_mm=0.0))
    kw = {} if resample_to is None else dict(resample_to=resample_to)
    return abd.BundleDict(out, **kw)


def mpat_bd(fornix=True, synthseg_cst=False):
    """MPAT's default bundle dictionary (see the module docstring); ``synthseg_cst``: the
    corticospinal tracts from SynthSeg endpoints (synthseg_cst_bd) instead of pyAFQ's."""
    import AFQ.api.bundle_dict as abd

    bd = abd.default_bd() + abd.slf_bd() + abd.callosal_bd() + hcc_bd()
    bd = bd + fornix_bd() if fornix else bd
    if synthseg_cst:
        cst = synthseg_cst_bd()
        bd = bd[[n for n in bd if n not in set(cst)]] + cst
    return bd
