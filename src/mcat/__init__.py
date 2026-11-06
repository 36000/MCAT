"""MCAT: Markov chain absorbing tractography.

Bundle densities as exact expectations over infinitely many streamlines: the probabilistic
tracker is an absorbing Markov chain on (white-matter voxel, direction) states whose
transitions come from MITTENS's analytic path geometry, and bundle selection (include /
exclude / endpoint ROIs) is solved with forward and backward passes over that chain.
pyAFQ provides the fODF, tissue, seeds, bundle definitions and warped ROIs.

    from AFQ.api.participant import ParticipantAFQ
    import mcat
    afq = ParticipantAFQ(dwi, bval, bvec, t1, output_dir)
    mcat.run(afq, n_proc=8, max_mem="16GB")            # -> <output_dir>/mcat/
    mcat.view(f"{output_dir}/mcat", bundles=["LeftArcuate"])   # in a notebook
"""
from .afq import run
from .chain import Chain, ChainConfig
from .core import bundle_core
from .fields import BundleFields, endpoint_patches, flux_orientation, restrict_endpoints
from .solve import prepare_launches, segment_bundles


def view(*args, **kwargs):
    """ipyniivue widget with bundle densities and cores (see mcat.viewer.view)."""
    from .viewer import view as _view

    return _view(*args, **kwargs)


__all__ = ["BundleFields", "Chain", "ChainConfig", "bundle_core", "endpoint_patches", "flux_orientation",
           "prepare_launches", "restrict_endpoints", "run", "segment_bundles", "view"]
