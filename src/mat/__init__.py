"""MAT: Markov absorbing tractography.

Bundle densities as exact expectations over infinitely many streamlines: the probabilistic
tracker is an absorbing Markov chain on (white-matter voxel, direction) states whose
transitions come from MITTENS's analytic path geometry, and bundle selection (include /
exclude / endpoint ROIs) is solved with forward and backward passes over that chain.
pyAFQ provides the fODF, tissue, seeds, bundle definitions and warped ROIs.

    from AFQ.api.participant import ParticipantAFQ
    import mat
    afq = ParticipantAFQ(dwi, bval, bvec, t1, output_dir)
    mat.run(afq, n_proc=8, max_mem="16GB")            # -> <output_dir>/mat/
    mat.view(f"{output_dir}/mat", bundles=["LeftArcuate"])   # in a notebook
"""
from .afq import run
from .chain import Chain, ChainConfig
from .core import bundle_core
from .solve import prepare_launches, segment_bundles


def view(*args, **kwargs):
    """ipyniivue widget with bundle densities and cores (see mat.viewer.view)."""
    from .viewer import view as _view

    return _view(*args, **kwargs)


__all__ = ["Chain", "ChainConfig", "bundle_core", "prepare_launches", "run", "segment_bundles", "view"]
