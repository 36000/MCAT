# Python API

## Running MCAT

```{eval-rst}
.. autofunction:: mcat.run

.. autofunction:: mcat.view

.. autofunction:: mcat.viewer.save_html
```

## The chain and the solves

```{eval-rst}
.. autoclass:: mcat.ChainConfig
   :members:

.. autoclass:: mcat.Chain
   :members: build, lin, seed_peaks, mul_forward, mul_backward, absorb_rhs, absorbed_from, to_volume

.. autofunction:: mcat.prepare_launches

.. autofunction:: mcat.segment_bundles

.. autoclass:: mcat.solve.BundleResult

.. autoclass:: mcat.solve.Launches

.. autofunction:: mcat.solve.solve_series

.. autofunction:: mcat.solve.bicgstab
```

## Geometry

```{eval-rst}
.. autofunction:: mcat.geometry.build_geometry

.. autofunction:: mcat.geometry.pmf_and_cone_mass

.. autofunction:: mcat.geometry.visitation_odf
```

## Cores

```{eval-rst}
.. autofunction:: mcat.bundle_core

.. autofunction:: mcat.core.write_cores
```

## pyAFQ integration

```{eval-rst}
.. autofunction:: mcat.afq.tissue_labels

.. autofunction:: mcat.afq.bundle_specs

.. autofunction:: mcat.afq.seal_26
```

## Spatial moments

```{eval-rst}
.. automodule:: mcat.moments
   :no-members:

.. autofunction:: mcat.moments.stabilize
```
