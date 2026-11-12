# Python API

## Running MPAT

```{eval-rst}
.. autofunction:: mpat.run

.. autofunction:: mpat.view

.. autofunction:: mpat.save_html
```

## The chain and the solves

```{eval-rst}
.. autoclass:: mpat.ChainConfig
   :members:

.. autoclass:: mpat.Chain
   :members: build, lin, seed_peaks, mul_forward, mul_backward, absorb_rhs, absorbed_from, to_volume

.. autofunction:: mpat.prepare_launches

.. autofunction:: mpat.segment_bundles

.. autoclass:: mpat.solve.BundleResult

.. autoclass:: mpat.solve.Launches

.. autofunction:: mpat.solve.solve_series

.. autofunction:: mpat.solve.bicgstab
```

## Geometry

```{eval-rst}
.. autofunction:: mpat.geometry.build_geometry

.. autofunction:: mpat.geometry.pmf_and_cone_mass

.. autofunction:: mpat.geometry.visitation_odf
```

## Cores

```{eval-rst}
.. autofunction:: mpat.bundle_core

.. autofunction:: mpat.core.write_cores
```

## pyAFQ integration

```{eval-rst}
.. autofunction:: mpat.afq.tissue_labels

.. autofunction:: mpat.afq.bundle_specs

.. autofunction:: mpat.afq.seal_26
```

## Spatial moments

```{eval-rst}
.. automodule:: mpat.moments
   :no-members:

.. autofunction:: mpat.moments.stabilize
```
