# Viewing bundles

`mcat.view` returns an [ipyniivue](https://github.com/niivue/ipyniivue) widget for Jupyter
(JupyterLab, Notebook 7, VS Code, Voila). It shows a 3-D render and three slices. Each
bundle's density is drawn on a log scale from 1e-3 of its peak, and its core as a tube:

```python
import mcat
mcat.view(f"{output_dir}/mcat", t1=t1_masked, bundles=["LeftArcuate", "LeftCorticospinal"])
```

## Visitation-ODF glyphs

With `glyphs=True` (or a list of bundle names), the viewer also draws the visitation ODFs as
glyphs, like fODF glyphs. Where the fODF has a crossing, the glyph shows only the lobe the
bundle uses ({numref}`fig-visitodf`). Glyphs follow the slices: every voxel at or above
`glyph_floor` (1e-2 of the peak) on the three slices through the crosshair gets one, redrawn
as you click or scroll. The slices start at the core's midpoint.

| option | |
|---|---|
| `fodf=` | an SH image on the same grid (pyAFQ's `*_model-csd_param-wm_dwimap.nii.gz`), drawn as translucent grey glyphs at the same voxels, for comparison |
| `glyph_planes=(x, y, z)` | start the slices elsewhere (mm); `"all"` draws every glyph at once, which is heavy |
| `glyph_norm=False` | size glyphs by visits instead of normalising each one |
| `glyph_color` | `"direction"` or `"bundle"` |
| `floor` | the faintest density shown, as a fraction of each bundle's peak |
| `show_density`, `density_opacity`, `core_radius` | display settings |

## A standalone web page

```python
mcat.viewer.save_html("bundles.html", folder, glyphs=True)
```

writes the same scene as a standalone web page, to open in any browser or share. The page has
controls for:
- the layout
- each bundle's density, core and glyphs
- the opacities
- whether scrolling moves the slices or zooms

The data are embedded, and the page meshes the glyphs itself. NiiVue loads from a CDN; pass
`niivue_js=` a local `niivue.umd.js` (npm `@niivue/niivue`, `dist/`) to embed it and work
offline.
