# Installation

MCAT needs Python 3.10 or newer. It depends on [MITTENS](https://github.com/mattcieslak/MITTENS)
v2 (`mittens-dmri`), which is not on PyPI yet, so pip builds it from GitHub. That build needs a
Rust toolchain ([rustup.rs](https://rustup.rs)) and `maturin`:

```bash
pip install maturin
pip install "mcat-tractography[viewer] @ git+https://github.com/36000/MCAT.git"
```

Extras:

| extra | for |
|---|---|
| `viewer` | `mcat.view` in notebooks ([ipyniivue](https://github.com/niivue/ipyniivue)) |
| `test` | the test suite (`pytest`) |
| `docs` | building this documentation |

The other dependencies are numpy, scipy, nibabel, numba, DIPY, scikit-image and pyAFQ (≥ 3.4).

## Building the documentation

```bash
pip install -e ".[docs,test]"
sphinx-build -b html docs docs/_build/html
```

The figures and animations on {doc}`algorithm` are rendered at build time by the scripts in
`docs/animations/`, which run MCAT on a small phantom. That takes a few minutes the first time.
The results are cached and re-rendered only when a script or MCAT's source changes. Two
environment variables control this:

- `MCAT_DOCS_ANIMATIONS=skip` skips rendering for quick edits to the text.
- `MCAT_DOCS_ANIMATIONS=gif` writes GIFs instead of MP4s.
