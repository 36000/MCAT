"""Sphinx configuration for the MCAT documentation (MyST Markdown)."""
from pathlib import Path

# docs/
#   source/      the pages (MyST Markdown) and conf.py
#   build/       everything generated (html/, ...)
HERE = Path(__file__).resolve().parent
DOCS = HERE.parent

project = "MCAT"
author = "The MCAT developers"
copyright = "2026, the MCAT developers"
try:
    from importlib.metadata import version as _v

    release = _v("mcat-tractography")
except Exception:  # pragma: no cover
    release = "0.1.0"
version = release

extensions = ["myst_parser"]

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

html_theme = "pydata_sphinx_theme"
html_title = "MCAT"
html_theme_options = {
    "github_url": "https://github.com/rokemlab/MCAT",
    "navigation_with_keys": True,
}
html_context = {"default_mode": "light"}
html_sidebars = {"**": []}
html_show_sourcelink = False
