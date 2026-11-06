"""Sphinx configuration for the MCAT documentation (MyST Markdown)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_ext"))

project = "MCAT"
author = "The MCAT developers"
copyright = "2026, the MCAT developers"
try:
    from importlib.metadata import version as _v

    release = _v("mcat-tractography")
except Exception:  # pragma: no cover
    release = "0.1.0"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.programoutput",
    "mcat_animations",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "_generated", "animations", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

# ------------------------------------------------------------------ MyST
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "attrs_block",
                          "attrs_inline", "substitution"]
myst_heading_anchors = 3
myst_dmath_double_inline = True

numfig = True
numfig_format = {"figure": "Figure %s"}

# ------------------------------------------------------------------ API docs
autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "dipy": ("https://docs.dipy.org/stable/", None),
}

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

# ------------------------------------------------------------------ HTML
html_theme = "pydata_sphinx_theme"
html_title = "MCAT"
(HERE / "_generated").mkdir(exist_ok=True)   # rendered by the mcat_animations extension
html_static_path = ["_static", "_generated"]
html_css_files = ["mcat.css"]
html_theme_options = {
    "github_url": "https://github.com/36000/MCAT",
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
html_context = {"default_mode": "light"}
html_sidebars = {"**": []}          # navigation lives in the header; pages use the full width
html_show_sourcelink = False

linkcheck_ignore = [r"https://github\.com/36000/MCAT.*"]

# ------------------------------------------------------------------ build environment
import os  # noqa: E402

# the `mcat` console script lives next to the interpreter running Sphinx (virtualenvs)
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

# ------------------------------------------------------------------ docstrings
# MCAT's docstrings use plain aligned "name    description" blocks rather than numpydoc.
# Keep them as written: such blocks are shown as preformatted text.
import re  # noqa: E402

_ROW = re.compile(r"^\s*[\w*][\w, *.=()\[\]'\"-]*\s{2,}\S")


def _plain_blocks(app, what, name, obj, options, lines):
    out, i = [], 0
    while i < len(lines):
        if _ROW.match(lines[i]):
            j = i
            block = []
            while j < len(lines) and lines[j].strip() and (
                    _ROW.match(lines[j]) or (block and lines[j].startswith(" ") and
                                            len(lines[j]) - len(lines[j].lstrip()) > 0)):
                block.append(lines[j])
                j += 1
            if out and out[-1].strip():
                out.append("")
            out += ["::", ""] + ["    " + b for b in block] + [""]
            i = j
        else:
            out.append(lines[i])
            i += 1
    lines[:] = out


def setup(app):
    app.connect("autodoc-process-docstring", _plain_blocks)
