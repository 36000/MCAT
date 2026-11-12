"""Sphinx extension: render the docs' animations and figures, and the directives that show them.

Layout (``docs/``)::

    source/        the pages (Sphinx's srcdir)
    animations/    anim_*.py and fig_*.py, with the shared phantom
    build/generated/   their rendered output (served as static files)
    build/cache/       the phantom's chains (see animations/phantom.py)

At ``builder-inited`` every ``animations/anim_*.py`` and ``animations/fig_*.py`` is rendered
(by ``animations/_render.py``, one subprocess each, in parallel) into ``build/generated/``,
unless its output is up to date: a hash of the script, the shared phantom module and MPAT's
source is kept in ``build/generated/.hashes.json``.

``MPAT_DOCS_ANIMATIONS``:  ``mp4`` (default), ``gif`` (no ffmpeg needed), or ``skip`` (render
nothing; missing outputs become placeholders, for quick text edits).

In MyST, an animation (a numbered figure holding a video; ``:plain:`` for an unnumbered one)::

    ```{animation} forward
    :name: fig-forward

    Caption, with *markup* and $math$.
    ```

and a static figure rendered by ``animations/fig_<name>.py``::

    ```{generated-figure} tissue_states
    :name: fig-phantom

    Caption.
    ```
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

log = logging.getLogger(__name__)
BLANK_PNG = base64.b64decode(  # 1x1 transparent PNG: a placeholder in skip mode
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def paths(srcdir):
    """(animation scripts, rendered output, MPAT's package source) for docs/source = srcdir."""
    docs = Path(srcdir).resolve().parent
    return docs / "animations", docs / "build" / "generated", docs.parent / "src" / "mpat"


def _mode():
    return os.environ.get("MPAT_DOCS_ANIMATIONS", "mp4").lower()


def _digest(files):
    h = hashlib.sha1()
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def render_all(app):
    adir, out, pkg = paths(app.srcdir)
    out.mkdir(parents=True, exist_ok=True)
    mode = _mode()
    if mode == "skip":
        log.info("[mpat] MPAT_DOCS_ANIMATIONS=skip: not rendering animations")
        for s in adir.glob("fig_*.py"):
            f = out / f"{s.stem}.png"
            if not f.exists():
                f.write_bytes(BLANK_PNG)
        return
    shared = _digest([adir / "phantom.py", adir / "_render.py", *sorted(pkg.glob("*.py"))])
    hfile = out / ".hashes.json"
    hashes = json.loads(hfile.read_text()) if hfile.exists() else {}
    ext = "gif" if mode == "gif" else "mp4"
    todo = []
    for s in sorted(adir.glob("anim_*.py")) + sorted(adir.glob("fig_*.py")):
        key = f"{s.stem}:{mode}"
        h = shared + _digest([s])
        outs = [out / f"{s.stem}.png"] + ([out / f"{s.stem}.{ext}"] if s.stem.startswith("anim_") else [])
        if hashes.get(key) == h and all(o.exists() for o in outs):
            continue
        todo.append((s, key, h))
    if not todo:
        log.info("[mpat] animations up to date")
        return
    env = dict(os.environ, MPLBACKEND="Agg")
    runner = str(adir / "_render.py")
    log.info("[mpat] building the phantom's chains")
    t0 = time.time()
    r = subprocess.run([sys.executable, runner, "--warm"], env=env, capture_output=True, text=True)
    if r.returncode:
        log.warning("[mpat] phantom warm-up failed:\n%s", r.stderr[-3000:])
        return

    def one(item):
        s, key, h = item
        t = time.time()
        p = subprocess.run([sys.executable, runner, str(s), str(out), ext], env=env,
                           capture_output=True, text=True)
        return s, key, h, p, time.time() - t

    workers = int(os.environ.get("MPAT_DOCS_JOBS", max(1, (os.cpu_count() or 2) // 2)))
    log.info("[mpat] rendering %d animations/figures with %d workers", len(todo), workers)
    with ThreadPoolExecutor(workers) as ex:
        for s, key, h, p, dt in ex.map(one, todo):
            if p.returncode:
                log.warning("[mpat] %s failed:\n%s", s.name, p.stderr[-3000:])
                continue
            hashes[key] = h
            log.info("[mpat]   %s (%.0fs)", s.stem, dt)
    hfile.write_text(json.dumps(hashes, indent=1))
    log.info("[mpat] done in %.0fs", time.time() - t0)


class _Generated(SphinxDirective):
    required_arguments = 1
    has_content = True
    option_spec = {"name": directives.unchanged, "width": directives.unchanged,
                   "alt": directives.unchanged}
    prefix = ""

    def stem(self):
        name = self.arguments[0]
        return name if name.startswith(self.prefix) else self.prefix + name

    def generated(self, filename):
        """(absolute path, image uri relative to this document) of a rendered file."""
        _, out, _ = paths(self.env.srcdir)
        f = out / filename
        self.env.note_dependency(str(f))
        here = Path(self.env.srcdir) / Path(self.env.docname).parent
        return f, Path(os.path.relpath(f, here)).as_posix()

    def caption(self, fig, plain=False):
        """Directive content: the first paragraph is the caption, the rest a legend."""
        if not self.content:
            return
        box = nodes.Element()
        self.state.nested_parse(self.content, self.content_offset, box)
        if plain:
            fig += box.children
            return
        first = box[0] if len(box) and isinstance(box[0], nodes.paragraph) else None
        rest = list(box)
        if first is not None:
            cap = nodes.caption(first.rawsource, "", *first.children)
            cap.source, cap.line = first.source, first.line
            fig += cap
            rest = rest[1:]
        if rest:
            fig += nodes.legend("", *rest)


class Animation(_Generated):
    """A numbered figure holding a looping, scrubbable video (its poster for other builders)."""

    option_spec = dict(_Generated.option_spec, plain=directives.flag)
    prefix = "anim_"

    def run(self):
        stem = self.stem()
        ext = "gif" if _mode() == "gif" else "mp4"
        video, _ = self.generated(f"{stem}.{ext}")
        poster, poster_uri = self.generated(f"{stem}.png")
        plain = "plain" in self.options        # not a numbered figure (e.g. on the landing page)
        fig = (nodes.container if plain else nodes.figure)(classes=["mpat-animation"])
        if not plain:
            self.add_name(fig)
        static = "../" * self.env.docname.count("/") + "_static/"
        width = self.options.get("width", "100%")
        alt = self.options.get("alt", self.arguments[0])
        if video.exists():
            if ext == "mp4":
                html = (f'<video class="mpat-video" controls loop muted autoplay playsinline '
                        f'preload="metadata" poster="{static}{stem}.png" style="width:{width}" '
                        f'aria-label="{alt}"><source src="{static}{stem}.mp4" type="video/mp4">'
                        f'<img src="{static}{stem}.png" alt="{alt}"></video>')
            else:
                html = f'<img class="mpat-video" src="{static}{stem}.gif" alt="{alt}" style="width:{width}">'
            fig += nodes.raw("", html, format="html")
            if poster.exists():
                fig += addnodes.only("", nodes.image(uri=poster_uri, alt=alt), expr="not html")
        else:
            fig += nodes.paragraph("", f"[animation '{stem}' not rendered: set MPAT_DOCS_ANIMATIONS=mp4]",
                                   classes=["mpat-placeholder"])
        self.caption(fig, plain)
        return [fig]


class GeneratedFigure(_Generated):
    """A numbered figure showing the image rendered by ``animations/fig_<name>.py``."""

    prefix = "fig_"

    def run(self):
        _, uri = self.generated(f"{self.stem()}.png")
        fig = nodes.figure(classes=["mpat-figure"])
        self.add_name(fig)
        img = nodes.image(uri=uri, alt=self.options.get("alt", self.arguments[0]))
        if "width" in self.options:
            img["width"] = self.options["width"]
        fig += img
        self.caption(fig)
        return [fig]


def setup(app):
    app.connect("builder-inited", render_all)
    app.add_directive("animation", Animation)
    app.add_directive("generated-figure", GeneratedFigure)
    return dict(version="0.2", parallel_read_safe=True, parallel_write_safe=True)
