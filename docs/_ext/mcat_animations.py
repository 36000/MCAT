"""Sphinx extension: render the docs' animations and figures, and the ``animation`` directive.

At ``builder-inited`` every ``animations/anim_*.py`` and ``animations/fig_*.py`` is rendered
(by ``animations/_render.py``, one subprocess each, in parallel) into ``_generated/``, unless
its output is up to date: a hash of the script, the shared phantom module and MCAT's source is
kept in ``_generated/.hashes.json``.

``MCAT_DOCS_ANIMATIONS``:  ``mp4`` (default), ``gif`` (no ffmpeg needed), or ``skip`` (render
nothing; missing outputs become placeholders, for quick text edits).

In MyST::

    ```{animation} forward
    :name: fig-forward

    Caption, with *markup* and $math$.
    ```

``:plain:`` makes it an unnumbered container instead of a figure.
"""
from __future__ import annotations

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
OUT = "_generated"


def _mode():
    return os.environ.get("MCAT_DOCS_ANIMATIONS", "mp4").lower()


def _digest(paths):
    h = hashlib.sha1()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def render_all(app):
    src = Path(app.srcdir)
    adir, out = src / "animations", src / OUT
    out.mkdir(exist_ok=True)
    mode = _mode()
    if mode == "skip":
        log.info("[mcat] MCAT_DOCS_ANIMATIONS=skip: not rendering animations")
        import base64

        blank = base64.b64decode(  # 1x1 transparent PNG, so static figures still resolve
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
        for s in adir.glob("fig_*.py"):
            f = out / f"{s.stem}.png"
            if not f.exists():
                f.write_bytes(blank)
        return
    pkg = sorted((src.parent / "src" / "mcat").glob("*.py"))
    shared = _digest([adir / "phantom.py", adir / "_render.py", *pkg])
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
        log.info("[mcat] animations up to date")
        return
    env = dict(os.environ, MPLBACKEND="Agg")
    runner = str(adir / "_render.py")
    log.info("[mcat] building the phantom's chains")
    t0 = time.time()
    r = subprocess.run([sys.executable, runner, "--warm"], env=env, capture_output=True, text=True)
    if r.returncode:
        log.warning("[mcat] phantom warm-up failed:\n%s", r.stderr[-3000:])
        return

    def one(item):
        s, key, h = item
        t = time.time()
        p = subprocess.run([sys.executable, runner, str(s), str(out), ext], env=env,
                           capture_output=True, text=True)
        return s, key, h, p, time.time() - t

    workers = int(os.environ.get("MCAT_DOCS_JOBS", max(1, (os.cpu_count() or 2) // 2)))
    log.info("[mcat] rendering %d animations/figures with %d workers", len(todo), workers)
    with ThreadPoolExecutor(workers) as ex:
        for s, key, h, p, dt in ex.map(one, todo):
            if p.returncode:
                log.warning("[mcat] %s failed:\n%s", s.name, p.stderr[-3000:])
                continue
            hashes[key] = h
            log.info("[mcat]   %s (%.0fs)", s.stem, dt)
    hfile.write_text(json.dumps(hashes, indent=1))
    log.info("[mcat] done in %.0fs", time.time() - t0)


class Animation(SphinxDirective):
    """A numbered figure holding a looping, scrubbable video (poster image elsewhere)."""

    required_arguments = 1
    has_content = True
    option_spec = {"name": directives.unchanged, "width": directives.unchanged,
                   "alt": directives.unchanged, "plain": directives.flag}

    def run(self):
        name = self.arguments[0]
        stem = name if name.startswith("anim_") else f"anim_{name}"
        ext = "gif" if _mode() == "gif" else "mp4"
        gen = Path(self.env.srcdir) / OUT
        video, poster = gen / f"{stem}.{ext}", gen / f"{stem}.png"
        for f in (video, poster):
            self.env.note_dependency(str(f))
        plain = "plain" in self.options        # not a numbered figure (e.g. on the landing page)
        fig = (nodes.container if plain else nodes.figure)(classes=["mcat-animation"])
        if not plain:
            self.add_name(fig)
        depth = self.env.docname.count("/")
        static = "../" * depth + "_static/"
        width = self.options.get("width", "100%")
        alt = self.options.get("alt", name)
        if video.exists():
            if ext == "mp4":
                html = (f'<video class="mcat-video" controls loop muted autoplay playsinline '
                        f'preload="metadata" poster="{static}{stem}.png" style="width:{width}" '
                        f'aria-label="{alt}"><source src="{static}{stem}.mp4" type="video/mp4">'
                        f'<img src="{static}{stem}.png" alt="{alt}"></video>')
            else:
                html = f'<img class="mcat-video" src="{static}{stem}.gif" alt="{alt}" style="width:{width}">'
            fig += nodes.raw("", html, format="html")
            if poster.exists():
                img = nodes.image(uri=f"/{OUT}/{stem}.png", alt=alt)
                fig += addnodes.only("", img, expr="not html")
        else:
            fig += nodes.paragraph("", f"[animation '{stem}' not rendered: set MCAT_DOCS_ANIMATIONS=mp4]",
                                   classes=["mcat-placeholder"])
        if self.content:
            box = nodes.Element()
            self.state.nested_parse(self.content, self.content_offset, box)
            first = box[0] if len(box) and isinstance(box[0], nodes.paragraph) else None
            if first is not None and plain:
                fig += box.children
                rest = []
            elif first is not None:
                cap = nodes.caption(first.rawsource, "", *first.children)
                cap.source, cap.line = first.source, first.line
                fig += cap
                rest = list(box[1:])
            else:
                rest = list(box)
            if rest:
                fig += nodes.legend("", *rest)
        return [fig]


def setup(app):
    app.connect("builder-inited", render_all)
    app.add_directive("animation", Animation)
    return dict(version="0.1", parallel_read_safe=True, parallel_write_safe=True)
