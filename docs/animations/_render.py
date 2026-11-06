"""Render one docs animation (``anim_*.py``) or figure (``fig_*.py``).

    python _render.py <script> <out_dir> [mp4|gif]
    python _render.py --warm               # build and cache the phantom's chains first

An ``anim_*`` script defines ``make()`` returning a dict with ``fig``, ``update`` (frame index
-> None), ``frames`` (int), and optionally ``fps`` (default 12) and ``poster`` (frame drawn for
the poster image, default the last).  A ``fig_*`` script's ``make()`` returns a figure.
Outputs: ``<out_dir>/<name>.mp4`` (or .gif) and ``<name>.png``, ``name`` the script's stem.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _even(fig):
    """H.264 (yuv420p) needs even pixel dimensions."""
    w, h = fig.get_size_inches() * fig.dpi
    fig.set_size_inches(2 * round(w / 2) / fig.dpi, 2 * round(h / 2) / fig.dpi)


def render(script, out_dir, fmt="mp4"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    import phantom

    phantom.style()
    script = Path(script)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(script.stem, script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    made = mod.make()
    if script.stem.startswith("fig_"):
        made.savefig(out_dir / f"{script.stem}.png")
        plt.close(made)
        return
    fig, update, n = made["fig"], made["update"], int(made["frames"])
    fps = made.get("fps", 12)
    _even(fig)
    anim = FuncAnimation(fig, update, frames=range(n), interval=1000 / fps, blit=False, repeat=False)
    if fmt == "gif":
        anim.save(out_dir / f"{script.stem}.gif", writer=PillowWriter(fps=fps))
    else:
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,
                              extra_args=["-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart"])
        anim.save(out_dir / f"{script.stem}.mp4", writer=writer)
    update(made.get("poster", n - 1))
    fig.savefig(out_dir / f"{script.stem}.png")
    plt.close(fig)


def warm():
    """Build (and cache) every chain the scripts use, once, before they run in parallel."""
    import phantom

    for moments in (False, True):
        c = phantom.build_chain(moments=moments)
        phantom.launches(c)
    phantom.build_chain(double_odf=False)


if __name__ == "__main__":
    if sys.argv[1] == "--warm":
        warm()
    else:
        render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "mp4")
