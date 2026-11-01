"""Regression: vignette intensity must actually change the output."""
import shutil
import subprocess

import pytest

from render_worker.edits.vignette import vignette_filter


def test_angle_scales_with_intensity():
    import re
    ang = lambda i: float(re.search(r"vignette=([\d.]+)", vignette_filter(0, 1, i)).group(1))
    assert ang(0.0) == 0.0
    assert ang(0.2) < ang(0.4) < ang(0.8)
    assert ang(5.0) <= 1.5708 + 1e-3  # clamped to PI/2


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_edges_not_black_at_default_intensity(tmp_path):
    from PIL import Image
    out = tmp_path / "v.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=white:s=640x360:d=1,format=yuv420p",
         "-vf", vignette_filter(0, 1, 0.4), "-frames:v", "1", str(out)], check=True)
    im = Image.open(out).convert("L")
    assert im.getpixel((2, 2)) > 60          # corners dimmed, not crushed to black
    assert im.getpixel((320, 180)) > 250     # centre untouched
