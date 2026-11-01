"""Regression: extracted LUT must not lift blacks / dim whites (washed-out look)."""
import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("color_matcher")

from style_worker import lut_extract as le


def _scene_frames(n=50):
    rng = np.random.default_rng(1)
    return [np.clip(rng.uniform(30, 220, 3) + rng.normal(0, 60, (144, 256, 3)), 0, 255).astype(np.uint8)
            for _ in range(n)]


@pytest.mark.parametrize("strength", [0.3, 0.5])
def test_lut_preserves_black_and_white(tmp_path, strength):
    path, _ = le._extract_lut_color_matcher(_scene_frames(), str(tmp_path), strength=strength)
    rows = [l.split() for l in open(path) if l[0].isdigit()]
    data = np.array(rows, dtype=np.float32)
    assert data[0].max() < 0.02          # black stays black
    assert data[-1].min() > 0.98         # white stays white


def _apply_cube(path, rgb):
    """Nearest-neighbour cube lookup, enough for a sanity check."""
    d = np.array([l.split() for l in open(path) if l[0].isdigit()], dtype=np.float32).reshape(33, 33, 33, 3)
    r, g, b = [int(round(c / 255 * 32)) for c in rgb]
    return d[b, g, r] * 255


def test_tone_lut_keeps_colour_with_neutral_white_reference(tmp_path):
    """A grey subject on a white studio backdrop must not desaturate vivid footage."""
    ref = np.full((144, 256, 3), 250, np.uint8)            # white backdrop
    ref[40:110, 60:200] = (110, 110, 112)                  # grey 'car'
    ref[110:, :] = (35, 35, 38)                            # dark floor
    path, _ = le._extract_lut_tone_style([ref] * 12, str(tmp_path), strength=0.5)

    neon_green = (60, 220, 40)
    out = _apply_cube(path, neon_green)
    sat = lambda c: (max(c) - min(c)) / max(max(c), 1)
    assert sat(out) > 0.9 * sat(neon_green)                # colour kept (old LUT halved it)

    sky = (70, 130, 210)
    assert sat(_apply_cube(path, sky)) > 0.85 * sat(sky)

    data = np.array([l.split() for l in open(path) if l[0].isdigit()], dtype=np.float32)
    assert data[0].max() < 0.01 and data[-1].min() > 0.99  # black/white pinned
