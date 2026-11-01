# Copyright (c) 2025 Devayan Dewri. All rights reserved.
# Licensed under the Elastic License 2.0 - see LICENSE in the repo root.
# Commercial SaaS use is prohibited without written permission.
"""Extract LUT from reference video using color-matcher."""

import os
from typing import Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    from color_matcher import ColorMatcher
    _HAS_COLOR_MATCHER = True
except ImportError:
    ColorMatcher = None  # type: ignore[assignment,misc]
    _HAS_COLOR_MATCHER = False

try:
    import colour
except ImportError:
    colour = None  # type: ignore[assignment]

from shared_py.logging_config import StructuredLogger
from shared_py.models import StyleAnalysis
from shared_py.storage import get_storage

logger = StructuredLogger("style_worker.lut_extract")


LUT_SIZE = 33


def sample_frames(video_path: str, n_samples: int = 50) -> list:
    """Sample n frames evenly across the video, skipping first/last 0.5s."""
    if cv2 is None:
        logger.warning("cv2 not available, cannot sample frames")
        return []
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(fps * 0.5)

    valid_range = (skip_frames, max(skip_frames + 1, total_frames - skip_frames))
    indices = np.linspace(valid_range[0], valid_range[1], n_samples, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Skip dark/letterboxed frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = gray.mean()
            if 15 < mean_brightness < 240:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def _build_identity_lut_image(lut_size: int = LUT_SIZE) -> np.ndarray:
    """Build a 2D image that encodes all RGB triples of a 3D identity LUT."""
    grid = np.arange(lut_size, dtype=np.float32) / (lut_size - 1) * 255.0
    r = np.broadcast_to(grid.reshape(lut_size, 1, 1), (lut_size, lut_size, lut_size))
    g = np.broadcast_to(grid.reshape(1, lut_size, 1), (lut_size, lut_size, lut_size))
    b = np.broadcast_to(grid.reshape(1, 1, lut_size), (lut_size, lut_size, lut_size))
    identity = np.stack([r, g, b], axis=-1)
    return identity.reshape(lut_size, lut_size * lut_size, 3)


def _lut_image_to_cube(lut_image: np.ndarray, lut_size: int = LUT_SIZE) -> np.ndarray:
    """Reshape a 2D LUT image back to (lut_size, lut_size, lut_size, 3)."""
    if lut_image.shape != (lut_size, lut_size * lut_size, 3):
        raise ValueError(f"Unexpected LUT image shape {lut_image.shape}")
    return lut_image.reshape(lut_size, lut_size, lut_size, 3)


def _write_cube_file(path: str, lut_data: np.ndarray, lut_size: int = LUT_SIZE) -> None:
    """Write a 3D LUT as a .cube file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lut_data = np.clip(lut_data, 0, 255).astype(np.float32)
    with open(path, "w", encoding="utf-8") as f:
        f.write("TITLE \"Extracted LUT\"\n")
        f.write(f"LUT_3D_SIZE {lut_size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        for b in range(lut_size):
            for g in range(lut_size):
                for r in range(lut_size):
                    c = lut_data[r, g, b] / 255.0
                    f.write(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")


def _pooled_reference(frames: list, width: int, height: int) -> np.ndarray:
    """Stack ALL sampled frames into one reference image.

    The old code used ``np.median`` across frames from different scenes, which
    collapses to a flat, low-contrast grey and made the LUT crush the tonal
    range. Pooling keeps the real colour/contrast distribution of the video.
    """
    stacked = np.concatenate([f.astype(np.float32) for f in frames], axis=0)
    return cv2.resize(stacked, (width, height), interpolation=cv2.INTER_AREA)


def _anchor_black_white(lut: np.ndarray) -> np.ndarray:
    """Pin neutral black to exactly 0 and neutral white to exactly 255.

    Histogram matching moves the LUT's black/white points to the reference's
    min/max, which lifts blacks and dims whites (washed-out look). We add a
    smooth offset that fades along the luma axis (full correction at the
    black/white corners, none in between), so the colour character of the
    look is kept but the range compression is removed. Unlike a per-channel
    stretch this stays stable when the matcher mixes channels.
    """
    lut = lut.astype(np.float32)
    size = lut.shape[0]
    grid = np.arange(size, dtype=np.float32) / (size - 1)
    r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
    t = (0.2126 * r + 0.7152 * g + 0.0722 * b)[..., None]  # 0 at black, 1 at white
    err_black = 0.0 - lut[0, 0, 0]
    err_white = 255.0 - lut[-1, -1, -1]
    # Quadratic falloff so mid-tones keep the reference's look.
    out = lut + err_black * (1.0 - t) ** 2 + err_white * t ** 2
    return np.clip(out, 0.0, 255.0)


# Typical natural-footage luma percentiles (CIE L*, 0-100). The reference's
# percentiles are compared against these to get its *tone* character
# (deeper shadows, darker mids, ...) independent of what is in the frame.
_TYPICAL_L = (15.0, 50.0, 85.0)
_MAX_L_SHIFT = 12.0  # never move an anchor more than this many L* points


def _tone_curve_from_frames(frames: list) -> Tuple[np.ndarray, dict]:
    """Return (curve, info): a monotone L* -> L* map sampled at L*=0..100 (step 0.5).

    Pixels that are clipped white/black are excluded, so a white studio
    backdrop or letterbox bars cannot drag the whole look toward white/black.
    """
    px = np.concatenate([f[::6, ::6].reshape(-1, 3) for f in frames]).astype(np.float32) / 255.0
    lab = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    luma = lab[:, 0]
    kept = luma[(luma > 3.0) & (luma < 97.0)]
    if kept.size < 1000:
        kept = luma
    ref_p5, ref_p50, ref_p95 = np.percentile(kept, [5, 50, 95])

    anchors_in = [0.0, *_TYPICAL_L, 100.0]
    anchors_out = [0.0]
    for typ, ref in zip(_TYPICAL_L, (ref_p5, ref_p50, ref_p95)):
        anchors_out.append(typ + float(np.clip(ref - typ, -_MAX_L_SHIFT, _MAX_L_SHIFT)))
    anchors_out.append(100.0)

    xs = np.arange(0.0, 100.0 + 1e-6, 0.5)
    curve = np.interp(xs, anchors_in, anchors_out)
    # Smooth the piecewise-linear kinks, keep endpoints pinned and monotone.
    k = np.ones(15) / 15.0
    padded = np.concatenate([np.full(7, curve[0]), curve, np.full(7, curve[-1])])
    curve = np.convolve(padded, k, mode="valid")
    curve = np.maximum.accumulate(np.clip(curve, 0.0, 100.0))
    curve[0], curve[-1] = 0.0, 100.0
    info = {"ref_p5": float(ref_p5), "ref_p50": float(ref_p50), "ref_p95": float(ref_p95)}
    return curve, info


def _extract_lut_tone_style(
    frames: list,
    output_dir: str,
    strength: float = 0.5,
    lut_size: int = LUT_SIZE,
    asset_id: Optional[str] = None,
) -> Tuple[Optional[str], StyleAnalysis]:
    """Content-robust LUT: transfers the reference's TONE only (shadows/mids/highlights).

    Colour (hue and saturation) of the footage is left untouched, so a neutral
    grey-car-on-white-studio reference no longer desaturates vivid clips. Black
    and white points are fixed, so nothing is lifted or washed out.
    """
    curve, _info = _tone_curve_from_frames(frames)
    strength = float(np.clip(strength, 0.0, 1.0))

    identity = _build_identity_lut_image(lut_size).reshape(lut_size, lut_size, lut_size, 3)
    rgb01 = (identity / 255.0).astype(np.float32).reshape(-1, 1, 3)
    lab = cv2.cvtColor(rgb01, cv2.COLOR_RGB2LAB).reshape(-1, 3)

    new_l = np.interp(lab[:, 0], np.arange(0.0, 100.0 + 1e-6, 0.5), curve)
    lab[:, 0] = lab[:, 0] * (1.0 - strength) + new_l * strength
    out = cv2.cvtColor(lab.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    lut_data = np.clip(out * 255.0, 0.0, 255.0).reshape(lut_size, lut_size, lut_size, 3)
    lut_data = _anchor_black_white(lut_data)

    os.makedirs(output_dir, exist_ok=True)
    cube_path = os.path.join(output_dir, "style.cube")
    _write_cube_file(cube_path, lut_data, lut_size)

    median_frame = np.median(np.stack([cv2.resize(f, (160, 90)) for f in frames]), axis=0).astype(np.float32)
    analysis = _build_style_analysis(frames, median_frame, cube_path)
    if asset_id:
        storage_key = f"luts/{asset_id}/global.cube"
        try:
            get_storage().put(cube_path, storage_key, content_type="application/vnd.adobe.cube")
            analysis.lut_storage_key = storage_key
        except Exception as e:
            logger.warning("Failed to persist LUT to storage", asset_id=asset_id, error=str(e))
    return cube_path, analysis


def _extract_lut_color_matcher(
    frames: list,
    output_dir: str,
    strength: float = 0.5,
    lut_size: int = LUT_SIZE,
    asset_id: Optional[str] = None,
) -> Tuple[Optional[str], StyleAnalysis]:
    """Extract LUT using HM-MVGD-HM color matching.

    Blends the matched LUT with identity by `strength` so 0.0 = no effect and
    1.0 = full reference style.
    """
    median_frame = np.median(np.stack(frames), axis=0).astype(np.float32)

    identity_lut_img = _build_identity_lut_image(lut_size)
    ref_resized = _pooled_reference(
        frames, identity_lut_img.shape[1], identity_lut_img.shape[0]
    )

    matcher = ColorMatcher(method="hm-mvgd-hm")
    matched = matcher.transfer(src=identity_lut_img, ref=ref_resized, method="hm-mvgd-hm")
    matched_lut = _lut_image_to_cube(matched, lut_size)

    # Blend with identity to allow adjustable strength
    identity_cube = _build_identity_lut_image(lut_size).reshape(lut_size, lut_size, lut_size, 3)
    blended = identity_cube * (1.0 - strength) + matched_lut * strength
    blended = _anchor_black_white(blended)

    os.makedirs(output_dir, exist_ok=True)
    cube_path = os.path.join(output_dir, "style.cube")
    _write_cube_file(cube_path, blended, lut_size)

    analysis = _build_style_analysis(frames, median_frame, cube_path)
    if asset_id:
        storage_key = f"luts/{asset_id}/global.cube"
        try:
            get_storage().put(cube_path, storage_key, content_type="application/vnd.adobe.cube")
            analysis.lut_storage_key = storage_key
        except Exception as e:
            logger.warning("Failed to persist LUT to storage", asset_id=asset_id, error=str(e))
    return cube_path, analysis


def _extract_lut_reinhard(
    frames: list,
    output_dir: str,
    strength: float = 0.5,
    lut_size: int = LUT_SIZE,
    asset_id: Optional[str] = None,
) -> Tuple[Optional[str], StyleAnalysis]:
    """Fallback LUT extraction using a simple Reinhard-style mean/std transfer."""
    median_frame = np.median(np.stack(frames), axis=0).astype(np.uint8)

    identity_lut = _build_identity_lut_image(lut_size).reshape(
        lut_size, lut_size, lut_size, 3
    )

    frames_arr = np.stack(frames).reshape(-1, 3).astype(np.float32)
    n_samples = min(100000, len(frames_arr))
    sample_idx = np.linspace(0, len(frames_arr) - 1, n_samples, dtype=int)
    samples = frames_arr[sample_idx]

    mean_color = samples.mean(axis=0)
    std_color = samples.std(axis=0) + 1e-6

    identity_flat = identity_lut.reshape(-1, 3)
    ref_mean = mean_color
    ref_std = std_color
    id_mean = 128.0
    id_std = 60.0

    transformed = ((identity_flat - id_mean) / id_std) * ref_std + ref_mean
    transformed = np.clip(transformed, 0, 255)
    lut_data = transformed.reshape(lut_size, lut_size, lut_size, 3)

    # Apply strength blend
    identity_flat_orig = identity_lut.reshape(-1, 3)
    lut_data = identity_flat_orig * (1.0 - strength) + lut_data * strength
    lut_data = lut_data.reshape(lut_size, lut_size, lut_size, 3)
    lut_data = _anchor_black_white(lut_data)

    os.makedirs(output_dir, exist_ok=True)
    cube_path = os.path.join(output_dir, "style.cube")
    _write_cube_file(cube_path, lut_data, lut_size)

    analysis = _build_style_analysis(frames, median_frame, cube_path)
    if asset_id:
        storage_key = f"luts/{asset_id}/global.cube"
        try:
            get_storage().put(cube_path, storage_key, content_type="application/vnd.adobe.cube")
            analysis.lut_storage_key = storage_key
        except Exception as e:
            logger.warning("Failed to persist LUT to storage", asset_id=asset_id, error=str(e))
    return cube_path, analysis


def _build_style_analysis(
    frames: list,
    median_frame: np.ndarray,
    cube_path: str,
) -> StyleAnalysis:
    """Compute style metadata from sampled frames."""
    frames_arr = np.stack(frames).reshape(-1, 3).astype(np.float32)
    n_samples = min(100000, len(frames_arr))
    sample_idx = np.linspace(0, len(frames_arr) - 1, n_samples, dtype=int)
    samples = frames_arr[sample_idx]

    mean_color = samples.mean(axis=0)
    std_color = samples.std(axis=0)

    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(samples)
        palette = [
            f"#{int(c[2]):02x}{int(c[1]):02x}{int(c[0]):02x}"
            for c in kmeans.cluster_centers_
        ]
    except Exception:
        logger.warning("Failed to compute color palette")
        palette = []

    saturation = 0.5
    if cv2 is not None:
        median_bgr = median_frame.astype(np.uint8)
        saturation = float(cv2.cvtColor(median_bgr, cv2.COLOR_RGB2HSV)[:, :, 1].mean() / 128.0)

    return StyleAnalysis(
        color_palette=palette,
        contrast_level=float(std_color.mean() / 60.0),
        saturation_level=saturation,
        brightness_level=float(mean_color.mean() / 128.0),
        lut_extracted=True,
        lut_storage_key=cube_path,
    )


def extract_lut_from_reference(
    video_path: str,
    output_dir: str,
    strength: float = 0.5,
    asset_id: Optional[str] = None,
) -> Tuple[Optional[str], StyleAnalysis]:
    """Extract a .cube LUT from reference video and return style analysis.

    Tries HM-MVGD-HM color matching first, falls back to a Reinhard-style
    transfer if color-matcher is unavailable or fails.
    """
    try:
        frames = sample_frames(video_path, n_samples=50)
    except FileNotFoundError:
        logger.warning("Reference video not found, cannot extract LUT", path=video_path)
        return None, StyleAnalysis(lut_extracted=False)

    if len(frames) < 10:
        return None, StyleAnalysis(lut_extracted=False)

    # Preferred: tone-only transfer. It is robust to reference content (white
    # studio backdrops, grey cars, letterboxing) and never desaturates footage.
    # Set AVE_LUT_METHOD=color_matcher to force the old global colour matching.
    if cv2 is not None and os.environ.get("AVE_LUT_METHOD", "tone").lower() != "color_matcher":
        try:
            return _extract_lut_tone_style(frames, output_dir, strength, asset_id=asset_id)
        except Exception as e:
            logger.warning("tone-style LUT extraction failed, falling back", error=str(e))

    if _HAS_COLOR_MATCHER and cv2 is not None:
        try:
            return _extract_lut_color_matcher(frames, output_dir, strength, asset_id=asset_id)
        except Exception as e:
            logger.warning("color-matcher LUT extraction failed, falling back", error=str(e))

    if colour is None:
        return None, StyleAnalysis(lut_extracted=False)

    try:
        return _extract_lut_reinhard(frames, output_dir, strength, asset_id=asset_id)
    except Exception as e:
        logger.warning("Reinhard LUT extraction skipped due to memory limit", error=str(e))
        return None, StyleAnalysis(lut_extracted=False)