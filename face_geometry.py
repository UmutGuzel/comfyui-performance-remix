"""Face box tracking math for the face pass. Only numpy, so it is testable without ComfyUI or OpenCV."""

import numpy as np


def fill_missing(detections):
    """Turn per-frame value tuples or None into an (N, K) array, filling gaps by linear interpolation.

    Frames before the first or after the last known value take the nearest known value.
    Raises ValueError when no frame has a value.
    """
    found = [i for i, d in enumerate(detections) if d is not None]
    if not found:
        raise ValueError("no face found in any frame")
    known = np.array([detections[i] for i in found], dtype=np.float64)
    frames = np.arange(len(detections))
    return np.stack([np.interp(frames, found, known[:, k]) for k in range(known.shape[1])], axis=1)


def decompose_similarity(matrix):
    """Split a 2x3 similarity transform into (scale, angle in radians, tx, ty)."""
    a, b = matrix[0, 0], matrix[1, 0]
    return float(np.hypot(a, b)), float(np.arctan2(b, a)), float(matrix[0, 2]), float(matrix[1, 2])


def compose_similarity(scale, angle, tx, ty):
    """Build the 2x3 similarity transform for (scale, angle, tx, ty)."""
    c, s = scale * np.cos(angle), scale * np.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)


def smooth_series(values, window):
    """Fill gaps (None) by interpolation, then smooth over time. Returns an (N, K) array."""
    return smooth(fill_missing(values), window)


def smooth(values, window):
    """Centered moving average over time; the ends are padded with the edge values."""
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be an odd number of at least 1")
    if window == 1 or len(values) == 1:
        return np.array(values, dtype=np.float64)
    half = window // 2
    padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(values.shape[1])], axis=1)


def square_boxes(track, frame_width, frame_height, scale):
    """One fixed-size square crop per frame, centered on the face and kept inside the frame.

    The side is the median face size times `scale`, capped at the shorter frame side, so the
    crop does not change size from frame to frame. Returns a list of [x1, y1, side] integers.
    """
    side = float(np.median(track[:, 2]) * scale)
    side = min(side, frame_width, frame_height)
    boxes = []
    for cx, cy, _ in track:
        x1 = min(max(cx - side / 2, 0.0), frame_width - side)
        y1 = min(max(cy - side / 2, 0.0), frame_height - side)
        boxes.append([int(round(x1)), int(round(y1)), int(round(side))])
    return boxes
