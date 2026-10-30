"""Face pass: re-render only the face at high resolution, then put it back.

In a full-body 480p shot the face is about 60 pixels wide, too small for the video model
to keep the reference's identity. "Face Track and Crop" follows the face through the
frames and cuts a fixed-size square close-up around it; the close-ups are re-rendered with
the reference face (with the Long Motion Transfer node); "Face Paste" aligns each new face
to the original landmarks, matches its color and blends it back with a soft oval.

Face detection uses OpenCV's YuNet (MIT license), bundled in models/.
"""

import logging
import os

import cv2
import numpy as np
import torch

import comfy.utils

from .face_geometry import (compose_similarity, decompose_similarity, fill_missing, smooth,
                            smooth_series, square_boxes)

log = logging.getLogger("performance-remix")
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_detection_yunet_2023mar.onnx")
_detector = None


def _detect(bgr, min_score):
    """Largest face in a BGR uint8 image as (cx, cy, size, landmarks 5x2), or None."""
    global _detector
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(MODEL, "", (320, 320), min_score, 0.3, 50)
    height, width = bgr.shape[:2]
    _detector.setInputSize((width, height))
    _detector.setScoreThreshold(min_score)
    _, faces = _detector.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    face = faces[np.argmax(faces[:, 2] * faces[:, 3])]
    x, y, w, h = face[:4]
    return x + w / 2, y + h / 2, max(w, h), face[4:14].reshape(5, 2).astype(np.float32)


def _to_bgr(image):
    return cv2.cvtColor((image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _to_tensor(frames):
    stacked = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]).astype(np.float32) / 255.0
    return torch.from_numpy(stacked)


def _lab_stats(image, weight):
    """Per-channel LAB mean and spread where weight > 0.5, as six numbers, or None if the area is tiny."""
    inside = weight > 0.5
    if inside.sum() < 50:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    return tuple([float(lab[..., c][inside].mean()) for c in range(3)] + [float(lab[..., c][inside].std() + 1e-3) for c in range(3)])


def _apply_lab(image, source_stats, target_stats):
    """Move image from source_stats to target_stats, channel by channel, in LAB."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        lab[..., c] = (lab[..., c] - source_stats[c]) * (target_stats[c + 3] / source_stats[c + 3]) + target_stats[c]
    return cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _oval(size, cx, cy, face, oval_width, oval_height, feather):
    weight = np.zeros((size, size), np.float32)
    cv2.ellipse(weight, (int(round(cx)), int(round(cy + face * 0.05))), (max(1, int(face * oval_width)), max(1, int(face * oval_height))), 0, 0, 360, 1.0, -1)
    if feather > 0:
        weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=max(1.0, size * feather))
    return weight


class FaceTrackCrop:
    CATEGORY = "performance-remix"
    RETURN_TYPES = ("IMAGE", "PR_FACE_TRACK", "INT")
    RETURN_NAMES = ("face_crops", "face_track", "frames_with_face")
    FUNCTION = "run"
    DESCRIPTION = "Follows the largest face through the frames and cuts a fixed-size, time-smoothed square close-up around it."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "crop_size": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 16, "tooltip": "Side of the output close-ups in pixels."}),
            "box_scale": ("FLOAT", {"default": 2.4, "min": 1.2, "max": 4.0, "step": 0.1, "tooltip": "Crop side as a multiple of the median face size."}),
            "smooth_frames": ("INT", {"default": 7, "min": 1, "max": 31, "step": 2, "tooltip": "Moving-average window over time; odd."}),
            "min_score": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 0.99, "step": 0.01, "tooltip": "Detector confidence needed to count a face."}),
        }}

    def run(self, images, crop_size, box_scale, smooth_frames, min_score):
        frames = [_to_bgr(image) for image in images]
        height, width = frames[0].shape[:2]
        found = [_detect(f, min_score) for f in frames]
        detected = sum(d is not None for d in found)
        try:
            track = smooth(fill_missing([None if d is None else d[:3] for d in found]), smooth_frames)
        except ValueError:
            log.warning("Face Track and Crop: no face in %d frames; face pass will pass frames through", len(frames))
            side = min(width, height)
            center = [cv2.resize(f[(height - side) // 2:(height + side) // 2, (width - side) // 2:(width + side) // 2], (crop_size, crop_size)) for f in frames]
            return (_to_tensor(center), {"boxes": [], "frame_size": [width, height], "crop_size": crop_size}, 0)
        boxes = square_boxes(track, width, height, box_scale)
        crops = [cv2.resize(f[y:y + s, x:x + s], (crop_size, crop_size), interpolation=cv2.INTER_LANCZOS4) for f, (x, y, s) in zip(frames, boxes)]
        log.info("Face Track and Crop: face in %d of %d frames, crop %d px -> %d px", detected, len(frames), boxes[0][2], crop_size)
        return (_to_tensor(crops), {"boxes": boxes, "frame_size": [width, height], "crop_size": crop_size}, detected)


class FacePaste:
    CATEGORY = "performance-remix"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    DESCRIPTION = "Aligns each re-rendered face to the original face landmarks, matches its color and blends it back with a soft oval."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The original frames."}),
            "face_images": ("IMAGE", {"tooltip": "The re-rendered close-ups, one per frame."}),
            "face_track": ("PR_FACE_TRACK",),
            "oval_width": ("FLOAT", {"default": 0.5, "min": 0.2, "max": 1.0, "step": 0.01, "tooltip": "Half-width of the blend oval as a fraction of the face box."}),
            "oval_height": ("FLOAT", {"default": 0.6, "min": 0.2, "max": 1.2, "step": 0.01, "tooltip": "Half-height of the blend oval as a fraction of the face box."}),
            "feather": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.2, "step": 0.005, "tooltip": "Blur of the oval edge as a fraction of the crop side."}),
            "color_match": ("BOOLEAN", {"default": True}),
            "min_score": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 0.99, "step": 0.01}),
        }, "optional": {
            "smooth_frames": ("INT", {"default": 3, "min": 1, "max": 31, "step": 2, "tooltip": "Frames over which alignment, oval and color are averaged; 1 uses each frame's own values. Larger values lag fast head moves."}),
        }}

    def run(self, images, face_images, face_track, oval_width, oval_height, feather, color_match, min_score, smooth_frames=3):
        boxes = face_track["boxes"]
        if not boxes:
            log.warning("Face Paste: empty face track, frames passed through")
            return (images,)
        count = min(len(images), len(face_images), len(boxes))
        if count < len(images):
            log.warning("Face Paste: %d frames, %d faces, %d boxes; pasting the first %d", len(images), len(face_images), len(boxes), count)
        frames = [_to_bgr(images[i]) for i in range(len(images))]
        # Pass 1: measure each frame's alignment and oval.
        crops, faces, transforms, ovals = [], [], [], []
        for i in range(count):
            x, y, s = boxes[i]
            crop = frames[i][y:y + s, x:x + s].copy()
            face = cv2.resize(_to_bgr(face_images[i]), (s, s), interpolation=cv2.INTER_AREA)
            old, new = _detect(crop, min_score), _detect(face, min_score)
            matrix = cv2.estimateAffinePartial2D(new[3], old[3])[0] if old is not None and new is not None else None
            transforms.append(decompose_similarity(matrix) if matrix is not None else None)
            ovals.append(old[:3] if old is not None else None)
            crops.append(crop); faces.append(face)
        aligned = sum(t is not None for t in transforms)
        side = boxes[0][2]
        # Pass 2: smooth over time, filling frames without a measurement from their neighbours.
        smooth_t = smooth_series(transforms, smooth_frames) if aligned else np.tile([1.0, 0.0, 0.0, 0.0], (count, 1))
        smooth_o = smooth_series(ovals, smooth_frames) if any(o is not None for o in ovals) else np.tile([side / 2, side / 2, side / 2.4], (count, 1))
        warped, weights, src_stats, dst_stats = [], [], [], []
        for i in range(count):
            s = boxes[i][2]
            face = cv2.warpAffine(faces[i], compose_similarity(*smooth_t[i]), (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            weight = _oval(s, *smooth_o[i], oval_width, oval_height, feather)
            warped.append(face); weights.append(weight)
            src_stats.append(_lab_stats(face, weight)); dst_stats.append(_lab_stats(crops[i], weight))
        if color_match and any(v is not None for v in src_stats) and any(v is not None for v in dst_stats):
            src_stats, dst_stats = smooth_series(src_stats, smooth_frames), smooth_series(dst_stats, smooth_frames)
        else:
            color_match = False
        # Pass 3: color-match and blend.
        pbar = comfy.utils.ProgressBar(count)
        for i in range(count):
            x, y, s = boxes[i]
            face = _apply_lab(warped[i], src_stats[i], dst_stats[i]) if color_match else warped[i]
            w = weights[i][..., None]
            frames[i][y:y + s, x:x + s] = (crops[i] * (1 - w) + face * w).astype(np.uint8)
            pbar.update(1)
        log.info("Face Paste: %d frames pasted, %d aligned by landmarks, smoothed over %d frames", count, aligned, smooth_frames)
        return (_to_tensor(frames),)


NODE_CLASS_MAPPINGS = {"PerformanceRemixFaceTrackCrop": FaceTrackCrop, "PerformanceRemixFacePaste": FacePaste}
NODE_DISPLAY_NAME_MAPPINGS = {"PerformanceRemixFaceTrackCrop": "Face Track and Crop", "PerformanceRemixFacePaste": "Face Paste"}
