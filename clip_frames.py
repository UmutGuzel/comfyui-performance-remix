"""Decode a clip range straight to the size the engine works at.

ComfyUI's Get Video Components decodes the whole range at the file's own size as float32 and stacks it, so the
frames exist twice while it runs. The save node at the end of each workflow then reads that node's audio and frame
rate, so the full-size frames stay in memory for the whole run: about 18 GB for 30 s of 1080x1920 at 24 fps, plus as
much again while stacking. This node decodes one frame at a time, scales it to the working size as it goes and also
returns the audio and the frame rate, so no full-size copy of the clip is ever held.
"""
from fractions import Fraction

import av
import numpy as np

SIZE_MODES = ["shorter side", "width x height"]


def scaled_size(src_w, src_h, mode, shorter_side, width, height):
    """(scaled w, scaled h, output w, output h) for a frame of src_w x src_h.

    "shorter side" scales the shorter side down to shorter_side and never up (the workflow's own resize nodes
    follow); "width x height" scales to cover width x height, up or down, and the frame is then cut to it in the
    center.
    """
    if src_w < 1 or src_h < 1:
        raise ValueError("the clip has no frame size")
    if mode == "shorter side":
        scale = min(1.0, shorter_side / min(src_w, src_h))
        w, h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
        return w, h, w, h
    if mode == "width x height":
        scale = max(width / src_w, height / src_h)
        return max(width, round(src_w * scale)), max(height, round(src_h * scale)), width, height
    raise ValueError(f"unknown size mode {mode!r}")


def _quadrant(frame):
    rotation = getattr(frame, "rotation", 0) or 0
    return int(round(rotation // 90)) % 4


def decode_frames(source, start, duration, mode, shorter_side=720, width=704, height=1248):
    """The frames of [start, start + duration) as uint8 RGB arrays at the working size, and the frame rate.

    duration 0 reads to the end. Frames are scaled one at a time (area filter going down, bicubic going up).
    """
    if hasattr(source, "seek"):
        source.seek(0)
    end = start + duration if duration > 0 else float("inf")
    frames = []
    with av.open(source, mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = Fraction(stream.average_rate) if stream.average_rate else Fraction(30)
        if start > 0:
            container.seek(int(start / stream.time_base), stream=stream)
        size = None
        for frame in container.decode(stream):
            t = float(frame.time) if frame.time is not None else len(frames) / float(rate)
            if t < start - 1e-6:
                continue
            if t >= end - 1e-6:
                break
            quadrant = _quadrant(frame)
            if size is None:
                shown_w, shown_h = (frame.height, frame.width) if quadrant % 2 else (frame.width, frame.height)
                size = scaled_size(shown_w, shown_h, mode, shorter_side, width, height)
                interpolation = "AREA" if size[0] <= shown_w else "BICUBIC"
            sw, sh, ow, oh = size
            coded_w, coded_h = (sh, sw) if quadrant % 2 else (sw, sh)
            img = frame.reformat(width=coded_w, height=coded_h, format="rgb24", interpolation=interpolation).to_ndarray()
            if quadrant:
                img = np.rot90(img, k=quadrant, axes=(0, 1))
            top, left = (sh - oh) // 2, (sw - ow) // 2
            frames.append(np.ascontiguousarray(img[top:top + oh, left:left + ow]))
    if not frames:
        raise ValueError("no video frames in the selected range")
    return frames, rate


def read_audio(source, start, end):
    """Source audio for [start, end) as (planar float samples, sample rate, layout), or None without audio."""
    if hasattr(source, "seek"):
        source.seek(0)
    with av.open(source, mode="r") as container:
        if not container.streams.audio:
            return None
        astream = container.streams.audio[0]
        to_float = av.audio.resampler.AudioResampler(format="fltp")
        chunks, rate, layout = [], None, None
        for frame in container.decode(astream):
            rate = rate or frame.sample_rate
            layout = layout or frame.layout.name
            chunks.extend(f.to_ndarray() for f in to_float.resample(frame))
        chunks.extend(f.to_ndarray() for f in to_float.resample(None))
        if not chunks:
            return None
        samples = np.concatenate(chunks, axis=1).astype(np.float32)
    first = int(round(start * rate))
    last = samples.shape[1] if end == float("inf") else min(samples.shape[1], int(round(end * rate)))
    samples = np.ascontiguousarray(samples[:, first:last])
    if samples.shape[1] == 0:
        return None
    layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(samples.shape[0], layout)
    return samples, int(rate), layout


class ClipFrames:
    CATEGORY = "performance-remix"
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("images", "audio", "fps", "width", "height")
    FUNCTION = "run"
    DESCRIPTION = ("Get Video Components at the working size: decodes the clip range one frame at a time and scales each "
                   "frame as it goes, so long clips never sit in memory at full size. Same first three outputs as Get "
                   "Video Components.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "size": (SIZE_MODES, {"default": "shorter side", "tooltip": "shorter side: scale down only, keep the shape. "
                                      "width x height: scale to cover and cut to exactly that size in the center."}),
                "shorter_side": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 8}),
                "width": ("INT", {"default": 704, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1248, "min": 64, "max": 4096, "step": 8}),
            },
        }

    def run(self, video, size, shorter_side, width, height):
        import torch

        source = video.get_stream_source()
        start, duration = video.get_active_trim_window() if hasattr(video, "get_active_trim_window") else (0.0, 0.0)
        frames, rate = decode_frames(source, start, duration, size, shorter_side, width, height)
        h, w = frames[0].shape[:2]
        images = torch.empty((len(frames), h, w, 3), dtype=torch.float32)
        for i in range(0, len(frames), 64):  # convert in chunks and let go of the uint8 frames as they are copied
            images[i:i + 64] = torch.from_numpy(np.stack(frames[i:i + 64])).float().div_(255.0)
            frames[i:i + 64] = [None] * len(frames[i:i + 64])
        end = start + duration if duration > 0 else float("inf")
        found = read_audio(source, start, end)
        audio = None
        if found is not None:
            samples, sample_rate, _ = found
            audio = {"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": sample_rate}
        return (images, audio, float(rate), w, h)


NODE_CLASS_MAPPINGS = {"PerformanceRemixClipFrames": ClipFrames}
NODE_DISPLAY_NAME_MAPPINGS = {"PerformanceRemixClipFrames": "Clip Frames at Working Size"}
