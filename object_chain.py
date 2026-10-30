"""Object swaps of any length: short clips padded to what LTX-2.5 handles, long clips joined from 5-second windows.

LTX-2.5 makes 8k+1 frames, and its inpainting lost the new object partway on 2 s and 3 s clips (89 frames and fewer
at 30 fps) while every 145-frame run kept it. Pad Frames holds the last frame until the clip has at least min_frames
frames and a count LTX makes; the workflow cuts the result back to the clip's own count, so the output is exactly as
long as the input. The app renders longer clips in windows of 5 s, each starting from the frame the window before it
redrew at that time (Frame at Time). Join Object Windows puts the windows on the clip's own frame times, crossfades
each join and adds the clip's audio, writing frames straight to the encoder so memory does not grow with the length.
"""
import logging
import math
import os
import time
from fractions import Fraction

import av
import numpy as np

log = logging.getLogger("performance-remix")
MAX_WINDOWS = 8


def padded_length(frames, min_frames):
    """The smallest 8k+1 frame count that is at least frames and min_frames."""
    target = max(int(frames), int(min_frames), 1)
    return (target - 1 + 7) // 8 * 8 + 1


def frame_index_at(seconds, fps):
    """Index of the first frame at or after `seconds` into a clip that starts at frame 0."""
    return max(0, math.ceil(seconds * fps - 1e-6))


def parse_starts(text):
    starts = [float(s) for s in str(text).replace(";", ",").split(",") if s.strip()]
    if not starts or starts[0] != 0 or any(b <= a for a, b in zip(starts, starts[1:])):
        raise ValueError(f"window starts must begin at 0 and increase: {text!r}")
    return starts


def window_offsets(times, starts):
    """For each window start (seconds, on the clock of `times`), the index of the first frame at or after it."""
    return [sum(1 for t in times if t < s - 1e-6) for s in starts]


def join_plan(offsets, counts, total, crossfade):
    """What every output frame is made of: a list, per frame, of (window, local index, weight of that window).

    Window k covers frames [offsets[k], offsets[k] + counts[k]). Where window k + 1 starts, both windows are blended
    over up to `crossfade` frames (window k + 1 starts from window k's frame there, so the blend starts identical);
    after that window k + 1 alone is used. A frame no window covers repeats the frame before it.
    """
    n = len(offsets)
    if n != len(counts) or n == 0:
        raise ValueError("one frame count per window is needed")
    plan = []
    for g in range(total):
        # the last window that has started by frame g and still covers it
        k = max((i for i in range(n) if offsets[i] <= g < offsets[i] + counts[i]), default=None)
        if k is None:
            plan.append(list(plan[-1]) if plan else [])
            continue
        if k > 0 and g - offsets[k] < crossfade and offsets[k - 1] + counts[k - 1] > g:
            blend = min(crossfade, offsets[k - 1] + counts[k - 1] - offsets[k])
            w = (g - offsets[k] + 1) / (blend + 1)
            plan.append([(k - 1, g - offsets[k - 1], 1 - w), (k, g - offsets[k], w)])
        else:
            plan.append([(k, g - offsets[k], 1.0)])
    if not plan or not plan[0]:
        raise ValueError("the first window must start at the first frame")
    return plan


def frame_times(source, start, duration):
    """The times of the frames in [start, start + duration), by the same rule as clip_frames.decode_frames."""
    if hasattr(source, "seek"):
        source.seek(0)
    end = start + duration if duration > 0 else float("inf")
    times = []
    with av.open(source, mode="r") as container:
        stream = container.streams.video[0]
        rate = Fraction(stream.average_rate) if stream.average_rate else Fraction(30)
        if start > 0:
            container.seek(int(start / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            t = float(frame.time) if frame.time is not None else len(times) / float(rate)
            if t < start - 1e-6:
                continue
            if t >= end - 1e-6:
                break
            times.append(t)
    return times, rate


class PadFrames:
    CATEGORY = "performance-remix"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frames")
    FUNCTION = "run"
    DESCRIPTION = ("Holds the last frame until the clip has at least min_frames frames and a count LTX makes (8k+1). "
                   "frames is the clip's own count, to cut the result back to.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "min_frames": ("INT", {"default": 145, "min": 1, "max": 4097, "step": 8,
                                   "tooltip": "shorter clips lost the new object partway in tests; 145 is 5 s at 29 fps"}),
        }}

    def run(self, images, min_frames):
        import torch

        n = images.shape[0]
        extra = padded_length(n, min_frames) - n
        if extra:
            images = torch.cat([images, images[-1:].expand(extra, -1, -1, -1)], dim=0)
        return (images, n)


class FrameAt:
    CATEGORY = "performance-remix"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    DESCRIPTION = "The frame at a time into the clip: the first one at or after `seconds`, the last frame past the end."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0}),
            "seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.001}),
        }}

    def run(self, images, fps, seconds):
        i = min(images.shape[0] - 1, frame_index_at(seconds, fps))
        return (images[i:i + 1].clone(),)


class JoinWindows:
    CATEGORY = "performance-remix"
    RETURN_TYPES = ("VIDEO",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    DESCRIPTION = ("Joins object-swap windows into one video on the clip's own frame times, with a crossfade at each "
                   "join and the clip's audio. Frames go straight to the encoder.")

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"window_{i}": ("VIDEO",) for i in range(1, MAX_WINDOWS + 1)}
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "the clip range the windows were made from (Load Video, Video Slice)"}),
                "starts": ("STRING", {"default": "0", "tooltip": "each window's start in seconds from the range start, "
                                      "comma separated, in the order of window_1, window_2, ..."}),
                "crossfade_frames": ("INT", {"default": 8, "min": 0, "max": 60}),
                "crf": ("INT", {"default": 16, "min": 0, "max": 51}),
                "filename_prefix": ("STRING", {"default": "performance-remix/motion_object"}),
            },
            "optional": optional,
        }

    def run(self, video, starts, crossfade_frames, crf, filename_prefix, **windows):
        import folder_paths
        from comfy_api.latest import InputImpl
        from comfy_api.latest._input_impl.video_types import BT709_NCL, set_video_color_properties

        from .clip_frames import read_audio

        rel = parse_starts(starts)
        clips = [windows.get(f"window_{i}") for i in range(1, len(rel) + 1)]
        if any(c is None for c in clips):
            raise ValueError(f"{len(rel)} window starts need window_1 to window_{len(rel)}")
        source = video.get_stream_source()
        start, duration = video.get_active_trim_window() if hasattr(video, "get_active_trim_window") else (0.0, 0.0)
        times, rate = frame_times(source, start, duration)
        offsets = window_offsets(times, [start + s for s in rel])
        counts = [len(frame_times(c.get_stream_source(), 0.0, 0.0)[0]) for c in clips]
        plan = join_plan(offsets, counts, len(times), crossfade_frames)
        log.info("Join object windows: %d frames from %d windows at %s, counts %s", len(times), len(clips), offsets, counts)

        out_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        out_path = os.path.join(out_dir, f"{name}_{counter:05}_.mp4")
        audio = read_audio(source, start, start + duration if duration > 0 else float("inf"))
        t0 = time.time()
        out = av.open(out_path, mode="w")
        try:
            stream, astream, size, written = None, None, None, 0
            held = {}  # output frame -> [(weight, frame)] waiting for the next window
            needed = {}  # (window, local index) -> output frames that use it
            for g, parts in enumerate(plan):
                for k, i, w in parts:
                    needed.setdefault((k, i), []).append((g, w))
            for k, clip in enumerate(clips):
                src = clip.get_stream_source()
                if hasattr(src, "seek"):
                    src.seek(0)
                with av.open(src, mode="r") as container:
                    for i, frame in enumerate(container.decode(container.streams.video[0])):
                        uses = needed.get((k, i))
                        if not uses:
                            continue
                        img = frame.to_ndarray(format="rgb24")
                        if size is None:
                            size = (img.shape[1], img.shape[0])
                            stream = out.add_stream("libx264", rate=rate)
                            stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
                            stream.options = {"crf": str(crf), "preset": "slow"}
                            set_video_color_properties(stream.codec_context, "sRGB")
                            if audio is not None:
                                astream = out.add_stream("aac", rate=audio[1], layout=audio[2])
                        elif (img.shape[1], img.shape[0]) != size:
                            img = np.asarray(av.VideoFrame.from_ndarray(img, format="rgb24").reformat(width=size[0], height=size[1]).to_ndarray(format="rgb24"))
                        for g, w in uses:
                            held.setdefault(g, []).append((w, img))
                        # write every frame whose parts have all arrived, in order
                        while written in held and len(held[written]) == len(plan[written]):
                            parts = held.pop(written)
                            mixed = parts[0][1] if len(parts) == 1 else sum(wt * f.astype(np.float32) for wt, f in parts).round().clip(0, 255).astype(np.uint8)
                            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(mixed), format="rgb24").reformat(format="yuv420p", dst_colorspace=BT709_NCL)
                            set_video_color_properties(vf, "sRGB")
                            vf.pts = written
                            for packet in stream.encode(vf):
                                out.mux(packet)
                            written += 1
            if written != len(plan):
                raise ValueError(f"the windows cover {written} of {len(plan)} frames")
            for packet in stream.encode(None):
                out.mux(packet)
            if astream is not None:
                samples, sample_rate, layout = audio
                af = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
                af.sample_rate, af.pts = sample_rate, 0
                out.mux(astream.encode(af))
                out.mux(astream.encode(None))
        finally:
            out.close()
        log.info("Join object windows: %d frames in %.1f s -> %s", written, time.time() - t0, out_path)
        result = InputImpl.VideoFromFile(out_path)
        return {"ui": {"images": [{"filename": os.path.basename(out_path), "subfolder": subfolder, "type": "output"}], "animated": (True,)}, "result": (result,)}


NODE_CLASS_MAPPINGS = {
    "PerformanceRemixPadFrames": PadFrames,
    "PerformanceRemixFrameAt": FrameAt,
    "PerformanceRemixJoinWindows": JoinWindows,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PerformanceRemixPadFrames": "Pad Frames for LTX",
    "PerformanceRemixFrameAt": "Frame at Time",
    "PerformanceRemixJoinWindows": "Join Object Windows",
}
