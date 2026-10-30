"""Streaming finish: SeedVR2 upscale, light grade and H.264 export, one segment at a time.

The graph version of the finish keeps every frame of the clip in system RAM several times
at the output size, so long clips run out of RAM. This node reads the
video a segment at a time, runs the same SeedVR2 steps as the official template on that
segment, crossfades the frames it shares with the previous segment, and hands the frames
straight to the encoder. Memory depends on `segment_frames`, not on the clip length.
"""

import logging
import os
import time
from fractions import Fraction

import av
import numpy as np
import torch

import comfy.utils
import folder_paths
import nodes
from comfy_api.latest import InputImpl
from comfy_api.latest._input_impl.video_types import BT709_NCL, set_video_color_properties
from comfy_extras.nodes_post_processing import Sharpen
from comfy_extras.nodes_seedvr import SeedVR2Conditioning, SeedVR2PostProcessing, SeedVR2Preprocess

from .clip_frames import read_audio

log = logging.getLogger("performance-remix")
try:
    import psutil
except ImportError:  # only used for logging memory
    psutil = None


def _rss_gb():
    return psutil.Process().memory_info().rss / 1e9 if psutil else float("nan")


class StreamingFinish:
    CATEGORY = "performance-remix"
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    DESCRIPTION = ("SeedVR2 upscale, light grade and H.264 export of a whole video, processed one segment at a time "
                   "so memory does not grow with the clip length. Keeps the source audio.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "model": ("MODEL", {"tooltip": "SeedVR2 model."}),
            "vae": ("VAE", {"tooltip": "SeedVR2 VAE."}),
            "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05}),
            "segment_frames": ("INT", {"default": 49, "min": 9, "max": 241, "tooltip": "Frames processed at once. Memory grows with this, not with the clip length."}),
            "overlap": ("INT", {"default": 4, "min": 0, "max": 32, "tooltip": "Frames shared by neighbouring segments and crossfaded."}),
            "color_correction": (["lab", "wavelet", "adain", "none"], {"default": "lab"}),
            "sharpen": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Unsharp strength after the upscale; 0 turns it off."}),
            "contrast": ("FLOAT", {"default": 1.03, "min": 0.5, "max": 1.5, "step": 0.01, "tooltip": "1.0 leaves contrast unchanged."}),
            "crf": ("INT", {"default": 16, "min": 0, "max": 51, "tooltip": "H.264 quality; lower is better and larger."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            "filename_prefix": ("STRING", {"default": "performance-remix/finish"}),
        }}

    def _process(self, frames, model, vae, scale, color_correction, sharpen, contrast, seed):
        """SeedVR2 on one segment, as in the official template: resize, pad, encode, 1 step, decode, color match."""
        images = torch.from_numpy(np.stack(frames)).float().div_(255.0)
        height, width = images.shape[1], images.shape[2]
        resized = comfy.utils.common_upscale(images.movedim(-1, 1), int(round(width * scale)), int(round(height * scale)), "lanczos", "disabled").movedim(1, -1)
        padded = SeedVR2Preprocess.execute(resized).args[0]
        latent = nodes.VAEEncodeTiled().encode(vae, padded, 512, 128, 64, 8)[0]
        positive, negative = SeedVR2Conditioning.execute(model, latent).args
        sampled = nodes.common_ksampler(model, seed, 1, 1.0, "euler", "simple", positive, negative, latent, denoise=1.0)[0]
        decoded = nodes.VAEDecodeTiled().decode(vae, sampled, 512, 128, 64, 8)[0]
        out = SeedVR2PostProcessing.execute(decoded, resized, color_correction).args[0]
        if out.dim() == 5:
            out = out.reshape(-1, *out.shape[-3:])
        if sharpen > 0:
            out = Sharpen.execute(out, 1, 1.0, sharpen).args[0]
        if contrast != 1.0:
            out = ((out - 0.5) * contrast + 0.5).clamp(0.0, 1.0)
        return (out[..., :3].clamp(0, 1).mul(255.0).round().to(torch.uint8).cpu().numpy())

    def run(self, video, model, vae, scale, segment_frames, overlap, color_correction, sharpen, contrast, crf, seed, filename_prefix):
        if overlap >= segment_frames:
            raise ValueError("overlap must be smaller than segment_frames")
        source = video.get_stream_source()
        start, duration = video.get_active_trim_window() if hasattr(video, "get_active_trim_window") else (0.0, 0.0)
        end = start + duration if duration > 0 else float("inf")
        out_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        out_path = os.path.join(out_dir, f"{name}_{counter:05}_.mp4")

        audio = read_audio(source, start, end)
        state = {"out": None, "stream": None, "audio": None, "written": 0, "held": None, "segments": 0}
        t0 = time.time()

        def write(frames):
            if state["out"] is None:
                # Every stream must exist before the first packet is written.
                h, w = frames.shape[1], frames.shape[2]
                state["out"] = av.open(out_path, mode="w")
                stream = state["out"].add_stream("libx264", rate=rate)
                stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
                stream.options = {"crf": str(crf), "preset": "slow"}
                set_video_color_properties(stream.codec_context, "sRGB")
                state["stream"] = stream
                if audio is not None:
                    state["audio"] = state["out"].add_stream("aac", rate=audio[1], layout=audio[2])
            for f in frames:
                vf = av.VideoFrame.from_ndarray(f, format="rgb24").reformat(format="yuv420p", dst_colorspace=BT709_NCL)
                set_video_color_properties(vf, "sRGB")
                vf.pts = state["written"]; state["written"] += 1
                for packet in state["stream"].encode(vf):
                    state["out"].mux(packet)

        def flush(buffer, final):
            seg_t = time.time()
            out = self._process(buffer, model, vae, scale, color_correction, sharpen, contrast, seed)
            held = state["held"]
            if held is not None:
                n = len(held)
                w = (np.arange(1, n + 1, dtype=np.float32) / (n + 1))[:, None, None, None]
                out[:n] = (held.astype(np.float32) * (1 - w) + out[:n].astype(np.float32) * w).round().astype(np.uint8)
            if final or overlap == 0:
                write(out); state["held"] = None
            else:
                write(out[:-overlap]); state["held"] = out[-overlap:].copy()
            state["segments"] += 1
            log.info("Streaming finish: segment %d, %d frames in %.1f s, process RAM %.1f GB", state["segments"], len(buffer), time.time() - seg_t, _rss_gb())

        try:
            if hasattr(source, "seek"):
                source.seek(0)
            with av.open(source, mode="r") as container:
                vstream = container.streams.video[0]
                rate = vstream.average_rate or Fraction(30, 1)
                buffer, seen = [], 0
                for frame in container.decode(vstream):
                    t = float(frame.time) if frame.time is not None else seen / float(rate)
                    if t < start - 1e-6:
                        continue
                    if t >= end - 1e-6:
                        break
                    buffer.append(frame.to_ndarray(format="rgb24")); seen += 1
                    if len(buffer) == segment_frames:
                        flush(buffer, final=False)
                        buffer = buffer[-overlap:] if overlap else []
                if seen == 0:
                    raise ValueError("no video frames in the selected range")
                if state["segments"] == 0 or len(buffer) > overlap:
                    flush(buffer, final=True)
                elif state["held"] is not None:
                    write(state["held"]); state["held"] = None
            for packet in state["stream"].encode(None):
                state["out"].mux(packet)
            audio_seconds = 0.0
            if state["audio"] is not None:
                samples, sample_rate, layout = audio
                frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
                frame.sample_rate, frame.pts = sample_rate, 0
                state["out"].mux(state["audio"].encode(frame))
                state["out"].mux(state["audio"].encode(None))
                audio_seconds = samples.shape[1] / sample_rate
        finally:
            if state["out"] is not None:
                state["out"].close()
        log.info("Streaming finish: %d frames, %d segments, audio %.2f s, %.1f s total -> %s", state["written"], state["segments"], audio_seconds, time.time() - t0, out_path)
        result = InputImpl.VideoFromFile(out_path)
        return {"ui": {"images": [{"filename": os.path.basename(out_path), "subfolder": subfolder, "type": "output"}], "animated": (True,)}, "result": (result,)}


NODE_CLASS_MAPPINGS = {"PerformanceRemixStreamingFinish": StreamingFinish}
NODE_DISPLAY_NAME_MAPPINGS = {"PerformanceRemixStreamingFinish": "Streaming Finish (SeedVR2)"}
