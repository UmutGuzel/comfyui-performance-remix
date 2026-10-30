"""Long motion transfer with SCAIL-2 in one node.

SCAIL-2 renders 81 frames per pass from a pose video, a colored SAM 3 mask video of the
person and a reference image. The official template makes longer clips with "Extend"
segments added by hand: each one reads the pose video 76 frames further on and anchors
on the last 5 decoded frames of the segment before. This node runs that chain for the
whole clip in one run through ComfyUI's WanSCAILToVideo, with the template's sampling
(SamplerCustom, the denoised output is decoded). The sampler and sigmas come in from the
graph, because the template builds its schedule from the model before the shift patch.
"""

import logging
import time

import torch

import comfy.model_management
import comfy.utils
import nodes
from comfy_extras.nodes_custom_sampler import SamplerCustom
from comfy_extras.nodes_scail import WanSCAILToVideo

from .pieces import piece_starts, plan_pieces

log = logging.getLogger("performance-remix")


def _pad(frames, length):
    if frames.shape[0] >= length:
        return frames[:length]
    return torch.cat([frames, frames[-1:].expand(length - frames.shape[0], -1, -1, -1)], dim=0)


class SCAIL2LongMotionTransfer:
    CATEGORY = "performance-remix"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "pieces")
    FUNCTION = "run"
    DESCRIPTION = (
        "Renders a whole pose video with SCAIL-2 as a chain of pieces in one run; "
        "each piece anchors on the last frames of the one before."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "SCAIL-2 model, already patched (LoRAs, sampling shift)."}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP vision features of the approved still."}),
                "reference_image": ("IMAGE", {"tooltip": "The approved still: who to animate."}),
                "reference_image_mask": ("IMAGE", {"tooltip": "Colored mask of the person in the still, from Create SCAIL-2 Colored Mask."}),
                "pose_video": ("IMAGE", {"tooltip": "The whole clip at the output size (multiples of 32)."}),
                "pose_video_mask": ("IMAGE", {"tooltip": "Colored mask video of the person in the clip, same length and size as pose_video."}),
                "replacement_mode": ("BOOLEAN", {"default": True, "tooltip": "True keeps the clip's background and replaces the person; false animates the still."}),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS", {"tooltip": "The template: BasicScheduler, simple, 6 steps with the lightx2v LoRA, on the model before the shift patch."}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "1 with the lightx2v LoRA, 5 without."}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "piece_length": ("INT", {"default": 81, "min": 9, "max": 161, "step": 4, "tooltip": "Frames per piece, 4n + 1; the model was trained on 81."}),
                "overlap_frames": ("INT", {"default": 5, "min": 1, "max": 33, "step": 4, "tooltip": "Frames each piece anchors on from the one before, 4n + 1; the model was trained on 5."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "0 renders the whole video."}),
            },
        }

    def run(self, model, positive, negative, vae, clip_vision_output, reference_image, reference_image_mask, pose_video,
            pose_video_mask, replacement_mode, sampler, sigmas, noise_seed, cfg, pose_strength, piece_length,
            overlap_frames, max_frames):
        if (overlap_frames - 1) % 4 != 0:
            raise ValueError("overlap_frames must be 4n + 1")
        if pose_video_mask.shape[0] != pose_video.shape[0]:
            raise ValueError(f"pose_video has {pose_video.shape[0]} frames but pose_video_mask has {pose_video_mask.shape[0]}")
        total = pose_video.shape[0] if max_frames == 0 else min(max_frames, pose_video.shape[0])
        height = (pose_video.shape[1] // 32) * 32
        width = (pose_video.shape[2] // 32) * 32
        lengths = plan_pieces(total, piece_length, step=4, overlap=overlap_frames)
        starts = piece_starts(lengths, overlap=overlap_frames)
        log.info("SCAIL-2 long motion transfer: %d frames at %dx%d in %d pieces %s, overlap %d",
                 total, width, height, len(lengths), lengths, overlap_frames)

        pbar = comfy.utils.ProgressBar(len(lengths))
        frames = None
        previous = None
        written = 0
        for index, (start, length) in enumerate(zip(starts, lengths)):
            comfy.model_management.throw_exception_if_processing_interrupted()
            started = time.time()
            piece_positive, piece_negative, latent, _ = WanSCAILToVideo.execute(
                positive, negative, vae, width, height, length, 1, pose_strength, 0.0, 1.0, 0, overlap_frames,
                replacement_mode=replacement_mode, reference_image=reference_image, clip_vision_output=clip_vision_output,
                pose_video=_pad(pose_video[start:start + length], length),
                pose_video_mask=_pad(pose_video_mask[start:start + length], length),
                reference_image_mask=reference_image_mask, previous_frames=previous,
            ).args
            denoised = SamplerCustom.execute(model, True, noise_seed + index, cfg, piece_positive, piece_negative,
                                             sampler, sigmas, latent).args[1]
            images = nodes.VAEDecode().decode(vae, denoised)[0]
            previous = images
            keep = images if index == 0 else images[overlap_frames:]
            keep = keep[:total - written]
            if frames is None:
                frames = torch.empty((total,) + tuple(images.shape[1:]), dtype=images.dtype)
            frames[written:written + keep.shape[0]] = keep
            written += keep.shape[0]
            pbar.update(1)
            log.info("SCAIL-2 piece %d/%d: frames %d-%d in %.1f s", index + 1, len(lengths), start, start + length - 1,
                     time.time() - started)

        return (frames, len(lengths))


NODE_CLASS_MAPPINGS = {"PerformanceRemixSCAIL2Long": SCAIL2LongMotionTransfer}
NODE_DISPLAY_NAME_MAPPINGS = {"PerformanceRemixSCAIL2Long": "SCAIL-2: Long Motion Transfer"}
