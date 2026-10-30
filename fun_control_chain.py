"""Long motion transfer with Wan2.2 Fun Control A14B in one node.

Fun Control renders 81 frames per pass from a control video (pose, depth or edges) and a
reference image. Longer clips are made by chaining passes: each piece starts from the
last `overlap_frames` frames of the previous piece, given to the model as start frames,
and reads the control video from the same clip frame, so motion carries across the
join. ComfyUI's Wan22FunControlToVideo already encodes start frames; its node does not
expose that input, so this node calls it directly. Sampling follows the official
template: the high-noise model for the first steps, the low-noise model for the rest.
"""

import logging
import time

import torch

import comfy.model_management
import comfy.samplers
import comfy.utils
import nodes
from comfy_extras.nodes_wan import Wan22FunControlToVideo

from .pieces import piece_starts, plan_pieces

log = logging.getLogger("performance-remix")


class WanFunControlLongMotionTransfer:
    CATEGORY = "performance-remix"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "pieces")
    FUNCTION = "run"
    DESCRIPTION = (
        "Renders a whole control video with Wan2.2 Fun Control as a chain of pieces in one run; "
        "each piece starts from the last frames of the one before."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high_noise": ("MODEL", {"tooltip": "Fun Control high-noise model, already patched (LoRA, sampling shift)."}),
                "model_low_noise": ("MODEL", {"tooltip": "Fun Control low-noise model, already patched."}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "ref_image": ("IMAGE", {"tooltip": "The approved still: who to animate."}),
                "control_video": ("IMAGE", {"tooltip": "Control frames for the whole clip at the output size (multiples of 16)."}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "switch_step": ("INT", {"default": 2, "min": 1, "max": 100, "tooltip": "Step where sampling moves from the high-noise to the low-noise model."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "piece_length": ("INT", {"default": 81, "min": 9, "max": 161, "step": 4, "tooltip": "Frames per piece, 4n + 1; the model was trained on 81."}),
                "overlap_frames": ("INT", {"default": 5, "min": 1, "max": 33, "step": 4, "tooltip": "Frames each piece repeats from the one before, 4n + 1; more frames carry motion speed across the join."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "0 renders the whole video."}),
            },
        }

    def run(self, model_high_noise, model_low_noise, positive, negative, vae, ref_image, control_video, noise_seed,
            steps, switch_step, cfg, sampler_name, scheduler, piece_length, overlap_frames, max_frames):
        if switch_step >= steps:
            raise ValueError("switch_step must be lower than steps")
        if (overlap_frames - 1) % 4 != 0:
            raise ValueError("overlap_frames must be 4n + 1")
        total = control_video.shape[0] if max_frames == 0 else min(max_frames, control_video.shape[0])
        height = (control_video.shape[1] // 16) * 16
        width = (control_video.shape[2] // 16) * 16
        lengths = plan_pieces(total, piece_length, step=4, overlap=overlap_frames)
        starts = piece_starts(lengths, overlap=overlap_frames)
        log.info("Fun Control long motion transfer: %d frames at %dx%d in %d pieces %s, overlap %d",
                 total, width, height, len(lengths), lengths, overlap_frames)

        pbar = comfy.utils.ProgressBar(len(lengths))
        sampler = nodes.KSamplerAdvanced()
        frames = None
        previous = None
        written = 0
        for index, (start, length) in enumerate(zip(starts, lengths)):
            comfy.model_management.throw_exception_if_processing_interrupted()
            started = time.time()
            control = control_video[start:start + length]
            if control.shape[0] < length:
                control = torch.cat([control, control[-1:].expand(length - control.shape[0], -1, -1, -1)], dim=0)
            start_frames = previous[-overlap_frames:] if previous is not None else None
            piece_positive, piece_negative, latent = Wan22FunControlToVideo.execute(
                positive, negative, vae, width, height, length, 1,
                ref_image=ref_image, start_image=start_frames, control_video=control,
            ).args
            seed = noise_seed + index
            latent = sampler.sample(model_high_noise, "enable", seed, steps, cfg, sampler_name, scheduler,
                                    piece_positive, piece_negative, latent, 0, switch_step, "enable")[0]
            latent = sampler.sample(model_low_noise, "disable", seed, steps, cfg, sampler_name, scheduler,
                                    piece_positive, piece_negative, latent, switch_step, 10000, "disable")[0]
            images = nodes.VAEDecode().decode(vae, latent)[0]
            previous = images
            keep = images if index == 0 else images[overlap_frames:]
            keep = keep[:total - written]
            if frames is None:
                frames = torch.empty((total,) + tuple(images.shape[1:]), dtype=images.dtype)
            frames[written:written + keep.shape[0]] = keep
            written += keep.shape[0]
            pbar.update(1)
            log.info("Fun Control piece %d/%d: frames %d-%d in %.1f s", index + 1, len(lengths), start, start + length - 1,
                     time.time() - started)

        return (frames, len(lengths))


NODE_CLASS_MAPPINGS = {"PerformanceRemixFunControlLong": WanFunControlLongMotionTransfer}
NODE_DISPLAY_NAME_MAPPINGS = {"PerformanceRemixFunControlLong": "Wan 2.2 Fun Control: Long Motion Transfer"}
