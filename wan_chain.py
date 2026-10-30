"""Long motion transfer with Wan-Animate-2 in one node.

Wan-Animate-2 renders 81 frames per pass. Longer clips are made by chaining passes:
each piece continues from the last frame of the previous piece and reads the driving
video one frame before where the previous piece stopped. This node runs that chain
internally with ComfyUI's own WanAnimate2ToVideo, SamplerCustom, TrimVideoLatent and
VAEDecode, so a workflow needs one node for any clip length.
"""

import logging
import time

import torch

import comfy.model_management
import comfy.utils
import nodes
from comfy_extras.nodes_custom_sampler import SamplerCustom
from comfy_extras.nodes_wan import TrimVideoLatent, WanAnimate2ToVideo

from .pieces import plan_pieces

log = logging.getLogger("performance-remix")


class WanAnimate2LongMotionTransfer:
    CATEGORY = "performance-remix"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "pieces")
    FUNCTION = "run"
    DESCRIPTION = (
        "Animates the reference image with the motion of the whole driving video, "
        "rendering it as a chain of Wan-Animate-2 pieces in one run."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Wan-Animate-2 model, already patched (LoRA, cache, sampling shift)."}),
                "positive": ("CONDITIONING", {"tooltip": "Appearance and background prompt."}),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "reference_image": ("IMAGE", {"tooltip": "The approved still: who to animate."}),
                "pose_video": ("IMAGE", {"tooltip": "Driving video frames at the output size."}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "piece_length": ("INT", {"default": 81, "min": 5, "max": 161, "step": 4, "tooltip": "Frames per piece; 81 is what the model was trained on."}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "reference_image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "0 renders the whole video."}),
            },
            "optional": {
                "pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Sampling percent at which the pose influence starts."}),
                "pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Sampling percent at which the pose influence ends; lower keeps the choreography but frees fine detail such as the face."}),
                "positive_pose": ("CONDITIONING", {"tooltip": "Motion prompt for the pose branch."}),
                "clip_vision_output": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP vision of the reference image."}),
                "clip_vision_output_pose": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP vision of the first driving frame."}),
            },
        }

    def run(self, model, positive, negative, vae, sampler, sigmas, reference_image, pose_video,
            noise_seed, cfg, piece_length, pose_strength, reference_image_strength, max_frames,
            positive_pose=None, clip_vision_output=None, clip_vision_output_pose=None,
            pose_start_percent=0.0, pose_end_percent=1.0):
        if pose_start_percent > pose_end_percent:
            raise ValueError("pose_start_percent must not be greater than pose_end_percent")
        total = pose_video.shape[0] if max_frames == 0 else min(max_frames, pose_video.shape[0])
        pose_video = pose_video[:total]
        height = (pose_video.shape[1] // 16) * 16
        width = (pose_video.shape[2] // 16) * 16
        lengths = plan_pieces(total, piece_length)
        log.info("Long motion transfer: %d frames at %dx%d in %d pieces %s", total, width, height, len(lengths), lengths)

        pbar = comfy.utils.ProgressBar(len(lengths))
        decoded = []
        previous = None
        offset = 0
        for index, length in enumerate(lengths):
            comfy.model_management.throw_exception_if_processing_interrupted()
            started = time.time()
            cond = WanAnimate2ToVideo.execute(
                positive, negative, vae, width, height, length, 1, offset,
                reference_image=reference_image, pose_video=pose_video,
                clip_vision_output=clip_vision_output, positive_pose=positive_pose,
                clip_vision_output_pose=clip_vision_output_pose, continue_motion=previous,
                pose_strength=pose_strength, pose_start_percent=pose_start_percent,
                pose_end_percent=pose_end_percent, reference_image_strength=reference_image_strength,
            ).args
            piece_positive, piece_negative, latent, trim_latent, trim_image, offset = cond
            sampled = SamplerCustom.execute(model, True, noise_seed + index, cfg, piece_positive,
                                            piece_negative, sampler, sigmas, latent).args[0]
            trimmed = TrimVideoLatent.execute(sampled, trim_latent).args[0]
            images = nodes.VAEDecode().decode(vae, trimmed)[0]
            previous = images
            decoded.append(images[trim_image:] if index > 0 else images)
            pbar.update(1)
            log.info("Piece %d/%d: %d frames in %.1f s", index + 1, len(lengths), length, time.time() - started)

        frames = torch.cat(decoded, dim=0)[:total]
        return (frames, len(lengths))


NODE_CLASS_MAPPINGS = {"PerformanceRemixWanAnimate2Long": WanAnimate2LongMotionTransfer}
NODE_DISPLAY_NAME_MAPPINGS = {"PerformanceRemixWanAnimate2Long": "Wan Animate 2: Long Motion Transfer"}
