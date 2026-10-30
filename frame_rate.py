"""Frame-rate change for a batch of frames.

Viggle-Animate expects its driving clip at 24 fps, and ComfyUI's core LoadVideo keeps
the file's own rate. This node keeps the clip's duration and picks, for every output
frame, the source frame nearest in time, the way ffmpeg's fps filter does; the clip's
audio therefore stays in sync when the frames are saved at the new rate.
"""


def resample_indices(frame_count, source_fps, target_fps):
    """Source frame index for each output frame when changing source_fps to target_fps."""
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("frame rates must be positive")
    count = max(1, round(frame_count * target_fps / source_fps))
    return [min(frame_count - 1, int(i * source_fps / target_fps + 0.5)) for i in range(count)]


class ChangeFrameRate:
    CATEGORY = "performance-remix"
    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "run"
    DESCRIPTION = "Changes the frame rate of a batch of frames by picking the nearest source frame; the duration stays the same."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_fps": ("FLOAT", {"default": 25.0, "min": 0.01, "max": 1000.0, "step": 0.001, "tooltip": "The clip's frame rate, from Get Video Components."}),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.001}),
            },
        }

    def run(self, images, source_fps, target_fps):
        if abs(source_fps - target_fps) < 1e-6:
            return (images, target_fps)
        return (images[resample_indices(images.shape[0], source_fps, target_fps)], target_fps)


NODE_CLASS_MAPPINGS = {"PerformanceRemixChangeFrameRate": ChangeFrameRate}
NODE_DISPLAY_NAME_MAPPINGS = {"PerformanceRemixChangeFrameRate": "Change Frame Rate"}
