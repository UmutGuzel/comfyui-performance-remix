"""Object swap nodes: fit a new object of another shape into a clip, and paste the redraw back.

Object Area sets how far the redraw area reaches past the old outline, so a new object of
another shape fits: it measures how far the new object in the approved look (aligned with
frame 0) reaches past the old object's mask, as a share of the old object's height, and
widens every frame by a base margin plus that share of the object's height in that frame.
The margin then grows when the object comes closer to the camera. Areas in `keep`, such as
heads, are never taken in.

Paste Object keeps the original everywhere except where an object changed: the old
object's mask (tracked in the clip) joined with the new object's mask (tracked in the
redraw), widened a little and feathered. Video models that refine the whole frame or blend
with a wide halo otherwise leave changes around the object, such as a redrawn face next to a
new jacket. The redraw is first colour-matched to the clip (the median difference in a band
just outside the area, averaged over time). With `blend_levels` above 0 the paste is a
multi-band (Laplacian pyramid) blend: fine detail switches at the mask edge while brightness
and colour fade over about 2^levels pixels, so a redrawn background that is slightly
brighter or softer than the clip does not show as a patch. Widen the area by at least that
much, so the fade stays in plain background and the old object's colour does not bleed in.

Masks track the object, not its shadow, so the old object's shadow would stay in the
clip under the new object. With `shadow_reach` above 0, the area also takes in nearby
patches where the colour-matched redraw clearly differs from the clip (blurred, so thin
detail such as road seams does not count): the redraw removes the old shadow and draws the
new one, so those are exactly the patches that differ. Only patches touching the object
area and within shadow_reach x the area's height are taken.
"""

import cv2
import numpy as np


def grow_mask(mask, grow_px):
    """Widen a boolean mask by grow_px pixels in every direction (round brush).

    Above 32 px it uses the exact distance to the mask, whose time does not depend on grow_px: a
    dilation with a 200 px round kernel took 1.2 s per 1792x1024 frame, the distance 0.01 s.
    """
    if grow_px < 0:
        raise ValueError("grow_px must be 0 or more")
    if grow_px == 0 or not mask.any():
        return mask.copy()
    if grow_px > 32:
        distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return distance <= grow_px + 0.5  # the half pixel matches the reach of OpenCV's round kernel
    size = 2 * grow_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def color_offsets(generated, original, masks, band_px, smooth_frames, min_pixels=50):
    """Per-frame RGB offset that brings `generated` to `original`, measured just outside the masks.

    generated, original: float (N, H, W, 3); masks: bool (N, H, W), the redrawn area.
    The offset is the median difference in the band, so a shadow or other change covering part of
    the band does not pull it. Frames whose band has fewer than min_pixels pixels take the nearest
    measured frame's offset; the offsets are then averaged over a window of smooth_frames frames.
    """
    if generated.shape != original.shape or masks.shape != generated.shape[:3]:
        raise ValueError("generated, original and masks must cover the same frames and size")
    if band_px < 1 or smooth_frames < 1:
        raise ValueError("band_px and smooth_frames must be at least 1")
    count = generated.shape[0]
    offsets = np.full((count, 3), np.nan, dtype=np.float32)
    for t in range(count):
        band = grow_mask(masks[t], band_px) & ~masks[t]
        if band.sum() >= min_pixels:
            offsets[t] = np.median(original[t][band] - generated[t][band], axis=0)
    measured = np.where(~np.isnan(offsets[:, 0]))[0]
    if measured.size == 0:
        return np.zeros((count, 3), dtype=np.float32)
    nearest = measured[np.abs(np.arange(count)[:, None] - measured[None, :]).argmin(axis=1)]
    offsets = offsets[nearest]
    half = smooth_frames // 2
    padded = np.concatenate([np.repeat(offsets[:1], half, axis=0), offsets, np.repeat(offsets[-1:], half, axis=0)])
    kernel = np.ones(2 * half + 1, dtype=np.float32) / (2 * half + 1)
    return np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(3)], axis=1).astype(np.float32)


def laplacian_blend(inside, outside, weight, levels):
    """Multi-band blend of two float (H, W, 3) images: weight 1 takes `inside`, 0 takes `outside`.

    Each Laplacian pyramid level is mixed with the Gaussian pyramid of the weight, so low
    frequencies fade over about 2**levels pixels while detail switches at the mask edge.
    levels 0 is a plain alpha blend.
    """
    if levels <= 0:
        w = weight[..., None]
        return inside * w + outside * (1.0 - w)
    lap_in, lap_out, weights = [], [], []
    g_in, g_out, g_w = inside.astype(np.float32), outside.astype(np.float32), weight.astype(np.float32)
    for _ in range(levels):
        if min(g_w.shape) < 2:
            break
        d_in, d_out, d_w = cv2.pyrDown(g_in), cv2.pyrDown(g_out), cv2.pyrDown(g_w)
        size = (g_in.shape[1], g_in.shape[0])
        lap_in.append(g_in - cv2.pyrUp(d_in, dstsize=size))
        lap_out.append(g_out - cv2.pyrUp(d_out, dstsize=size))
        weights.append(g_w)
        g_in, g_out, g_w = d_in, d_out, d_w
    w = g_w[..., None]
    result = g_in * w + g_out * (1.0 - w)
    for l_in, l_out, l_w in zip(reversed(lap_in), reversed(lap_out), reversed(weights)):
        w = l_w[..., None]
        result = cv2.pyrUp(result, dstsize=(l_in.shape[1], l_in.shape[0])) + l_in * w + l_out * (1.0 - w)
    return result


def shadow_area(area, matched, original, reach_px, threshold, blur_sigma):
    """Widen a bool (H, W) paste area by nearby patches where `matched` and `original` clearly differ.

    The difference is taken on blurred images (max over channels). Patches must touch `area` and lie
    within reach_px of it; small holes in the result are closed.
    """
    if reach_px <= 0 or not area.any():
        return area
    size = int(blur_sigma * 3) * 2 + 1
    diff = np.abs(cv2.GaussianBlur(matched, (size, size), blur_sigma) - cv2.GaussianBlur(original, (size, size), blur_sigma)).max(axis=2)
    candidates = (diff > threshold) & grow_mask(area, reach_px)
    count, labels = cv2.connectedComponents((candidates | area).astype(np.uint8), connectivity=8)
    touching = np.unique(labels[area])
    widened = np.isin(labels, touching[touching > 0])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(widened.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0


def mask_height(mask):
    """Rows from the mask's top to its bottom, 0 for an empty mask."""
    rows = np.where(mask.any(axis=1))[0]
    return int(rows[-1] - rows[0] + 1) if rows.size else 0


def object_area(masks, look_mask, base_px, max_ratio, min_ratio=0.0, quantile=0.99, keep=None):
    """Old-object masks widened so the new object from the look fits in every frame.

    masks: bool (N, H, W), the old object tracked in the clip; look_mask: bool (H, W), the new object
    in the look, aligned with frame 0. The reach is the distance from the old mask that covers
    `quantile` of the new object's pixels (stray pixels do not count), divided by the old object's
    height and kept between min_ratio and max_ratio. min_ratio covers sides the look does not show,
    such as a pickup's bed when the car turns. Each frame grows by base_px + ratio x its object height.
    keep: optional bool (N, H, W), such as heads, that the widening never takes in (the old mask
    itself stays). Returns the widened masks and the ratio.
    """
    if look_mask.shape != masks.shape[1:]:
        raise ValueError("look_mask must have the frames' size")
    if keep is not None and keep.shape != masks.shape:
        raise ValueError("keep must cover the same frames and size as masks")
    if base_px < 0 or min_ratio < 0 or max_ratio < min_ratio:
        raise ValueError("base_px and min_ratio must be 0 or more, and max_ratio at least min_ratio")
    first = next((t for t in range(len(masks)) if masks[t].any()), None)
    if first is None:
        return masks.copy(), 0.0
    ratio = min_ratio
    if look_mask.any():
        distance = cv2.distanceTransform((~masks[first]).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        reach = float(np.quantile(distance[look_mask], quantile)) / mask_height(masks[first])
        ratio = min(max(reach, min_ratio), max_ratio)
    widened = np.stack([grow_mask(m, base_px + int(round(ratio * mask_height(m)))) for m in masks])
    if keep is not None:
        widened = masks | (widened & ~keep)
    return widened, ratio


def paste_object(original, generated, masks_old, masks_new, grow_px, feather_px, band_px, smooth_frames, blend_levels=0,
                 shadow_reach=0.0, shadow_threshold=0.06, shadow_blur=6.0, keep=None):
    """Original frames with the colour-matched redraw pasted inside (old | new mask), widened and feathered.

    original, generated: float (N, H, W, 3); masks_old, masks_new: bool (N, H, W).
    blend_levels 0 pastes with the soft mask; above 0 it uses a multi-band blend (laplacian_blend).
    shadow_reach above 0 also takes nearby patches where the redraw differs from the clip (shadow_area),
    up to shadow_reach x the height of the object area.
    keep: optional bool (N, H, W), such as heads, that neither the widening nor the shadow patches take
    in: a face next to a jacket that the redraw changed stays the clip's (the object masks still win).
    Returns the frames and the soft paste masks (N, H, W) in 0..1.
    """
    if original.shape != generated.shape or masks_old.shape != original.shape[:3] or masks_new.shape != masks_old.shape:
        raise ValueError("frames and masks must cover the same frames and size")
    if keep is not None and keep.shape != masks_old.shape:
        raise ValueError("keep must cover the same frames and size as the masks")
    if feather_px < 0:
        raise ValueError("feather_px must be 0 or more")
    objects = masks_old | masks_new
    free = ~keep if keep is not None else np.ones_like(objects)
    area = np.stack([o | (grow_mask(o, grow_px) & f) for o, f in zip(objects, free)])
    offsets = color_offsets(generated, original, area, band_px, smooth_frames)

    def matched(t):  # one frame at a time: a full colour-matched copy of a long clip does not fit in memory
        return np.clip(generated[t] + offsets[t], 0.0, 1.0)

    if shadow_reach > 0:
        for t in range(len(area)):
            reach = int(round(shadow_reach * mask_height(area[t])))
            area[t] = objects[t] | (shadow_area(area[t], matched(t), original[t], reach, shadow_threshold, shadow_blur) & free[t])
        # measure the colour shift again outside the widened area
        offsets = color_offsets(generated, original, area, band_px, smooth_frames)
    soft = area.astype(np.float32)
    frames = np.empty(original.shape, dtype=np.float32)
    for t in range(len(area)):
        if feather_px > 0:
            size = 2 * feather_px + 1
            # the area itself stays fully redrawn; only the outside edge fades
            soft[t] = np.maximum(cv2.GaussianBlur(soft[t], (size, size), feather_px / 2.0), area[t])
        frames[t] = np.clip(laplacian_blend(matched(t), original[t], soft[t], blend_levels), 0.0, 1.0)
    return frames, soft


class PasteObject:
    CATEGORY = "performance-remix"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "mask")
    FUNCTION = "run"
    DESCRIPTION = (
        "Keeps the original frames except where the object changed: the old object's mask joined with the new "
        "object's mask, widened and feathered; the redraw is colour-matched to the original first."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE", {"tooltip": "The clip's frames, aligned with the redraw."}),
                "generated": ("IMAGE", {"tooltip": "The redrawn frames."}),
                "masks_old": ("MASK", {"tooltip": "The old object, tracked in the clip; scaled to the frames."}),
                "masks_new": ("MASK", {"tooltip": "The new object, tracked in the redraw; scaled to the frames."}),
                "grow_px": ("INT", {"default": 8, "min": 0, "max": 256}),
                "feather_px": ("INT", {"default": 6, "min": 0, "max": 64}),
                "band_px": ("INT", {"default": 24, "min": 1, "max": 256, "tooltip": "Band outside the area where colours are compared."}),
                "smooth_frames": ("INT", {"default": 9, "min": 1, "max": 101}),
                "blend_levels": ("INT", {"default": 0, "min": 0, "max": 8, "tooltip": "0 pastes with the soft mask; 3 to 5 blends brightness over about 2^levels px while detail stays sharp. Widen the area at least that much."}),
                "shadow_reach": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Also take nearby patches where the redraw differs from the clip (the old shadow it removed, the new shadow it drew), up to this many times the object area's height. 0 turns it off."}),
                "shadow_threshold": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 1.0, "step": 0.005}),
                "shadow_blur": ("FLOAT", {"default": 6.0, "min": 0.5, "max": 32.0, "step": 0.5, "tooltip": "Blur before comparing, so thin detail does not count."}),
            },
            "optional": {
                "keep": ("MASK", {"tooltip": "Areas that always stay the original outside the object masks, such as heads (SAM 3 \"head\" and \"face\")."}),
            },
        }

    def run(self, original, generated, masks_old, masks_new, grow_px, feather_px, band_px, smooth_frames, blend_levels=0,
            shadow_reach=0.0, shadow_threshold=0.06, shadow_blur=6.0, keep=None):
        import torch
        import torch.nn.functional as F

        count = min(original.shape[0], generated.shape[0])
        height, width = generated.shape[1], generated.shape[2]
        if original.shape[1:3] != (height, width):
            raise ValueError("original and generated must have the same size")

        def fit(masks):
            masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1]).float()
            if masks.shape[-2:] != (height, width):
                masks = F.interpolate(masks[:, None], size=(height, width), mode="bilinear")[:, 0]
            if masks.shape[0] < count:  # a tracker may return fewer frames: repeat the last one
                masks = torch.cat([masks, masks[-1:].expand(count - masks.shape[0], -1, -1)])
            return (masks[:count] > 0.5).cpu().numpy()

        frames, soft = paste_object(original[:count, ..., :3].cpu().numpy(), generated[:count, ..., :3].cpu().numpy(),
                                    fit(masks_old), fit(masks_new), grow_px, feather_px, band_px, smooth_frames, blend_levels,
                                    shadow_reach, shadow_threshold, shadow_blur, None if keep is None else fit(keep))
        return (torch.from_numpy(frames), torch.from_numpy(soft))


class ObjectArea:
    CATEGORY = "performance-remix"
    RETURN_TYPES = ("MASK", "FLOAT")
    RETURN_NAMES = ("mask", "ratio")
    FUNCTION = "run"
    DESCRIPTION = (
        "Widens the old object's mask in every frame so the new object from the approved look fits: by a base "
        "margin plus how far the new object reaches past the old outline, scaled with the object's size."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK", {"tooltip": "The old object in every frame (SAM 3 Track to Mask)."}),
                "look_mask": ("MASK", {"tooltip": "The new object in the approved look (SAM 3 Detect); the first mask is used, stretched to the frames' size."}),
                "base_px": ("INT", {"default": 16, "min": 0, "max": 256, "tooltip": "Margin when the new object fits the old outline."}),
                "min_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Smallest extra reach, as a share of the object's height: covers sides the look does not show, such as a car's side when it turns."}),
                "max_ratio": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Largest extra reach, as a share of the object's height."}),
            },
            "optional": {
                "keep": ("MASK", {"tooltip": "Areas the widening never takes in, such as heads (SAM 3 \"head\" and \"face\"), so the redraw leaves them alone."}),
            },
        }

    def run(self, masks, look_mask, base_px, min_ratio, max_ratio, keep=None):
        import torch
        import torch.nn.functional as F

        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
        count, height, width = masks.shape
        look = look_mask.reshape(-1, look_mask.shape[-2], look_mask.shape[-1])[:1].float()
        if look.shape[-2:] != (height, width):
            look = F.interpolate(look[:, None], size=(height, width), mode="bilinear")[:, 0]
        keep_masks = None
        if keep is not None:
            keep = keep.reshape(-1, keep.shape[-2], keep.shape[-1]).float()
            if keep.shape[-2:] != (height, width):
                keep = F.interpolate(keep[:, None], size=(height, width), mode="bilinear")[:, 0]
            if keep.shape[0] < count:  # a tracker may return fewer frames: repeat the last one
                keep = torch.cat([keep, keep[-1:].expand(count - keep.shape[0], -1, -1)])
            keep_masks = (keep[:count] > 0.5).cpu().numpy()
        widened, ratio = object_area((masks > 0.5).cpu().numpy(), (look[0] > 0.5).cpu().numpy(), base_px, max_ratio, min_ratio,
                                     keep=keep_masks)
        return (torch.from_numpy(widened.astype(np.float32)), ratio)


NODE_CLASS_MAPPINGS = {
    "PerformanceRemixObjectArea": ObjectArea,
    "PerformanceRemixPasteObject": PasteObject,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PerformanceRemixObjectArea": "Object Area (fit the new object)",
    "PerformanceRemixPasteObject": "Paste Object onto Original",
}
