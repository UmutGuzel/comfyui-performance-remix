# ComfyUI Performance Remix nodes

Custom nodes for swapping a person or an object in a video with ComfyUI. They are the
nodes behind [Performance Remix](https://github.com/UmutGuzel/performance-remix), and each one
also works on its own in any workflow.

| Node | What it does |
|---|---|
| SCAIL-2: Long Motion Transfer | replaces the person in a clip of any length, in one run |
| Wan 2.2 Fun Control: Long Motion Transfer | animates an image along a pose, depth or edge video of any length |
| Wan Animate 2: Long Motion Transfer | animates an image with a driving video of any length |
| Face Track and Crop, Face Paste | re-render small faces in wide shots as close-ups and blend them back |
| Object Area, Paste Object | fit a new object of another shape into a clip, then paste it back with its shadow |
| Pad Frames for LTX, Frame at Time, Join Object Windows | object swaps of any length: pad short clips, chain 5-second windows, join them |
| Clip Frames at Working Size | loads a clip frame by frame at the size the model works at, using less memory |
| Change Frame Rate | changes a clip's frame rate and keeps its length and audio sync |
| Streaming Finish (SeedVR2) | upscales and grades a video in segments, so memory does not grow with its length |

## Install

**Windows, ComfyUI portable**: in PowerShell, in the `ComfyUI_windows_portable` folder:

```powershell
git clone https://github.com/UmutGuzel/comfyui-performance-remix ComfyUI\custom_nodes\comfyui-performance-remix
.\python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\comfyui-performance-remix\requirements.txt
```

**Linux, or any manual install**: with ComfyUI's Python environment active:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/UmutGuzel/comfyui-performance-remix
pip install -r comfyui-performance-remix/requirements.txt
```

Restart ComfyUI. Requires ComfyUI 0.34 or newer (for the core `WanAnimate2ToVideo` and
`WanSCAILToVideo` nodes) and `opencv-python` 4.8 or newer. The nodes appear in the
**performance-remix** category.

## Clip Frames at Working Size

A lower-memory **Get Video Components**. The core node decodes the clip at full size and
keeps those frames in memory until the workflow ends: about 18 GB for 30 s of 1080x1920 at
24 fps. This node decodes one frame at a time and scales it to the working size straight
away. Its first three outputs match Get Video Components (`images`, `audio`, `fps`), and it
also returns the `width` and `height` it decoded at.

| Input | Meaning |
|---|---|
| video | from Load Video and Video Slice |
| size | `shorter side`: scale down so the shorter side is `shorter_side`, never up, keep the shape. `width x height`: scale to cover `width` x `height` and cut to exactly that size in the center |
| shorter_side | 720 by default |
| width, height | 704 x 1248 by default |

Frames are scaled with an area filter going down and bicubic going up, from 8-bit RGB.
Clip rotation in the file is applied.

## SCAIL-2: Long Motion Transfer

Replaces the person in a whole clip with the person in a still, with SCAIL-2, as a chain
of pieces in one run. SCAIL-2 renders 81 frames per pass from the clip's frames, a
colored SAM 3 mask video of the person and the still with its own mask. The official
template makes longer clips with "Extend" segments added by hand; this node runs that
chain for any length: piece n + 1 reads the clip `piece_length - overlap_frames` frames
further on and anchors on the last `overlap_frames` decoded frames of piece n, which
are then dropped. The last piece is padded to a legal length with its last frame and
trimmed back. Each piece runs ComfyUI's own `WanSCAILToVideo`, `SamplerCustom` (the
denoised output is decoded, as in the template) and `VAEDecode`.

| Input | Meaning |
|---|---|
| model | SCAIL-2 with the DPO LoRA, the lightx2v LoRA at 0.8 and `ModelSamplingSD3` shift 5 |
| positive, negative | prompt and negative prompt |
| vae | Wan 2.1 VAE |
| clip_vision_output | CLIP vision of the still |
| reference_image, reference_image_mask | the still and its mask from **Create SCAIL-2 Colored Mask** |
| pose_video, pose_video_mask | the whole clip at the output size (multiples of 32) and its mask video, same length |
| replacement_mode | true keeps the clip's background and replaces the person; false animates the still |
| sampler, sigmas | `KSamplerSelect` euler and `BasicScheduler` simple, 6 steps, on the model before the shift patch, as in the template |
| noise_seed | piece n uses seed + n |
| cfg | 1.0 with the lightx2v LoRA |
| pose_strength | passed to every piece |
| piece_length, overlap_frames | 81 and 5, the values the model was trained with |
| max_frames | 0 renders the whole video |

Outputs: `images`, all frames in order, and `pieces`.

On a 32 GB GPU with the int8 model at 704x1248, a 6-second clip renders in two pieces in
about 12 minutes, with no visible jump at the join.

## Change Frame Rate

Viggle-Animate expects its driving clip at 24 fps, and ComfyUI's core Load Video keeps
the file's own rate. This node keeps the clip's duration and picks, for every output
frame, the source frame nearest in time (as ffmpeg's `fps` filter does): 25 fps loses one
frame in 25, 30 fps one in 5. Its `fps` output goes to Create Video, so the clip's audio
stays in sync.

| Input | Meaning |
|---|---|
| images | the clip's frames |
| source_fps | the clip's frame rate, from Get Video Components or Clip Frames at Working Size |
| target_fps | 24 by default |

## Wan 2.2 Fun Control: Long Motion Transfer

Renders a whole control video (pose, depth or edges) with Wan2.2 Fun Control A14B as a
chain of pieces in one run. Each piece gets the last `overlap_frames` frames of the piece
before as start frames, and reads the control video from the same clip frame, so motion
speed carries across the join; the repeated frames are dropped. ComfyUI's
`Wan22FunControlToVideo` already encodes start frames, but its node does not show that
input, so this node calls it directly. Sampling follows the official 4-step template:
the high-noise model up to `switch_step`, the low-noise model after it.

| Input | Meaning |
|---|---|
| model_high_noise, model_low_noise | the two Fun Control models, each with its lightx2v LoRA and `ModelSamplingSD3` |
| positive, negative | prompt and negative prompt |
| vae | Wan 2.1 VAE |
| ref_image | the image to animate, used as a reference |
| control_video | control frames for the whole clip at the output size |
| noise_seed | piece n uses seed + n |
| steps, switch_step, cfg, sampler_name, scheduler | 4, 2, 1.0, euler, simple with the 4-step LoRAs |
| piece_length | frames per piece, 81 by default |
| overlap_frames | frames repeated from the piece before, 4n + 1, 5 by default |
| max_frames | 0 renders the whole video |

On a 32 GB GPU, an 11.2-second clip at 576x1024 with DWPose control renders in five pieces
in about 8 minutes. It follows the body more closely than Wan-Animate-2 on the same clip.

## Wan Animate 2: Long Motion Transfer

Animates a reference image with the motion of a whole driving video in one node and
one run, at any length.

Wan-Animate-2 renders 81 frames per pass. This node chains passes for you: each piece
continues from the last frame of the piece before and reads the driving video one
frame before where that piece stopped, which is how the model's own extension input
is meant to be used. The duplicated frame is dropped, the last piece is shortened to
the smallest legal length, and the result is trimmed to the exact frame count of the
driving video. Each piece is sampled with ComfyUI's own `WanAnimate2ToVideo`,
`SamplerCustom`, `TrimVideoLatent` and `VAEDecode`, so results match the official
template.

Memory stays at the size of one 81-frame pass, whatever the clip length.

| Input | Meaning |
|---|---|
| model | Wan-Animate-2, already patched with the LoRA, `WanAnimate2Cache` and `ModelSamplingSD3` |
| positive, negative | appearance and background prompt, negative prompt |
| positive_pose | optional motion prompt for the pose branch |
| vae | Wan 2.1 VAE |
| sampler, sigmas | from `KSamplerSelect` and `BasicScheduler` |
| reference_image | the image to animate |
| pose_video | driving video frames, already resized to the output size |
| clip_vision_output, clip_vision_output_pose | optional CLIP vision of the reference and of the first driving frame |
| noise_seed | piece n uses seed + n |
| cfg | 1.0 with the lightx2v LoRA |
| piece_length | frames per piece, 81 by default |
| pose_strength, reference_image_strength | passed to every piece |
| max_frames | 0 renders the whole video |

Outputs: `images`, all frames in order, and `pieces`, the number of pieces rendered.

On a 32 GB GPU with the int8 model at 480x848, a 21-second clip renders in eight pieces in
about 18 minutes, with 18.6 GB of GPU memory at peak and no visible jump at the joins.

The node also takes optional `pose_start_percent` and `pose_end_percent`, passed to every
piece: the pose branch only runs inside that part of the sampling.

## Face pass: Face Track and Crop, Face Paste

In a full-body 480p shot the face is about 60 pixels wide, too small for the video model
to keep the reference's identity, so it comes out as a stranger. The face pass re-renders
only the face at 512x512 and puts it back. On an 11-second test, identity similarity to the
reference rose from 0.13 to 0.47.

1. **Face Track and Crop** finds the largest face in every frame, fills frames without a
   detection by interpolating between neighbours, smooths the position with a moving
   average over `smooth_frames`, and cuts a fixed-size square (`box_scale` times the
   median face size) resized to `crop_size`. Outputs the close-ups, the track, and the
   number of frames with a detected face. Run it on the reference image too, with
   `smooth_frames` 1, to get a matching reference close-up.
2. Re-render the close-ups with **Wan Animate 2: Long Motion Transfer**, using the
   reference close-up as `reference_image` and the video close-ups as `pose_video`.
3. **Face Paste** scales each new face to its box, aligns it to the original face's five
   landmarks with a similarity transform (shift, rotation, uniform scale), matches its
   color to the original in LAB space inside the blend oval, and blends it with a
   feathered oval, so hair, veil and hands stay from the original frame. It works in
   three passes: it measures the alignment, oval and color statistics of every frame,
   fills frames without a detection from their neighbours and averages each value over
   `smooth_frames`, then warps, matches and blends with the averaged values.

   `smooth_frames` 3, the default, keeps the face as steady as the main pass. Larger
   windows lag behind fast head moves.

If no face is found, Face Track and Crop returns centre crops and an empty track, and
Face Paste passes the frames through unchanged.

Faces are detected with OpenCV's YuNet, bundled in `models/` (MIT license in
`models/LICENSE-yunet.txt`). Cropping and pasting an 11-second clip takes about 35 seconds.

## Streaming Finish (SeedVR2)

Upscales a whole video with SeedVR2, applies a light grade, and saves an H.264 MP4 with
the source audio, one segment at a time, so memory does not grow with the clip length.

The official SeedVR2 template holds every frame in RAM several times at the output size,
so long clips run out of memory. This node reads `segment_frames` frames from the video file,
runs the template's steps on them (2x lanczos resize, SeedVR2 pre-processing, tiled VAE
encode, one sampling step, tiled decode, color post-processing), applies sharpen and
contrast, crossfades the `overlap` frames it shares with the previous segment, and passes
the frames straight to the encoder. Audio for the same time range is added as AAC.

| Input | Default | Meaning |
|---|---|---|
| video | | a video from Load Video, optionally trimmed with Video Slice |
| model, vae | | SeedVR2 model and VAE |
| scale | 2.0 | upscale factor |
| segment_frames | 49 | frames processed at once; memory follows this |
| overlap | 4 | frames shared by neighbouring segments, crossfaded |
| color_correction | lab | SeedVR2 color match to the input |
| sharpen | 0.15 | unsharp strength; 0 turns it off |
| contrast | 1.03 | 1.0 leaves contrast unchanged |
| crf | 16 | H.264 quality |
| seed | 0 | sampling seed |
| filename_prefix | performance-remix/finish | output file name |

Output: the saved video (also shown in ComfyUI's output panel). The output is tagged
BT.709, like ComfyUI's own Save Video.

## Object swap: Object Area, Paste Object

Two nodes around a video model that redraws one object in a clip (the app uses LTX-2.5
inpainting), so the new object may have another shape and only the object changes.

**Object Area** widens the old object's mask in every frame so the new object from the
approved look fits. It measures how far the new object in the look (aligned with frame 0)
reaches past the old mask, as a share of the old object's height, and grows each frame by
`base_px` plus that share of the object's height in that frame, so the margin grows when
the object comes closer to the camera.

| Input | Meaning |
|---|---|
| masks | the old object in every frame (SAM 3 Track to Mask) |
| look_mask | the new object in the approved look (SAM 3 Detect), stretched to the frames' size |
| base_px | margin when the new object fits the old outline, 16 by default |
| min_ratio, max_ratio | the share is kept between them, 0.2 and 0.6; the minimum covers sides the look does not show, such as a pickup's bed when the car turns |
| keep | optional: areas the widening never takes in, such as heads (SAM 3 "head" and "face") |

Outputs: `mask`, the widened masks, and `ratio`, the share used.

**Paste Object** keeps the clip everywhere except the old object and the new one (tracked
in the redraw), widened by `grow_px`: the redraw is colour-matched to the clip (median
difference in a `band_px` band, averaged over `smooth_frames`), and with `blend_levels`
above 0 pasted with a multi-band (Laplacian pyramid) blend, so brightness fades over about
2^levels pixels while detail switches at the edge. With `shadow_reach` above 0 the area
also takes in nearby patches where the blurred redraw differs from the clip by more than
`shadow_threshold`, if they touch the area and lie within `shadow_reach` times its height:
the old shadow the redraw removed and the new one it drew. Areas in the optional `keep`
mask stay the clip's outside the object masks.

| Input | Meaning |
|---|---|
| original, generated | the clip's frames and the redraw, same size |
| masks_old, masks_new | the old object (tracked in the clip) and the new one (tracked in the redraw) |
| grow_px, feather_px | widening and soft edge, 24 and 6 in the app |
| band_px, smooth_frames | colour match band and time window, 24 and 9 |
| blend_levels | 0 pastes with the soft mask; 4 in the app |
| shadow_reach, shadow_threshold, shadow_blur | 0.5, 0.06, 6.0 in the app; reach 0 turns it off |
| keep | optional: areas that stay the clip's, such as heads |

Outputs: `images` and the paste `mask`. 81 frames at 1792x1024 take about 22 s on the
CPU.

## Object swaps of any length: Pad Frames for LTX, Frame at Time, Join Object Windows

LTX-2.5 makes 8k+1 frames, and in tests its inpainting lost the new object partway on 2- and
3-second clips while every 145-frame run kept it.

**Pad Frames for LTX** holds the last frame until the clip has at least `min_frames` (145)
frames and an 8k+1 count. Its `frames` output is the clip's own count: cut the result back to
it with core **Image From Batch** before saving, and the output is exactly as long as the
input.

Longer clips are made in 5-second windows. **Frame at Time** picks the frame at `seconds`
into a window (the first frame at or after it); the next window starts from that frame as its
first frame. **Join Object Windows** takes the clip range the windows came from (Load Video,
Video Slice) and up to eight window videos with their start times (`starts`, seconds from
the range start, comma separated). It places every window on the clip's own frame times,
crossfades `crossfade_frames` frames where the next window starts, adds the clip's audio and
writes the frames straight to an H.264 file, so memory does not grow with the length.

## Tests

Run each file in `tests/` with ComfyUI's Python, for example
`python tests/test_clip_frames.py`.

## License

MIT, see `LICENSE`. Bundled YuNet detector: MIT, see `models/LICENSE-yunet.txt`. Model
licenses are separate: SCAIL-2 is MIT, Wan2.2 Fun Control and Wan-Animate-2 are Apache
2.0.
