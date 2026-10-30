import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from object_swap import color_offsets, grow_mask, laplacian_blend, mask_height, object_area, paste_object, shadow_area  # noqa: E402


def square_mask(size=64, top=24, left=24, side=16):
    mask = np.zeros((size, size), dtype=bool)
    mask[top:top + side, left:left + side] = True
    return mask


class GrowMaskTest(unittest.TestCase):
    def test_zero_keeps_the_mask(self):
        mask = square_mask()
        self.assertTrue(np.array_equal(grow_mask(mask, 0), mask))

    def test_grows_by_the_given_pixels(self):
        grown = grow_mask(square_mask(), 4)
        rows = np.where(grown.any(axis=1))[0]
        cols = np.where(grown.any(axis=0))[0]
        self.assertEqual((rows[0], rows[-1]), (20, 43))
        self.assertEqual((cols[0], cols[-1]), (20, 43))
        self.assertTrue(grown[square_mask()].all())

    def test_large_growth_is_round(self):
        mask = square_mask(size=200, top=92, left=92, side=16)
        grown = grow_mask(mask, 40)
        rows = np.where(grown.any(axis=1))[0]
        self.assertEqual((rows[0], rows[-1]), (52, 147))
        self.assertTrue(grown[mask].all())
        self.assertTrue(grown[92 - 28, 92 - 28])  # 39.6 px from the corner: inside
        self.assertFalse(grown[92 - 30, 92 - 30])  # 42.4 px from the corner: outside

    def test_empty_mask_stays_empty(self):
        self.assertFalse(grow_mask(np.zeros((8, 8), dtype=bool), 5).any())

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            grow_mask(square_mask(), -1)


class ColorOffsetsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.original = rng.random((5, 64, 64, 3), dtype=np.float32) * 0.5 + 0.25
        self.masks = np.stack([square_mask() for _ in range(5)])

    def test_finds_a_constant_shift(self):
        shift = np.array([-0.02, -0.03, -0.04], dtype=np.float32)
        generated = self.original + shift
        offsets = color_offsets(generated, self.original, self.masks, 6, 3)
        self.assertTrue(np.allclose(offsets, -shift, atol=1e-5))

    def test_ignores_what_changed_inside_the_area(self):
        generated = self.original.copy()
        generated[self.masks] = 1.0  # a bright new object inside the redrawn area
        offsets = color_offsets(generated, self.original, self.masks, 6, 1)
        self.assertTrue(np.allclose(offsets, 0.0, atol=1e-6))

    def test_smoothing_spreads_a_one_frame_jump(self):
        generated = self.original.copy()
        generated[2] -= 0.09
        offsets = color_offsets(generated, self.original, self.masks, 6, 3)
        self.assertTrue(np.allclose(offsets[2], 0.03, atol=1e-5))
        self.assertTrue(np.allclose(offsets[0], 0.0, atol=1e-6))

    def test_frames_without_a_band_take_the_nearest_measurement(self):
        masks = self.masks.copy()
        masks[1] = True  # the whole frame is redrawn: no band to measure
        generated = self.original - 0.05
        generated[1] = 0.0
        offsets = color_offsets(generated, self.original, masks, 6, 1)
        self.assertTrue(np.allclose(offsets[1], 0.05, atol=1e-5))

    def test_no_band_anywhere_gives_no_shift(self):
        masks = np.ones((5, 64, 64), dtype=bool)
        self.assertFalse(color_offsets(self.original, self.original, masks, 6, 3).any())


class PasteObjectTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.original = rng.random((3, 64, 64, 3), dtype=np.float32) * 0.5 + 0.25
        self.generated = self.original + 0.04  # the redraw is shifted everywhere, like a model's colour drift
        self.old = np.stack([square_mask(left=10) for _ in range(3)])
        self.new = np.stack([square_mask(left=30) for _ in range(3)])
        self.generated[:, 24:40, 30:46] = 0.95  # the new object

    def test_far_from_both_objects_the_original_is_kept(self):
        frames, soft = paste_object(self.original, self.generated, self.old, self.new, 2, 3, 6, 1)
        far = soft == 0
        self.assertTrue(far.any())
        self.assertTrue(np.allclose(frames[far], self.original[far]))

    def test_inside_both_masks_the_colour_matched_redraw_is_used(self):
        frames, soft = paste_object(self.original, self.generated, self.old, self.new, 2, 3, 6, 1)
        self.assertTrue(np.all(soft[self.old] == 1.0) and np.all(soft[self.new] == 1.0))
        # the old object's area gets the redraw minus its colour shift
        self.assertTrue(np.allclose(frames[self.old], self.original[self.old], atol=1e-5))
        self.assertTrue(np.allclose(frames[:, 30:34, 34:42], 0.91, atol=1e-5))

    def test_feather_only_fades_outward(self):
        _, soft = paste_object(self.original, self.generated, self.old, self.new, 0, 4, 6, 1)
        hard = self.old | self.new
        ring = (soft > 0) & ~hard
        self.assertTrue(ring.any())
        self.assertTrue(np.all(soft[ring] < 1.0))

    def test_rejects_mismatched_masks(self):
        with self.assertRaises(ValueError):
            paste_object(self.original, self.generated, self.old[:2], self.new, 2, 3, 6, 1)


class LaplacianBlendTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.outside = rng.random((128, 128, 3), dtype=np.float32) * 0.6 + 0.2
        self.inside = self.outside + 0.1  # same scene, slightly brighter, like a redrawn background
        self.step = np.zeros((128, 128), dtype=np.float32)
        self.step[:, 64:] = 1.0

    def test_full_weight_reproduces_each_image(self):
        ones, zeros = np.ones((128, 128), np.float32), np.zeros((128, 128), np.float32)
        self.assertTrue(np.allclose(laplacian_blend(self.inside, self.outside, ones, 4), self.inside, atol=1e-5))
        self.assertTrue(np.allclose(laplacian_blend(self.inside, self.outside, zeros, 4), self.outside, atol=1e-5))

    def test_brightness_step_is_spread_out(self):
        hard = laplacian_blend(self.inside, self.outside, self.step, 0) - self.outside
        multi = laplacian_blend(self.inside, self.outside, self.step, 4) - self.outside
        # with a hard paste the whole 0.1 offset appears between two neighbouring columns
        self.assertAlmostEqual(float(np.abs(np.diff(hard.mean(axis=(0, 2)))).max()), 0.1, places=4)
        self.assertLess(float(np.abs(np.diff(multi.mean(axis=(0, 2)))).max()), 0.05)

    def test_far_from_the_edge_each_side_is_kept(self):
        out = laplacian_blend(self.inside, self.outside, self.step, 3)
        self.assertTrue(np.allclose(out[:, :24], self.outside[:, :24], atol=2e-3))
        self.assertTrue(np.allclose(out[:, 104:], self.inside[:, 104:], atol=2e-3))

    def test_paste_object_with_levels_keeps_far_pixels(self):
        original = np.stack([self.outside] * 2)
        generated = original + 0.05
        old = np.zeros((2, 128, 128), dtype=bool)
        old[:, 56:72, 56:72] = True
        frames, soft = paste_object(original, generated, old, old, 4, 2, 8, 1, blend_levels=3)
        self.assertTrue(np.allclose(frames[:, :20], original[:, :20], atol=2e-3))
        self.assertTrue(np.allclose(frames[:, 60:68, 60:68], original[:, 60:68, 60:68], atol=1e-3))  # colour-matched redraw


class ShadowAreaTest(unittest.TestCase):
    """A 128x128 road with a car area; the clip has the old car's shadow below it, the redraw does not."""

    def setUp(self):
        self.original = np.full((128, 128, 3), 0.5, dtype=np.float32)
        self.redraw = self.original.copy()
        self.area = np.zeros((128, 128), dtype=bool)
        self.area[40:64, 40:88] = True
        self.original[64:76, 36:92] = 0.25  # the old shadow, touching the area from below

    def test_old_shadow_is_taken_in(self):
        widened = shadow_area(self.area, self.redraw, self.original, 24, 0.06, 2.0)
        self.assertTrue(widened[66:74, 44:84].all())
        self.assertTrue(widened[self.area].all())

    def test_far_and_detached_differences_are_left_alone(self):
        self.original[110:120, 0:20] = 0.2  # far away
        self.original[20:26, 100:120] = 0.2  # near but not touching the area
        widened = shadow_area(self.area, self.redraw, self.original, 24, 0.06, 2.0)
        self.assertFalse(widened[110:120, 0:20].any())
        self.assertFalse(widened[20:26, 100:120].any())

    def test_thin_seams_do_not_count(self):
        original = np.full((128, 128, 3), 0.5, dtype=np.float32)
        original[64, 20:108] = 0.3  # a one-pixel road seam right below the area
        widened = shadow_area(self.area, self.redraw, original, 24, 0.06, 4.0)
        self.assertFalse(widened[70:].any())

    def test_reach_zero_keeps_the_area(self):
        self.assertTrue(np.array_equal(shadow_area(self.area, self.redraw, self.original, 0, 0.06, 2.0), self.area))

    def test_paste_object_removes_the_old_shadow(self):
        original = np.stack([self.original] * 2)
        redraw = np.stack([self.redraw] * 2)
        masks = np.stack([self.area] * 2)
        without, _ = paste_object(original, redraw, masks, masks, 0, 0, 8, 1)
        with_shadow, _ = paste_object(original, redraw, masks, masks, 0, 0, 8, 1, shadow_reach=0.6, shadow_blur=2.0)
        self.assertLess(float(without[:, 66:74, 44:84].mean()), 0.3)
        self.assertGreater(float(with_shadow[:, 66:74, 44:84].mean()), 0.45)


class ObjectAreaTest(unittest.TestCase):
    """A 20 px tall old object at frame 0 that is 40 px tall (closer to the camera) at frame 1."""

    def setUp(self):
        self.masks = np.zeros((2, 160, 160), dtype=bool)
        self.masks[0, 70:90, 60:100] = True
        self.masks[1, 60:100, 40:120] = True

    def test_same_shape_grows_by_the_base_only(self):
        widened, ratio = object_area(self.masks, self.masks[0], 6, 0.6)
        self.assertEqual(ratio, 0.0)
        for t in range(2):
            self.assertTrue(np.array_equal(widened[t], grow_mask(self.masks[t], 6)))

    def test_taller_new_object_scales_with_the_object(self):
        look = self.masks[0].copy()
        look[60:70, 60:100] = True  # the new object reaches 10 px above the old one
        widened, ratio = object_area(self.masks, look, 4, 0.6)
        self.assertAlmostEqual(ratio, 0.5, delta=0.05)
        self.assertTrue(widened[0][look].all())
        # frame 1: the object is twice as tall, so it grows by 4 + about 20 px
        self.assertEqual(mask_height(widened[1]), 40 + 2 * (4 + round(ratio * 40)))

    def test_reach_is_capped(self):
        look = np.ones((160, 160), dtype=bool)
        _, ratio = object_area(self.masks, look, 4, 0.3)
        self.assertEqual(ratio, 0.3)

    def test_minimum_reach_covers_unseen_sides(self):
        widened, ratio = object_area(self.masks, self.masks[0], 0, 0.6, min_ratio=0.2)
        self.assertEqual(ratio, 0.2)
        self.assertEqual(mask_height(widened[1]), 40 + 2 * 8)
        _, no_look = object_area(self.masks, np.zeros((160, 160), dtype=bool), 0, 0.6, min_ratio=0.2)
        self.assertEqual(no_look, 0.2)
        with self.assertRaises(ValueError):
            object_area(self.masks, self.masks[0], 0, 0.1, min_ratio=0.2)

    def test_no_new_object_found_grows_by_the_base(self):
        widened, ratio = object_area(self.masks, np.zeros((160, 160), dtype=bool), 6, 0.6)
        self.assertEqual(ratio, 0.0)
        self.assertTrue(np.array_equal(widened[1], grow_mask(self.masks[1], 6)))

    def test_old_object_missing_in_early_frames_uses_the_first_found(self):
        masks = self.masks.copy()
        masks[0] = False
        look = masks[1].copy()
        look[50:60, 40:120] = True  # 10 px above the 40 px object: ratio 0.25
        widened, ratio = object_area(masks, look, 0, 0.6)
        self.assertFalse(widened[0].any())
        self.assertAlmostEqual(ratio, 0.25, delta=0.03)

    def test_no_old_object_anywhere_changes_nothing(self):
        empty = np.zeros((2, 32, 32), dtype=bool)
        widened, ratio = object_area(empty, np.ones((32, 32), dtype=bool), 6, 0.6)
        self.assertEqual(ratio, 0.0)
        self.assertFalse(widened.any())

    def test_rejects_a_look_mask_of_another_size(self):
        with self.assertRaises(ValueError):
            object_area(self.masks, np.zeros((10, 10), dtype=bool), 6, 0.6)

    def test_widening_leaves_kept_areas_alone(self):
        keep = np.zeros_like(self.masks)
        keep[:, 40:70, 60:100] = True  # a head right above the object, overlapping its top row at frame 1
        widened, _ = object_area(self.masks, self.masks[0], 12, 0.6, keep=keep)
        self.assertFalse(widened[0][40:70, 60:100].any())
        self.assertTrue(widened[1][self.masks[1]].all())  # the object itself stays, even under the head
        self.assertTrue(widened[0][90:100, 60:100].all())  # below the object it still grows


class PasteObjectKeepTest(unittest.TestCase):
    """The redraw changed a face right above a jacket; the face must stay the clip's."""

    def setUp(self):
        self.original = np.full((2, 96, 96, 3), 0.5, dtype=np.float32)
        self.redraw = self.original.copy()
        self.jacket = np.zeros((2, 96, 96), dtype=bool)
        self.jacket[:, 40:80, 30:66] = True
        self.redraw[:, 40:80, 30:66] = 0.9  # the new jacket
        self.redraw[:, 14:40, 36:60] = 0.2  # the face, redrawn differently, touching the jacket
        self.face = np.zeros((2, 96, 96), dtype=bool)
        self.face[:, 14:40, 36:60] = True

    def test_without_keep_the_changed_face_is_taken(self):
        frames, _ = paste_object(self.original, self.redraw, self.jacket, self.jacket, 4, 0, 8, 1, shadow_reach=0.8, shadow_blur=1.0)
        self.assertLess(float(frames[:, 18:34, 40:56].mean()), 0.3)

    def test_with_keep_the_face_stays_and_the_jacket_is_pasted(self):
        frames, soft = paste_object(self.original, self.redraw, self.jacket, self.jacket, 4, 0, 8, 1, shadow_reach=0.8, shadow_blur=1.0,
                                    keep=self.face)
        self.assertTrue(np.allclose(frames[:, 14:40, 36:60], 0.5))
        self.assertFalse(soft[:, 14:40, 36:60].any())
        self.assertTrue(np.allclose(frames[:, 44:76, 34:62], 0.9))

    def test_rejects_a_keep_mask_of_another_size(self):
        with self.assertRaises(ValueError):
            paste_object(self.original, self.redraw, self.jacket, self.jacket, 4, 0, 8, 1, keep=self.face[:1])


if __name__ == "__main__":
    unittest.main()
