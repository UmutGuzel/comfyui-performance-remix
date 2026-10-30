import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_geometry import (compose_similarity, decompose_similarity, fill_missing,  # noqa: E402
                           smooth, smooth_series, square_boxes)


class SimilarityTest(unittest.TestCase):
    def test_round_trip(self):
        m = compose_similarity(1.3, 0.2, 5.0, -7.0)
        np.testing.assert_allclose(decompose_similarity(m), (1.3, 0.2, 5.0, -7.0), atol=1e-5)


class SmoothSeriesTest(unittest.TestCase):
    def test_fills_gaps_and_reduces_jitter(self):
        rng = np.random.default_rng(0)
        truth = np.linspace(0, 10, 60)
        noisy = [(t + rng.normal(0, 1.0), 1.0) if i % 10 else None for i, t in enumerate(truth)]
        out = smooth_series(noisy, 7)
        self.assertEqual(out.shape, (60, 2))
        known = np.array([n[0] for n in noisy if n is not None])
        raw_err = np.abs(known - truth[[i for i, n in enumerate(noisy) if n is not None]]).mean()
        self.assertLess(np.abs(out[:, 0] - truth)[5:-5].mean(), raw_err)

    def test_any_width(self):
        self.assertEqual(fill_missing([(1, 2, 3, 4, 5), None]).shape, (2, 5))


class FillMissingTest(unittest.TestCase):
    def test_interpolates_gaps(self):
        out = fill_missing([(0, 0, 10), None, (10, 20, 30)])
        np.testing.assert_allclose(out[1], [5, 10, 20])

    def test_holds_edges(self):
        out = fill_missing([None, (4, 4, 8), None])
        np.testing.assert_allclose(out[0], [4, 4, 8])
        np.testing.assert_allclose(out[2], [4, 4, 8])

    def test_no_face_raises(self):
        with self.assertRaises(ValueError):
            fill_missing([None, None])


class SmoothTest(unittest.TestCase):
    def test_constant_stays_constant(self):
        values = np.full((9, 3), 7.0)
        np.testing.assert_allclose(smooth(values, 5), values)

    def test_removes_single_frame_jump(self):
        values = np.zeros((7, 3)); values[3] = 7.0
        out = smooth(values, 7)
        self.assertLess(out[3, 0], 1.5)
        self.assertEqual(out.shape, values.shape)

    def test_rejects_even_window(self):
        with self.assertRaises(ValueError):
            smooth(np.zeros((4, 3)), 4)


class SquareBoxesTest(unittest.TestCase):
    def test_fixed_side_from_median(self):
        track = np.array([[100, 100, 50], [110, 100, 60], [120, 100, 70]], dtype=float)
        boxes = square_boxes(track, 480, 848, 2.0)
        self.assertTrue(all(b[2] == 120 for b in boxes))

    def test_boxes_stay_inside_the_frame(self):
        track = np.array([[5, 5, 60], [475, 840, 60]], dtype=float)
        for x1, y1, side in square_boxes(track, 480, 848, 2.4):
            self.assertGreaterEqual(x1, 0); self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x1 + side, 480); self.assertLessEqual(y1 + side, 848)

    def test_side_capped_at_shorter_frame_side(self):
        track = np.array([[240, 424, 400]], dtype=float)
        self.assertEqual(square_boxes(track, 480, 848, 2.4)[0][2], 480)


if __name__ == "__main__":
    unittest.main()
