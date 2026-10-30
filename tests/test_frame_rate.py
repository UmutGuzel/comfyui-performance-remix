import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame_rate import resample_indices  # noqa: E402


class ResampleIndicesTest(unittest.TestCase):
    def test_same_rate_keeps_every_frame(self):
        self.assertEqual(resample_indices(10, 24, 24), list(range(10)))

    def test_25_to_24_drops_one_frame_per_second(self):
        indices = resample_indices(250, 25, 24)
        self.assertEqual(len(indices), 240)
        self.assertEqual(len(set(indices)), 240)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 249)

    def test_30_to_24_keeps_duration(self):
        indices = resample_indices(375, 29.97, 24)
        self.assertEqual(len(indices), 300)
        self.assertTrue(all(b > a for a, b in zip(indices, indices[1:])))
        self.assertLessEqual(indices[-1], 374)

    def test_lower_to_higher_rate_repeats_frames(self):
        indices = resample_indices(16, 16, 24)
        self.assertEqual(len(indices), 24)
        self.assertEqual(max(indices), 15)

    def test_single_frame(self):
        self.assertEqual(resample_indices(1, 30, 24), [0])

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            resample_indices(0, 25, 24)
        with self.assertRaises(ValueError):
            resample_indices(10, 0, 24)


if __name__ == "__main__":
    unittest.main()
