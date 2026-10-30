import os
import sys
import tempfile
import unittest

import av
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from object_chain import frame_index_at, frame_times, join_plan, padded_length, parse_starts, window_offsets  # noqa: E402


def write_clip(path, frames, fps):
    with av.open(path, mode="w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i % 256, dtype=np.uint8), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


class PaddedLength(unittest.TestCase):
    def test_short_clips_grow_to_the_minimum(self):
        for n in (1, 30, 89, 145):
            self.assertEqual(padded_length(n, 145), 145)

    def test_longer_clips_round_up_to_8k_plus_1(self):
        self.assertEqual(padded_length(150, 145), 153)
        self.assertEqual(padded_length(146, 145), 153)
        self.assertEqual(padded_length(153, 145), 153)
        for n in range(1, 400):
            p = padded_length(n, 1)
            self.assertTrue(p >= n and (p - 1) % 8 == 0 and p - n < 8, n)


class FrameIndex(unittest.TestCase):
    def test_first_frame_at_or_after(self):
        self.assertEqual(frame_index_at(0, 30), 0)
        self.assertEqual(frame_index_at(4.5, 30), 135)
        self.assertEqual(frame_index_at(4.5, 25), 113)  # 112.5 -> the frame after
        self.assertEqual(frame_index_at(1 / 30, 30), 1)


class Starts(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_starts("0, 4.5,9"), [0.0, 4.5, 9.0])
        for bad in ("", "1, 2", "0, 4, 4", "0, 5, 3"):
            with self.assertRaises(ValueError):
                parse_starts(bad)


class JoinPlan(unittest.TestCase):
    def test_one_window_is_the_clip(self):
        plan = join_plan([0], [10], 10, 8)
        self.assertEqual(plan, [[(0, i, 1.0)] for i in range(10)])

    def test_two_windows_blend_where_the_second_starts(self):
        # window 0: frames 0-149, window 1 from frame 135 (4.5 s at 30 fps) to the end at 180
        plan = join_plan([0, 135], [150, 45], 180, 8)
        self.assertEqual(len(plan), 180)
        self.assertEqual(plan[134], [(0, 134, 1.0)])
        first = plan[135]
        self.assertEqual([p[:2] for p in first], [(0, 135), (1, 0)])
        self.assertAlmostEqual(first[1][2], 1 / 9)
        self.assertEqual([p[:2] for p in plan[142]], [(0, 142), (1, 7)])
        self.assertEqual(plan[143], [(1, 8, 1.0)])
        self.assertEqual(plan[179], [(1, 44, 1.0)])
        for parts in plan:
            self.assertAlmostEqual(sum(w for _, _, w in parts), 1.0)

    def test_blend_never_longer_than_the_overlap(self):
        plan = join_plan([0, 8], [10, 10], 18, 8)  # 2 frames of overlap
        self.assertEqual([p[:2] for p in plan[8]], [(0, 8), (1, 0)])
        self.assertEqual([p[:2] for p in plan[9]], [(0, 9), (1, 1)])
        self.assertEqual(plan[10], [(1, 2, 1.0)])

    def test_a_gap_repeats_the_frame_before(self):
        plan = join_plan([0], [8], 10, 8)
        self.assertEqual(plan[8], plan[7])
        self.assertEqual(plan[9], plan[7])

    def test_first_window_must_start_at_zero(self):
        with self.assertRaises(ValueError):
            join_plan([2], [10], 12, 8)


class Times(unittest.TestCase):
    def test_window_offsets_follow_the_frame_times(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "clip.mp4")
            write_clip(path, 60, 25)
            times, rate = frame_times(path, 0.0, 0.0)
            self.assertEqual(len(times), 60)
            self.assertEqual(float(rate), 25.0)
            self.assertEqual(window_offsets(times, [0.0, 1.0, 1.02]), [0, 25, 26])
            part, _ = frame_times(path, 1.0, 1.0)
            self.assertEqual(len(part), 25)
            self.assertAlmostEqual(part[0], 1.0)


if __name__ == "__main__":
    unittest.main()
