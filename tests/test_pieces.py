import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pieces import frames_produced, legal_length, piece_starts, plan_pieces, plan_segments  # noqa: E402


class PlanSegmentsTest(unittest.TestCase):
    def test_short_clip_is_one_segment(self):
        self.assertEqual(plan_segments(40, 49, 4), [(0, 40)])

    def test_neighbours_share_the_overlap_and_cover_the_clip(self):
        for total in range(1, 800, 7):
            ranges = plan_segments(total, 49, 4)
            self.assertEqual(ranges[0][0], 0); self.assertEqual(ranges[-1][1], total)
            for (s1, e1), (s2, e2) in zip(ranges, ranges[1:]):
                self.assertEqual(e1 - s2, 4, total)
                self.assertLessEqual(e2 - s2, 49)

    def test_no_overlap(self):
        self.assertEqual(plan_segments(100, 40, 0), [(0, 40), (40, 80), (80, 100)])

    def test_rejects_overlap_not_smaller_than_segment(self):
        with self.assertRaises(ValueError):
            plan_segments(100, 4, 4)


class LegalLengthTest(unittest.TestCase):
    def test_rounds_up_to_4n_plus_1(self):
        self.assertEqual(legal_length(66), 69)
        self.assertEqual(legal_length(81), 81)
        self.assertEqual(legal_length(82), 85)

    def test_minimum_is_5(self):
        self.assertEqual(legal_length(1), 5)


class PlanPiecesTest(unittest.TestCase):
    def test_626_frames_matches_the_tested_chain(self):
        lengths = plan_pieces(626, 81)
        self.assertEqual(lengths, [81, 81, 81, 81, 81, 81, 81, 69])
        self.assertEqual(frames_produced(lengths), 629)

    def test_exactly_one_piece(self):
        self.assertEqual(plan_pieces(81, 81), [81])

    def test_short_clip_uses_one_short_piece(self):
        self.assertEqual(plan_pieces(50, 81), [53])

    def test_one_frame_past_a_piece_adds_a_minimal_piece(self):
        self.assertEqual(plan_pieces(82, 81), [81, 5])

    def test_every_plan_covers_the_clip_without_waste(self):
        for total in range(1, 700):
            lengths = plan_pieces(total, 81)
            produced = frames_produced(lengths)
            self.assertGreaterEqual(produced, total)
            self.assertLess(produced - total, 5, total)
            self.assertTrue(all((n - 1) % 4 == 0 and 5 <= n <= 81 for n in lengths), total)

    def test_rejects_illegal_piece_length(self):
        with self.assertRaises(ValueError):
            plan_pieces(100, 80)


class LtxPiecesTest(unittest.TestCase):
    def test_rounds_up_to_8n_plus_1(self):
        self.assertEqual(legal_length(1, 8), 9)
        self.assertEqual(legal_length(17, 8), 17)
        self.assertEqual(legal_length(18, 8), 25)

    def test_337_frames(self):
        lengths = plan_pieces(337, 81, 8)
        self.assertEqual(lengths, [81, 81, 81, 81, 17])
        self.assertEqual(piece_starts(lengths), [0, 80, 160, 240, 320])
        self.assertEqual(frames_produced(lengths), 337)

    def test_every_plan_covers_the_clip_without_waste(self):
        for total in range(1, 700):
            lengths = plan_pieces(total, 81, 8)
            produced = frames_produced(lengths)
            self.assertGreaterEqual(produced, total)
            self.assertLess(produced - total, 9, total)
            self.assertTrue(all((n - 1) % 8 == 0 and 9 <= n <= 81 for n in lengths), total)

    def test_rejects_wan_length_that_is_not_8n_plus_1(self):
        with self.assertRaises(ValueError):
            plan_pieces(100, 77, 8)


class OverlapTest(unittest.TestCase):
    def test_337_frames_with_5_frame_overlap(self):
        lengths = plan_pieces(337, 81, overlap=5)
        self.assertEqual(lengths, [81, 81, 81, 81, 33])
        self.assertEqual(piece_starts(lengths, overlap=5), [0, 76, 152, 228, 304])
        self.assertEqual(frames_produced(lengths, overlap=5), 337)

    def test_every_plan_covers_the_clip_without_waste(self):
        for total in range(1, 700):
            lengths = plan_pieces(total, 81, overlap=5)
            produced = frames_produced(lengths, overlap=5)
            self.assertGreaterEqual(produced, total)
            self.assertLess(produced - total, 5, total)
            starts = piece_starts(lengths, overlap=5)
            self.assertTrue(all(s + n <= produced for s, n in zip(starts, lengths)), total)

    def test_rejects_overlap_that_leaves_no_new_frames(self):
        with self.assertRaises(ValueError):
            plan_pieces(337, 9, overlap=5)


class PieceStartsTest(unittest.TestCase):
    def test_each_piece_starts_on_the_last_frame_of_the_one_before(self):
        lengths = plan_pieces(626, 81)
        starts = piece_starts(lengths)
        for (s1, n1), s2 in zip(zip(starts, lengths), starts[1:]):
            self.assertEqual(s2, s1 + n1 - 1)
        self.assertEqual(starts[0], 0)


if __name__ == "__main__":
    unittest.main()
