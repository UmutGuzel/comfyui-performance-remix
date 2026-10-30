import os
import sys
import tempfile
import unittest

import av
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip_frames import decode_frames, read_audio, scaled_size  # noqa: E402


def write_clip(path, frames=48, fps=24, width=320, height=180, audio_seconds=0.0):
    """A test clip whose frame i is flat gray at 4 * i, with an optional 440 Hz tone."""
    with av.open(path, mode="w") as out:
        video = out.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        video.options = {"crf": "0"}
        audio = out.add_stream("aac", rate=48000, layout="mono") if audio_seconds else None
        for i in range(frames):
            img = np.full((height, width, 3), 4 * i, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode(None):
            out.mux(packet)
        if audio is not None:
            t = np.arange(int(48000 * audio_seconds)) / 48000
            tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[None, :]
            frame = av.AudioFrame.from_ndarray(tone, format="fltp", layout="mono")
            frame.sample_rate, frame.pts = 48000, 0
            for packet in audio.encode(frame):
                out.mux(packet)
            for packet in audio.encode(None):
                out.mux(packet)


class ScaledSizeTest(unittest.TestCase):
    def test_shorter_side_scales_down_and_keeps_the_shape(self):
        self.assertEqual(scaled_size(1080, 1920, "shorter side", 720, 0, 0), (720, 1280, 720, 1280))
        self.assertEqual(scaled_size(1920, 1080, "shorter side", 1024, 0, 0), (1820, 1024, 1820, 1024))

    def test_shorter_side_never_scales_up(self):
        self.assertEqual(scaled_size(640, 360, "shorter side", 720, 0, 0), (640, 360, 640, 360))

    def test_width_x_height_covers_then_cuts(self):
        self.assertEqual(scaled_size(1080, 1920, "width x height", 0, 704, 1248), (704, 1252, 704, 1248))
        # a landscape clip into a portrait size: scaled up to cover the height, cut in width
        self.assertEqual(scaled_size(1920, 1080, "width x height", 0, 704, 1248), (2219, 1248, 704, 1248))

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            scaled_size(0, 10, "shorter side", 720, 0, 0)
        with self.assertRaises(ValueError):
            scaled_size(10, 10, "fit", 720, 0, 0)


class DecodeFramesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.clip = os.path.join(cls.dir.name, "clip.mp4")
        write_clip(cls.clip, audio_seconds=2.0)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_range_and_size(self):
        frames, rate = decode_frames(self.clip, 0.5, 1.0, "shorter side", shorter_side=90)
        self.assertEqual(rate, 24)
        self.assertEqual(len(frames), 24)
        self.assertEqual(frames[0].shape, (90, 160, 3))
        self.assertEqual(frames[0].dtype, np.uint8)
        # frame 12 is gray 48, frame 35 is gray 140
        self.assertAlmostEqual(float(frames[0].mean()), 48, delta=2)
        self.assertAlmostEqual(float(frames[-1].mean()), 140, delta=2)

    def test_whole_clip_when_duration_is_zero(self):
        frames, _ = decode_frames(self.clip, 0, 0, "shorter side", shorter_side=720)
        self.assertEqual(len(frames), 48)
        self.assertEqual(frames[0].shape, (180, 320, 3))  # not scaled up

    def test_exact_size(self):
        frames, _ = decode_frames(self.clip, 0, 0.5, "width x height", width=64, height=96)
        self.assertEqual(len(frames), 12)
        self.assertEqual(frames[0].shape, (96, 64, 3))

    def test_empty_range(self):
        with self.assertRaises(ValueError):
            decode_frames(self.clip, 5.0, 1.0, "shorter side")

    def test_audio_range(self):
        samples, rate, layout = read_audio(self.clip, 0.5, 1.5)
        self.assertEqual((rate, layout), (48000, "mono"))
        self.assertEqual(samples.shape, (1, 48000))
        self.assertIsNone(read_audio(self.clip, 5.0, 6.0))


if __name__ == "__main__":
    unittest.main()
